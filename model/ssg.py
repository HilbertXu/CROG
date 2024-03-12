import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2

from collections import OrderedDict
from math import sqrt
import numpy as np
from itertools import product
from utils.box_utils import match, crop, ones_crop, make_anchors



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       self.norm_layer(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample, self.norm_layer)]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        print(f'\nBackbone is initiated with {path}.\n')


class PredictionModule(nn.Module):
    def __init__(self, cfg, coef_dim=32):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim
        self.gr_coef_dim = coef_dim / 2

        self.upfeature = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))
        self.bbox_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        self.coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        if cfg.with_grasp_masks:
            # Generate 4 grasp masks
            self.grasp_coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim * 4,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())


    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)

        grasp_coef_layer = self.grasp_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4, self.coef_dim)

        
        return conf, box, coef, grasp_coef_layer


class ProtoNet(nn.Module):
    def __init__(self, coef_dim):
        super().__init__()
        self.proto1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proto2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
        self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                        nn.ReLU(inplace=True)) for _ in self.in_channels])

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True)),
                                                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True))])

        self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

    def forward(self, backbone_outs):
        p5_1 = self.lat_layers[2](backbone_outs[2])
        p5_upsample = self.upsample_module[1](p5_1)

        p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
        p4_upsample = self.upsample_module[0](p4_1)

        p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

        p5 = self.pred_layers[2](p5_1)
        p4 = self.pred_layers[1](p4_1)
        p3 = self.pred_layers[0](p3_1)

        p6 = self.downsample_layers[0](p5)
        p7 = self.downsample_layers[1](p6)

        return p3, p4, p5, p6, p7


class SSG(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        if cfg.backbone == "resnet":
            self.backbone = ResNet(layers=cfg.resnet_layers)
            if cfg.path_to_pretrained_resnet and not cfg.resume:
                self.backbone.init_backbone(cfg.path_to_pretrained_resnet)
            if cfg.with_depth:
                with torch.no_grad():
                    # Add extra depth channel for net.backbone
                    weight = self.backbone.conv1.weight.clone()
                    self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    self.backbone.conv1.weight[:,:3] = weight
            self.fpn = FPN(in_channels=cfg.fpn_in_channels)

        else:
            raise NotImplementedError
        
        self.proto_net = ProtoNet(coef_dim=cfg.num_protos)
        self.prediction_layers = PredictionModule(cfg, coef_dim=cfg.num_protos)

        self.anchors = []
        scales = [int(cfg.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in cfg.anchor_strides]
        for i, size in enumerate(fpn_fm_shape):
            self.anchors += make_anchors(cfg, size, size, scales[i])

        if self.training:
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes, kernel_size=1)

        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    
    def forward(self, data_dict):
        if self.cfg.with_depth:
            img = torch.cat([data_dict["rgb"], data_dict["depth"]], dim=1)
        else:
            img = data_dict["rgb"]
        x = self.backbone(img)
        x = self.fpn(x[1:4])
        protos = self.proto_net(x[0])
        protos = protos.permute(0, 2, 3, 1).contiguous()

        class_pred, box_pred, ins_coef_pred, grasp_coef_pred = [], [], [], []

        for aa in x:
            class_p, box_p, ing_coef_p, grasp_coef_p = self.prediction_layers(aa)
            class_pred.append(class_p)
            box_pred.append(box_p)
            ins_coef_pred.append(ing_coef_p)
            grasp_coef_pred.append(grasp_coef_p)
        
        class_pred = torch.cat(class_pred, dim=1)
        box_pred = torch.cat(box_pred, dim=1)
        ins_coef_pred = torch.cat(ins_coef_pred, dim=1)
        grasp_coef_pred = torch.cat(grasp_coef_pred, dim=1)
        
        output_dict = {
                "anchors": self.anchors,
                "protos": protos,
                "cls_pred": F.softmax(class_pred, -1),
                "box_pred": box_pred,
                "ins_coef_pred": ins_coef_pred,
                "grasp_coef_pred": grasp_coef_pred,
            }

        if self.training:
            seg_pred = self.semantic_seg_conv(x[0])
            loss_dict = self.compute_loss(
                class_pred, 
                box_pred, 
                ins_coef_pred, 
                grasp_coef_pred, 
                protos, seg_pred, 
                data_dict, output_dict
            )
            return output_dict, loss_dict
        else:
            return output_dict
    


    def compute_loss(
        self, 
        class_pred, box_pred, ins_coef_pred, grasp_coef_pred, 
        protos, seg_pred, 
        data_dict, output_dict
    ):
        device = class_pred.device
        class_gt = [None] * len(data_dict["bboxes"])
        batch_size = box_pred.size(0)

        if isinstance(self.anchors, list):
            self.anchors = torch.tensor(self.anchors, device=device).reshape(-1, 4)

        num_anchors = self.anchors.shape[0]

        all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)
        anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        for i in range(batch_size):
            box_gt = data_dict["bboxes"][i][:, :-1]
            class_gt[i] = data_dict["bboxes"][i][:, -1].long()
            all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                                  self.anchors, class_gt[i])

        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # anchor_max_gt: the corresponding max IoU gt box for each anchor
        # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
        assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and \
               (not anchor_max_i.requires_grad), 'Incorrect computation graph, check the grad.'

        # only compute losses from positive samples
        pos_bool = conf_gt > 0

        loss_c = self.category_loss(class_pred, conf_gt, pos_bool)
        loss_b = self.box_loss(box_pred, all_offsets, pos_bool)
        if self.cfg.intermidiate_output:
            loss_m = self.lincomb_mask_loss(ins_coef_pred, protos, data_dict["ins_masks"], pos_bool, anchor_max_i, anchor_max_gt, output_dict)
        else:
            loss_m = self.lincomb_mask_loss(ins_coef_pred, protos, data_dict["ins_masks"], pos_bool, anchor_max_i, anchor_max_gt)
        loss_g = self.lincomb_grasp_masks_loss(grasp_coef_pred, protos, data_dict["grasp_masks"], pos_bool, anchor_max_i, anchor_max_gt)
        loss_s = self.semantic_seg_loss(seg_pred, data_dict["sem_mask"], data_dict["labels"])

        return {
            "loss_cls": loss_c,
            "loss_box": loss_b,
            "loss_ins": loss_m,
            "loss_sem": loss_s,
            "loss_qua": loss_g["qua"],
            "loss_sin": loss_g["sin"],
            "loss_cos": loss_g["cos"],
            "loss_wid": loss_g["wid"]
        }

    def category_loss(self, class_p, conf_gt, pos_bool, np_ratio=3):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)

        batch_conf_max = batch_conf.max()

        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

        # Hard Negative Mining
        mark = mark.reshape(class_p.size(0), -1)
        mark[pos_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = pos_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
        neg_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[pos_bool] = 0
        neg_bool[conf_gt < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        class_p_mined = class_p[(pos_bool + neg_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_mined = conf_gt[(pos_bool + neg_bool)]

        return self.cfg.alpha_conf * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum') / num_pos.sum()

    
    def box_loss(self, box_p, all_offsets, pos_bool):
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]
        pos_offsets = all_offsets[pos_bool, :]

        return self.cfg.alpha_bbox * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos

    

    def lincomb_mask_loss(self, ins_coef_p, protos, ins_masks_gt, pos_bool, anchor_max_i, anchor_max_gt, output_dict=None):
        proto_h, proto_w = protos.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0

        inter_mask_p = []
        inter_mask_gt = []

        for i in range(ins_coef_p.shape[0]):
            downsampled_masks = F.interpolate(ins_masks_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            pos_coef = ins_coef_p[i][pos_bool[i]]

            if pos_anchor_i.size(0) == 0:
                continue
            
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_anchor_i = pos_anchor_i[select]
                pos_anchor_box = pos_anchor_box[select]
            
            num_pos = pos_coef.size(0)
            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

            mask_p = torch.sigmoid(protos[i] @ pos_coef.t())
            mask_p = crop(mask_p, pos_anchor_box)

            if output_dict is not None and self.cfg.intermidiate_output:
                inter_mask_p.append(mask_p.data)
                inter_mask_gt.append(pos_mask_gt.data)

            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')

            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos
            
            loss_m += torch.sum(mask_loss)
        
        if len(inter_mask_p) > 0 and output_dict is not None:
            inter_mask_p = torch.cat(inter_mask_p, dim=-1).permute(2, 0, 1)
            inter_mask_gt = torch.cat(inter_mask_gt, dim=-1).permute(2, 0, 1)

            output_dict["inter_mask_p"] = inter_mask_p
            output_dict["inter_mask_gt"] = inter_mask_gt


        return self.cfg.alpha_ins * loss_m / proto_h / proto_w / total_pos_num

    

    def lincomb_grasp_masks_loss(self, grasp_coef_p, protos, grasp_masks_gt, pos_bool, anchor_max_i, anchor_max_gt):
        proto_h, proto_w = protos.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_dict = {
            "qua": 0.0,
            "sin": 0.0,
            "cos": 0.0,
            "wid": 0.0
        }
        for i in range(grasp_coef_p.shape[0]):
            for idx, key in enumerate(grasp_masks_gt.keys()):
                downsampled_masks = F.interpolate(grasp_masks_gt[key][i].unsqueeze(0), (proto_h, proto_w), mode='bilinear', align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
                pos_anchor_i = anchor_max_i[i][pos_bool[i]]
                pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
                pos_coef = grasp_coef_p[i, pos_bool[i], idx, :]

                if pos_anchor_i.size(0) == 0:
                    continue
                
                old_num_pos = pos_coef.size(0)
                if old_num_pos > self.cfg.masks_to_train:
                    perm = torch.randperm(pos_coef.size(0))
                    select = perm[:self.cfg.masks_to_train]
                    pos_coef = pos_coef[select]
                    pos_anchor_i = pos_anchor_i[select]
                    pos_anchor_box = pos_anchor_box[select]
                
                num_pos = pos_coef.size(0)

                pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

                mask_p = torch.sigmoid(protos[i] @ pos_coef.t())
                if key == "cos":
                    mask_p = ones_crop(mask_p, pos_anchor_box)
                else:
                    mask_p = crop(mask_p, pos_anchor_box) 
                    # if key == "qua":
                    #     for j in range(mask_p.shape[-1]):
                    #         m_p = (mask_p[:, :, j].data.cpu().numpy() * 255).astype(int)
                    #         m_g = (pos_mask_gt[:, :, j].data.cpu().numpy() * 255).astype(int)
                    #         cv2.imwrite(f"./test/{j}_qua_p.png", m_p)
                    #         cv2.imwrite(f"./test/{j}_qua_g.png", m_g)
                loss = F.smooth_l1_loss(mask_p, pos_mask_gt, reduction="none")
                anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
                loss = loss.sum(dim=(0, 1)) / anchor_area
                
                if old_num_pos > num_pos:
                    loss *= old_num_pos / num_pos
                
                loss_dict[key] += self.cfg.alpha_grasp * torch.sum(loss) / proto_h /proto_w / total_pos_num
        
        return loss_dict
    

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0).unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
            for j in range(downsampled_masks.size(0)):
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return self.cfg.alpha_sem * loss_s / mask_h / mask_w / batch_size