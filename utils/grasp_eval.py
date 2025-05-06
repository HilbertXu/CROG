import cv2
import numpy as np
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from .box_utils import crop, box_iou


@torch.no_grad()
def draw_lincomb(ids_p, proto_data, masks, img_name, target_dir=None):
    for kdx in range(masks.shape[0]):
        target_id = int(ids_p[kdx])
        # jdx = kdx + -1
        coeffs = masks[kdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))

        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4, 8)
        p_h, p_w, _ = proto_data.size()
        arr_img = np.zeros([p_h * arr_h, p_w * arr_w])
        arr_run = np.zeros([p_h * arr_h, p_w * arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = (1 / (1 + np.exp(-running_total)))

                arr_img[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (proto_data[:, :, idx[i]] / torch.max(
                    proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (running_total_nonlin > 0.5).astype(np.float)

        arr_img = ((arr_img + 1) * 127.5).astype('uint8')
        arr_img = cv2.applyColorMap(arr_img, cv2.COLORMAP_WINTER)
        if target_dir is None:
            cv2.imwrite(f'results/ocid/lincomb_{img_name}', arr_img)
        else:
            if not os.path.exists(f'{target_dir}/{target_id}'):
                os.makedirs(f'{target_dir}/{target_id}')

            cv2.imwrite(f'{target_dir}/{target_id}/lincomb_{img_name}', arr_img)


@torch.no_grad()
def fast_nms(cfg, box_pred_kept, cls_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept):
    cls_pred_kept, idx = cls_pred_kept.sort(1, descending=True)

    idx = idx[:, :cfg.top_k]
    cls_pred_kept = cls_pred_kept[:, :cfg.top_k]
    num_classes, num_dets = idx.size()

    box_pred_kept = box_pred_kept[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)
    ins_coef_pred_kept = ins_coef_pred_kept[idx.reshape(-1), :].reshape(num_classes, num_dets, cfg.num_protos)
    grasp_coef_pred_kept = grasp_coef_pred_kept[idx.reshape(-1), :].reshape(num_classes, num_dets, 4, cfg.num_protos)

    # Calculate IoU between predicted boxes
    iou = box_iou(box_pred_kept, box_pred_kept)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_pred_kept.device)[:, None].expand_as(keep)
    class_ids = class_ids[keep]
    cls_pred_kept = cls_pred_kept[keep]
    box_pred_kept = box_pred_kept[keep]
    ins_coef_pred_kept = ins_coef_pred_kept[keep]
    grasp_coef_pred_kept = grasp_coef_pred_kept[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    cls_pred_kept, idx = cls_pred_kept.sort(0, descending=True)
    idx = idx[:cfg.max_detections]
    cls_pred_kept = cls_pred_kept[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_pred_kept = box_pred_kept[idx]
    ins_coef_pred_kept = ins_coef_pred_kept[idx]
    grasp_coef_pred_kept = grasp_coef_pred_kept[idx]

    
    return class_ids, cls_pred_kept, box_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept



# IMPORTANT
# For batch_size=1 only
@torch.no_grad()
def ssg_post_processing(cfg, output_dict, data_dict):
    ori_size = data_dict["ori_size"]
    ori_h, ori_w = ori_size
    input_size = max(ori_h, ori_w)
    protos = output_dict["protos"].squeeze()
    cls_pred = output_dict["cls_pred"].squeeze()
    box_pred = output_dict["box_pred"].squeeze()
    ins_coef_pred = output_dict["ins_coef_pred"].squeeze()
    grasp_coef_pred = output_dict["grasp_coef_pred"].squeeze()
    anchors = torch.tensor(output_dict["anchors"], device=protos.device).reshape(-1, 4).squeeze()
    # print("anchors", anchors.shape)
    # print("cls_pred", cls_pred.shape)
    B, N = cls_pred.shape[:2]

    cls_pred = cls_pred.transpose(1, 0).contiguous()

    # Exclude the background class
    cls_pred = cls_pred[1:, :]
    # get the max score class of 19248 predicted boxes
    cls_pred_max, _ = torch.max(cls_pred, dim=0)
    # print("cls_pred_max", cls_pred_max.shape)

    # filter predicted boxes according the class score
    keep = (cls_pred_max > cfg.nms_score_thre) # [N_anchors]
    # print(anchors.shape, cls_pred.shape, box_pred.shape, ins_coef_pred.shape, grasp_coef_pred.shape)
    # print(keep.shape)
    anchors_kept = anchors[keep, :]
    cls_pred_kept = cls_pred[:, keep]
    box_pred_kept = box_pred[keep, :]
    ins_coef_pred_kept = ins_coef_pred[keep, :]
    grasp_coef_pred_kept = grasp_coef_pred[keep, :]

    # decode boxes
    box_pred_kept = torch.cat((anchors_kept[:, :2] + box_pred_kept[:, :2] * 0.1 * anchors_kept[:, 2:],
                          anchors_kept[:, 2:] * torch.exp(box_pred_kept[:, 2:] * 0.2)), 1)
    box_pred_kept[:, :2] -= box_pred_kept[:, 2:] / 2
    box_pred_kept[:, 2:] += box_pred_kept[:, :2]
    box_pred_kept = torch.clip(box_pred_kept, min=0., max=1.)

    # Fast NMS
    class_ids, cls_pred_kept, box_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept = fast_nms(cfg, box_pred_kept, cls_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept)

    keep = (cls_pred_kept > 0.3)
    if not keep.any():
        print("No valid instance")
    else:
        class_ids = class_ids[keep]
        cls_pred_kept = cls_pred_kept[keep]
        box_pred_kept = box_pred_kept[keep]
        ins_coef_pred_kept = ins_coef_pred_kept[keep]   
        grasp_coef_pred_kept = grasp_coef_pred_kept[keep]

    class_ids = (class_ids + 1)
    class_ids = class_ids.cpu().numpy()
    
    # if cfg.vis_protos:
    #     ones_coef = torch.ones(pos_coef_p.shape).float().cuda()
    #     # print("ProtoTypes")
    #     draw_lincomb(ids_p, proto_p, ones_coef, "prototypes.png", target_dir)

    #     # print("Semantic")
    #     draw_lincomb(ids_p, proto_p, coef_p, "cogr-sem.png", target_dir)
    #     # print("Grasp pos")
    #     draw_lincomb(ids_p, proto_p, pos_coef_p, "cogr-gr-pos.png", target_dir)
    #     # print("Grasp sin")
    #     draw_lincomb(ids_p, proto_p, sin_coef_p, "cogr-gr-sin.png", target_dir)
    #     # print("Grasp cos")
    #     draw_lincomb(ids_p, proto_p, cos_coef_p, "cogr-gr-cos.png", target_dir)
    #     # print("Grasp wid")
    #     draw_lincomb(ids_p, proto_p, wid_coef_p, "cogr-gr-wid.png", target_dir)

    ins_masks = torch.sigmoid(torch.matmul(protos, ins_coef_pred_kept.t())).contiguous()
    grasp_qua_masks = torch.sigmoid(torch.matmul(protos, grasp_coef_pred_kept[:, 0, :].t())).contiguous()
    grasp_sin_masks = torch.matmul(protos, grasp_coef_pred_kept[:, 1, :].t()).contiguous()
    grasp_cos_masks = torch.matmul(protos, grasp_coef_pred_kept[:, 2, :].t()).contiguous()
    grasp_wid_masks = torch.sigmoid(torch.matmul(protos, grasp_coef_pred_kept[:, 3, :].t())).contiguous()

    ins_masks = crop(ins_masks, box_pred_kept).permute(2,0,1)
    grasp_qua_masks = crop(grasp_qua_masks, box_pred_kept).permute(2,0,1)
    grasp_sin_masks = crop(grasp_sin_masks, box_pred_kept).permute(2,0,1)
    grasp_cos_masks = crop(grasp_cos_masks, box_pred_kept).permute(2,0,1)
    grasp_wid_masks = crop(grasp_wid_masks, box_pred_kept).permute(2,0,1)

    ins_masks = F.interpolate(ins_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    ins_masks.gt_(0.5)
    grasp_qua_masks = F.interpolate(grasp_qua_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    grasp_sin_masks = F.interpolate(grasp_sin_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    grasp_cos_masks = F.interpolate(grasp_cos_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    grasp_wid_masks = F.interpolate(grasp_wid_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)

    ins_masks = ins_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_qua_masks = grasp_qua_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_sin_masks = grasp_sin_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_cos_masks = grasp_cos_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_wid_masks = grasp_wid_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()

    grasp_ang_masks = []
    for i in range(ins_masks.shape[0]):
        grasp_qua_masks[i] = gaussian(grasp_qua_masks[i], 2.0, preserve_range=True)
        ang_mask = (np.arctan2(grasp_sin_masks[i], grasp_cos_masks[i]) / 2.0)
        grasp_ang_masks.append(ang_mask)
    grasp_ang_masks = np.asarray(grasp_ang_masks)
    scale = np.array([ori_w, ori_w, ori_w, ori_w])
    box_pred_kept = box_pred_kept.cpu().numpy() * scale

    ins_grasp_rects_top1 = []
    ins_grasp_rects_top5 = []
    for i in range(ins_masks.shape[0]):
        grasps_top1, _ = detect_grasps(grasp_qua_masks[i], grasp_sin_masks[i], grasp_cos_masks[i], grasp_wid_masks[i], 1)
        grasps_top5, _ = detect_grasps(grasp_qua_masks[i], grasp_sin_masks[i], grasp_cos_masks[i], grasp_wid_masks[i], 5)
        ins_grasp_rects_top1.append(grasps_top1)
        ins_grasp_rects_top5.append(grasps_top5)
    

    return {
        "cls": class_ids,
        "bboxes": box_pred_kept,
        "ins_masks": ins_masks,
        "grasps_top1": ins_grasp_rects_top1,
        "grasps_top5": ins_grasp_rects_top5,
        "grasp_masks": (grasp_qua_masks, grasp_ang_masks, grasp_wid_masks)
    }




def visualization(img, mask, grasp_masks, grasps, text, save_path=None):
    grasp_qua_mask, grasp_ang_mask, grasp_wid_mask = grasp_masks
    
    fig = plt.figure(figsize=(25, 10))
    
    # draw grasp rectangles in image
    tmp = img.copy()
    for rect in grasps:
        center_x, center_y, width, height, theta = rect
        box = ((center_x, center_y), (width, height), -(theta+180))
        box = cv2.boxPoints(box)
        box = np.intp(box)
        ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in box]
        tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
        tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
        tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
        tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
    # tmp = cv2.rectangle(tmp, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
    tmp = cv2.putText(tmp, text, (0,10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
    
    # draw predicted instance mask in image
    msk_img = (img * 0.3).astype(np.uint8).copy()
    mask = mask.astype(np.uint8)
    msk_img[mask, 0] = 255
    
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(img/255.)
    ax.set_title('RGB')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(tmp/255.)
    ax.set_title('predicted grasps')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(mask)
    ax.set_title('predicted instance mask')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_qua_mask, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp quality')
    ax.axis('off')
    plt.colorbar(plot)
    
    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_ang_mask, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp Angle')
    ax.axis('off')
    plt.colorbar(plot)
    
    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_wid_mask, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp Width')
    ax.axis('off')
    plt.colorbar(plot)
    
    plt.suptitle(f"{text}", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path)
    

def detect_grasps(grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask, num_grasps=5):
    grasps = []
    max_width = 100
    local_max = peak_local_max(grasp_quality_mask, min_distance=2, threshold_abs=0.4, num_peaks=num_grasps)
    grasp_angle_mask = (np.arctan2(grasp_sin_mask, grasp_cos_mask) / 2.0)
    
    for p_array in local_max:
        grasp_point = tuple(p_array)
        grasp_angle = grasp_angle_mask[grasp_point] / np.pi * 180
        grasp_width = grasp_wid_mask[grasp_point]

        grasps.append([float(grasp_point[1]), float(grasp_point[0]), grasp_width*max_width, 20, grasp_angle])
    
    return grasps, grasp_angle_mask


def calculate_iou(rect_p, rect_gt, shape=(480, 640), angle_threshold=30):
    if abs(rect_p[4] - rect_gt[4]) > angle_threshold and abs(rect_p[4] + rect_gt[4]) > angle_threshold:
        return 0
    
    center_x, center_y, w_rect, h_rect, theta, _ = rect_gt
    gt_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
    gt_box = cv2.boxPoints(gt_r_rect)
    gt_box = np.int0(gt_box)
    rr1, cc1 = polygon(gt_box[:, 0], gt_box[:,1], shape)

    mask_rr = rr1 < shape[1]
    rr1 = rr1[mask_rr]
    cc1 = cc1[mask_rr]

    mask_cc = cc1 < shape[0]
    cc1 = cc1[mask_cc]
    rr1 = rr1[mask_cc]

    center_x, center_y, w_rect, h_rect, theta = rect_p
    p_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
    p_box = cv2.boxPoints(p_r_rect)
    p_box = np.int0(p_box)
    rr2, cc2 = polygon(p_box[:, 0], p_box[:,1], shape)

    mask_rr = rr2 < shape[1]
    rr2 = rr2[mask_rr]
    cc2 = cc2[mask_rr]

    mask_cc = cc2 < shape[0]
    cc2 = cc2[mask_cc]
    rr2 = rr2[mask_cc]

    area = np.zeros(shape)
    area[cc1, rr1] += 1
    area[cc2, rr2] += 1

    union = np.sum(area > 0)
    intersection = np.sum(area == 2)

    if union <= 0:
        return 0
    else:
        return intersection / union


def calculate_max_iou(rects_p, rects_gt):
    max_iou = 0
    for rect_gt in rects_gt:
        for rect_p in rects_p:
            iou = calculate_iou(rect_p, rect_gt)
            # print("==============================")
            # print(rect_p, rect_gt, iou)
            if iou > max_iou:
                max_iou = iou
    return max_iou


def calculate_jacquard_index(grasp_preds, grasp_targets, iou_threshold=0.25):
    j_index = 0
    grasp_preds = np.asarray(grasp_preds)
    grasp_targets = np.asarray(grasp_targets)

    grasp_targets[:, 3] = 20
    grasp_targets[:, 2] = np.clip(grasp_targets[:, 2], 0, 100)
    
    iou = calculate_max_iou(grasp_preds, grasp_targets)
    if iou > iou_threshold:
        j_index = 1
    
    return j_index