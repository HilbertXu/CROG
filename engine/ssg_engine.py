import os
import time

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU, get_seg_image)
from utils.grasp_eval import (ssg_post_processing, detect_grasps, calculate_iou, calculate_max_iou, calculate_jacquard_index, visualization)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    cls_loss_meter = AverageMeter('Loss_cls', ':2.4f')
    box_loss_meter = AverageMeter('Loss_box', ':2.4f')
    ins_loss_meter = AverageMeter('Loss_ins', ':2.4f')
    sem_loss_meter = AverageMeter('Loss_sem', ':2.4f')
    qua_loss_metter = AverageMeter('Loss_qua', ':2.4f')
    sin_loss_metter = AverageMeter('Loss_sin', ':2.4f')
    cos_loss_metter = AverageMeter('Loss_cos', ':2.4f')
    wid_loss_metter = AverageMeter('Loss_wid', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')

    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr, 
            cls_loss_meter, box_loss_meter, ins_loss_meter, sem_loss_meter,
            qua_loss_metter, sin_loss_metter, cos_loss_metter, wid_loss_metter, 
            iou_meter, pr_meter
        ],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))
    
    model.train()
    time.sleep(2)
    end = time.time()

    for i, data_dict in enumerate(train_loader):
        data_dict["rgb"] = data_dict["rgb"].cuda(non_blocking=True)
        data_dict["depth"] = data_dict["depth"].cuda(non_blocking=True)
        data_dict["labels"] = [x.cuda(non_blocking=True) for x in data_dict["labels"]]
        data_dict["bboxes"] = [x.cuda(non_blocking=True) for x in data_dict["bboxes"]]
        data_dict["sem_mask"] = [x.cuda(non_blocking=True) for x in data_dict["sem_mask"]]
        data_dict["ins_masks"] = [x.cuda(non_blocking=True) for x in data_dict["ins_masks"]]
        data_dict["grasp_masks"]["qua"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["qua"]]
        data_dict["grasp_masks"]["sin"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["sin"]]
        data_dict["grasp_masks"]["cos"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["cos"]]
        data_dict["grasp_masks"]["wid"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["wid"]]
        
        output_dict, loss_dict = model(data_dict)
        loss = 0.0
        for k, v in loss_dict.items():
            loss += v

        optimizer.zero_grad()
        loss.backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        iou, pr5 = trainMetricGPU(output_dict["inter_mask_p"], output_dict["inter_mask_gt"], 0.35, 0.5, sigmoid=False)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        cls_loss_meter.update(loss_dict["loss_cls"].item(), data_dict["rgb"].size(0))
        box_loss_meter.update(loss_dict["loss_box"].item(), data_dict["rgb"].size(0))
        ins_loss_meter.update(loss_dict["loss_ins"].item(), data_dict["rgb"].size(0))
        sem_loss_meter.update(loss_dict["loss_sem"].item(), data_dict["rgb"].size(0))
        qua_loss_metter.update(loss_dict["loss_qua"].item(), data_dict["rgb"].size(0))
        sin_loss_metter.update(loss_dict["loss_sin"].item(), data_dict["rgb"].size(0))
        cos_loss_metter.update(loss_dict["loss_cos"].item(), data_dict["rgb"].size(0))
        wid_loss_metter.update(loss_dict["loss_wid"].item(), data_dict["rgb"].size(0))
        iou_meter.update(iou, data_dict["rgb"].size(0)) #@TODO add iou & pr evaluation
        pr_meter.update(pr5, data_dict["rgb"].size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)

            if dist.get_rank() in [-1, 0]:
                wandb.log(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss_cls": cls_loss_meter.val,
                        "training/loss_box": box_loss_meter.val,
                        "training/loss_ins": ins_loss_meter.val,
                        "training/loss_sem": sem_loss_meter.val,
                        "training/loss_qua": qua_loss_metter.val,
                        "training/loss_sin": sin_loss_metter.val,
                        "training/loss_cos": cos_loss_metter.val,
                        "training/loss_wid": wid_loss_metter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                    },
                    step=epoch * len(train_loader) + (i + 1))



@torch.no_grad()
def validate(val_loader, model, epoch, args):
    model.eval()
    time.sleep(2)

    num_grasps = [1,5]
    J_index = [0, 0]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]
    
    

    for i, data_dict in enumerate(val_loader):

        data_dict["rgb"] = data_dict["rgb"].cuda(non_blocking=True)
        data_dict["depth"] = data_dict["depth"].cuda(non_blocking=True)
        data_dict["labels"] = [x.cuda(non_blocking=True) for x in data_dict["labels"]]
        data_dict["bboxes"] = [x.cuda(non_blocking=True) for x in data_dict["bboxes"]]
        data_dict["sem_mask"] = [x.cuda(non_blocking=True) for x in data_dict["sem_mask"]]
        data_dict["ins_masks"] = [x.cuda(non_blocking=True) for x in data_dict["ins_masks"]]
        data_dict["grasp_masks"]["qua"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["qua"]]
        data_dict["grasp_masks"]["sin"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["sin"]]
        data_dict["grasp_masks"]["cos"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["cos"]]
        data_dict["grasp_masks"]["wid"] = [x.cuda(non_blocking=True) for x in data_dict["grasp_masks"]["wid"]]
        
        output_dict = model(data_dict)
        post_dict = ssg_post_processing(args, output_dict, data_dict)

        for idx, num_g in enumerate(num_grasps):
            for obj_grasp_rects_gt in data_dict["grasp_rects"][0]:
                flag_top1 = False
                for obj_grasp_rects_p in post_dict["grasps_top1"]:
                    j_index_top1 = calculate_jacquard_index(obj_grasp_rects_p, obj_grasp_rects_gt)
                    if j_index_top1 > 0:
                        flag_top1 = True
                        break
                if flag_top1:
                    num_correct_grasps[0] += 1
                num_total_grasps[0] += 1
                
                flag_top5 = False
                for obj_grasp_rects_p in post_dict["grasps_top5"]:
                    j_index_top5 = calculate_jacquard_index(obj_grasp_rects_p, obj_grasp_rects_gt)
                    if j_index_top5 > 0:
                        flag_top5 =  True
                        break
                if flag_top5:
                    num_correct_grasps[1] += 1
                num_total_grasps[1] += 1
        # Use first 100 samples for validation to save time
        if i > 100:
            break
    
    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i]/num_total_grasps[i]
    
    head = 'Evaluation: Epoch=[{}/{}]  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * J_index[0], 100. * J_index[1])
    logger.info(head)

    return J_index



@torch.no_grad()
def visualization(dataset, model, epoch, args):
    model.eval()
    time.sleep(2)

    tgt_dir = os.path.join(args.output_folder, "vis", f"epoch-{epoch}")
    os.makedirs(tgt_dir, exist_ok=True)

    vis_idx = np.random.randint(len(dataset), size=1)[0]
    data = dataset[vis_idx]
    data_dict = dataset.collate_fn([data])

    data_dict["rgb"] = data_dict["rgb"].cuda()
    data_dict["depth"] = data_dict["depth"].cuda()
    
    output_dict = model(data_dict)
    post_dict = ssg_post_processing(args, output_dict, data_dict)

    cls_pred = post_dict["cls"]
    bboxes_pred = post_dict["bboxes"]
    ins_masks_pred = post_dict["ins_masks"]
    qua_masks_pred, ang_masks_pred, wid_masks_pred = post_dict["grasp_masks"]
    grasps_top1 = post_dict["grasps_top1"]
    grasps_top5 = post_dict["grasps_top5"]

    img_path = os.path.join(dataset.root_dir, data_dict["scene_id"][0], "rgb", data_dict["img_f"][0])
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    for i in range(bboxes_pred.shape[0]):
        tmp = img.copy()
        cls_p = cls_pred[i]
        bbox = bboxes_pred[i]
        ins_mask = ins_masks_pred[i].astype(int)
        qua_mask_p = qua_masks_pred[i]
        ang_mask_p = ang_masks_pred[i]
        wid_mask_p = wid_masks_pred[i]
        grasp_top5_p = grasps_top5[i]

        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(1, 4, 1)
        ax.imshow(tmp[:, :, ::-1])
        ax.set_title('RGB')
        ax.axis('off')

        mask = np.asarray([0, 0, 255]).reshape(1, 1, 3).repeat(480, axis=0).repeat(640, axis=1)
        ins_mask = np.expand_dims(ins_mask, axis=-1).repeat(3, axis=-1)
        mask *= ins_mask
        img_with_mask = (tmp * 0.4 + mask * 0.6).astype(int)
        
        ax = fig.add_subplot(1, 4, 2)
        ax.imshow(img_with_mask[:, :, ::-1])
        ax.set_title('ins mask')
        ax.axis('off')

        # Put bounding box
        tmp = cv2.rectangle(tmp, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 2)
        tmp = cv2.putText(tmp, dataset.idx_to_class[cls_p], (int(bbox[0]),int(bbox[1])-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)

        ax = fig.add_subplot(1, 4, 3)
        ax.imshow(tmp[:, :, ::-1])
        ax.set_title('bounding box')
        ax.axis('off')

        # Put grasp rects
        grasps = dataset.grasp_transforms.inverse(grasp_top5_p)
        for entry in grasps:
            ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
            tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
            tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)

        ax = fig.add_subplot(1, 4, 4)
        ax.imshow(tmp[:, :, ::-1])
        ax.set_title('grasps')
        ax.axis('off')

        plt.savefig(os.path.join(tgt_dir, f"sample_{i}.png"))
        plt.cla()
        plt.clf()
        plt.close()

