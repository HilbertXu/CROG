import cv2
from cv2 import resize
import numpy as np
import random


class DataAugmentor(object):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mean = np.asarray([0.48145466, 0.4578275,
                                  0.40821073], dtype=np.float32)
        self.std = np.asarray([0.26862954, 0.26130258,
                                 0.27577711], dtype=np.float32)
        self.mode = mode
    

    def _random_mirror(self, data_dict):
        img = data_dict["rgb"]
        bboxes = data_dict["bboxes"][:, :4]
        if random.randint(0, 1):
            _, width, _ = img.shape
            data_dict["rgb"] = img[:, ::-1]
            data_dict["depth"] = data_dict["depth"][:, ::-1]
            data_dict["ins_masks"] = data_dict["ins_masks"][:, :, ::-1]
            data_dict["grasp_masks"]["qua"] = data_dict["grasp_masks"]["qua"][:, :, ::-1]
            data_dict["grasp_masks"]["ang"] = data_dict["grasp_masks"]["ang"][:, :, ::-1]
            data_dict["grasp_masks"]["wid"] = data_dict["grasp_masks"]["wid"][:, :, ::-1]

            bboxes[:, 0::2] = width - bboxes[:, 2::-2]
            data_dict["bboxes"][:, :4] = bboxes
    

    def _random_brightness(self, img, delta=32):
        img += random.uniform(-delta, delta)
        return np.clip(img, 0., 255.)

    
    def _random_contrast(self, img, lower=0.7, upper=1.3):
        img *= random.uniform(lower, upper)
        return np.clip(img, 0., 255.)

    
    def _random_saturation(self, img, delta=15.):
        img[:, :, 0] += random.uniform(-delta, delta)
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img
    

    def _random_hue(self, img, delta=15.):
        img[:, :, 0] += random.uniform(-delta, delta)
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img

    
    def _photometric_distort(self, data_dict):
        img = data_dict["rgb"].astype(np.float32)
        
        if random.randint(0, 1):
            img = self._random_brightness(img)
        if random.randint(0, 1):
            img = self._random_contrast(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = self._random_saturation(img)
        img = self._random_hue(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0., 255.)

        data_dict["rgb"] = img
    

    def _pad_to_square(self, data_dict):
        img = data_dict["rgb"]
        img_h, img_w = img.shape[:2]
        if img_h == img_w:
            pass
        else:
            pad_size = max(img_h, img_w)
            pad_img = np.zeros((pad_size, pad_size, 3), dtype='float32')
            pad_img[:, :, :] = self.mean
            pad_depth = np.zeros((pad_size, pad_size), dtype='float32')
            pad_ins_masks = np.zeros((data_dict["ins_masks"].shape[0], pad_size, pad_size), dtype='float32')
            pad_qua_masks = np.zeros((data_dict["grasp_masks"]["qua"].shape[0], pad_size, pad_size), dtype='float32')
            pad_ang_masks = np.zeros((data_dict["grasp_masks"]["ang"].shape[0], pad_size, pad_size), dtype='float32')
            pad_wid_masks = np.zeros((data_dict["grasp_masks"]["wid"].shape[0], pad_size, pad_size), dtype='float32')


            if self.mode=="train":
                if img_h < img_w:
                    random_y1 = random.randint(0, img_w - img_h)
                    pad_img[random_y1: random_y1 + img_h, :, :] = data_dict["rgb"]
                    pad_depth[random_y1: random_y1 + img_h, :] = data_dict["depth"]
                    pad_ins_masks[:, random_y1: random_y1 + img_h, :] = data_dict["ins_masks"]
                    pad_qua_masks[:, random_y1: random_y1 + img_h, :] = data_dict["grasp_masks"]["qua"]
                    pad_ang_masks[:, random_y1: random_y1 + img_h, :] = data_dict["grasp_masks"]["ang"]
                    pad_wid_masks[:, random_y1: random_y1 + img_h, :] = data_dict["grasp_masks"]["wid"]
                    data_dict["bboxes"][:, [1, 3]] += random_y1

                if img_h > img_w:
                    random_x1 = random.randint(0, img_h - img_w)
                    pad_img[:, random_x1: random_x1 + img_w, :] = data_dict["rgb"]
                    pad_depth[:, random_x1: random_x1 + img_w] = data_dict["depth"]
                    pad_ins_masks[:, :, random_x1: random_x1 + img_w] = data_dict["ins_masks"]
                    pad_qua_masks[:, :, random_x1: random_x1 + img_w] = data_dict["grasp_masks"]["qua"]
                    pad_ang_masks[:, :, random_x1: random_x1 + img_w] = data_dict["grasp_masks"]["ang"]
                    pad_wid_masks[:, :, random_x1: random_x1 + img_w] = data_dict["grasp_masks"]["wid"]
                    data_dict["bboxes"][:, [0, 2]] += random_x1
            elif self.mode in ["test", "val"]:
                pad_img[0: img_h, 0: img_w, :] = data_dict["rgb"]
                pad_depth[0: img_h, 0: img_w] = data_dict["depth"]
                pad_ins_masks[:, 0: img_h, 0: img_w] = data_dict["ins_masks"]
                pad_qua_masks[:, 0: img_h, 0: img_w] = data_dict["grasp_masks"]["qua"]
                pad_ang_masks[:, 0: img_h, 0: img_w] = data_dict["grasp_masks"]["ang"]
                pad_wid_masks[:, 0: img_h, 0: img_w] = data_dict["grasp_masks"]["wid"]


            data_dict["rgb"] = pad_img
            data_dict["depth"] = pad_depth
            data_dict["ins_masks"] = pad_ins_masks
            data_dict["grasp_masks"]["qua"] = pad_qua_masks
            data_dict["grasp_masks"]["ang"] = pad_ang_masks
            data_dict["grasp_masks"]["wid"] = pad_wid_masks
    

    def _resize(self, data_dict):
        ori_size = data_dict["rgb"].shape[0]
        tgt_size = self.cfg.img_size
        scale = tgt_size / ori_size

        data_dict["rgb"] = cv2.resize(data_dict["rgb"], (tgt_size, tgt_size))
        data_dict["depth"] = cv2.resize(data_dict["depth"], (tgt_size, tgt_size))
        data_dict["ins_masks"] = cv2.resize(data_dict["ins_masks"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["ins_masks"].shape[0] > 1 else \
                                     np.expand_dims(cv2.resize(data_dict["ins_masks"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["grasp_masks"]["qua"] = cv2.resize(data_dict["grasp_masks"]["qua"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["grasp_masks"]["qua"].shape[0] > 1 else \
                                    np.expand_dims(cv2.resize(data_dict["grasp_masks"]["qua"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["grasp_masks"]["ang"] = cv2.resize(data_dict["grasp_masks"]["ang"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["grasp_masks"]["ang"].shape[0] > 1 else \
                                    np.expand_dims(cv2.resize(data_dict["grasp_masks"]["ang"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["grasp_masks"]["wid"] = cv2.resize(data_dict["grasp_masks"]["wid"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["grasp_masks"]["wid"].shape[0] > 1 else \
                                    np.expand_dims(cv2.resize(data_dict["grasp_masks"]["wid"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["bboxes"][:, :4] *= scale

    
    def _normalize_boxes(self, data_dict):
        h, w = data_dict["rgb"].shape[:2]
        data_dict["bboxes"][:, [0, 2]] /= w
        data_dict["bboxes"][:, [1, 3]] /= h
    
    def _normalize_img(self, data_dict):
        img = data_dict["rgb"] / 255.
        # img = (img - self.mean) / self.std
        img = img[:, :, (2,1,0)]
        img = np.transpose(img, (2, 0, 1))
        data_dict["rgb"] = img



    def __call__(self, data_dict):
        if self.mode == "train":
            self._photometric_distort(data_dict)
            self._random_mirror(data_dict)
        self._pad_to_square(data_dict)
        self._resize(data_dict)
        self._normalize_boxes(data_dict)
        self._normalize_img(data_dict)