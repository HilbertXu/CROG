import os
from typing import List, Union
import json
import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from copy import deepcopy
import functools
from skimage.measure import regionprops
from shapely.geometry import Polygon
from skimage.draw import polygon
from skimage.filters import gaussian
import matplotlib.pyplot as plt

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
from .augmentation import DataAugmentor

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    },
    'cocostuff': {
        "train": 965042,
        'val': 42095,
        'val-test': 42095
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)



class RefOCIDGraspDataset(Dataset):
    def __init__(self, root_path, input_size, word_length, mode="train"):
        super().__init__()
        json_path = os.path.join(root_path, f"{mode}_expressions.json")
        with open(json_path, "r") as f:
            self.meta_data = json.load(f)
        
        self.root_path = root_path
        self.keys = list(self.meta_data.keys())
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mode = mode
        
        self.cls_names = cls_names
        self.colors = colors
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        
        self.target_save_dir = os.path.join("./", mode)
        os.makedirs(self.target_save_dir, exist_ok=True)

        
    def __len__(self):
        return len(self.keys)
    

    def visualization(self, rgb, depth, masks, rects, grasp_masks, bbox, obj_cls, sentence):
        cls_list = list(self.cls_names.keys())

        print(rgb.shape, depth.shape, len(rects), masks.shape)
        mask_color = self.colors[str(obj_cls)]
        rgb_with_grasp = deepcopy(rgb)

        color_masks = np.repeat(masks[:, :, np.newaxis], 3, axis=-1) * mask_color
        rgb = (rgb * 255).astype(np.uint8)
        img_fuse = (rgb * 0.3 + color_masks * 0.7).astype(np.uint8)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_fuse, (x1, y1), (x2, y2), (255, 0, 0), 2)


        print(f'\nimg shape: {rgb.shape}')
        print('----------------boxes----------------')
        for rect in rects:
            print(rect)
        print('----------------labels---------------')
        print([cls_list[int(i)] for i in [obj_cls]], '\n')

        for rect in rects:
            name = cls_list[int(obj_cls)]
            color = self.colors[str(obj_cls)]
            center_x, center_y, width, height, theta, cls_id = rect
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.int0(box)
            cv2.drawContours(rgb_with_grasp, [box], 0, color.tolist(), 2)
        

        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(rgb)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(img_fuse)
        ax.set_title('Masks & Bboxes')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(rgb_with_grasp)
        ax.set_title('Grasps')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plot = ax.imshow(grasp_masks[0], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 6)
        plot = ax.imshow(grasp_masks[1], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 7)
        plot = ax.imshow(grasp_masks[2], cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
        ax.set_title('Angle')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 8)
        plot = ax.imshow(grasp_masks[3], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Width')
        ax.axis('off')
        plt.colorbar(plot)

        plt.suptitle(f"{sentence}", fontsize=20)
        plt.tight_layout()
        plt.savefig("./visualization.png")

    # @functools.lru_cache(maxsize=None)
    def _load_bbox(self, bbox):
        bbox = bbox.replace("[", "").replace("]", "").split(",")
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        return [x, y, w, h]

    # @functools.lru_cache(maxsize=None)
    def _load_rgb(self, path):
        image = cv2.imread(os.path.join(self.root_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        return image
    
    # @functools.lru_cache(maxsize=None)
    def _load_depth(self, path, factor=1000.):
        depth = cv2.imread(os.path.join(self.root_path, path), cv2.IMREAD_UNCHANGED) / factor
        depth = 1 - (depth / np.max(depth))

        return depth
    
    # @functools.lru_cache(maxsize=None)
    def _load_annos(self, path, target_cls):
        cls_annos_path = os.path.join(os.path.join(self.root_path, path), str(target_cls))
        file_id = cls_annos_path.split("/")[-2]
        cls_annos_path = os.path.join(cls_annos_path, f"{file_id}.txt")

        grasps_list = []
        with open(cls_annos_path, 'r') as f:
            points_list = []
            for count, line in enumerate(f):
                line = line.rstrip()
                [x, y] = line.split(' ')

                x = float(x)
                y = float(y)

                pt = (x, y)
                points_list.append(pt)

                if len(points_list) == 4:
                    p1, p2, p3, p4 = points_list
                    center_x = (p1[0] + p3[0]) / 2
                    center_y = (p1[1] + p3[1]) / 2
                    width  = np.sqrt((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]))
                    height = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                    
                    # @NOTE
                    # Along x+ is 0 degree, increase by rotating anti-clockwise
                    # If you want to use opencv boxPoints & drawContours to visualize grasps
                    # Remember to take -theta as param :-)
                    theta = np.arctan2(p4[0] - p1[0], p4[1] - p1[1]) * 180 / np.pi
                    if theta > 0:
                        theta = theta-90
                    else:
                        theta = theta+90


                    grasps_list.append([center_x, center_y, width, height, theta, int(target_cls)])
                    points_list = []

        return grasps_list

    # @functools.lru_cache(maxsize=None)
    def _load_mask(self, path, target_cls):
        masks = cv2.imread(os.path.join(self.root_path, path), cv2.IMREAD_UNCHANGED)
        if target_cls == -1:
            # for loading instance masks
            return masks
        else:
            # For loading target semantic masks
            target_masks = (masks == target_cls)
            return target_masks

    def _match_masks_with_ref(self, bbox, ins_masks, masks):
        # Preparing bounding box for calculating IoU
        x1, y1, x2, y2 = bbox
        w, h = (x2-x1), (y2-y1)
        vertices = [[x1, y1], [x1+w, y1], [x2, y2], [x1, y1+h]]
        poly1 = Polygon(vertices)

        # Keep only the instance masks with correct class label
        ins_masks = ins_masks * masks
        ins_props = regionprops(ins_masks)

        max_iou = 0.0
        ins_idx = 0

        for ins in ins_props:
            _x1, _y1, _x2, _y2 = ins.bbox[1], ins.bbox[0], ins.bbox[3], ins.bbox[2]
            _w, _h = (_x2-_x1), (_y2-_y1)
            ins_vertices = [[_x1, _y1], [_x1+_w, _y1], [_x2, _y2], [_x1, _y1+_h]]
            poly2 = Polygon(ins_vertices)

            if poly1.intersects(poly2): 
                intersect = poly1.intersection(poly2).area
                union = poly1.union(poly2).area
                iou = intersect/union

                if iou > max_iou:
                    max_iou = iou
                    ins_idx = ins.label
        
        ins_masks = (ins_masks == ins_idx)

        return ins_masks

    def _match_grasps_with_ref(self, rects, ins_masks):
        # Check if the center of grasp falls in the target instance mask
        grasps = []
        for rect in rects:
            c_x, c_y = int(rect[0]), int(rect[1])
            if ins_masks[c_y, c_x]:
                grasps.append(rect)
        
        return grasps

    def _filter_grasps(self, rects):
        angles = []
        for rect in rects:
            angles.append(rect[4])
            
    
    def _generate_grasp_masks(self, grasps, width, height):
        pos_out = np.zeros((height, width))
        ang_out = np.zeros((height, width))
        wid_out = np.zeros((height, width))
        for rect in grasps:
            center_x, center_y, w_rect, h_rect, theta, cls_id = rect
            width_factor = float(100)

            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.int0(box)

            rr, cc = polygon(box[:, 0], box[:,1])

            mask_rr = rr < width
            rr = rr[mask_rr]
            cc = cc[mask_rr]

            mask_cc = cc < height
            cc = cc[mask_cc]
            rr = rr[mask_cc]


            pos_out[cc, rr] = 1.0
            if theta < 0:
                ang_out[cc, rr] = int(theta + 180)
            else:
                ang_out[cc, rr] = int(theta)
            # Adopt width normalize accoding to class 
            wid_out[cc, rr] = np.clip(w_rect, 0.0, width_factor) / width_factor
        
        qua_out = (gaussian(pos_out, 3, preserve_range=True) * 255).astype(np.uint8)
        pos_out = (pos_out * 255).astype(np.uint8)
        ang_out = ang_out.astype(np.uint8)
        wid_out = (gaussian(wid_out, 3, preserve_range=True) * 255).astype(np.uint8)
        
        return [pos_out, qua_out, ang_out, wid_out]
    
    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


    def convert(self, img, mask=None, grasp_quality_mask=None, grasp_sin_masks=None, grasp_cos_masks=None, grasp_width_mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        
        if grasp_quality_mask is not None:
            grasp_quality_mask = torch.from_numpy(grasp_quality_mask)
            if not isinstance(grasp_quality_mask, torch.FloatTensor):
                grasp_quality_mask = grasp_quality_mask.float()

        if grasp_sin_masks is not None:
            grasp_sin_masks = torch.from_numpy(grasp_sin_masks)
            if not isinstance(grasp_sin_masks, torch.FloatTensor):
                grasp_sin_masks = grasp_sin_masks.float()
        
        if grasp_cos_masks is not None:
            grasp_cos_masks = torch.from_numpy(grasp_cos_masks)
            if not isinstance(grasp_cos_masks, torch.FloatTensor):
                grasp_cos_masks = grasp_cos_masks.float()

        if grasp_width_mask is not None:
            grasp_width_mask = torch.from_numpy(grasp_width_mask)
            if not isinstance(grasp_width_mask, torch.FloatTensor):
                grasp_width_mask = grasp_width_mask.float()
        return img, mask, grasp_quality_mask, grasp_sin_masks, grasp_cos_masks, grasp_width_mask


    def __getitem__(self, index):
        data_dict = {}

        key = self.keys[index]
        ref_data = self.meta_data[key]
        obj_cls = int(self.cls_names[ref_data["class"]])

        # Get path to other data
        scene_path = ref_data["scene_path"]
        depth_path = scene_path.replace("rgb", "depth")
        annos_path = scene_path.replace("rgb", "Annotations_per_class")[:-4]
        masks_path = scene_path.replace("rgb", "seg_mask_labeled_combi")
        ins_masks_path = scene_path.replace("rgb", "seg_mask_instances_combi")

        # Read data
        rgb = self._load_rgb(scene_path) # [0 - 255]
        depth = self._load_depth(depth_path)
        annos = self._load_annos(annos_path, obj_cls)
        masks = self._load_mask(masks_path, obj_cls)
        ins_masks = self._load_mask(ins_masks_path, -1)
        bbox = self._load_bbox(ref_data["bbox"])
        sentence = ref_data["sentence"]

        # ins_masks = ins_masks * masks

        ins_masks = (self._match_masks_with_ref(bbox, ins_masks, masks) * 255).astype(np.uint8)

        grasps = self._match_grasps_with_ref(annos, ins_masks)

        assert rgb.shape[:2] == depth.shape

        height, width = depth.shape

        grasp_masks = self._generate_grasp_masks(grasps, width, height)

        grasp_quality_masks = grasp_masks[1]
        grasp_angle_masks = grasp_masks[2]
        grasp_width_masks = grasp_masks[3]

        # Image transforms
        img_size = rgb.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, True)
        rgb = cv2.warpAffine(rgb,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_CUBIC,
                                borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        if self.mode == 'train':
            ins_masks = cv2.warpAffine(ins_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            grasp_quality_masks = cv2.warpAffine(grasp_quality_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            grasp_angle_masks = cv2.warpAffine(grasp_angle_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            grasp_width_masks = cv2.warpAffine(grasp_width_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            # Normalize and pre-process target masks
            ins_masks = ins_masks / 255.
            grasp_quality_masks = grasp_quality_masks / 255.
            grasp_angle_masks = grasp_angle_masks * np.pi / 180.
            grasp_width_masks = grasp_width_masks / 255.
            grasp_sin_masks = np.sin(2 * grasp_angle_masks)
            grasp_cos_masks = np.cos(2 * grasp_angle_masks)
            
            word_vec = tokenize(sentence, self.word_length, True).squeeze(0)

            # self.visualization(rgb, depth, ins_masks, grasps, (grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks), bbox, obj_cls, sentence)

            rgb, ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks = self.convert(
                rgb, ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks
            )

            # np.savez(
            #     os.path.join(self.target_save_dir, f"{key}.npz"),
            #     img=rgb.cpu().numpy(),
            #     word_vec=word_vec.cpu().numpy(),
            #     ins_masks=ins_masks.cpu().numpy(),
            #     grasp_quality_masks=grasp_quality_masks.cpu().numpy(),
            #     grasp_sin_masks=grasp_sin_masks.cpu().numpy(),
            #     grasp_cos_masks=grasp_cos_masks.cpu().numpy(),
            #     grasp_width_masks=grasp_width_masks.cpu().numpy(),
            #     grasps=grasps
            # )


            return rgb, (ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks), word_vec
        
        elif self.mode == 'val':
            word_vec = tokenize(sentence, self.word_length, True).squeeze(0)

            ins_masks = ins_masks / 255.
            grasp_quality_masks = grasp_quality_masks / 255.
            grasp_angle_masks = grasp_angle_masks * np.pi / 180.
            grasp_width_masks = grasp_width_masks / 255.
            grasp_sin_masks = np.sin(2 * grasp_angle_masks)
            grasp_cos_masks = np.cos(2 * grasp_angle_masks)

            rgb = self.convert(rgb)[0]
            params = {
                # 'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            # np.savez(
            #     os.path.join(self.target_save_dir, f"{key}.npz"),
            #     img=rgb.cpu().numpy(),
            #     word_vec=word_vec.cpu().numpy(),
            #     ins_masks=ins_masks,
            #     grasp_quality_masks=grasp_quality_masks,
            #     grasp_sin_masks=grasp_sin_masks,
            #     grasp_cos_masks=grasp_cos_masks,
            #     grasp_width_masks=grasp_width_masks,
            #     grasps=grasps,
            #     params=params
            # )
            return rgb, (ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks), word_vec, params
        elif self.mode == 'test':
            rgb = self.convert(rgb)[0]
            word_vec = tokenize(sentence, self.word_length, True).squeeze(0)
            params = {
                # 'ori_img': ori_img,
                # 'seg_id': seg_id,
                # 'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sentence

            }
            # np.savez(
            #     os.path.join(self.target_save_dir, f"{key}.npz"),
            #     img=rgb.cpu().numpy(),
            #     word_vec=word_vec.cpu().numpy(),
            #     params=params
            # )
            return rgb, word_vec, params



class GraspTransforms:
    # Class for converting cv2-like rectangle formats and generate grasp-quality-angle-width masks

    def __init__(self, width_factor=100, width=640, height=480):
        self.width_factor = width_factor
        self.width = width 
        self.height = height

    def __call__(self, grasp_rectangles, target):
        # grasp_rectangles: (M, 4, 2)
        M = grasp_rectangles.shape[0]
        p1, p2, p3, p4 = np.split(grasp_rectangles, 4, axis=1)
        
        center_x = (p1[..., 0] + p3[..., 0]) / 2
        center_y = (p1[..., 1] + p3[..., 1]) / 2
        
        width  = np.sqrt((p1[..., 0] - p4[..., 0]) * (p1[..., 0] - p4[..., 0]) + (p1[..., 1] - p4[..., 1]) * (p1[..., 1] - p4[..., 1]))
        height = np.sqrt((p1[..., 0] - p2[..., 0]) * (p1[..., 0] - p2[..., 0]) + (p1[..., 1] - p2[..., 1]) * (p1[..., 1] - p2[..., 1]))
        
        theta = np.arctan2(p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1]) * 180 / np.pi
        theta = np.where(theta > 0, theta - 90, theta + 90)

        target = np.tile(np.array([[target]]), (M,1))

        return np.concatenate([center_x, center_y, width, height, theta, target], axis=1)

    def inverse(self, grasp_rectangles):
        boxes = []
        for rect in grasp_rectangles:
            center_x, center_y, width, height, theta = rect[:5]
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.intp(box)
            boxes.append(box)
        return boxes

    def generate_masks(self, grasp_rectangles):
        pos_out = np.zeros((self.height, self.width))
        ang_out = np.zeros((self.height, self.width))
        wid_out = np.zeros((self.height, self.width))
        for rect in grasp_rectangles:
            center_x, center_y, w_rect, h_rect, theta = rect[:5]
            
            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.intp(box)

            rr, cc = polygon(box[:, 0], box[:,1])

            mask_rr = rr < self.width
            rr = rr[mask_rr]
            cc = cc[mask_rr]

            mask_cc = cc < self.height
            cc = cc[mask_cc]
            rr = rr[mask_cc]
            pos_out[cc, rr] = 1.0
            if theta < 0:
                ang_out[cc, rr] = int(theta + 180)
            else:
                ang_out[cc, rr] = int(theta)
            # Adopt width normalize accoding to class 
            wid_out[cc, rr] = np.clip(w_rect, 0.0, self.width_factor) / self.width_factor
        
        qua_out = (gaussian(pos_out, 3, preserve_range=True) * 255).astype(np.uint8)
        pos_out = (pos_out * 255).astype(np.uint8)
        ang_out = ang_out.astype(np.uint8)
        wid_out = (gaussian(wid_out, 3, preserve_range=True) * 255).astype(np.uint8)
        
        
        return {'pos': pos_out, 
                'qua': qua_out, 
                'ang': ang_out, 
                'wid': wid_out}



class OCIDVLGDataset(Dataset):
    
    """ OCID-Vision-Language-Grasping dataset with referring expressions and grasps """

    def __init__(self, 
                 root_dir,
                 split, 
                 transform_img = None,
                 transform_grasp = GraspTransforms(),
                 input_size = 416,
                 word_length = 20,
                 with_depth = True, 
                 with_segm_mask = True,
                 with_grasp_masks = True,
                 version="multiple"
    ):
        super(OCIDVLGDataset, self).__init__()
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, "data_split")
        self.split_map = {'train': 'train_expressions.json', 
                          'val': 'val_expressions.json',
                          'test': 'test_expressions.json'
                         }
        self.split = split
        self.refer_dir = os.path.join(root_dir, "refer", version)
        
        self.transform_img = transform_img
        self.transform_grasp = transform_grasp
        self.with_depth = with_depth
        self.with_segm_mask = with_segm_mask
        self.with_grasp_masks = with_grasp_masks
        # assert (self.transform_grasp and self.with_grasp_masks) or (not self.transform_grasp and not self.with_grasp_masks)

        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

        self._load_dicts()
        self._load_split()

    def _load_dicts(self):
        cwd = os.getcwd()
        os.chdir(self.root_dir)
        from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
        cnames_inv = {int(v):k for k,v in cnames.items()}
        subnames_inv = {v:k for k,v in subnames.items()}
        self.class_names = cnames 
        self.idx_to_class = cnames_inv
        self.class_instance_names = subnames
        self.idx_to_class_instance = subnames_inv
        self.instance_idx_to_class_idx = sub_to_class
        os.chdir(cwd)

    def _load_split(self):
        refer_data = json.load(open(os.path.join(self.refer_dir, self.split_map[self.split])))
        self.seq_paths, self.img_names, self.scene_ids = [], [], []
        self.bboxes, self.grasps = [], []
        self.sent_to_index, self.sent_indices = {}, []
        self.rgb_paths, self.depth_paths, self.mask_paths = [], [], []
        self.targets, self.sentences, self.semantics, self.objIDs = [], [], [], []
        n = 0
        for item in refer_data['data']:
            seq_path, im_name = item['image_filename'].split(',')
            self.seq_paths.append(seq_path)
            self.img_names.append(im_name)
            self.scene_ids.append(item['image_filename'])
            self.bboxes.append(item['box'])
            self.grasps.append(item['grasps'])
            self.objIDs.append(item['answer'])
            self.targets.append(item['target'])
            self.sentences.append(item['question'])
            self.semantics.append(item['program'])
            self.rgb_paths.append(os.path.join(seq_path, "rgb", im_name))
            self.depth_paths.append(os.path.join(seq_path, "depth", im_name))
            self.mask_paths.append(os.path.join(seq_path, "seg_mask_instances_combi", im_name))
            self.sent_indices.append(item['question_index'])
            self.sent_to_index[item['question_index']] = n
            n += 1
            
    def get_index_from_sent(self, sent_id):
        return self.sent_to_index[sent_id]

    def get_sent_from_index(self, n):
        return self.sent_indices[n]
    
    def _load_sent(self, sent_id):
        n = self.get_index_from_sent(sent_id)
        
        scene_id = self.scene_ids[n]
       
        img_path = os.path.join(self.root_dir, self.rgb_paths[n])
        img = self.get_image_from_path(img_path)
        
        x, y, w, h = self.bboxes[n]
        bbox = np.asarray([x, y, x+w, y+h])
        
        sent = self.sentences[n]
        
        target = self.targets[n]
        target_idx = self.class_instance_names[target]
        objID = self.objIDs[n]
        
        grasps = np.asarray(self.grasps[n])
        
        result = {'img': self.transform_img(img) if self.transform_img else img, 
                  'grasps':  self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
                  'grasp_rects': self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
                  'sentence': sent,
                  'target': target,
                  'objID': objID,
                  'bbox': bbox,
                  'target_idx': target_idx,
                  'sent_id': sent_id,
                  'scene_id': scene_id,
                  'img_path': img_path
                 }
        
        if self.with_depth:
            depth_path = os.path.join(self.root_dir, self.depth_paths[n])
            depth = self.get_depth_from_path(depth_path)
            result = {**result, 'depth': torch.from_numpy(depth) if self.transform_img else depth}

        if self.with_segm_mask:
            mask_path = os.path.join(self.root_dir, self.mask_paths[n])
            msk_full = self.get_mask_from_path(mask_path)
            msk = np.where(msk_full == objID, True, False)
            result = {**result, 'mask': torch.from_numpy(msk) if self.transform_img else msk}

        if self.with_grasp_masks:
            grasp_masks = self.transform_grasp.generate_masks(result['grasp_rects'])
            result = {**result, 'grasp_masks': grasp_masks}
        
        result = self.preprocess(result)
        
        return result
    
    def get_transform_mat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


    def preprocess(self, data):
        img = data["img"]
        sent = data["sentence"]
        if np.max(data["mask"]) <= 1.0:
            ins_mask = (data["mask"] * 255).astype(np.uint8)
        else:
            ins_mask = data["mask"]
        
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_ang_mask = data["grasp_masks"]["ang"]
        grasp_wid_mask = data["grasp_masks"]["wid"]

        img_size = img.shape[:2]
        mat, mat_inv = self.get_transform_mat(img_size, True)

        img = cv2.warpAffine(
            img, mat, self.input_size, flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        )

        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)

        ins_mask = cv2.warpAffine(ins_mask,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
        grasp_qua_mask = cv2.warpAffine(grasp_qua_mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)
        
        grasp_ang_mask = cv2.warpAffine(grasp_ang_mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)
        
        grasp_wid_mask = cv2.warpAffine(grasp_wid_mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)


        ins_mask = ins_mask / 255.
        grasp_qua_mask = grasp_qua_mask / 255.
        grasp_ang_mask = grasp_ang_mask * np.pi / 180.
        grasp_wid_mask = grasp_wid_mask / 255.
        grasp_sin_mask = np.sin(2 * grasp_ang_mask)
        grasp_cos_mask = np.cos(2 * grasp_ang_mask)

        word_vec = tokenize(sent, self.word_length, True).squeeze(0)

        data["img"] = img
        data["mask"] = ins_mask
        data["grasp_masks"]["qua"] = grasp_qua_mask
        data["grasp_masks"]["ang"] = grasp_ang_mask
        data["grasp_masks"]["wid"] = grasp_wid_mask
        data["grasp_masks"]["sin"] = grasp_sin_mask
        data["grasp_masks"]["cos"] = grasp_cos_mask
        data["word_vec"] = word_vec
        data["inverse"] = mat_inv
        data["ori_size"] = np.array(img_size)
        
        # del data["sentence"]
        
        return data

    def __len__(self):
        return len(self.sent_indices)
    
    def __getitem__(self, n):
        sent_id = self.get_sent_from_index(n)
        data = self._load_sent(sent_id)
        
        return data
    
    @staticmethod
    def transform_grasp_inv(grasp_pt):
        pass
    
    # @functools.lru_cache(maxsize=None)
    def get_image_from_path(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(os.path.exists(path))
            print(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img

    # @functools.lru_cache(maxsize=None)
    def get_mask_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # @functools.lru_cache(maxsize=None)
    def get_depth_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000. # mm -> m
    
    def get_image(self, n):
        img_path = os.path.join(self.root_dir, self.imgs[n])
        return self.get_image_from_path(img_path)
    
    def get_annotated_image(self, n, text=True):
        sample = self.__getitem__(n)
        
        img, sent, grasps, bbox = sample['img'], sample['sentence'], sample['grasp_rects'], sample['bbox']
        if isinstance(img, torch.FloatTensor):
            img = img.permute(1,2,0)
            img = (img.cpu().numpy() * 255).astype(np.uint8)
        if self.transform_img:
            img = np.asarray(tfn.to_pil_image(img))
        if self.transform_grasp:
            #grasps = list(map(self.transform_grasp_inv, list(grasps)))
            grasps = self.transform_grasp.inverse(grasps)

        tmp = img.copy()
        for entry in grasps:
            ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
            tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
            tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
        
        tmp = cv2.rectangle(tmp, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
        if text:
            tmp = cv2.putText(tmp, sent, (0,10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
        return tmp

    def visualization(self, n, save_path):
        s = self.__getitem__(n)

        rgb = s['img']
        if isinstance(rgb, torch.FloatTensor):
            rgb = rgb.permute(1,2,0)
            rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        depth = (0xff * s['depth'] / 3).astype(np.uint8)
        ii = self.get_annotated_image(n, text=False)
        sentence = s['sentence']
        msk = s['mask'].astype(np.uint8) / 255
        # msk_img = (rgb * 0.3).astype(np.uint8).copy()
        # msk_img[msk, 0] = 255

        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(rgb)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(msk)
        ax.set_title('Segm Mask')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(ii)
        ax.set_title('Box & Grasp')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plot = ax.imshow(s['grasp_masks']['qua'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Grasp quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 6)
        plot = ax.imshow(s['grasp_masks']['sin'], cmap='rainbow', vmin=-1, vmax=1)
        ax.set_title('Angle-cosine')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 7)
        plot = ax.imshow(s['grasp_masks']['cos'], cmap='rainbow', vmin=-1, vmax=1)
        ax.set_title('Angle-sine')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 8)
        plot = ax.imshow(s['grasp_masks']['wid'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Width')
        ax.axis('off')
        plt.colorbar(plot)

        plt.suptitle(f"{sentence}", fontsize=20)
        plt.tight_layout()
        print("save")
        plt.savefig(os.path.join(save_path, f"sample_{n}.png"))
    
    @staticmethod
    def collate_fn(batch):
        return {
            "img": torch.stack([x["img"] for x in batch]),
            "depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch]),
            "mask": torch.stack([torch.from_numpy(x["mask"]).float() for x in batch]),
            "grasp_masks" : {
                "qua": torch.stack([torch.from_numpy(x["grasp_masks"]["qua"]).float() for x in batch]),
                "sin": torch.stack([torch.from_numpy(x["grasp_masks"]["sin"]).float() for x in batch]),
                "cos": torch.stack([torch.from_numpy(x["grasp_masks"]["cos"]).float() for x in batch]),
                "wid": torch.stack([torch.from_numpy(x["grasp_masks"]["wid"]).float() for x in batch])
            },
            "word_vec": torch.stack([x["word_vec"].long() for x in batch]),
            "grasps": [x["grasps"] for x in batch],
            "target": [x["target"] for x in batch],
            "sentence": [x["sentence"] for x in batch],
            "bbox": [x["bbox"] for x in batch],
            "target_idx": [x["target_idx"] for x in batch],
            "sent_id": [x["sent_id"] for x in batch],
            "scene_id": [x["scene_id"] for x in batch],
            "inverse": [x["inverse"] for x in batch],
            "ori_size": [x["ori_size"] for x in batch],
            "img_path": [x["img_path"] for x in batch]
        }
            
        


class OCIDGraspDataset(Dataset):
    
    """ OCID-Grasp dataset """

    def __init__(self, 
                 cfg,
                 split):
        self.cfg = cfg
        self.split = split
        self.root_dir = cfg.root_dir
        self.img_size = cfg.img_size
        self.depth_factor = cfg.depth_factor
        self.with_grasp_masks = cfg.with_grasp_masks
        self.with_sem_masks = cfg.with_sem_masks
        self.with_ins_masks = cfg.with_ins_masks
        self.with_depth = cfg.with_depth
        self.grasp_transforms = GraspTransforms()

        aug_mode = "train" if self.split == "training_0" else "test"
        self.data_augmentor = DataAugmentor(cfg, mode=aug_mode)
        # self.data_augmentor = DataAugmentor(cfg)
        
        self._load_dicts()
        self.num_classes = len(cnames)

        with open(os.path.join(cfg.root_dir, "data_split", split + ".txt"), "r") as fid:
            self.meta = [x.strip().split(',') for x in fid.readlines()]


    def _load_dicts(self):
        cwd = os.getcwd()
        os.chdir(self.root_dir)
        from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
        cnames_inv = {int(v):k for k,v in cnames.items()}
        subnames_inv = {v:k for k,v in subnames.items()}
        self.class_names = cnames 
        self.idx_to_class = cnames_inv
        self.class_instance_names = subnames
        self.idx_to_class_instance = subnames_inv
        self.instance_idx_to_class_idx = sub_to_class
        os.chdir(cwd)

    
    def _get_rgb_image(self, scene_id, img_f, data_dict):
        img_path = os.path.join(self.root_dir, scene_id, "rgb", img_f)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        data_dict["rgb"] = img

    
    def _get_depth_image(self, scene_id, img_f, data_dict):
        depth_path = os.path.join(self.root_dir, scene_id, "depth", img_f)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / float(self.depth_factor)
        depth = 1 - (depth / np.max(depth))
        data_dict["depth"] = depth
        
    
    def _get_sem_mask(self, scene_id, img_f, data_dict):
        sem_mask = cv2.imread(os.path.join(self.root_dir, scene_id, "seg_mask_labeled_combi", img_f), cv2.IMREAD_UNCHANGED)
        data_dict["sem_mask"] = sem_mask
        return sem_mask

    
    def _get_ins_mask(self, scene_id, img_f, data_dict):
        # Load semantic mask first if the with_sem_masks set to False
        if "sem_mask" not in data_dict.keys():
            sem_mask = cv2.imread(os.path.join(self.root_dir, scene_id, "seg_mask_labeled_combi", img_f), cv2.IMREAD_UNCHANGED)
        else:
            sem_mask = data_dict["sem_mask"]
        ins_mask = cv2.imread(os.path.join(self.root_dir, scene_id, "seg_mask_instances_combi", img_f), cv2.IMREAD_UNCHANGED)

        labels     = []
        bboxes     = []
        ins_masks  = []

        props = regionprops(sem_mask)
        for prop in props:
            cls_id = prop.label
            
            # Get binary mask for each semantic class
            bin_mask = (sem_mask == cls_id).astype('int8')
            # Get corresponding semantic mask (may contains multiple instances)
            cls_ins_mask = (ins_mask * bin_mask)

            # Get regions for each instance
            ins_props = regionprops(cls_ins_mask)
            for ins in ins_props:
                labels.append(cls_id)
                bboxes.append([ins.bbox[1], ins.bbox[0], ins.bbox[3], ins.bbox[2], cls_id])
                mask = (cls_ins_mask == ins.label).astype('int8').astype('float32')
                ins_masks.append(mask)
        
        bboxes = np.array(bboxes).astype('float32')
        labels = np.array(labels)
        ins_masks  = np.array(ins_masks)

        data_dict["bboxes"] = bboxes
        data_dict["labels"] = labels
        data_dict["ins_masks"] = ins_masks


    
    def _get_per_cls_grasp_rects(self, scene_id, img_f, data_dict):
        anno_path = os.path.join(self.root_dir, scene_id, "Annotations_per_class", img_f[:-4])
        grasps_list = []
        for cls_id in os.listdir(anno_path):
            grasp_path = os.path.join(anno_path, cls_id, img_f[:-4]+".txt")
            with open(grasp_path, 'r') as f:
                points_list = []
                for count, line in enumerate(f):
                    line = line.rstrip()
                    [x, y] = line.split(' ')

                    x = float(x)
                    y = float(y)

                    pt = (x, y)
                    points_list.append(pt)

                    if len(points_list) == 4:
                        p1, p2, p3, p4 = points_list
                        center_x = (p1[0] + p3[0]) / 2
                        center_y = (p1[1] + p3[1]) / 2
                        width  = np.sqrt((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]))
                        height = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                        
                        # @NOTE
                        # Along x+ is 0 degree, increase by rotating anti-clockwise
                        # If you want to use opencv boxPoints & drawContours to visualize grasps
                        # Remember to take -theta as param :-)
                        theta = np.arctan2(p4[0] - p1[0], p4[1] - p1[1]) * 180 / np.pi
                        if theta > 0:
                            theta = theta-90
                        else:
                            theta = theta+90


                        grasps_list.append([center_x, center_y, width, height, theta, int(cls_id)])
                        points_list = []
        data_dict["raw_grasp_rects"] = grasps_list



    def _get_grasp_mask(self, scene_id, img_f, data_dict):
        grasp_rects = data_dict["raw_grasp_rects"]
        data_dict["grasp_masks"] = {}
        bboxes = data_dict["bboxes"]
        labels = data_dict["labels"]
        masks = data_dict["ins_masks"] # @NOTE there may exist instance with no grasp annotations, thus we need to filter them out
        assert bboxes.shape[0] == masks.shape[0], "inconsistent bounding boxes and instances, check the data"
        num_ins = bboxes.shape[0]

        ins_grasp_rects = []
        ins_grasp_masks = []
        ins_bboxes = []
        ins_masks = []
        ins_labels = []

        for i in range(num_ins):
            box = bboxes[i]
            mask = masks[i]
            label = labels[i]
            tmp = []
            for rect in grasp_rects:
                center_x, center_y, w, h = rect[:4]
                cls_id = rect[-1]
                # Grasp rect and bbox should have the same cls_id
                if int(cls_id) == int(box[4]):
                    # Center of grasp rect in bbox
                    if mask[int(center_y), int(center_x)]:
                        tmp.append(rect)
            if len(tmp) > 0:
                ins_grasp_masks.append(self.grasp_transforms.generate_masks(tmp))
                ins_grasp_rects.append(tmp)
                ins_bboxes.append(box)
                ins_masks.append(mask)
                ins_labels.append(label)

        data_dict["bboxes"] = np.asarray(ins_bboxes)
        data_dict["labels"] = np.asarray(ins_labels)
        data_dict["ins_masks"] = np.asarray(ins_masks)
        data_dict["ins_grasp_rects"] = ins_grasp_rects
        data_dict["grasp_masks"]["qua"] = np.asarray([gm["qua"] / 255 for gm in ins_grasp_masks])
        data_dict["grasp_masks"]["ang"] = np.asarray([gm["ang"] for gm in ins_grasp_masks])
        data_dict["grasp_masks"]["wid"] = np.asarray([gm["wid"] / 255 for gm in ins_grasp_masks])


    def __len__(self):
        return len(self.meta)

    
    def __getitem__(self, index):
        data_dict = {}
        scene_id, img_f = self.meta[index]
        data_dict["scene_id"] = scene_id
        data_dict["img_f"] = img_f

        img_path = os.path.join(self.root_dir, scene_id, "rgb", img_f)
        img = cv2.imread(img_path)
        data_dict["rgb"] = img
        data_dict["ori_size"] = img.shape[:2]

        if self.with_depth:
            self._get_depth_image(scene_id, img_f, data_dict)
        if self.with_sem_masks:
            self._get_sem_mask(scene_id, img_f, data_dict)
        if self.with_ins_masks:
            self._get_ins_mask(scene_id, img_f, data_dict)
        if self.with_grasp_masks:
            self._get_per_cls_grasp_rects(scene_id, img_f, data_dict)
            self._get_grasp_mask(scene_id, img_f, data_dict)
        
        
        self.data_augmentor(data_dict)

        data_dict["grasp_masks"]["sin"] = np.sin(2 * data_dict["grasp_masks"]["ang"])
        data_dict["grasp_masks"]["cos"] = np.cos(2 * data_dict["grasp_masks"]["ang"])

        return data_dict


    def visualization(self, index, tgt_dir, with_preprocessing=False):
        data_dict = {}
        scene_id, img_f = self.meta[index]
        data_dict["scene_id"] = scene_id
        data_dict["img_f"] = img_f

        img_path = os.path.join(self.root_dir, scene_id, "rgb", img_f)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        data_dict["rgb"] = img

        if self.with_depth:
            self._get_depth_image(scene_id, img_f, data_dict)
        if self.with_sem_masks:
            self._get_sem_mask(scene_id, img_f, data_dict)
        if self.with_ins_masks:
            self._get_ins_mask(scene_id, img_f, data_dict)
        if self.with_grasp_masks:
            self._get_per_cls_grasp_rects(scene_id, img_f, data_dict)
            self._get_grasp_mask(scene_id, img_f, data_dict)
        if with_preprocessing:
            self.data_augmentor(data_dict)
            img = data_dict["rgb"].transpose((1,2,0))
        # img = img / 255.

        num_ins = data_dict["bboxes"].shape[0]
        
        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(np.clip(img[:, :, ::-1], 0.0, 1.0)*255)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(data_dict["depth"], cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(data_dict["sem_mask"])
        ax.set_title('Segm Mask')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(tgt_dir, f"raw-data.png"))
        plt.clf()
        plt.cla()
        plt.close()


        for i in range(num_ins):
            fig = plt.figure(figsize=(20, 2))

            ins_box = data_dict["bboxes"][i]
            ins_mask = data_dict["ins_masks"][i]
            ins_label = self.idx_to_class[data_dict["labels"][i]]
            ins_grasp_rects = data_dict["ins_grasp_rects"][i]
            ins_grasp_qua_masks = data_dict["grasp_masks"]["qua"][i]
            ins_grasp_ang_masks = data_dict["grasp_masks"]["ang"][i]
            ins_grasp_wid_masks = data_dict["grasp_masks"]["wid"][i]

            grasps = self.grasp_transforms.inverse(ins_grasp_rects)
            tmp = img.copy()
            h, w, c = tmp.shape
            for entry in grasps:
                ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
                tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
                tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
                tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
                tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
            if ins_box[0] <= 1:
                tmp = cv2.rectangle(tmp, (int(ins_box[0]*w),int(ins_box[1]*h)), (int(ins_box[2]*w),int(ins_box[3]*h)), (0,0,0xff), 2)
            else:
                tmp = cv2.rectangle(tmp, (int(ins_box[0]),int(ins_box[1])), (int(ins_box[2]),int(ins_box[3])), (0,0,0xff), 2)

            ax = fig.add_subplot(1, 5, 1)
            ax.imshow(tmp[:, :, ::-1])
            ax.set_title('Bboxes & Grasps')
            ax.axis('off')

            ax = fig.add_subplot(1, 5, 2)
            tmp_mask = np.expand_dims(ins_mask, axis=-1).repeat(3, axis=-1)
            ax.imshow(tmp_mask*0.6 + tmp[:, :, ::-1] *0.4)
            ax.set_title('ins mask')
            ax.axis('off')

            ax = fig.add_subplot(1, 5, 3)
            plot = ax.imshow(ins_grasp_qua_masks, cmap='jet', vmin=0, vmax=1)
            ax.set_title('Grasp quality')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(1, 5, 4)
            plot = ax.imshow(ins_grasp_ang_masks, cmap='rainbow', vmin=-1, vmax=1)
            ax.set_title('Grasp angle')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(1, 5, 5)
            plot = ax.imshow(ins_grasp_wid_masks, cmap='rainbow', vmin=0, vmax=1)
            ax.set_title('Grasp width')
            ax.axis('off')
            plt.colorbar(plot)

            plt.tight_layout()
            plt.savefig(os.path.join(tgt_dir, f"ins-{i}-{ins_label}.png"))


    @staticmethod
    def collate_fn(batch):
        return {
            "scene_id": [x["scene_id"] for x in batch],
            "img_f": [x["img_f"] for x in batch],
            "ori_size": batch[0]["ori_size"],
            "rgb": torch.stack([torch.from_numpy(x["rgb"]) for x in batch]),
            "depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch]).unsqueeze(1),
            "labels": [torch.from_numpy(x["labels"]).long() for x in batch],
            "bboxes": [torch.from_numpy(x["bboxes"]) for x in batch],
            "ins_masks": [torch.from_numpy(x["ins_masks"]).float() for x in batch],
            "sem_mask": torch.stack([torch.from_numpy(x["sem_mask"]).float() for x in batch]),
            "grasp_rects": [x["ins_grasp_rects"] for x in batch],
            "grasp_masks" : {
                "qua": [torch.from_numpy(x["grasp_masks"]["qua"]).float() for x in batch],
                "sin": [torch.from_numpy(x["grasp_masks"]["sin"]).float() for x in batch],
                "cos": [torch.from_numpy(x["grasp_masks"]["cos"]).float() for x in batch],
                "wid": [torch.from_numpy(x["grasp_masks"]["wid"]).float() for x in batch]
            },
        }
            