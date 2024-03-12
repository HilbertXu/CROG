import argparse
from curses import meta
import os
import warnings
import json

import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import inference_with_grasp
from model import build_segmenter
from utils.dataset import OCIDVLGDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    
    save_path = os.path.join("./results", args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.resume))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    meta_data = json.load(open("./refer_types.json"))
    print(meta_data.keys())

    for key in meta_data.keys():
        logger.info(f"Start testing {key} expressions")
        indices = meta_data[key]

        # build dataset & dataloader
        test_data = OCIDVLGDataset(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split='test',
                                version=args.version)
        test_data = torch.utils.data.Subset(test_data, indices)
        test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=12,
                                              shuffle=False,
                                              num_workers=8,
                                              pin_memory=True,
                                              collate_fn=OCIDVLGDataset.collate_fn)

        # inference
        inference_with_grasp(test_loader, model, args)


if __name__ == '__main__':
    main()
