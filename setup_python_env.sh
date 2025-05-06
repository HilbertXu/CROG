#!/bin/bash

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install scikit-image scipy tqdm loguru triton wandb pycocotools opencv-python lmdb pyarrow shapely ftfy regex 
pip install 'numpy<2'
