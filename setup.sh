#!/usr/bin/env bash
# Needs an NVIDIA GPU + CUDA 12.8. Request checkpoint access first:
#   https://huggingface.co/facebook/sam3   then:  hf auth login
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
pip install -e third_party/sam3
pip install opencv-python pillow numpy matplotlib
