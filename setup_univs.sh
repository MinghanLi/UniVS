#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# Create a conda environment with Python 3.10
conda create --name univs python=3.10 -y

# Activate the conda environment
conda activate univs

# Install PyTorch and related packages
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Clone and install Detectron2
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

# Navigate back to the working directory
cd ..

# Clone and install UniVS
git clone git@github.com:MinghanLi/UniVS.git
cd UniVS
pip install -r requirements.txt

# Compile MSDeformAttn
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

# Install davis2017_evaluation
cd ../../../univs/evaluation/davis2017-evaluation
# Install it - Python 3.6 or higher required
python setup.py install

echo "Setup completed successfully."