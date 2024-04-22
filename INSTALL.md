# Installation

## Requirements
- Linux or macOS with Python ≥ 3.10
- Torch == 2.1.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation (0.16.1+cu121).
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

- After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

  `CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

  ```bash
  cd mask2former/modeling/pixel_decoder/ops
  sh make.sh
  ```

## Example conda environment setup
```bash
conda create --name univs python=3.10 -y
conda activate univs
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone git@github.com:MinghanLi/UniVS.git
cd UniVS
pip install -r requirements.txt
# compile MSDeformAttn
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

# install davis2017_evaluation
cd univs/evaluation/davis2017-evaluation
# Install it - Python 3.6 or higher required
python setup.py install
```
