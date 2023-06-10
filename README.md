# Reliable Student: Addressing Noise in Semi-Supervised 3D Object Detection

This is the official implementation of the paper "Reliable Student: Addressing Noise in Semi-Supervised 3D Object Detection".

## Results

Evaluation results on the KITTI dataset. 


| Labeled Percent         |       | 1%  |      |       | 2%  |      |
|-------------------------|-------|-----|------|-------|-----|------|
| Method                  | Car   | Pedestrian | Cyclist |  Car  | Pedestrian | Cyclist |
| PV-RCNN                 | 74.1  | 31.7 | 28.8 | 76.8  | 40.4 | 42.3 |
| 3DIoUMatch (Baseline)   | 76.4  | 35.7 | 36.0 | 78.9  | 47.0 | 53.3 |
| ReliableStudent         | 77.0  | 41.9 | 36.4 | 79.5  | 53.0 | 59.0 |


## Installation

```shell
# Create a conda environment
conda create -n reliable_student python=3.8
conda activate reliable_student

# Install PyTorch 1.7.1 with CUDA 11.0
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# Install spconv 1.2.1
git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive
cd spconv
python setup.py bdist_wheel
python -m pip install ./dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl
cd ..

# Clone the repo, install requirements and compile
git clone https://github.com/fnozarian/ReliableStudent.git
cd ReliableStudent
python -m pip install -r requirements.txt
python setup.py develop
```

## Getting Started

### Data Preparation
Please refer to [Data Preparation](https://github.com/yezhen17/3DIoUMatch-PVRCNN#data-preparation) to generate the data splits or use the baseline data we provide.

### Pre-training
```shell
# Single GPU (e.g., A100-80GB)
python -u pretrain.py \
    --cfg_file ./cfgs/kitti_models/pv_rcnn.yaml \
    --batch_size 8 \
    --split train_0.01_1 \
    --extra_tag train_0.01_1 \
    --ckpt_save_interval 4 \
    --repeat 10 \
    --dbinfos kitti_dbinfos_train_0.01_1_37.pkl

# Multi-GPU with Slurm (e.g., A100-40GB x2)
srun -p A100-40GB --job-name=pretrain_0.01_1 --ntasks=2 \
    --gpus-per-task=1 --cpus-per-task=3 \
    --mem=60GB --kill-on-bad-exit=1 python -u pretrain.py \
    --cfg_file ./cfgs/kitti_models/pv_rcnn.yaml --batch_size 8 \
    --launcher slurm --split train_0.01_1 --extra_tag train_0.01_1 \
    --ckpt_save_interval 4 --repeat 10 \
    --dbinfos kitti_dbinfos_train_0.01_1_37.pkl
```

### Training
```shell
# Single GPU (e.g., A100-80GB)
python -u train.py \
    --cfg_file ./cfgs/kitti_models/pv_rcnn_ssl_60.yaml \
    --batch_size 8 \
    --launcher slurm \
    --split train_0.01_1 \
    --extra_tag train_0.01_1 \
    --ckpt_save_interval 2 \
    --pretrained_model "../output/cfgs/kitti_models/pv_rcnn/split_0.01_1/ckpt/checkpoint_epoch_80.pth" \
    --repeat 5 \
    --dbinfos kitti_dbinfos_train_0.01_1_37.pkl
    
# Multi-GPU with Slurm (e.g., A100-40GB x2)
srun -p A100-40GB --job-name=train_0.01_1 --ntasks=2 \
    --gpus-per-task=1 --cpus-per-task=3 \
    --mem=60GB --kill-on-bad-exit=1 python -u train.py \
    --cfg_file ./cfgs/kitti_models/pv_rcnn_ssl_60.yaml \
    --batch_size 8 --launcher slurm --split train_0.01_1 \
    --extra_tag train_0.01_1 --ckpt_save_interval 2 \
    --pretrained_model "../output/cfgs/kitti_models/pv_rcnn/split_0.01_1/ckpt/checkpoint_epoch_80.pth" \
    --repeat 5 --dbinfos kitti_dbinfos_train_0.01_1_37.pkl
```

## Acknowledgements

Our code is based on [3DIoUMatch-PVRCNN](https://github.com/THU17cyz/3DIoUMatch-PVRCNN) and [OpenPCDet v0.5](https://github.com/open-mmlab/OpenPCDet/tree/v0.5.0).
Thanks OpenPCDet Development Team for their awesome codebase.

This work has been funded by the German Ministry for Education and Research (BMB+F) in the project MOMENTUM.

## Citation

If you find this work useful, please consider citing:
```
@InProceedings{Nozarian_2023_CVPR,
    author    = {Nozarian, Farzad and Agarwal, Shashank and Rezaeianaran, Farzaneh and Shahzad, Danish and Poibrenski, Atanas and M\"uller, Christian and Slusallek, Philipp},
    title     = {Reliable Student: Addressing Noise in Semi-Supervised 3D Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {4980-4989}
}
```
