# 
<h2 align="center"> <a href="https://motion3-to-4.github.io/">Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2503.24391-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2601.14253) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://motion3-to-4.github.io/) 

[Hongyuan Chen](https://hyuanChen.github.io/),
[Xingyu Chen](https://rover-xingyu.github.io/),
[Youjia Zhang](https://youjiazhang.github.io/),
[Zexiang Xu](https://zexiangxu.github.io/),
[Anpei Chen](https://apchenstu.github.io/),
</h5>
<div align="center">

Motion 3-to-4 reconstructs 3D motion from videos for 4D synthesis in a **feedforward** mannar within seconds.
</div>

## Quick Start

For users who want to quickly try the inference:

```bash
git clone https://github.com/Inception3D/Motion324.git
cd Motion324

# 1. Setup environment
conda create -n Motion324 python=3.11
conda activate Motion324
pip install -r requirements.txt
# Install Hunyuan3D-2.0 components(optional)
cd scripts/hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../../..
cd scripts/hy3dgen/texgen/differentiable_renderer
python3 setup.py install
cd ../../../..

# 2. Download pre-trained checkpoints and place in experiments/checkpoints/

# 3. Run inference
chmod +x ./scripts/4D_from_existing.sh
./scripts/4D_from_existing.sh ./examples/chili.glb ./examples/chili.mp4 ./examples/output

# Hunyuan needed
chmod +x ./scripts/4D_from_video.sh
./scripts/4D_from_video.sh ./examples/tiger.mp4
```

## 1. Preparation

### Checkpoints

**Download**: Please download the pre-trained checkpoint from [here](https://huggingface.co/River-Chen/Motion324/tree/main) and place it in `experiments/checkpoints/`.

### Environment Details
#### Setup up base environment
```bash
conda create -n Motion324 python=3.11
conda activate Motion324
pip install -r requirements.txt
```
The code has been tested with Python 3.11 + Pytorch 2.4.1 + CUDA 12.4.

#### Setup Hunyuan3D-2.0 Components
```bash
# Install custom rasterizer
cd scripts/hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../../..

# Install differentiable renderer
cd scripts/hy3dgen/texgen/differentiable_renderer
python3 setup.py install
cd ../../../..
```
#### Setup Blender
Download and install Blender for 4D asset rendering.

Our results is rendered with [blender-4.0.0-linux-x64](https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz), using the scripts which is modified from [bpy-renderer](https://github.com/huanngzh/bpy-renderer).

Installation steps:
```bash
# Download Blender
wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xf blender-4.0.0-linux-x64.tar.xz

# Add Blender to PATH (optional, or use full path in scripts)
export PATH=$PATH:$(pwd)/blender-4.0.0-linux-x64
```

**Note**: As we use [xformers](https://github.com/facebookresearch/xformers) `memory_efficient_attention` with [flash_attn](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1), the GPU device compute capability needs > 8.0. Otherwise, it would pop up an error. Check your GPU compute capability in [CUDA GPUs Page](https://developer.nvidia.com/cuda-gpus#compute).

### Dataset
- [ ] TODO: Dataset will be provided. Please check back for updates.

Update the dataset path in `configs/dyscene.yaml`:
```yaml
training:
  dataset_path: /path/to/your/dataset
  train_lst: /path/to/name_list
```

## 2. Training

Before training, you need to follow the instructions [here](https://docs.wandb.ai/guides/track/public-api-guide/#:~:text=You%20can%20generate%20an%20API,in%20the%20upper%20right%20corner.) to generate the Wandb key file for logging and save it in the `configs` folder as `api_keys.yaml`.

### Training Command

The default training uses `configs/dyscene.yaml`:

```bash
torchrun --nproc_per_node 8 --nnodes 1 --master_port 12344 train.py --config configs/dyscene.yaml
```

### Training Configuration

Key training parameters in `configs/dyscene.yaml`:
You can override any config parameter via command line:

```bash
torchrun --nproc_per_node 8 --nnodes 1 --master_port 12346 train.py --config configs/dyscene.yaml \
    training.batch_size_per_gpu=32
```

## 3. Inference

### Generate 4D animation from a single video input

**Input**: Video file (`.mp4/.avi/.mov`) or image directory (use `./scripts/images2video.py` to convert images to video first)

**Output**: 
- Processed frames and mesh files in `{video_name}_processed/`
- Animation output in `{video_name}_processed/animation/` (FBX format)

**Example**:
```bash
chmod +x ./scripts/4D_from_video.sh
./scripts/4D_from_video.sh ./examples/tiger.mp4
```

### Reconstruct 4D from an existing mesh and video

**Inputs**:
- `data_dir`: Mesh file (`.glb` or `.fbx`) - FBX files will be automatically converted to GLB
- `video_path`: Video file (`.mp4/.avi/.mov`) or image directory
- `output_dir`: Output directory for results

**Output**: 
- Animated mesh files (GLB format) in the specified output directory
- Segmented videos if segmentation is enabled

**Example**:
```bash
chmod +x ./scripts/4D_from_existing.sh
./scripts/4D_from_existing.sh ./examples/chili.glb ./examples/chili.mp4 ./examples/output
```

## 4. Evaluation

You can evaluate video results with the provided `evaluation.py` script. 

Example:

```bash
python ./evaluation/evaluation.py \
--gt_paths /paths/to/gt_videos.mp4 \
--result_paths /paths/to/results_videos.mp4
```
This compares the generated result video(s) to the ground-truth and outputs metrics such as FVD, LPIPS, DreamSim, and CLIP Loss. 

You can evaluate mesh geometry with the provided `evaluation_pcd.py` script. 

Example:

```bash
python ./evaluation/evaluation_pcd.py \
--gt_path /paths/to/name_pointclouds \
--result_path /paths/to/mesh.fbx
```
This compares your mesh result with the ground-truth point cloud and evaluates the geometric error between them. 
It outputs metrics such as Chamfer Distance and F-score.

## 5. Citation 

If you find this work useful in your research, please consider citing:

```bibtex
@article{chen2026motion3to4,
    title={Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis},
    author={Hongyuan, Chen and Xingyu, Chen and Youjia Zhang, and Zexiang, Xu and Anpei, Chen},
    journal={arXiv preprint arXiv:2601.14253},
    year={2026}
}
```
## 6. Acknowledgments
- [LVSM](https://github.com/Haian-Jin/LVSM) (for code architecture reference)
- [V2M4](https://github.com/WindVChen/V2M4), [AnimateAnyMesh](https://github.com/JarrentWu1031/AnimateAnyMesh) (for code reference)
- [bpy-renderer](https://github.com/huanngzh/bpy-renderer) (for rendering results)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) (for 3D generation)
  
## 7. License

This project is licensed under the [CC BY-NC-SA 4.0 License](LICENSE.md) - see the LICENSE.md file for details.
