<div align="center">
<h3>S<sup>2</sup>M<sup>2</sup>: Scalable Stereo Matching Model for Reliable Depth Estimation (ICCV 2025)</h3>
<h5>Junhong Min¹<sup>*</sup>, Youngpil Jeon¹, Jimin Kim¹, Minyong Choi¹</h5>
<p align="center">
  <a href="https://arxiv.org/abs/2507.13229">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2507.13229-b31b1b?style=for-the-badge&logo=arxiv">
  </a>
  <a href="https://junhong-3dv.github.io/s2m2-project">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-S²M²%20Page-brightgreen?style=for-the-badge&logo=googlechrome">
  </a>
</p>

[comment]: <> (<p align="center">)

[comment]: <> (  <img src="fig/thumbnail.png" width="90%">)

[comment]: <> (</p>)

</div>


## 🤗 Notice
>This repository and its contents are **not related to any official Samsung Electronics products**.  
>All resources are provided **solely for non-commercial research and education purposes**.
---

## ✨ Key Features
### 🧩 Model
- Scalable stereo matching architecture  
- State-of-the-art performance on ETH3D (1st), Middlebury V3 (1st), and Booster (1st)
- Joint estimation of disparity, occlusion, and confidence
- Supports negative disparity estimation
- Optimal under the pinhole camera model with ideal stereo rectification (vertical disparity < 2px)

### ⚙️ Code
- ✅ FP16 / FP32 inference
- ✅ TorchScript/ONNX/TensorRT export
- ❌ Training pipeline (not included)

> Note: The publicly released model weights differ from the version used for Middlebury and ETH but are identical to the one used in Booster benchmark. This implementation replaces the dynamic attention-based refinement module with an UNet for stable ONNX export. It also includes an additional M variant and extended training data with transparent objects. 

---

## 🚀 Performance

> Detailed benchmark results and visualizations are available on the [Project Page](https://junhong-3dv.github.io/s2m2-project).

Inference Speed (FPS)** on **NVIDIA RTX 5090 (float16 + `refine_iter=3`):

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>CH</th>
            <th>NTR</th>
            <th>Inference</th>
            <th>640x480</th>
            <th>1216x1024</th>
            <th>2432x2048</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>S</td>
            <td rowspan=2>128</td>
            <td rowspan=2>1</td>
            <td>torch.compile</td>
            <td>59.8</td>
            <td>26.4</td>
            <td>6.6</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>124.0</td>
            <td>59.4</td>
            <td>7.3</td>
        </tr>
        <tr>
            <td rowspan=2>M</td>
            <td rowspan=2>192</td>
            <td rowspan=2>2</td>
            <td>torch.compile</td>
            <td>32.4</td>
            <td>12.5</td>
            <td>3.1</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>66.7</td>
            <td>18.3</td>
            <td>3.8</td>
        </tr>
        <tr>
            <td rowspan=2>L</td>
            <td rowspan=2>256</td>
            <td rowspan=2>3</td>
            <td>torch.compile</td>
            <td>25.8</td>
            <td>7.6</td>
            <td>2.0</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>46.6</td>
            <td>11.2</td>
            <td>2.4</td>
        </tr>
        <tr>
            <td rowspan=2>XL</td>
            <td rowspan=2>384</td>
            <td rowspan=2>3</td>
            <td>torch.compile</td>
            <td>16.3</td>
            <td>4.3</td>
            <td>1.1</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>26.6</td>
            <td>6.4</td>
            <td>1.4</td>
        </tr>
    </tbody>
</table>

---

## 🔧 Installation

We recommend using Python 3.10, PyTorch 2.9, CUDA 12.9, CUDNN 9.1.0, and tensorRT 10.13.3 with Anaconda.

```bash
git clone https://github.com/junhong-3dv/s2m2
cd s2m2
```
```bash
conda env create -n s2m2 -f environment.yml
conda activate s2m2
```
```bash
pip install -e .
```


If the environment setup via .yml doesn’t work smoothly,
you can manually install the main dependencies with:
```bash
pip install torch torchvision opencv-python open3d onnx onnxruntime-gpu onnxscript tensorrt-cu12==10.13.3.9 --extra-index-url https://pypi.nvidia.com/tensorrt-cu12-libs

```
That should cover most of the required packages for running the demo.

## 🚀 Pre-trained Models and Inference

### 1. Download Pre-trained Models

Create a directory for weights and download the desired models from the links below.

```bash
mkdir weights
mkdir weights/pretrain_weights
```

| Model | Download | Model Size |
| :---: | :---: | :--: |
| **S** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH128NTR1.pth) | 26.5M | 
| **M** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH192NTR2.pth) | 80.4M | 
| **L** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH256NTR3.pth) | 181M | 
| **XL**| [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth) | 406M | 

### 2. Run Basic Demo
To generate a result for a single input, run `demo/visualize_2d_simple.py`.

```bash
python ./demo/visualize_2d_simple.py --model_type XL --num_refine 3 
```
| arg | default | type | help |
| :---: | :---: | :--: | :--: |
| --model_type | 'XL' | str | select model type: [S,M,L,XL] |
| --num_refine | 3 | int | number of local iterative refinement |
| --torch_compile | False | set_true | apply torch_compile  | 
| --allow_negative | False | set_true | allow negative disparity  | 


### 3. Run 3D Visualization Demo

To visualize the 3D output interactively, run `demo/visualize_3d_booster.py` or `demo/visualize_3d_middlebury.py`

```bash
python ./demo/visualize_3d_booster.py --model_type L 
```
For 'visualize_3d_middlebury.py --model_type XL ', result should be like below. 
<p align="center">
  <img src="fig/result_bicycle2.png" width="90%">
</p>
If you failed to reproduce this, let me know. 

## 🚀 Model Optimization (ONNX / TensorRT)

### 1. Export to ONNX (OPSET=18)

Use export_onnx.py to convert the model to ONNX:

```console
python demo/export_onnx.py --model_type $MODEL_TYPE --img_width $IMG_WIDHT --img_height $IMG_HEIGHT
```
### 2. Export to TensorRT (tested with 10.13.3)
Use export_tensorrt.py to build a TensorRT engine:

```console
python demo/export_tensorrt.py --model_type $MODEL_TYPE --img_width $IMG_WIDHT --img_height $IMG_HEIGHT --precision $PRECISION
```
or simply in the terminal
```console
trtexec --onnx={onnx_file_path} --saveEngine=./{trt_file_path} --fp16 --precisionConstraints=obey --layerPrecisions=node_linalg_vector_norm_2:fp32
```
Supported TensorRT precisions: fp32, tf32, fp16

## 🤖 ROS2 Wrapper (for Isaac ROS Nvblox)

An optional ROS2 wrapper lives at `ros2_ws/src/s2m2_ros2/`. It subscribes to
a rectified stereo pair and publishes a depth image plus `CameraInfo` in the
format Isaac ROS Nvblox consumes. Tested on **Humble** (Ubuntu 22.04),
**Jazzy** (Ubuntu 24.04), and **Kilted** (Ubuntu 24.04).

### Topics

- Subscribes to:
  - `left/image_rect` (`sensor_msgs/Image`, rgb8/bgr8/mono8)
  - `right/image_rect` (`sensor_msgs/Image`)
  - `left/camera_info` (`sensor_msgs/CameraInfo`)
  - `right/camera_info` (`sensor_msgs/CameraInfo`, used to derive the stereo baseline from `P[3]`)
- Publishes:
  - `depth/image` (`sensor_msgs/Image`, `32FC1` meters by default; switchable to `16UC1` mm)
  - `depth/camera_info` (mirrors the left rectified intrinsics)
  - `depth/confidence` (`32FC1`, s2m2's per-pixel confidence map; optional)

The depth pipeline applies the standard rectified-stereo formula
`Z = fx * baseline / disp`, masks invalid disparities to 0, and clamps depth
to `[min_depth_m, max_depth_m]`.

### Install

The wrapper is built with `colcon` (ament_python) and is *separate* from the
`pip install -e .` step above, but the two must share the **same Python
environment**.

#### Auto-detect (recommended)

`scripts/install.sh` reads `/etc/os-release`, picks the matching ROS2 distro,
sources it, then runs both install steps:

```bash
./scripts/install.sh
source ros2_ws/install/setup.bash
```

| Ubuntu | Default distro picked | Other supported |
| --- | --- | --- |
| 22.04 (Jammy) | Humble | — |
| 24.04 (Noble) | Jazzy | Kilted (`./scripts/install.sh --distro kilted`) |

Override the auto-detect with `--distro humble|jazzy|kilted`, or by exporting
`ROS_DISTRO` before running. Use `--skip-pip` / `--skip-colcon` to do only
half of the install.

#### Manual install

If you'd rather run the steps yourself, do them in this order in the same
Python env:

```bash
source /opt/ros/<distro>/setup.bash      # humble, jazzy, or kilted
pip install -e .                          # 1. installs the s2m2 Python package
cd ros2_ws
colcon build --packages-select s2m2_ros2 --symlink-install   # 2. builds the wrapper
source install/setup.bash
```

`torch` and `s2m2` are intentionally not declared as ROS dependencies
(rosdep cannot resolve them). The conda env where you ran `pip install -e .`
must also be active when you run `colcon build` and `ros2 launch`, otherwise
the node will fail with `ModuleNotFoundError: s2m2` or `torch`.

### Run

```bash
ros2 launch s2m2_ros2 s2m2_depth.launch.py \
    left_image_topic:=/stereo/left/image_rect \
    right_image_topic:=/stereo/right/image_rect \
    left_info_topic:=/stereo/left/camera_info \
    right_info_topic:=/stereo/right/camera_info \
    depth_image_topic:=/camera_0/depth/image \
    depth_info_topic:=/camera_0/depth/camera_info
```

Tune behaviour through `ros2_ws/src/s2m2_ros2/config/s2m2_depth.yaml`
(or `--ros-args -p key:=val`):

| param | default | meaning |
| --- | --- | --- |
| `model_type` | `L` | one of `S`, `M`, `L`, `XL` |
| `weights_dir` | `weights/pretrain_weights` | directory containing `CH*NTR*.pth` |
| `refine_iter` | `3` | s2m2 iterative refinement passes |
| `allow_negative_disparity` | `false` | flips s2m2's `use_positivity` flag |
| `backend` | `pytorch` | or `tensorrt` |
| `trt_engine_path` | `""` | required when `backend=tensorrt` |
| `device` | `cuda:0` | torch device |
| `depth_encoding` | `32FC1` | or `16UC1` (millimeters) |
| `min_depth_m` / `max_depth_m` | `0.2` / `20.0` | clamp range; outside → 0 |
| `min_confidence` | `0.0` | mask depth where s2m2 confidence < threshold |
| `publish_confidence` | `true` | also publish `depth/confidence` |
| `fx_fallback`, `baseline_m_fallback` | `0.0` | used only if `camera_info` topics never arrive |
| `sync_slop_s` | `0.05` | `ApproximateTimeSynchronizer` tolerance |
| `qos_reliability` | `reliable` | or `best_effort` |

### TensorRT backend

Build an engine via the s2m2 demo at the resolution your stereo camera streams
(post pad-to-32):

```bash
python demo/export_tensorrt.py --model_type L --img_width 1216 --img_height 1024 --precision fp16
```

Then launch with:

```bash
ros2 launch s2m2_ros2 s2m2_depth.launch.py \
    --ros-args -p backend:=tensorrt -p trt_engine_path:=/abs/path/to/engine.trt
```

The TensorRT engine is fixed-shape; if the incoming image (after pad-to-32)
does not match the engine's input dims, the node logs an error and skips
the frame.

### Wiring into Nvblox

Nvblox consumes `camera_*/depth/image` plus `camera_*/depth/camera_info`. The
default launch arguments already publish to `/camera_0/depth/...`, so a
typical setup is:

```bash
# terminal 1: stereo source publishing rectified left/right images + camera_info
ros2 launch isaac_ros_image_proc isaac_ros_image_proc.launch.py ...

# terminal 2: this node
ros2 launch s2m2_ros2 s2m2_depth.launch.py

# terminal 3: nvblox, subscribing to /camera_0/depth/...
ros2 launch nvblox_examples_bringup ...
```

Nvblox additionally requires a TF tree and odometry; that is the
responsibility of the rest of your robot stack and is intentionally out of
scope here.

### Limitations

- Inputs must already be rectified (use `isaac_ros_image_proc` or
  `stereo_image_proc` upstream).
- The node does not publish TF — Nvblox needs that from elsewhere.
- The TensorRT backend is fixed-shape; rebuild the engine when the camera
  resolution changes.
- The s2m2 occlusion mask is currently dropped; only disparity and
  confidence are exposed downstream.

## 📜 Citation

If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{min2025s2m2,
  title={{S\textsuperscript{2}M\textsuperscript{2}}: Scalable Stereo Matching Model for Reliable Depth Estimation},
  author={Junhong Min and Youngpil Jeon and Jimin Kim and Minyong Choi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
