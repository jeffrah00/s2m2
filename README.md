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

For a turnkey setup with an Intel RealSense + Nvblox, jump straight to
[Wiring into Nvblox](#wiring-into-nvblox) and use
`s2m2_realsense_nvblox.launch.py`.

For any other rectified stereo source, run the node directly with
`ros2 run` and remap topics to your camera:

```bash
ros2 run s2m2_ros2 stereo_depth_node --ros-args \
    -r left/image_rect:=/stereo/left/image_rect \
    -r right/image_rect:=/stereo/right/image_rect \
    -r left/camera_info:=/stereo/left/camera_info \
    -r right/camera_info:=/stereo/right/camera_info \
    -r depth/image:=/camera_0/depth/image \
    -r depth/camera_info:=/camera_0/depth/camera_info \
    --params-file <path>/install/s2m2_ros2/share/s2m2_ros2/config/s2m2_depth.yaml
```

Tune behaviour through `ros2_ws/src/s2m2_ros2/config/s2m2_depth.yaml`
(or `--ros-args -p key:=val`):

| param | default | meaning |
| --- | --- | --- |
| `model_type` | `S` | one of `S`, `M`, `L`, `XL` (lightest → heaviest). See "Choosing the s2m2 model size" below. |
| `weights_dir` | `weights/pretrain_weights` | directory containing `CH*NTR*.pth` |
| `refine_iter` | `3` | s2m2 iterative refinement passes |
| `allow_negative_disparity` | `false` | flips s2m2's `use_positivity` flag |
| `backend` | `pytorch` | or `tensorrt` |
| `trt_engine_path` | `""` | required when `backend=tensorrt`. Supports `{w}`/`{h}`/`{wxh}` placeholders, resolved from the first CameraInfo. |
| `device` | `cuda:0` | torch device |
| `depth_encoding` | `32FC1` | or `16UC1` (millimeters) |
| `min_depth_m` / `max_depth_m` | `0.2` / `20.0` | clamp range; outside → 0 |
| `min_confidence` | `0.0` | mask depth where s2m2 confidence < threshold |
| `publish_confidence` | `true` | also publish `depth/confidence` |
| `fx_fallback`, `baseline_m_fallback` | `0.0` | used only if `camera_info` topics never arrive |
| `sync_slop_s` | `0.05` | `ApproximateTimeSynchronizer` tolerance |
| `qos_reliability` | `reliable` | or `best_effort` |

#### Choosing the s2m2 model size

The default is `S` (lightest, fastest); higher tiers trade speed for
quality:

| `model_type` | feature channels | refinement transformers | trade-off |
| --- | --- | --- | --- |
| `S` | 128 | 1 | fastest, lowest VRAM (default) |
| `M` | 192 | 2 | balanced |
| `L` | 256 | 3 | higher quality |
| `XL` | 384 | 3 | best quality, slowest |

Three ways to override the default:

1. **Via the bundled launch** — point `s2m2_params_file` at your own YAML:
   ```bash
   ros2 launch s2m2_ros2 s2m2_realsense_nvblox.launch.py \
       s2m2_params_file:=/abs/path/to/my_params.yaml
   ```
   ```yaml
   # my_params.yaml
   s2m2_stereo_depth_node:
     ros__parameters:
       model_type: "L"   # or M / XL
   ```
2. **Via CLI param override** when launching the node directly:
   ```bash
   ros2 run s2m2_ros2 stereo_depth_node --ros-args -p model_type:=L
   ```
3. **By editing the shipped default** at
   `ros2_ws/src/s2m2_ros2/config/s2m2_depth.yaml` and rebuilding with
   `colcon build --packages-select s2m2_ros2`.

#### Camera resolution handling

The node auto-detects width and height from the first `CameraInfo` it
receives and logs `auto-detected camera resolution: <W>x<H>`. That
detection is used for two things:

- **Eager warmup** — the PyTorch backend runs a single warmup forward
  pass at the detected resolution before the first stereo image arrives,
  so first-frame latency stays low.
- **TensorRT engine selection** — if `trt_engine_path` contains `{w}`,
  `{h}`, or `{wxh}`, the placeholders are substituted and the matching
  engine is loaded on the first CameraInfo. Example:
  `trt_engine_path: '/abs/path/CH128NTR1_{wxh}_fp16.trt'` resolves to
  `/abs/path/CH128NTR1_848x480_fp16.trt` on a D435i at 848×480.

The node accepts **any** image dimensions — RealSense profiles like
848×480, 1280×720, and 1920×1080 are not divisible by 32, but s2m2's
internal `image_pad`/`image_crop` round-trip handles that natively, so
depth is published at the camera's native dimensions with K from
CameraInfo unchanged. Nvblox sees depth and color at the same size,
which keeps voxel coverage maximal.

### TensorRT backend

Build an engine via the s2m2 demo at the resolution your stereo camera streams
(post pad-to-32):

```bash
python demo/export_tensorrt.py --model_type L --img_width 1216 --img_height 1024 --precision fp16
```

Then run with:

```bash
ros2 run s2m2_ros2 stereo_depth_node --ros-args \
    -p backend:=tensorrt -p trt_engine_path:=/abs/path/to/engine.trt
```

(or set the same params under `s2m2_stereo_depth_node.ros__parameters` in
`config/s2m2_depth.yaml` and pass `--params-file …` so they take effect
under `s2m2_realsense_nvblox.launch.py` as well.)

The TensorRT engine is fixed-shape; if the incoming image (after pad-to-32)
does not match the engine's input dims, the node logs an error and skips
the frame.

### Wiring into Nvblox

Nvblox consumes `camera_*/depth/image` plus `camera_*/depth/camera_info`.
The supported integration is the bundled
`s2m2_realsense_nvblox.launch.py` described below, which wires s2m2 into
Isaac ROS Nvblox's RealSense example end-to-end.

Nvblox additionally requires a TF tree and odometry; that is the
responsibility of the rest of your robot stack and is intentionally out of
scope here.

#### Replacing RealSense depth in `realsense_example.launch.py`

`nvblox_examples_bringup`'s `realsense_example.launch.py` brings up the
RealSense driver, a `realsense_splitter_node` (which gates the IR projector),
and nvblox subscribed to:

| nvblox topic | RealSense source |
| --- | --- |
| `camera_0/depth/image` | `/camera0/realsense_splitter_node/output/depth` |
| `camera_0/depth/camera_info` | `/camera0/depth/camera_info` |
| `camera_0/color/image` | `/camera0/color/image_raw` |
| `camera_0/color/camera_info` | `/camera0/color/camera_info` |

##### One-shot launch (recommended)

`ros2_ws/src/s2m2_ros2/launch/s2m2_realsense_nvblox.launch.py` is a
**vendored copy** of `nvblox_examples_bringup/launch/realsense_example.launch.py`
(pinned to `NVIDIA-ISAAC-ROS/isaac_ros_nvblox` branch
[`release-4.3`](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/tree/release-4.3))
with a few small additions, each fenced by `# >>> s2m2:` … `# <<< s2m2`
markers:

1. four new launch args (`depth_source`, `s2m2_params_file`,
   `s2m2_depth_topic`, `s2m2_depth_info_topic`),
2. a `SetRemap` (gated on `depth_source==s2m2`) that diverts nvblox's
   `camera_0/depth/image` subscription to the s2m2 depth topic, and
3. a `Node(...)` (also gated on `depth_source==s2m2`) for
   `s2m2_stereo_depth_node`, subscribed to the
   `realsense_splitter_node`'s emitter-off IR pair.

`depth_source` defaults to `s2m2`, so a bare
`ros2 launch s2m2_ros2 s2m2_realsense_nvblox.launch.py` already runs
nvblox on s2m2 depth. Pass `depth_source:=realsense` to skip s2m2 and
fall back to the splitter-driven depth instead.

Run it with:

```bash
ros2 launch s2m2_ros2 s2m2_realsense_nvblox.launch.py
```

You inherit everything the upstream launch already does — RealSense
driver, splitter, VSLAM, nvblox, RViz/Foxglove, optional people
segmentation/detection, multi-camera, rosbag playback. s2m2 subscribes
to the splitter's clean (emitter-off) IR at
`/camera0/realsense_splitter_node/output/infra_{1,2}`, so we don't touch
the emitter config.

Args added by the s2m2 patch (every other arg comes straight from the
vendored upstream file — `mode`, `num_cameras`, `run_realsense`,
`rosbag`, `rosbag_args`, `log_level`, `container_name`,
`attach_to_container`, `use_foxglove_whitelist`,
`camera_serial_numbers`, `people_segmentation`, `multicam_urdf_path`):

| arg | default | purpose |
| --- | --- | --- |
| `s2m2_params_file` | `<share>/s2m2_ros2/config/s2m2_depth.yaml` | s2m2 node parameters |
| `s2m2_depth_topic` | `/camera0/s2m2/depth/image` | where s2m2 publishes; nvblox is remapped to subscribe here |
| `s2m2_depth_info_topic` | `/camera0/depth/camera_info` | CameraInfo for the s2m2 depth output |
| `depth_source` | `s2m2` | `s2m2` (default) routes nvblox to s2m2 depth and spawns the s2m2 node; `realsense` skips both and falls back to the upstream splitter-driven depth |

To re-vendor against a newer Isaac ROS release, refetch
`nvblox_examples_bringup/launch/realsense_example.launch.py` from the
new tag, then re-apply the four `# >>> s2m2: … # <<< s2m2` blocks (one
import, one args block, one `SetRemap`, one `Node`).

##### Manual three-terminal version

If you'd rather run each component yourself (e.g. you already have a
RealSense bringup file):

```bash
# terminal 1 — RealSense driver only (no splitter, IR emitter disabled)
ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera0 \
    enable_infra1:=true enable_infra2:=true enable_color:=true \
    depth_module.emitter_enabled:=0 \
    align_depth.enable:=false \
    rgb_camera.profile:=640x480x30 \
    depth_module.profile:=640x480x30

# terminal 2 — s2m2 stereo depth from the RealSense IR pair,
# publishing to /camera0/depth/image
ros2 run s2m2_ros2 stereo_depth_node --ros-args \
    -r left/image_rect:=/camera0/infra1/image_rect_raw \
    -r right/image_rect:=/camera0/infra2/image_rect_raw \
    -r left/camera_info:=/camera0/infra1/camera_info \
    -r right/camera_info:=/camera0/infra2/camera_info \
    -r depth/image:=/camera0/depth/image \
    -r depth/camera_info:=/camera0/depth/camera_info

# terminal 3 — nvblox alone, with depth/color remapped to our outputs
ros2 launch nvblox_examples_bringup nvblox.launch.py \
    --ros-args \
        -r camera_0/depth/image:=/camera0/depth/image \
        -r camera_0/depth/camera_info:=/camera0/depth/camera_info \
        -r camera_0/color/image:=/camera0/color/image_raw \
        -r camera_0/color/camera_info:=/camera0/color/camera_info
```

##### Verifying the pipeline

`rqt_graph` should show `s2m2_stereo_depth_node` as the only publisher on
`/camera0/depth/image` and `nvblox_node` as its subscriber. In RViz,
visualize `/nvblox_node/mesh` (or `/nvblox_node/esdf_pointcloud`) to
confirm the reconstruction is being driven by s2m2 depth.

A few RealSense-specific notes:
- s2m2 expects RGB inputs (3 channels). The node automatically replicates
  RealSense's `mono8` IR images to 3 channels, so no extra conversion is
  needed.
- Baseline is read from `/camera0/infra2/camera_info`'s `P[3]` (≈0.05 m on
  D435, ≈0.095 m on D455). If the right `camera_info` doesn't carry it,
  set `baseline_m_fallback` (50 mm for D435, 95 mm for D455).
- Drop `depth_module.emitter_enabled:=0` only if you're using a fixed mount
  with separate room lighting — leaving the dot projector on hurts s2m2
  badly because the dots break the assumption of natural-texture stereo.

References:
- [Isaac ROS Nvblox: RealSense tutorial](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_realsense.html)
- [Isaac ROS Nvblox: topics & services](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/isaac_ros_nvblox/api/topics_and_services.html)

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
