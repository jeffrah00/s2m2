# s2m2_ros2

A ROS2 (Jazzy) wrapper around [s2m2](../../..) that consumes a rectified stereo
pair and publishes a depth image suitable for [Isaac ROS Nvblox](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/).

## What it does

- Subscribes to:
  - `left/image_rect` (`sensor_msgs/Image`, rgb8/bgr8/mono8)
  - `right/image_rect` (`sensor_msgs/Image`)
  - `left/camera_info` (`sensor_msgs/CameraInfo`)
  - `right/camera_info` (`sensor_msgs/CameraInfo`, used to derive the stereo baseline from `P[3]`)
- Runs s2m2 on each synchronized stereo pair (PyTorch eager or TensorRT engine).
- Publishes:
  - `depth/image` (`sensor_msgs/Image`, `32FC1` meters by default; switchable to `16UC1` mm)
  - `depth/camera_info` (mirrors the left rectified intrinsics)
  - `depth/confidence` (`32FC1`, s2m2's per-pixel confidence map; optional)

The depth pipeline applies the standard rectified-stereo formula
`Z = fx * baseline / disp`, masks invalid disparities to 0, and clamps depth
to `[min_depth_m, max_depth_m]`.

## Install

`s2m2_ros2` is a separate package from `s2m2` and uses a different build
system (colcon / ament_python vs. pip). You must install both, in this
order, in the **same Python environment**:

**1. Install the s2m2 Python package** (so `import s2m2` works):

```bash
# from the repo root, in the conda env you intend to run the node from
pip install -e .
```

**2. Build the ROS2 wrapper** (so `ros2 launch s2m2_ros2 ...` works):

```bash
# same conda env, with ROS2 Jazzy sourced
cd ros2_ws
colcon build --packages-select s2m2_ros2 --symlink-install
source install/setup.bash
```

Why two steps: `s2m2` is a regular Python package distributed via
`pyproject.toml`; the ROS2 wrapper is an `ament_python` package built by
colcon and installed into a ROS overlay. They have different install
destinations and neither's installer knows about the other.

`torch` and `s2m2` are intentionally not declared as ROS dependencies
(rosdep cannot resolve them). They must be importable in the same Python
environment that runs the node — i.e. the env where you ran `pip install -e .`
must also be active when you run `colcon build` and `ros2 launch`, otherwise
the node will fail with `ModuleNotFoundError: s2m2` or `torch`.

## Run

```bash
ros2 launch s2m2_ros2 s2m2_depth.launch.py \
    left_image_topic:=/stereo/left/image_rect \
    right_image_topic:=/stereo/right/image_rect \
    left_info_topic:=/stereo/left/camera_info \
    right_info_topic:=/stereo/right/camera_info \
    depth_image_topic:=/camera_0/depth/image \
    depth_info_topic:=/camera_0/depth/camera_info
```

Tune behaviour through `config/s2m2_depth.yaml` (or `--ros-args -p key:=val`):

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

## Wiring into Nvblox

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

## Smoke test (no real cameras)

`ros2 run image_publisher image_publisher_node /tmp/left.png` and similar for
right, plus a hand-crafted `CameraInfo` publisher (`fx`, `baseline` matching
the dataset's `calib.txt`). Then

```bash
ros2 topic hz /camera_0/depth/image
ros2 run rqt_image_view rqt_image_view /camera_0/depth/image
```

should produce ~1 Hz depth output that visually matches running
`demo/visualize_2d_simple.py` on the same pair.

## Limitations

- Inputs must already be rectified (use `isaac_ros_image_proc` or
  `stereo_image_proc` upstream).
- The node does not publish TF — Nvblox needs that from elsewhere.
- The TensorRT backend is fixed-shape; rebuild the engine when the camera
  resolution changes.
- Occlusion mask from s2m2 is currently dropped; only disparity and
  confidence are exposed downstream.
