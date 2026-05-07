"""ROS2 node that runs s2m2 on a rectified stereo pair and publishes depth.

Output topics are aligned with Isaac ROS Nvblox conventions
(``camera_*/depth/image`` + ``camera_*/depth/camera_info``); remap them in
the launch file to fit your pipeline.
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__("s2m2_stereo_depth_node")

        ros_distro = os.environ.get("ROS_DISTRO", "<unknown>")
        self.get_logger().info(
            f"ROS_DISTRO={ros_distro}, python={sys.version.split()[0]}"
        )

        self._declare_params()
        self._read_params()

        self.bridge = CvBridge()
        self.backend = None
        self._left_info = None
        self._right_info = None
        self._fx = float(self.fx_fallback)
        self._baseline = float(self.baseline_m_fallback)
        self._warmed_up = False

        qos = self._build_qos()

        self.sub_left = Subscriber(self, Image, "left/image_rect", qos_profile=qos)
        self.sub_right = Subscriber(self, Image, "right/image_rect", qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_right],
            queue_size=10,
            slop=self.sync_slop_s,
        )
        self.sync.registerCallback(self._stereo_callback)

        self.create_subscription(
            CameraInfo, "left/camera_info", self._left_info_cb, 10
        )
        self.create_subscription(
            CameraInfo, "right/camera_info", self._right_info_cb, 10
        )

        self.pub_depth = self.create_publisher(Image, "depth/image", 10)
        self.pub_depth_info = self.create_publisher(CameraInfo, "depth/camera_info", 10)
        self.pub_conf = (
            self.create_publisher(Image, "depth/confidence", 10)
            if self.publish_confidence
            else None
        )

        self._init_backend()

        self.get_logger().info(
            f"s2m2 stereo depth node ready (backend={self.backend_name}, "
            f"model_type={self.model_type}, encoding={self.depth_encoding})"
        )

    def _declare_params(self):
        self.declare_parameter("model_type", "L")
        self.declare_parameter("weights_dir", "weights/pretrain_weights")
        self.declare_parameter("refine_iter", 3)
        self.declare_parameter("allow_negative_disparity", False)
        self.declare_parameter("backend", "pytorch")
        self.declare_parameter("trt_engine_path", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("depth_encoding", "32FC1")
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("publish_confidence", True)
        self.declare_parameter("fx_fallback", 0.0)
        self.declare_parameter("baseline_m_fallback", 0.0)
        self.declare_parameter("sync_slop_s", 0.05)
        self.declare_parameter("qos_reliability", "reliable")

    def _read_params(self):
        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.model_type = gp("model_type").string_value
        self.weights_dir = gp("weights_dir").string_value
        self.refine_iter = gp("refine_iter").integer_value
        self.allow_negative = gp("allow_negative_disparity").bool_value
        self.backend_name = gp("backend").string_value.lower()
        self.trt_engine_path = gp("trt_engine_path").string_value
        self.device_str = gp("device").string_value
        self.depth_encoding = gp("depth_encoding").string_value
        self.min_depth_m = gp("min_depth_m").double_value
        self.max_depth_m = gp("max_depth_m").double_value
        self.min_confidence = gp("min_confidence").double_value
        self.publish_confidence = gp("publish_confidence").bool_value
        self.fx_fallback = gp("fx_fallback").double_value
        self.baseline_m_fallback = gp("baseline_m_fallback").double_value
        self.sync_slop_s = gp("sync_slop_s").double_value
        self.qos_reliability = gp("qos_reliability").string_value.lower()

        if self.depth_encoding not in ("32FC1", "16UC1"):
            raise ValueError(
                f"depth_encoding must be 32FC1 or 16UC1, got {self.depth_encoding}"
            )

    def _warn_throttled(self, msg: str, sec: float = 5.0) -> None:
        # rclpy gained throttle_duration_sec on Logger.warn at different points
        # across distros; fall back to an unthrottled warn if it's missing.
        try:
            self.get_logger().warn(msg, throttle_duration_sec=sec)
        except TypeError:
            self.get_logger().warn(msg)

    def _build_qos(self) -> QoSProfile:
        qos = QoSPresetProfiles.SENSOR_DATA.value
        if self.qos_reliability == "reliable":
            qos.reliability = QoSReliabilityPolicy.RELIABLE
        else:
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        return qos

    def _init_backend(self):
        if self.backend_name == "pytorch":
            from s2m2_ros2.backends import TorchBackend

            self.backend = TorchBackend(
                weights_dir=self.weights_dir,
                model_type=self.model_type,
                refine_iter=self.refine_iter,
                use_positivity=not self.allow_negative,
                device=self.device_str,
            )
        elif self.backend_name == "tensorrt":
            from s2m2_ros2.backends import TensorRTBackend

            if not self.trt_engine_path:
                raise ValueError(
                    "backend=tensorrt requires trt_engine_path parameter"
                )
            self.backend = TensorRTBackend(
                engine_path=self.trt_engine_path, device=self.device_str
            )
        else:
            raise ValueError(
                f"Unknown backend '{self.backend_name}' (expected pytorch|tensorrt)"
            )

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg
        if msg.k[0] > 0.0:
            self._fx = float(msg.k[0])

    def _right_info_cb(self, msg: CameraInfo):
        self._right_info = msg
        # rectified stereo: P[0,3] = -fx * baseline → baseline = -P[3] / P[0]
        if msg.p[0] > 0.0 and msg.p[3] != 0.0:
            self._baseline = float(-msg.p[3] / msg.p[0])

    def _calibration_ready(self) -> bool:
        return self._fx > 0.0 and self._baseline > 0.0

    def _to_torch(self, img_msg: Image) -> torch.Tensor:
        if img_msg.encoding in ("rgb8", "bgr8"):
            cv_img = self.bridge.imgmsg_to_cv2(
                img_msg, desired_encoding="rgb8"
            )
        elif img_msg.encoding == "mono8":
            mono = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
            cv_img = np.repeat(mono[..., None], 3, axis=2)
        else:
            cv_img = self.bridge.imgmsg_to_cv2(
                img_msg, desired_encoding="rgb8"
            )
        # crop to multiples of 32 (mirror demo/visualize_2d_simple.py:55-62)
        h, w = cv_img.shape[:2]
        h32 = (h // 32) * 32
        w32 = (w // 32) * 32
        cv_img = cv_img[:h32, :w32]
        t = torch.from_numpy(cv_img).permute(2, 0, 1).unsqueeze(0).float()
        return t.to(self.backend.device)

    def _stereo_callback(self, left_msg: Image, right_msg: Image):
        if not self._calibration_ready():
            self._warn_throttled(
                "Skipping frame: no calibration (fx and baseline). "
                "Provide camera_info topics or fx_fallback / baseline_m_fallback params.",
                sec=5.0,
            )
            return

        try:
            left_t = self._to_torch(left_msg)
            right_t = self._to_torch(right_msg)
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        if not self._warmed_up:
            try:
                h, w = left_t.shape[-2:]
                self.backend.warmup(h, w)
            except Exception as e:
                self.get_logger().warn(f"Warmup pass failed (continuing): {e}")
            self._warmed_up = True

        try:
            disp, conf = self.backend.infer(left_t, right_t)
        except Exception as e:
            self.get_logger().error(f"s2m2 inference failed: {e}")
            return

        depth_m = self._disp_to_depth(disp, conf)
        depth_msg = self._depth_to_image_msg(depth_m, left_msg.header)
        info_msg = self._make_depth_camera_info(depth_msg)

        self.pub_depth.publish(depth_msg)
        self.pub_depth_info.publish(info_msg)

        if self.pub_conf is not None:
            conf_np = conf.detach().cpu().numpy().astype(np.float32)
            conf_msg = self.bridge.cv2_to_imgmsg(conf_np, encoding="32FC1")
            conf_msg.header = depth_msg.header
            self.pub_conf.publish(conf_msg)

    def _disp_to_depth(self, disp: torch.Tensor, conf: torch.Tensor) -> np.ndarray:
        valid = disp > 0
        depth = torch.zeros_like(disp)
        fxB = float(self._fx) * float(self._baseline)
        depth[valid] = fxB / disp[valid]

        in_range = (depth >= self.min_depth_m) & (depth <= self.max_depth_m)
        depth = torch.where(in_range, depth, torch.zeros_like(depth))

        if self.min_confidence > 0.0:
            depth = torch.where(
                conf >= self.min_confidence, depth, torch.zeros_like(depth)
            )
        return depth.detach().cpu().numpy().astype(np.float32)

    def _depth_to_image_msg(self, depth_m: np.ndarray, ref_header) -> Image:
        if self.depth_encoding == "16UC1":
            arr = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
            msg = self.bridge.cv2_to_imgmsg(arr, encoding="16UC1")
        else:
            msg = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        msg.header.stamp = ref_header.stamp
        msg.header.frame_id = ref_header.frame_id
        return msg

    def _make_depth_camera_info(self, depth_msg: Image) -> CameraInfo:
        info = CameraInfo()
        info.header = depth_msg.header
        info.width = int(depth_msg.width)
        info.height = int(depth_msg.height)
        if self._left_info is not None:
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            info.k = list(self._left_info.k)
            info.r = list(self._left_info.r) if any(self._left_info.r) else [
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
            ]
            info.p = list(self._left_info.p) if any(self._left_info.p) else [
                self._fx, 0.0, info.width / 2.0, 0.0,
                0.0, self._fx, info.height / 2.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
        else:
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            cx = info.width / 2.0
            cy = info.height / 2.0
            info.k = [self._fx, 0.0, cx, 0.0, self._fx, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [
                self._fx, 0.0, cx, 0.0,
                0.0, self._fx, cy, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
        return info


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
