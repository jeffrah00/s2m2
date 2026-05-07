"""Bring up RealSense + s2m2 + nvblox in a single launch.

Recommended one-shot setup for visualizing s2m2 depth in nvblox with a
RealSense camera. It replaces the depth source that
``nvblox_examples_bringup/realsense_example.launch.py`` normally uses
(the RealSense's own depth via ``realsense_splitter_node``) with s2m2
depth computed from the rectified IR pair.

Components (each toggleable via launch args):
  - realsense2_camera/rs_launch.py   (IR emitter disabled)
  - s2m2_ros2 stereo_depth_node       (consumes IR pair, publishes depth)
  - nvblox_examples_bringup nvblox.launch.py  (subscribes to s2m2 depth)

Set ``launch_realsense:=false`` if you already have a RealSense driver
running, or ``launch_nvblox:=false`` if you'd rather start nvblox yourself.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory("s2m2_ros2")
    default_params = os.path.join(pkg_share, "config", "s2m2_depth.yaml")

    args = [
        DeclareLaunchArgument(
            "launch_realsense", default_value="true",
            description="Bring up realsense2_camera with the IR emitter disabled.",
        ),
        DeclareLaunchArgument(
            "launch_nvblox", default_value="true",
            description="Include nvblox_examples_bringup/launch/perception/nvblox.launch.py.",
        ),
        DeclareLaunchArgument(
            "params_file", default_value=default_params,
            description="s2m2 stereo_depth_node parameter file.",
        ),
        DeclareLaunchArgument(
            "camera_namespace", default_value="camera0",
            description="RealSense topic namespace.",
        ),
        DeclareLaunchArgument(
            "image_profile", default_value="640x480x30",
            description="RealSense IR/depth profile (WxHxFPS).",
        ),
        DeclareLaunchArgument(
            "nvblox_launch_pkg", default_value="nvblox_examples_bringup",
            description="Package providing the nvblox launch file to include.",
        ),
        DeclareLaunchArgument(
            "nvblox_launch_file", default_value="launch/perception/nvblox.launch.py",
            description="Path (within nvblox_launch_pkg/share/) of the nvblox launch file.",
        ),
    ]

    cam_ns = LaunchConfiguration("camera_namespace")

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"), "launch", "rs_launch.py",
            ])
        ),
        launch_arguments={
            "camera_namespace": cam_ns,
            "enable_infra1": "true",
            "enable_infra2": "true",
            "enable_color": "true",
            "depth_module.emitter_enabled": "0",
            "align_depth.enable": "false",
            "rgb_camera.profile": LaunchConfiguration("image_profile"),
            "depth_module.profile": LaunchConfiguration("image_profile"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_realsense")),
    )

    s2m2 = Node(
        package="s2m2_ros2",
        executable="stereo_depth_node",
        name="s2m2_stereo_depth_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        remappings=[
            ("left/image_rect",  ["/", cam_ns, "/infra1/image_rect_raw"]),
            ("right/image_rect", ["/", cam_ns, "/infra2/image_rect_raw"]),
            ("left/camera_info",  ["/", cam_ns, "/infra1/camera_info"]),
            ("right/camera_info", ["/", cam_ns, "/infra2/camera_info"]),
            ("depth/image",       ["/", cam_ns, "/depth/image"]),
            ("depth/camera_info", ["/", cam_ns, "/depth/camera_info"]),
        ],
    )

    # SetRemap propagates into the included nvblox launch, so nvblox sees
    # our s2m2-driven depth and the RealSense color stream instead of the
    # splitter outputs that realsense_example.launch.py wires by default.
    nvblox = GroupAction(
        actions=[
            SetRemap(src="camera_0/depth/image",        dst=["/", cam_ns, "/depth/image"]),
            SetRemap(src="camera_0/depth/camera_info", dst=["/", cam_ns, "/depth/camera_info"]),
            SetRemap(src="camera_0/color/image",        dst=["/", cam_ns, "/color/image_raw"]),
            SetRemap(src="camera_0/color/camera_info", dst=["/", cam_ns, "/color/camera_info"]),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare(LaunchConfiguration("nvblox_launch_pkg")),
                        LaunchConfiguration("nvblox_launch_file"),
                    ])
                ),
            ),
        ],
        condition=IfCondition(LaunchConfiguration("launch_nvblox")),
    )

    return LaunchDescription(args + [realsense, s2m2, nvblox])
