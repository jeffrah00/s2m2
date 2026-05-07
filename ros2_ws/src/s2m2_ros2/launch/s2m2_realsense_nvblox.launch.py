"""Wrapper around nvblox_examples_bringup/launch/realsense_example.launch.py
that adds the s2m2 stereo depth node and redirects nvblox's depth
subscription to s2m2's output.

The original ``realsense_example.launch.py`` brings up:
    - the RealSense driver + ``realsense_splitter_node`` (which publishes
      *clean*, emitter-off IR pairs at ``/<ns>/realsense_splitter_node/output/infra_{1,2}``
      and emitter-on depth at ``/<ns>/realsense_splitter_node/output/depth``)
    - VSLAM, segmentation/detection (when ``mode`` selects them), nvblox, RViz/Foxglove

This launch keeps all of that and only adds two things:

  1. ``s2m2_stereo_depth_node`` subscribed to the splitter's emitter-off IR
     pair (with CameraInfo from the realsense driver), publishing depth on
     a fresh topic so it does not collide with the splitter's depth output.
  2. A scope-level ``SetRemap`` that overrides nvblox's
     ``camera_0/depth/image`` subscription to point at s2m2's depth topic.
     The splitter still publishes its own depth, but nobody subscribes to
     it; everything else (color, color/info, depth/info) keeps its
     original routing.

All public launch args of ``realsense_example.launch.py`` are exposed as
pass-throughs.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory("s2m2_ros2")
    default_params = os.path.join(pkg_share, "config", "s2m2_depth.yaml")

    args = [
        # s2m2-specific
        DeclareLaunchArgument(
            "params_file", default_value=default_params,
            description="s2m2 stereo_depth_node parameter file."),
        DeclareLaunchArgument(
            "camera_namespace", default_value="camera0",
            description="RealSense topic namespace used by realsense_example.launch.py."),
        DeclareLaunchArgument(
            "s2m2_depth_topic", default_value="/camera0/s2m2/depth/image",
            description="Topic where s2m2 publishes depth and where nvblox is redirected to subscribe."),
        DeclareLaunchArgument(
            "s2m2_depth_info_topic", default_value="/camera0/depth/camera_info",
            description="CameraInfo topic for the s2m2 depth output."),

        # Pass-through args for realsense_example.launch.py
        DeclareLaunchArgument("mode", default_value="static"),
        DeclareLaunchArgument("num_cameras", default_value="1"),
        DeclareLaunchArgument("run_realsense", default_value="True"),
        DeclareLaunchArgument("rosbag", default_value="None"),
        DeclareLaunchArgument("rosbag_args", default_value=""),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("attach_to_container", default_value="False"),
        DeclareLaunchArgument("container_name", default_value="nvblox_container"),
        DeclareLaunchArgument("use_foxglove_whitelist", default_value="True"),
        DeclareLaunchArgument("camera_serial_numbers", default_value=""),
        DeclareLaunchArgument(
            "people_segmentation",
            default_value="peoplesemsegnet_vanilla"),
    ]

    cam_ns = LaunchConfiguration("camera_namespace")
    s2m2_depth_topic = LaunchConfiguration("s2m2_depth_topic")
    s2m2_depth_info_topic = LaunchConfiguration("s2m2_depth_info_topic")

    s2m2_node = Node(
        package="s2m2_ros2",
        executable="stereo_depth_node",
        name="s2m2_stereo_depth_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        remappings=[
            ("left/image_rect",  ["/", cam_ns, "/realsense_splitter_node/output/infra_1"]),
            ("right/image_rect", ["/", cam_ns, "/realsense_splitter_node/output/infra_2"]),
            ("left/camera_info",  ["/", cam_ns, "/infra1/camera_info"]),
            ("right/camera_info", ["/", cam_ns, "/infra2/camera_info"]),
            ("depth/image",       s2m2_depth_topic),
            ("depth/camera_info", s2m2_depth_info_topic),
        ],
    )

    # SetRemap propagates into the included launches, including nvblox.launch.py
    # nested inside realsense_example.launch.py. nvblox subscribes internally
    # to camera_0/depth/image (with underscore) and the realsense profile
    # remaps that to the splitter's depth output by default; we override it
    # here to use s2m2's depth instead.
    nvblox_with_s2m2_depth = GroupAction(actions=[
        SetRemap(src="camera_0/depth/image", dst=s2m2_depth_topic),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("nvblox_examples_bringup"),
                    "launch",
                    "realsense_example.launch.py",
                ])
            ),
            launch_arguments={
                "mode": LaunchConfiguration("mode"),
                "num_cameras": LaunchConfiguration("num_cameras"),
                "run_realsense": LaunchConfiguration("run_realsense"),
                "rosbag": LaunchConfiguration("rosbag"),
                "rosbag_args": LaunchConfiguration("rosbag_args"),
                "log_level": LaunchConfiguration("log_level"),
                "attach_to_container": LaunchConfiguration("attach_to_container"),
                "container_name": LaunchConfiguration("container_name"),
                "use_foxglove_whitelist": LaunchConfiguration("use_foxglove_whitelist"),
                "camera_serial_numbers": LaunchConfiguration("camera_serial_numbers"),
                "people_segmentation": LaunchConfiguration("people_segmentation"),
            }.items(),
        ),
    ])

    return LaunchDescription(args + [nvblox_with_s2m2_depth, s2m2_node])
