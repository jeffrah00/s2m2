"""Launch s2m2 stereo depth node and wire its output to the Nvblox camera_0 namespace."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("s2m2_ros2")
    default_params = os.path.join(pkg_share, "config", "s2m2_depth.yaml")

    args = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("left_image_topic", default_value="/stereo/left/image_rect"),
        DeclareLaunchArgument("right_image_topic", default_value="/stereo/right/image_rect"),
        DeclareLaunchArgument("left_info_topic", default_value="/stereo/left/camera_info"),
        DeclareLaunchArgument("right_info_topic", default_value="/stereo/right/camera_info"),
        DeclareLaunchArgument("depth_image_topic", default_value="/camera_0/depth/image"),
        DeclareLaunchArgument("depth_info_topic", default_value="/camera_0/depth/camera_info"),
        DeclareLaunchArgument("depth_confidence_topic", default_value="/camera_0/depth/confidence"),
    ]

    node = Node(
        package="s2m2_ros2",
        executable="stereo_depth_node",
        name="s2m2_stereo_depth_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        remappings=[
            ("left/image_rect", LaunchConfiguration("left_image_topic")),
            ("right/image_rect", LaunchConfiguration("right_image_topic")),
            ("left/camera_info", LaunchConfiguration("left_info_topic")),
            ("right/camera_info", LaunchConfiguration("right_info_topic")),
            ("depth/image", LaunchConfiguration("depth_image_topic")),
            ("depth/camera_info", LaunchConfiguration("depth_info_topic")),
            ("depth/confidence", LaunchConfiguration("depth_confidence_topic")),
        ],
    )

    return LaunchDescription(args + [node])
