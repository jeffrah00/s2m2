from glob import glob
from setuptools import find_packages, setup

package_name = 's2m2_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='s2m2 contributors',
    maintainer_email='noreply@example.com',
    description='ROS2 wrapper for s2m2 stereo depth estimation, producing depth/CameraInfo for Nvblox.',
    license='See LICENSE.md in repository root',
    entry_points={
        'console_scripts': [
            'stereo_depth_node = s2m2_ros2.stereo_depth_node:main',
        ],
    },
)
