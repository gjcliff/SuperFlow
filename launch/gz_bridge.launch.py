from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def bridge_setup(context, *args, **kwargs):
    world_name = LaunchConfiguration("world_name").perform(context)
    return (
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/world/{world_name}/model/x500_mono_cam_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image",
                f"/world/{world_name}/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            ],
            remappings=[
                (
                    f"/world/{world_name}/model/x500_mono_cam_0/link/camera_link/sensor/imager/image",
                    "/camera/image_raw",
                ),
                (
                    f"/world/{world_name}/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info",
                    "/camera/camera_info",
                ),
            ],
            output="screen",
        ),
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="world_name",
                default_value="default",
                description="The name of the gz world you chose",
                choices=["default", "baylands"],
            ),
            OpaqueFunction(function=bridge_setup),
        ]
    )
