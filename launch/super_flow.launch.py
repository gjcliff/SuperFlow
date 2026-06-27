from time import time

from launch_ros.actions import Node

from launch import LaunchDescription

rec_id = str(int(time()))


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="px4_slam",
                namespace="",
                executable="super_flow",
                name="super_flow",
                parameters=[{"recording_id": rec_id}],
            ),
            Node(
                package="px4_slam",
                namespace="",
                executable="backend",
                name="backendsuper_flow",
                parameters=[{"recording_id": rec_id}],
            ),
        ]
    )
