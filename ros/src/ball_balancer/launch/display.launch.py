from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = "ball_balancer"
    share = get_package_share_directory(pkg)

    urdf_path = os.path.join(share, "urdf", "ball_balancer.urdf")

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    nodes = [
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            output="screen",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
        ),
    ]

    return LaunchDescription(nodes)