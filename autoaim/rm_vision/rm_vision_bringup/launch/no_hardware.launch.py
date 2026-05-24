import os
import sys
from ament_index_python.packages import get_package_share_directory
sys.path.append(os.path.join(get_package_share_directory('rm_vision_bringup'), 'launch'))


def generate_launch_description():

    from common import launch_params, robot_state_publisher, node_params
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer
    from launch_ros.actions import Node
    from launch.actions import DeclareLaunchArgument, Shutdown, TimerAction
    from launch import LaunchDescription
    from launch.conditions import IfCondition
    from launch.substitutions import LaunchConfiguration

    with_tracker = LaunchConfiguration('with_tracker')

    def get_camera_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='camera_node',
            parameters=[node_params],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    def get_camera_detector_container(camera_node):
        return ComposableNodeContainer(
            name='camera_detector_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                camera_node,
                ComposableNode(
                    package='armor_detector',
                    plugin='rm_auto_aim::ArmorDetectorNode',
                    name='armor_detector',
                    parameters=[node_params],
                    extra_arguments=[{'use_intra_process_comms': True}]
                )
            ],
            output='both',
            emulate_tty=True,
            ros_arguments=['--ros-args', '--log-level', 'debug'],
            on_exit=Shutdown(),
        )

    hik_camera_node = get_camera_node('hik_camera', 'hik_camera::HikCameraNode')

    camera = launch_params.get('camera', 'hik')
    if camera != 'hik':
        raise RuntimeError('launch_params.yaml: only "hik" camera is supported')
    cam_detector = get_camera_detector_container(hik_camera_node)

    tracker_node = Node(
        package='armor_tracker',
        executable='armor_tracker_node',
        output='both',
        emulate_tty=True,
        parameters=[node_params, {'target_frame': 'camera_optical_frame'}],
        ros_arguments=['--ros-args', '--log-level', 'debug'],
        condition=IfCondition(with_tracker),
    )

    delay_tracker_node = TimerAction(
        period=2.0,
        actions=[tracker_node],
        condition=IfCondition(with_tracker),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'with_tracker',
            default_value='false',
            description='Start armor_tracker in no_hardware (target_frame forced to camera_optical_frame)'
        ),
        robot_state_publisher,
        cam_detector,
        delay_tracker_node,
    ])
