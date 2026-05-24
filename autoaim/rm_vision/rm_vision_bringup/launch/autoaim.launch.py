import os
import sys
import yaml
from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess  
sys.path.append(os.path.join(get_package_share_directory('rm_vision_bringup'), 'launch'))


def generate_launch_description():

    from common import node_params, launch_params, robot_state_publisher, tracker_node
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription

    node_params_yaml = yaml.safe_load(open(node_params))
    armor_params = node_params_yaml.get('/armor_detector', {}).get('ros__parameters', {})
    detector_use_hik_sdk = bool(armor_params.get('input.use_hik_sdk', False))

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
            executable='component_container',
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
            ros_arguments=['--ros-args', '--log-level',
                           'armor_detector:='+launch_params['detector_log_level']],
            on_exit=Shutdown(),
        )

    def get_detector_only_container():
        return ComposableNodeContainer(
            name='camera_detector_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
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
            ros_arguments=['--ros-args', '--log-level',
                           'armor_detector:='+launch_params['detector_log_level']],
            on_exit=Shutdown(),
        )

    hik_camera_node = get_camera_node('hik_camera', 'hik_camera::HikCameraNode')


    if (launch_params['camera'] == 'hik'):
        if detector_use_hik_sdk:
            cam_detector = get_detector_only_container()
        else:
            cam_detector = get_camera_detector_container(hik_camera_node)

    delay_tracker_node = TimerAction(
        period=2.0,
        actions=[tracker_node],
    )
    
    serial_driver_node = Node(
        package='vision_serial_driver',
        executable='vision_serial_driver_node',
        parameters = [node_params],
    )

    attacker_node = Node(
        package='vision_attacker',
        executable='vision_attacker_node',
        parameters=[node_params],
        # ros_arguments=['--log-level', '--log-level', 'debug'],
    )

    return LaunchDescription([
        robot_state_publisher,
        cam_detector,
        delay_tracker_node,
        serial_driver_node,
        attacker_node,
    ])
