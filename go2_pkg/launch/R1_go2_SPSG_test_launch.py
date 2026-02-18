# go2_pkg/launch/go2_superpoint_launch.py
#=================================================================----
# ────────────────────────────────────────────────────────────────----
# Launches
#   • camera_generator_node   → /camera_test
#   • go2_superglue_node      → /superglue/matches  (+ /relative_pose)
#=================================================================----
# Run (defaults):
#   ros2 launch go2_pkg go2_superpoint_launch.py
#=================================================================----
# Common overrides:
#   ros2 launch go2_pkg go2_superpoint_launch.py display:=true
#   ros2 launch go2_pkg go2_superpoint_launch.py camera_index:=2
# ────────────────────────────────────────────────────────────────----
#=================================================================----
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description() -> LaunchDescription:
    # ─────────────── launch-time arguments ────────────────
    launch_args = [
        # shared
        DeclareLaunchArgument("camera_index",      default_value="0"),
        DeclareLaunchArgument("resize_long_edge",  default_value="960"),
        DeclareLaunchArgument("display",           default_value="True"),

        # SuperGlue intrinsics (optional)
        DeclareLaunchArgument("fx", default_value="0.0"),
        DeclareLaunchArgument("fy", default_value="0.0"),
        DeclareLaunchArgument("cx", default_value="0.0"),
        DeclareLaunchArgument("cy", default_value="0.0"),

        # camera_generator-specific
        DeclareLaunchArgument("frame_rate",        default_value="60"),
    ]

    # ─────────────── camera_generator_node ────────────────
    
    cam_node = Node(
        package="go2_pkg",
        executable="camera_generator_2_node",
        name="camera_generator_2_node",
        output="screen",
        parameters=[{
            "camera_index":      LaunchConfiguration("camera_index"),
            "frame_rate":        LaunchConfiguration("frame_rate"),
            "resize_long_edge":  LaunchConfiguration("resize_long_edge"),
        }],
    )
    
    cam_video_node = Node(
        package="go2_pkg",
        executable="R1_video_cam_T0_node",
        name="R1_video_cam_T0_node",
        output="screen",
    )

    # ─────────────── go2_ALoFTR_test_node ────────────────
    LoFTR_node = Node(
        package="go2_pkg",
        executable="R1_go2_SPSG_T0_c4_node",
        name="R1_go2_SPSG_T0_c4_node",
        output="screen",
    ) 
    
    
    # ── joystick driver node ────────────────────────────────────
    joy_driver = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
        parameters=[{
            "deadzone": 0.01,   # ignore small drifts
            "autorepeat_rate":100.0,
        }],
    )
    # ─────────────── go2_deep_feature_node ────────────────
    AIF_node = Node(
        package="go2_pkg",
        executable="R1_go2_deep_feature_T0_node",
        name="R1_go2_deep_feature_T0_node",
        output="screen",
    )
    #___________________ R1_go2_visualization_node ___________________
    Visualization = Node(
        package="go2_pkg",
        executable="R1_go2_visualization_T0_3_node",
        name="R1_go2_visualization_T0_3_node",
        output="screen",
    )
    
    
    return LaunchDescription(launch_args + [
                                            cam_video_node,
                                            LoFTR_node,
                                            AIF_node,
                                            joy_driver,
                                            Visualization
                                            ])


