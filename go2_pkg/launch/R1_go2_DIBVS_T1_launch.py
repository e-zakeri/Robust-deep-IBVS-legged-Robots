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

    # ─────────────── go2_ALoFTR_test_node ────────────────
    LoFTR_node = Node(
        package="go2_pkg",
        executable="R1_go2_AEoFTR_c4_node",
        name="R1_go2_AEoFTR_c4_node",
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
        executable="R1_go2_deep_feature_node",
        name="R1_go2_deep_feature_node",
        output="screen",
    )
    
    go2_node= Node(
        package="go2_pkg",
        executable="R1_go2_node",
        name="R1_go2_node",
        output="screen",
    )

    Desired_node= Node(
        package="go2_pkg",
        executable="R1_go2_desired_image_node",
        name="R1_go2_desired_image_node",
        output="screen",
    )   

    Joy_movement_command_node= Node(
        package="go2_pkg",
        executable="R1_go2_joy_movement_command_node",
        name="R1_go2_joy_movement_command_node",
        output="screen",
    )

    IBVS_node= Node(
        package="go2_pkg",
        executable="R1_go2_IBVS_Test_2FQSMC_T1_node",
        name="R1_go2_IBVS_Test_2FQSMC_T1_node",
        output="screen",
    )

    visualization_node= Node(
        package="go2_pkg",
        executable="R1_go2_visualization_node",
        name="R1_go2_visualization_node",
        output="screen",
    )

    Controllers_node= Node(
        package="go2_pkg",
        executable="R1_go2_controllers_node",
        name="R1_go2_controllers_node",
        output="screen",
    )
    
    DFE_node= Node(
        package="go2_pkg",
        executable="R1_go2_DFE_node",
        name="R1_go2_DFE_node",
        output="screen",
    )

    return LaunchDescription(launch_args + [joy_driver,
                                            Desired_node,
                                            go2_node,                                                                            
                                            Joy_movement_command_node,
                                            LoFTR_node,
                                            AIF_node,
                                            Controllers_node,
                                            #IBVS_node,
                                            visualization_node,    
                                            DFE_node,                                        
                                            ])


