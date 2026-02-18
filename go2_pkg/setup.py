from setuptools import setup
from glob import glob
import os

package_name = "go2_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],

    # ── Python-side runtime deps ─────────────────────────────────
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",
        "torch",
        "cv_bridge",          # apt: ros-<distro>-cv-bridge
        "sensor_msgs",
        "geometry_msgs",
        "tf_transformations",
    ],
    zip_safe=True,
    maintainer="YourName",
    maintainer_email="you@example.com",
    description="ROS 2 package for Unitree Go2 using WebRTC driver",
    license="Apache License 2.0",

    # ── data files to install (manifest, marker, launch, msg) ───
    data_files=[
        # ament resource index
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        # package manifest
        (f"share/{package_name}", ["package.xml"]),
        # all launch files in launch/
        (f"share/{package_name}/launch", glob("launch/*.py")),
        # custom messages
        (f"share/{package_name}/msg",    glob("msg/*.msg")),
    ],

    # ── entry-points (executables) ───────────────────────────────
    entry_points={
        "console_scripts": [
            "go2_node              = go2_pkg.go2_node:main",
            "go2_superpoint_node= go2_pkg.go2_superpoint_node:main",
            "go2_superglue_node    = go2_pkg.go2_superglue_node:main",
            "camera_generator_node = go2_pkg.camera_generator_node:main",
            "camera_generator_2_node= go2_pkg.camera_generator_2_node:main",
            "go2_joy_control_node  = go2_pkg.go2_joy_control_node:main",
            "go2_matched_points_node = go2_pkg.go2_matched_points_node:main",
            "go2_matched_points_2_node = go2_pkg.go2_matched_points_2_node:main",
            "go2_matched_points_3_node = go2_pkg.go2_matched_points_3_node:main",
            "go2_joy_movment_comand_2_node = go2_pkg.go2_joy_movment_comand_2_node:main",
            "go2_IBVS_LFD_P1_node  = go2_pkg.go2_IBVS_LFD_P1_node:main",
            "go2_IBVS_Test_2FQSMC_node = go2_pkg.go2_IBVS_Test_2FQSMC_node:main",
            "go2_superglue_T2_node = go2_pkg.go2_superglue_T2_node:main",
            "go2_IBVS_Test_2FQSMC_T0_node = go2_pkg.go2_IBVS_Test_2FQSMC_T0_node:main",
            "R1_go2_AEoFTR_c4_node= go2_pkg.R1_go2_AEoFTR_c4_node:main",
            "R1_go2_deep_feature_node= go2_pkg.R1_go2_deep_feature_node:main",
            "R1_go2_desired_image_T3_node= go2_pkg.R1_go2_desired_image_T3_node:main",
            "R1_go2_joy_movement_command_node = go2_pkg.R1_go2_joy_movement_command_node:main",
            "R1_go2_IBVS_Test_2FQSMC_T1_node = go2_pkg.R1_go2_IBVS_Test_2FQSMC_T1_node:main",
            "R1_go2_node= go2_pkg.R1_go2_node:main",
            "R1_go2_visualization_node= go2_pkg.R1_go2_visualization_node:main",
            "R1_go2_controllers_node= go2_pkg.R1_go2_controllers_node:main",
            "R1_go2_controllers_T3_node= go2_pkg.R1_go2_controllers_T3_node:main",
            "R1_go2_DFE_node= go2_pkg.R1_go2_DFE_node:main",
            "R1_go2_deep_feature_T0_node= go2_pkg.R1_go2_deep_feature_T0_node:main",
            "R1_go2_AEoFTR_T0_c4_node= go2_pkg.R1_go2_AEoFTR_T0_c4_node:main",
            "R1_go2_visualization_T0_node = go2_pkg.R1_go2_visualization_T0_node:main",
            "R1_go2_EoFTR_T0_c4_node= go2_pkg.R1_go2_EoFTR_T0_c4_node:main",
            "R1_go2_SPSG_T0_c4_node= go2_pkg.R1_go2_SPSG_T0_c4_node:main",
            "R1_video_cam_T0_node= go2_pkg.R1_video_cam_T0_node:main",
            "R1_go2_visualization_T0_3_node= go2_pkg.R1_go2_visualization_T0_3_node:main",
            "R1_go2_visualization_T0_2_node= go2_pkg.R1_go2_visualization_T0_2_node:main",
            "R1_go2_visualization_T0_F_node= go2_pkg.R1_go2_visualization_T0_F_node:main",
            "R1_go2_4P_T0_c4_node= go2_pkg.R1_go2_4P_T0_c4_node:main",
            "R1_go2_visualization_T0_F_2_node= go2_pkg.R1_go2_visualization_T0_F_2_node:main",
            "R1_go2_IB_T0_c4_node= go2_pkg.R1_go2_IB_T0_c4_node:main",
            "R1_go2_visualization_T0_F_3_node= go2_pkg.R1_go2_visualization_T0_F_3_node:main",


        ],
    },
)
