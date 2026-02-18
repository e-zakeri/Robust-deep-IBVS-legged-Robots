Robust Deep-Feature Image-Based Visual Servoing of Legged Robots

This repository contains the reference implementation of a Deep-Feature
Image-Based Visual Servoing (DF-IBVS) framework for legged robots. The
proposed method enables robust, markerless visual servo control using
adaptive transformer-based feature matching and a sliding-mode control
strategy designed for noisy and uncertain locomotion dynamics.

Overview The project implements a markerless Image-Based Visual Servoing
(IBVS) scheme for legged robots operating under vision uncertainty,
image noise, and locomotion-induced disturbances. The framework
integrates:

-   Adaptive LoFTR (ALoFTR) for matched-point detection
-   Deep image feature extraction using virtual image points
-   Full-rank interaction matrix formulation for 6-DOF control
-   Filtered Fuzzy Quasi-Sliding Mode Controller (FFQSMC)
-   Learning-From-Demonstration (LFD) based desired planner

The approach eliminates the need for artificial markers and
object-specific training while maintaining stable six-degree-of-freedom
control.

Associated Publication If you use this code, please cite:

Ehsan Zakeri, Wen-Fang Xie Robust Deep-Feature Image-Based Visual
Servoing of Legged Robots

(Publication details to be updated)

Key Contributions - Markerless deep-feature formulation for IBVS -
Adaptive transformer-based feature matching - Full-rank image
interaction matrix for 6-DOF control - Robust control under visual and
locomotion uncertainties - LFD-based desired image sequence generation

System Architecture The DF-IBVS pipeline consists of:

1.  RGB monocular image acquisition (robot's camera)
2.  Matched-point detection via Adaptive LoFTR (ALoFTR)
3.  Deep feature extraction from virtual image points
4.  Image-space control using FFQSMC
5.  Desired trajectory planning using LFD

Requirements

Hardware - Legged robot platform (validated on Unitree Go2 Pro) -
Monocular RGB camera - NVIDIA GPU recommended for real-time performance

Software - Ubuntu 22.04 - ROS 2 Humble - Python 3.10 or newer - OpenCV -
PyTorch

Dependencies Install Python dependencies:

"TBD"

Typical packages: - numpy - opencv-python - torch - torchvision - scipy

Usage

"TBD"

Learning-From-Demonstration (Optional) The LFD module records
demonstration image sequences and timestamps. During execution, desired
frames are selected via temporal scaling to regulate trajectory
reproduction speed.

Reproducibility Notes Performance depends on hardware capability, image
resolution, number of matched points, and ROS2 communication latency.

License This project is released under the MIT License.

Disclaimer This software is provided for research and academic purposes
only. Users are responsible for validating safety and stability before
deployment on physical robotic systems.

Contact

Ehsan Zakeri Concordia University ehsan.zakeri@concordia.ca
