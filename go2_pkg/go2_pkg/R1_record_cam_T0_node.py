#!/usr/bin/env python3

import os

# IMPORTANT: set before creating VideoCapture (ideally before any capture usage)
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "200000")  # increase from 4096

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class R1VideoCamNode(Node):
    def __init__(self):
        super().__init__("R1_video_cam_T0_node")

        self.video_path = "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/ALoFTR_test/Video_record_1.webm"

        self.bridge = CvBridge()
        self.pub_frame = self.create_publisher(Image, "/camera_test_2", 10)
        self.pub_desired = self.create_publisher(Image, "/camera_test_2_d", 10)

        self.cap = None
        self.first_frame_sent = False

        # Open once
        self._open_capture()

        # FPS (fallback 30)
        #fps = 0.0
        #if self.cap is not None and self.cap.isOpened():
        #    fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        #if fps <= 0.0 or fps > 120.0:
        #    fps = 20.0
        fps=15.0

        self.get_logger().info(f"Using publish FPS: {fps:.2f}")
        self.timer = self.create_timer(1.0 / fps, self.timer_cb)

    def _open_capture(self):
        # Always release old
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Prefer FFMPEG backend explicitly
        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video: {self.video_path}")
            self.cap = None
            return

        self.get_logger().info(f"Opened video: {self.video_path}")

    def timer_cb(self):
        if self.cap is None or not self.cap.isOpened():
            # try to recover
            self._open_capture()
            self.first_frame_sent = False
            return

        ret, frame = self.cap.read()

        if not ret or frame is None:
            # For WEBM, seeking often fails. Reopen instead of CAP_PROP_POS_FRAMES=0
            self.get_logger().warn("Video read failed (end/decoder issue). Re-opening video...")
            self._open_capture()
            self.first_frame_sent = False
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        if not self.first_frame_sent:
            self.pub_desired.publish(msg)
            self.first_frame_sent = True
            self.get_logger().info("Published desired frame (/camera_test_2_d)")

        self.pub_frame.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = R1VideoCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
