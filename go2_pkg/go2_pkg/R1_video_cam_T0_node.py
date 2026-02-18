#!/usr/bin/env python3
import os

os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "200000")

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class R1VideoCamNode(Node):
    def __init__(self):
        super().__init__("R1_video_cam_T0_node")

        #self.video_path = "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/ALoFTR_test/Video_record_1.webm"
        #self.video_path = "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/ALoFTR_test/Video_record_2.webm"
        #self.video_path = "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/ALoFTR_test/Video_record_3.webm"
        self.video_path = "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/ALoFTR_test/Video_record_6.mp4"

        self.topic_frame = "/camera_test"
        self.topic_desired = "/camera_test_d"

        self.desired_publish_period_s = 1.0
        self.start_delay_s = 10.0
        self.fps_fallback = 20.0

        self.bridge = CvBridge()
        self.pub_frame = self.create_publisher(Image, self.topic_frame, 10)
        self.pub_desired = self.create_publisher(Image, self.topic_desired, 10)

        self.cap = None
        self.frame_timer = None
        self.video_started = False

        self.desired_msg = None
        self.publish_dt = None

        if not self._open_capture():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        if not self._cache_first_frame():
            raise RuntimeError("Could not read first frame")

        self.desired_timer = self.create_timer(
            self.desired_publish_period_s,
            self.desired_timer_cb
        )

        self.start_timer = self.create_timer(
            self.start_delay_s,
            self.start_video_once_cb
        )

        self.get_logger().info("Node initialized")

    # ==============================================================
    # Helpers
    # ==============================================================

    def _open_capture(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video: {self.video_path}")
            self.cap = None
            return False

        self.get_logger().info(f"Opened video: {self.video_path}")
        return True

    def _get_video_fps(self) -> float:
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or fps > 120.0:
            fps = self.fps_fallback
        return fps

    def _cache_first_frame(self) -> bool:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False

        self.desired_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.get_logger().info("Cached desired frame")
        return True

    # ==============================================================
    # Timers
    # ==============================================================

    def desired_timer_cb(self):
        if self.desired_msg is not None:
            self.pub_desired.publish(self.desired_msg)

    def start_video_once_cb(self):
        if self.video_started:
            return

        try:
            self.start_timer.cancel()
        except Exception:
            pass

        if not self._open_capture():
            return

        fps = self._get_video_fps()
        self.publish_dt = 1.0 / fps

        self.get_logger().info(f"Starting video at FPS={fps:.2f}")

        self.video_started = True
        self.frame_timer = self.create_timer(self.publish_dt, self.frame_timer_cb)

    def frame_timer_cb(self):
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error("Capture lost. Stopping frame timer.")
            self._stop_video_only()
            return

        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.get_logger().info("Video finished. Stopping frame publishing.")
            self._stop_video_only()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_frame.publish(msg)

    # ==============================================================
    # Stop logic (NO shutdown)
    # ==============================================================

    def _stop_video_only(self):
        try:
            if self.frame_timer is not None:
                self.frame_timer.cancel()
                self.frame_timer = None
        except Exception:
            pass

        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

        self.get_logger().info("Video stopped. Node still alive.")


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
