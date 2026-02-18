#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import threading
import time
import os


class SimpleCameraNode(Node):
    def __init__(self):
        super().__init__('camera_generator_2_node')

        self.bridge = CvBridge()

        # ---- Open default camera (index 0) ----
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)  # or 1920
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # or 1080

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera (index 0)")

        self.get_logger().info("Camera opened using default settings")

        # ---- Publishers (same names as your original code) ----
        self.pub_cam = self.create_publisher(Image, "/camera_test", 1)
        self.pub_static = self.create_publisher(Image, "/camera_test_d", 1)
        self.pub_fps = self.create_publisher(Float32, "/camera_fps", 1)

        # ---- Joystick subscriber ----
        self.sub_joy = self.create_subscription(Joy, "/joy", self.joy_cb, 10)

        # ---- Desired image path (same path as your original code) ----
        self.static_image_path = (
            "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/"
            "ALoFTR_test/saved_image2.png"
        )

        # Ensure folder exists (do not require the file to exist anymore)
        static_dir = os.path.dirname(self.static_image_path)
        if static_dir and (not os.path.isdir(static_dir)):
            raise RuntimeError(f"Desired image folder not found: {static_dir}")

        # ---- Publish desired image every 1 second ----
        self.timer_static = self.create_timer(1.0, self.publish_static_image)

        # ---- FPS tracking ----
        self.frame_count = 0
        self.last_fps_time = time.time()

        # ---- Latest frame storage (for saving on joystick press) ----
        self._frame_lock = threading.Lock()
        self._latest_frame = None

        # ---- Joystick edge detection ----
        self._prev_btn_save = 0  # for buttons[1]

        # ---- Start camera thread ----
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

        self.get_logger().info("Started camera capture loop + static image @ 1 Hz")
        self.get_logger().info("Press joystick button #2 (index 1) to SAVE desired image.")

    # ---------------------------------------------------------
    #                     JOYSTICK CALLBACK
    # ---------------------------------------------------------
    def joy_cb(self, msg: Joy):
        # button #2 -> index 1
        if len(msg.buttons) <= 1:
            return

        btn = int(msg.buttons[1])
        rising_edge = (btn == 1 and self._prev_btn_save == 0)
        self._prev_btn_save = btn

        if not rising_edge:
            return

        # Save the latest camera frame as the desired image
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is None:
            self.get_logger().warn("Save requested, but no camera frame is available yet.")
            return

        ok = cv2.imwrite(self.static_image_path, frame)
        if ok:
            self.get_logger().info(f"Saved desired image to: {self.static_image_path}")
        else:
            self.get_logger().error(f"Failed to save desired image to: {self.static_image_path}")

    # ---------------------------------------------------------
    #                     CAMERA LOOP
    # ---------------------------------------------------------
    def capture_loop(self):
        while rclpy.ok() and self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warning("Camera returned an empty frame")
                time.sleep(0.01)
                continue

            # Store latest frame for joystick-triggered save
            with self._frame_lock:
                self._latest_frame = frame

            # Publish camera frame
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.pub_cam.publish(msg)

            # ---- FPS Tracking ----
            self.frame_count += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                fps = self.frame_count / (now - self.last_fps_time)
                self.pub_fps.publish(Float32(data=float(fps)))
                self.last_fps_time = now
                self.frame_count = 0

    # ---------------------------------------------------------
    #          STATIC / DESIRED IMAGE PUBLISHED @ 1 Hz
    # ---------------------------------------------------------
    def publish_static_image(self):
        img = cv2.imread(self.static_image_path)
        if img is None:
            # Don't spam error if the user hasn't saved yet
            self.get_logger().debug(f"Desired image not available yet: {self.static_image_path}")
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub_static.publish(msg)

    # ---------------------------------------------------------
    #                     CLEAN SHUTDOWN
    # ---------------------------------------------------------
    def destroy_node(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = SimpleCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
