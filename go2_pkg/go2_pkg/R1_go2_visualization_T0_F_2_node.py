#!/usr/bin/env python3

import os
import time
import csv

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge

import cv2


class ANImageViewerNode(Node):
    def __init__(self):
        super().__init__("R1_go2_visualization_T0_F_2_node")

        self.bridge = CvBridge()

        self.latest_img = None
        self.have_image = False

        self.last_n_matches = 0
        self.last_long_side = 0

        # ✅ TRUE processing time from LoFTR node
        self.last_proc_ms = 0.0

        # ✅ Feature error from 16D vector: [s_d(8), s(8)] -> e = s - s_d
        self.last_e = [0.0] * 8
        self.have_features = False

        self.disp_w = 800
        self.disp_h = 600

        # ---------------- Logging setup ----------------
        log_dir = "/home/ehsan/ros2_ws/src/go2_pkg/data/R1/Test0"
        os.makedirs(log_dir, exist_ok=True)

        self.log_path = os.path.join(log_dir, "Test0_ALoFTR_F_4p.csv")

        self._csv_file = open(self.log_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        self._csv_writer.writerow([
            "t_unix",
            "processing_time_ms",
            "n_matches",
            "incoming_w",
            "incoming_h",
            "long_side",
            # --- feature errors (8) ---
            "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7",
            "features_valid",
        ])
        self._csv_file.flush()

        # ---------------- Subscriptions ----------------
        self.sub_img = self.create_subscription(
            Image, "/annotated_image", self.image_cb, 10
        )

        self.sub_matches = self.create_subscription(
            Float32MultiArray, "/ALoFTR/matched_points", self.matches_cb, 10
        )

        # ✅ Subscribe to TRUE algorithm timing
        self.sub_time = self.create_subscription(
            Float32, "/LoFTR/processing_time_ms", self.time_cb, 10
        )

        # ✅ Subscribe to features: [s_d(8), s(8)]
        # Make sure your detector node publishes 16 floats on /AN_features.
        self.sub_features = self.create_subscription(
            Float32MultiArray, "/AN_features", self.features_cb, 10
        )

        self.timer = self.create_timer(1.0 / 30.0, self.display_cb)

        self.get_logger().info(f"Viewer started.\nLogging to: {self.log_path}")

    # ==============================================================
    # Callbacks
    # ==============================================================

    def image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.latest_img = img
        self.have_image = True

        h, w = img.shape[:2]
        self.last_long_side = int(max(w, h))

    def matches_cb(self, msg: Float32MultiArray):
        data = msg.data if msg.data is not None else []
        if len(data) >= 5:
            self.last_n_matches = int(len(data) // 5)
        else:
            self.last_n_matches = 0

    def time_cb(self, msg: Float32):
        """Receive processing time from LoFTR node."""
        self.last_proc_ms = float(msg.data)

    def features_cb(self, msg: Float32MultiArray):
        """
        Expected msg.data length = 16 floats:
          [s_d0..s_d7, s0..s7]
        error e = s - s_d (8 floats)

        If your upstream node still publishes 12 floats, this logger will mark features invalid.
        """
        data = list(msg.data) if msg.data is not None else []
        if len(data) < 16:
            self.have_features = False
            return

        sd = data[0:8]
        s  = data[8:16]
        self.last_e = [float(s[i] - sd[i]) for i in range(8)]
        self.have_features = True

    def display_cb(self):
        if not self.have_image or self.latest_img is None:
            return

        incoming_h, incoming_w = self.latest_img.shape[:2]

        disp = cv2.resize(
            self.latest_img,
            (self.disp_w, self.disp_h),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("AN_image", disp)
        key = cv2.waitKey(1) & 0xFF

        # -------- Log row --------
        try:
            e = self.last_e if self.have_features else [0.0] * 8
            self._csv_writer.writerow([
                f"{time.time():.6f}",
                f"{self.last_proc_ms:.3f}",
                str(int(self.last_n_matches)),
                str(int(incoming_w)),
                str(int(incoming_h)),
                str(int(max(incoming_w, incoming_h))),
                f"{e[0]:.6f}", f"{e[1]:.6f}", f"{e[2]:.6f}", f"{e[3]:.6f}",
                f"{e[4]:.6f}", f"{e[5]:.6f}", f"{e[6]:.6f}", f"{e[7]:.6f}",
                "1" if self.have_features else "0",
            ])
            self._csv_file.flush()
        except Exception as e:
            self.get_logger().warn(f"Failed to write log row: {e}")

        if key == 27 or key == ord('q'):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()

    # ==============================================================
    # Cleanup
    # ==============================================================

    def destroy_node(self):
        try:
            if self._csv_file:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ANImageViewerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
