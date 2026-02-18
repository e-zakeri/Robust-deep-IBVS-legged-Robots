#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import time


class ANImageViewerNode(Node):
    def __init__(self):
        super().__init__("R1_go2_visualization_node")

        self.bridge = CvBridge()

        # Latest frame storage
        self.latest_img = None
        self.have_image = False

        # Subscribe to AN_image
        self.sub = self.create_subscription(
            Image, "/AN_image", self.image_cb, 10
        )

        # Timer at ~30 FPS (33ms)
        self.timer = self.create_timer(1.0/30.0, self.display_cb)

        self.get_logger().info("ANImageViewerNode started. Showing /AN_image at 30 FPS.")

    # ==============================================================
    #                        Callbacks
    # ==============================================================

    def image_cb(self, msg: Image):
        """Receive image, convert to cv2, store latest."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.latest_img = img
        self.have_image = True

    def display_cb(self):
        """Display latest frame at ~30 FPS."""
        if not self.have_image or self.latest_img is None:
            return

        cv2.imshow("AN_image", self.latest_img)

        # process GUI events
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            self.get_logger().info("Quit requested")
            rclpy.shutdown()


# ==============================================================
#                            MAIN
# ==============================================================

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
