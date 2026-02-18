#!/usr/bin/env python3
"""
camera_generator_node.py

Grabs /dev/videoN with OpenCV-V4L2 and republishes each frame
as sensor_msgs/Image on /camera_test (encoding = bgr8).

Optional launch-time parameters
--------------------------------
camera_index       int   default 0      # /dev/videoX
frame_rate         int   default 30     # Hz cap + publish rate
resize_long_edge   int   default 0      # 0 = keep native; else shrink longest side
image_width        int   default 720    # 0 = leave native; else request this width
image_height       int   default 1024   # 0 = leave native; else request this height
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraGeneratorNode(Node):
    def __init__(self):
        super().__init__('camera_generator_node')

        # Declare parameters (with new defaults 720×1024)
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("frame_rate", 30)
        self.declare_parameter("resize_long_edge", 0)
        self.declare_parameter("image_width", 960)
        self.declare_parameter("image_height", 1280)

        # Read parameters
        self.cam_idx   = self.get_parameter("camera_index").value
        self.fps       = self.get_parameter("frame_rate").value
        self.long_edge = self.get_parameter("resize_long_edge").value
        self.width     = self.get_parameter("image_width").value
        self.height    = self.get_parameter("image_height").value

        # OpenCV video capture
        self.bridge = CvBridge()
        self.cap    = cv2.VideoCapture(self.cam_idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{self.cam_idx}")

        # Request capture resolution if specified
        if self.width  > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Query back actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera opened at {actual_w}×{actual_h} @ index {self.cam_idx}")

        # Publisher and timer
        self.pub   = self.create_publisher(Image, "/camera_test", 1)
        self.timer = self.create_timer(1.0 / self.fps, self.loop)

        self.get_logger().info(
            f"Publishing /camera_test @ {self.fps} Hz, "
            f"resize_long_edge={self.long_edge}, "
            f"requested {self.width}×{self.height}"
        )

    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("Empty frame – skipping")
            return

        # Optional shrink of longest side
        if self.long_edge > 0:
            h, w = frame.shape[:2]
            if max(h, w) > self.long_edge:
                scale = self.long_edge / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = CameraGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
