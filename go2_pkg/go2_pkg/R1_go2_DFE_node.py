#!/usr/bin/env python3
"""
R1_go2_DFE_node.py

DFE = (Desired-Feature Error) node.

Subscribes:
  /AN_features (std_msgs/Float32MultiArray)
    - length 6  -> interpreted as e_s directly
    - length 12 -> interpreted as [s_d(6), s(6)] and computes e_s = s - s_d

Publishes:
  /e_S (std_msgs/Float32MultiArray, length 6)
"""

from typing import Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# ========================= CONFIG =========================
NODE_NAME = "R1_go2_DFE_node"

FEATURES_TOPIC = "/AN_features"
ES_TOPIC       = "/e_S"          # publishes e_S (length 6)

QUEUE_SIZE = 10
# =========================================================


def parse_features(msg: Float32MultiArray) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      ("es", e_s, None)               if msg length == 6
      ("sd_s", s_d, s)                if msg length == 12
    """
    data = np.asarray(msg.data, dtype=float).reshape(-1)
    if data.size == 6:
        return "es", data, None
    if data.size == 12:
        s_d = data[:6]
        s   = data[6:]
        return "sd_s", s_d, s
    raise ValueError(f"{FEATURES_TOPIC} length {data.size} != 6 or 12")


class DFENode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        # ========================= PUB/SUB =========================
        self.es_pub = self.create_publisher(Float32MultiArray, ES_TOPIC, QUEUE_SIZE)
        self.create_subscription(Float32MultiArray, FEATURES_TOPIC, self.features_cb, QUEUE_SIZE)

        # ========================= STATE =========================
        self.e_s = np.zeros(6, dtype=float)
        self.have_es = False

        self.get_logger().info(f"{NODE_NAME} started.")
        self.get_logger().info(f"Subscribing: {FEATURES_TOPIC}")
        self.get_logger().info(f"Publishing:  {ES_TOPIC} (Float32MultiArray length 6)")

    # ========================= CALLBACKS =========================
    def features_cb(self, msg: Float32MultiArray):
        try:
            mode, v0, v1 = parse_features(msg)
        except Exception as ex:
            self.get_logger().warn_throttle(1.0, f"Bad {FEATURES_TOPIC}: {ex}")
            return

        if mode == "es":
            self.e_s = v0.astype(float)
        else:
            s_d = v0.astype(float)
            s   = v1.astype(float)
            self.e_s = (s - s_d)

        self.have_es = True

        out = Float32MultiArray()
        out.data = self.e_s.astype(float).reshape(6,).tolist()
        self.es_pub.publish(out)


# ============================== MAIN ===============================
def main(args=None):
    rclpy.init(args=args)
    node = DFENode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
