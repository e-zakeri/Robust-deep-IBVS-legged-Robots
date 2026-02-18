#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Float32MultiArray


# ------------------------------------------------------------
# ROI detection (unchanged)
# ------------------------------------------------------------
def _find_board_roi(frame_bgr: np.ndarray):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    mask = ((V > 120) & (S < 90)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0, 0, w, h)

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 0.10 * float(w * h):
        return (0, 0, w, h)

    x, y, ww, hh = cv2.boundingRect(c)
    return (x, y, x + ww, y + hh)


def _morph_clean(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


# ------------------------------------------------------------
# Red segmentation
# ------------------------------------------------------------
def _red_mask(frame_bgr, roi):
    x0, y0, x1, y1 = roi
    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0, 100, 40), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (168, 100, 40), (179, 255, 255))

    mask = cv2.bitwise_or(mask1, mask2)
    mask = _morph_clean(mask)
    return mask, (x0, y0)


def _largest_contour(mask, min_area=3000):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


# ------------------------------------------------------------
# Moment features (6D)
# ------------------------------------------------------------
def _moment_features(contour, offset, W, H):
    if contour is None:
        return [float("nan")] * 6, {"valid": False}

    M = cv2.moments(contour)
    if M["m00"] <= 1e-6:
        return [float("nan")] * 6, {"valid": False}

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]

    alpha = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    cov = np.array([[mu20, mu11],
                    [mu11, mu02]])

    eigvals = np.linalg.eigvalsh(cov)
    xs = np.sqrt(max(eigvals[1], 0.0)) / max(W, H)
    ys = np.sqrt(max(eigvals[0], 0.0)) / max(W, H)

    cx_full = cx + offset[0]
    cy_full = cy + offset[1]

    gx = cx_full / W
    gy = cy_full / H
    a  = M["m00"] / (W * H)

    aux = {"valid": True, "centroid": (cx_full, cy_full)}

    return [gx, gy, a, alpha, xs, ys], aux


# ------------------------------------------------------------
# Node
# ------------------------------------------------------------
class R1MomentRedNode(Node):
    def __init__(self):
        super().__init__("R1_go2_IB_T0_c4_node")

        self.bridge = CvBridge()
        self.image0_np = None
        self.image1_np = None
        self.processing = False

        self.sub_image1 = self.create_subscription(Image, "/camera_test", self.image1_cb, 10)
        self.sub_image0 = self.create_subscription(Image, "/camera_test_d", self.image0_cb, 10)

        self.pub_annotated = self.create_publisher(Image, "/annotated_image", 10)
        self.pub_features  = self.create_publisher(Float32MultiArray, "/AN_features", 10)
        self.pub_proc_time = self.create_publisher(Float32, "/LoFTR/processing_time_ms", 10)

        self.timer = self.create_timer(0.001, self.process_cb)

    def image1_cb(self, msg):
        self.image1_np = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def image0_cb(self, msg):
        self.image0_np = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def process_cb(self):
        if self.processing:
            return
        if self.image0_np is None or self.image1_np is None:
            return

        self.processing = True
        try:
            t0 = time.perf_counter()

            H, W = self.image1_np.shape[:2]
            roi = _find_board_roi(self.image1_np)

            mask_d, off_d = _red_mask(self.image0_np, roi)
            mask_c, off_c = _red_mask(self.image1_np, roi)

            c_d = _largest_contour(mask_d)
            c_c = _largest_contour(mask_c)

            s_d, _ = _moment_features(c_d, off_d, W, H)
            s_c, aux_c = _moment_features(c_c, off_c, W, H)

            msg = Float32MultiArray()
            msg.data = s_d + s_c
            self.pub_features.publish(msg)

            vis = self.image1_np.copy()
            vis[roi[1]:roi[3], roi[0]:roi[2]][mask_c > 0] = (0, 0, 255)

            if aux_c["valid"]:
                cx, cy = aux_c["centroid"]
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)

                contour_full = c_c.copy()
                contour_full[:, 0, 0] += roi[0]
                contour_full[:, 0, 1] += roi[1]
                cv2.drawContours(vis, [contour_full], -1, (0, 255, 255), 3)

            self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

            t1 = time.perf_counter()
            self.pub_proc_time.publish(Float32(data=(t1 - t0) * 1000.0))

        finally:
            self.processing = False


def main():
    rclpy.init()
    node = R1MomentRedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
