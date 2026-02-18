#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension


# ============================================================
# Board ROI (to avoid desktop icons / outside clutter)
# ============================================================
def _find_board_roi(frame_bgr: np.ndarray):
    """
    Detect the large bright/low-saturation rectangle (your board) and return ROI:
      (x0, y0, x1, y1) in pixel coords (inclusive-exclusive convention for slicing).
    If not found, returns full image ROI.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Board is bright and not very saturated (gray/white)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    mask = ((V > 120) & (S < 80)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0, 0, w, h)

    # largest contour
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 0.10 * float(w * h):  # too small -> fallback
        return (0, 0, w, h)

    x, y, ww, hh = cv2.boundingRect(c)

    # expand slightly (margin)
    margin = 10
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + ww + margin)
    y1 = min(h, y + hh + margin)
    return (x0, y0, x1, y1)


# ============================================================
# Color blob helpers (with circularity + size filtering)
# ============================================================
def _morph_clean(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def _contour_circularity(cnt) -> float:
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _best_round_blob_centroid(mask: np.ndarray, min_area: int, r_min: float, r_max: float, circ_min: float):
    """
    Pick the "best" contour by score = area * circularity, after filtering:
      - area >= min_area
      - enclosing circle radius within [r_min, r_max]
      - circularity >= circ_min
    Return (cx, cy, area, radius, circularity) or (nan,nan,0,0,0)
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (float("nan"), float("nan"), 0.0, 0.0, 0.0)

    best = None
    best_score = -1.0

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < float(min_area):
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        r = float(r)
        if r < r_min or r > r_max:
            continue

        circ = _contour_circularity(c)
        if circ < circ_min:
            continue

        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        score = area * circ
        if score > best_score:
            best_score = score
            best = (cx, cy, area, r, circ)

    if best is None:
        return (float("nan"), float("nan"), 0.0, 0.0, 0.0)
    return best


def _detect_4_colors_bgr_in_roi(frame_bgr: np.ndarray, roi, hsv_thresholds: dict,
                               min_area: int, r_min: float, r_max: float, circ_min: float):
    """
    Detect 4 colored circular markers in ROI.
    Returns:
      pts_px_full: dict color->(x,y) in FULL image pixel coords (float, may be nan)
      stats      : dict color->(area, radius, circularity)
      masks      : dict color->mask (uint8) in ROI coords
      roi_used   : roi tuple (x0,y0,x1,y1)
    """
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = roi
    x0 = int(max(0, min(w - 1, x0)))
    y0 = int(max(0, min(h - 1, y0)))
    x1 = int(max(x0 + 1, min(w, x1)))
    y1 = int(max(y0 + 1, min(h, y1)))

    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    pts_px_full = {}
    stats = {}
    masks = {}

    for key, ranges in hsv_thresholds.items():
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        mask = _morph_clean(mask)

        cx, cy, area, rad, circ = _best_round_blob_centroid(
            mask, min_area=min_area, r_min=r_min, r_max=r_max, circ_min=circ_min
        )

        # map back to full image coords
        if np.isfinite(cx) and np.isfinite(cy):
            pts_px_full[key] = (float(cx + x0), float(cy + y0))
        else:
            pts_px_full[key] = (float("nan"), float("nan"))

        stats[key] = (float(area), float(rad), float(circ))
        masks[key] = mask

    return pts_px_full, stats, masks, (x0, y0, x1, y1)


def _norm_xy(pt_px, w: int, h: int):
    x, y = pt_px
    if not np.isfinite(x) or not np.isfinite(y) or w <= 0 or h <= 0:
        return (float("nan"), float("nan"))
    return (float(x) / float(w), float(y) / float(h))


# ============================================================
# ROS2 Node
# ============================================================
class LoFTRCase4Node(Node):
    """
    Same node name as your original, but does 4-color circular marker detection.
    Publishes:
      /AN_features : 16 floats  [desired(8), current(8)]
      /ALoFTR/matched_points : 4 matches [x0,y0,x1,y1,conf] (normalized)
      /annotated_image : visualization
      /LoFTR/processing_time_ms : processing time (ms)
    """
    def __init__(self):
        super().__init__("R1_go2_4P_T0_c4_node")

        self.bridge = CvBridge()
        self.image0_np = None  # desired
        self.image1_np = None  # current
        self.processing = False

        # ---- joystick desired capture ----
        self.desired_button_index = 1  # button #2 -> index 1
        self._last_joy_buttons = []
        self._request_save_desired = False
        self.save_dir = os.path.expanduser("~/desired_images")
        os.makedirs(self.save_dir, exist_ok=True)
        self._desired_save_counter = 0

        # ---- detection params tuned for your screenshots ----
        # Use ROI board detection + circle filtering to avoid desktop icons.
        self.min_area = 800          # was 120; raise to ignore tiny icons
        self.r_min = 12.0            # minimum marker radius (pixels)
        self.r_max = 80.0            # maximum marker radius (pixels)
        self.circ_min = 0.55         # circularity filter

        # HSV ranges (OpenCV HSV: H=[0..179], S=[0..255], V=[0..255])
        # NOTE: Blue updated based on your screenshot: mean H ~100, S~255, V~184.
        self.hsv_th = {
            "R": [  # red wraps around
                (np.array([0, 90, 50]),    np.array([12, 255, 255])),
                (np.array([168, 90, 50]),  np.array([179, 255, 255])),
            ],
            "G": [(np.array([35, 70, 40]),  np.array([90, 255, 255]))],
            # BLUE tuned wider but still saturated to avoid gray board:
            "B": [(np.array([85, 120, 40]), np.array([130, 255, 255]))],
            "Y": [(np.array([15, 80, 70]),  np.array([45, 255, 255]))],
        }
        self.color_order = ["R", "G", "B", "Y"]  # fixed ordering

        # ---- pubs/subs ----
        self.sub_image1 = self.create_subscription(Image, "/camera_test", self.image1_cb, 10)
        self.sub_image0 = self.create_subscription(Image, "/camera_test_d", self.image0_cb, 10)
        self.sub_joy = self.create_subscription(Joy, "/joy", self.joy_cb, 10)

        self.pub_annotated = self.create_publisher(Image, "/annotated_image", 10)
        self.pub_matched = self.create_publisher(Float32MultiArray, "/ALoFTR/matched_points", 10)
        self.pub_features = self.create_publisher(Float32MultiArray, "/AN_features", 10)
        self.pub_proc_time = self.create_publisher(Float32, "/LoFTR/processing_time_ms", 10)

        self.process_timer = self.create_timer(0.001, self.process_timer_cb)

        self.get_logger().info("Color-4points node initialized (ROI+circle filtering).")

    # ---------------- callbacks ----------------
    def image1_cb(self, msg: Image):
        try:
            self.image1_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert /camera_test: {e}")

    def image0_cb(self, msg: Image):
        try:
            self.image0_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert /camera_test_d: {e}")

    def joy_cb(self, msg: Joy):
        buttons = list(msg.buttons) if msg.buttons is not None else []
        if len(buttons) <= self.desired_button_index:
            self._last_joy_buttons = buttons
            return

        prev = 0
        if len(self._last_joy_buttons) > self.desired_button_index:
            prev = int(self._last_joy_buttons[self.desired_button_index])
        curr = int(buttons[self.desired_button_index])
        self._last_joy_buttons = buttons

        if prev == 0 and curr == 1:
            self._request_save_desired = True

    def _handle_desired_capture_and_save(self):
        if not self._request_save_desired:
            return
        self._request_save_desired = False

        if self.image1_np is None:
            self.get_logger().warn("Desired capture requested, but no current image available yet.")
            return

        self.image0_np = self.image1_np.copy()

        self._desired_save_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"desired_{ts}_{self._desired_save_counter:06d}.png"
        fpath = os.path.join(self.save_dir, fname)

        try:
            ok = cv2.imwrite(fpath, self.image0_np)
        except Exception as e:
            self.get_logger().error(f"Failed to write desired image: {e}")
            ok = False

        if ok:
            self.get_logger().info(f"Saved desired image: {fpath}")
        else:
            self.get_logger().error(f"cv2.imwrite failed for: {fpath}")

    # ---------------- publishing helpers ----------------
    def _publish_proc_time_ms(self, ms: float):
        m = Float32()
        m.data = float(ms)
        self.pub_proc_time.publish(m)

    def _publish_matched_points_4(self, desired_pts_n, current_pts_n):
        """
        /ALoFTR/matched_points: 4 matches (N=4) flattened:
          [x0_n, y0_n, x1_n, y1_n, conf]
        conf=1 if both desired+current exist, else 0 (NaNs kept).
        """
        N = 4
        arr = np.zeros((N, 5), dtype=np.float32)

        for i, key in enumerate(self.color_order):
            x0, y0 = desired_pts_n[key]
            x1, y1 = current_pts_n[key]
            arr[i, 0] = np.float32(x0)
            arr[i, 1] = np.float32(y0)
            arr[i, 2] = np.float32(x1)
            arr[i, 3] = np.float32(y1)
            valid = np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)
            arr[i, 4] = np.float32(1.0 if valid else 0.0)

        msg = Float32MultiArray()
        msg.data = arr.flatten().tolist()

        dim0 = MultiArrayDimension()
        dim0.label = "matches"
        dim0.size = N
        dim0.stride = 5 * N

        dim1 = MultiArrayDimension()
        dim1.label = "components"
        dim1.size = 5
        dim1.stride = 5

        msg.layout.dim = [dim0, dim1]
        self.pub_matched.publish(msg)

    def _publish_features_16(self, desired_pts_n, current_pts_n):
        """
        /AN_features: 16 floats (normalized):
          desired: [xR,yR,xG,yG,xB,yB,xY,yY]
          current: [xR,yR,xG,yG,xB,yB,xY,yY]
        """
        vec = []
        for key in self.color_order:
            vec.extend(list(desired_pts_n[key]))
        for key in self.color_order:
            vec.extend(list(current_pts_n[key]))

        msg = Float32MultiArray()
        msg.data = [float(v) for v in vec]
        self.pub_features.publish(msg)

    def _publish_annotated(self, frame_bgr, roi, desired_pts_px, current_pts_px, stats_cur):
        vis = frame_bgr.copy()

        # draw ROI
        x0, y0, x1, y1 = roi
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)

        bgr_map = {
            "R": (0, 0, 255),
            "G": (0, 255, 0),
            "B": (255, 0, 0),
            "Y": (0, 255, 255),
        }

        # draw current points
        for key in self.color_order:
            cx, cy = current_pts_px[key]
            if np.isfinite(cx) and np.isfinite(cy):
                cv2.circle(vis, (int(cx), int(cy)), 10, bgr_map[key], 2)
                cv2.putText(vis, f"{key}", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_map[key], 2, cv2.LINE_AA)

        # draw desired points + lines
        for key in self.color_order:
            dx, dy = desired_pts_px[key]
            cx, cy = current_pts_px[key]
            if np.isfinite(dx) and np.isfinite(dy):
                cv2.circle(vis, (int(dx), int(dy)), 7, (255, 255, 255), 2)
                cv2.putText(vis, f"D{key}", (int(dx) + 10, int(dy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if np.isfinite(dx) and np.isfinite(dy) and np.isfinite(cx) and np.isfinite(cy):
                cv2.line(vis, (int(dx), int(dy)), (int(cx), int(cy)), bgr_map[key], 2)

        # optional: show current blob areas on screen (top-left)
        ytxt = 25
        for key in self.color_order:
            area, rad, circ = stats_cur[key]
            cv2.putText(vis, f"{key}.A={int(area)} r={rad:.1f} c={circ:.2f}",
                        (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_map[key], 2, cv2.LINE_AA)
            ytxt += 22

        try:
            msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            self.pub_annotated.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    # ---------------- main processing loop ----------------
    def process_timer_cb(self):
        if self.processing:
            return
        if self.image1_np is None:
            return

        self.processing = True
        try:
            self._handle_desired_capture_and_save()

            if self.image0_np is None or self.image1_np is None:
                return

            t0 = time.perf_counter()

            # Use ROI from CURRENT frame (stable) for both detections
            roi = _find_board_roi(self.image1_np)

            des_pts_px, _, _, _ = _detect_4_colors_bgr_in_roi(
                self.image0_np, roi, self.hsv_th,
                min_area=self.min_area, r_min=self.r_min, r_max=self.r_max, circ_min=self.circ_min
            )
            cur_pts_px, stats_cur, _, _ = _detect_4_colors_bgr_in_roi(
                self.image1_np, roi, self.hsv_th,
                min_area=self.min_area, r_min=self.r_min, r_max=self.r_max, circ_min=self.circ_min
            )

            h, w = self.image1_np.shape[:2]
            des_pts_n = {k: _norm_xy(des_pts_px[k], w, h) for k in self.color_order}
            cur_pts_n = {k: _norm_xy(cur_pts_px[k], w, h) for k in self.color_order}

            # Publish features + matched points
            self._publish_features_16(des_pts_n, cur_pts_n)
            self._publish_matched_points_4(des_pts_n, cur_pts_n)

            # Annotate on current frame (with desired overlay)
            self._publish_annotated(self.image1_np, roi, des_pts_px, cur_pts_px, stats_cur)

            t1 = time.perf_counter()
            self._publish_proc_time_ms((t1 - t0) * 1000.0)

        except Exception as e:
            self.get_logger().error(f"Error in process_timer_cb: {e}")
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = LoFTRCase4Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
