#!/usr/bin/env python3
import os
import sys
import time

import cv2
import numpy as np
import torch

from contextlib import nullcontext
from torch.cuda.amp import autocast

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension

torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)


# =========================
# Helpers
# =========================
def to_fixed_gray_np(img_bgr_or_gray: np.ndarray, out_w=800, out_h=600) -> np.ndarray:
    """Return uint8 grayscale resized to fixed WxH."""
    if img_bgr_or_gray is None:
        raise ValueError("Input image is None")

    img = img_bgr_or_gray
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def draw_points_on_image1_return(img1_gray_float_or_u8, mkpts0, mkpts1, title, max_points=20000):
    if img1_gray_float_or_u8 is None:
        return None

    if img1_gray_float_or_u8.dtype != np.uint8:
        img = np.clip(img1_gray_float_or_u8, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    else:
        img = img1_gray_float_or_u8.copy()

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    mkpts0 = mkpts0.copy() if mkpts0 is not None else np.zeros((0, 2), dtype=np.float32)
    mkpts1 = mkpts1.copy() if mkpts1 is not None else np.zeros((0, 2), dtype=np.float32)

    N = len(mkpts0)
    if N > max_points:
        idx = np.linspace(0, N - 1, max_points).astype(int)
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]

    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        p0 = (int(x0), int(y0))
        p1 = (int(x1), int(y1))
        cv2.circle(vis, p0, 3, (255, 0, 0), -1)   # desired
        cv2.circle(vis, p1, 3, (0, 255, 0), -1)   # current
        cv2.line(vis, p0, p1, (0, 165, 255), 1)

    cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


# =========================
# ROS2 Node (SuperPoint/SuperGlue, fixed input size)
# =========================
class SPSGFixedNode(Node):
    def __init__(self):
        super().__init__("R1_go2_SPSG_fixed_node")

        # -------- fixed input resolution ----------
        self.fixed_w = 800
        self.fixed_h = 600

        # -------- SuperGlue repo path (same style as your old code) ----------
        self.REPO_DIR = "/home/ehsan/SuperGluePretrainedNetwork"
        if self.REPO_DIR not in sys.path:
            sys.path.append(self.REPO_DIR)

        from models.matching import Matching
        from models.utils import frame2tensor

        self.frame2tensor = frame2tensor

        # -------- device / AMP ----------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        self.use_amp = True
        if self.device == "cuda" and self.use_amp:
            self.amp_ctx = autocast
        else:
            self.amp_ctx = nullcontext
        self.get_logger().info(f"AMP enabled: {self.use_amp}")

        # -------- SuperPoint/SuperGlue config (same as your old code) ----------
        config = {
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": -1,  # keep all
            },
            "superglue": {
                "weights": "indoor",
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2,
            }
        }
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ["keypoints", "scores", "descriptors"]

        # -------- ROS I/O ----------
        self.bridge = CvBridge()
        self.image0_np = None  # desired (BGR)
        self.image1_np = None  # current (BGR)
        self.processing = False
        self.enable_visualization = True

        # joystick desired capture (button #2 -> index 1)
        self.desired_button_index = 1
        self._last_joy_buttons = []
        self._request_save_desired = False
        self.save_dir = os.path.expanduser("~/desired_images")
        os.makedirs(self.save_dir, exist_ok=True)
        self._desired_save_counter = 0

        # cached desired SuperPoint data (like your old code)
        self.desired_data = None        # dict: keypoints0, scores0, descriptors0, image0
        self.desired_gray_u8 = None     # fixed-size gray uint8

        # stats
        self.curr_width = self.fixed_w
        self.curr_height = self.fixed_h
        self.last_n_matches = 0
        self.last_proc_ms = 0.0

        # visualization shared
        self.vis_img1_gray_u8 = None
        self.vis_mkpts0 = np.zeros((0, 2), dtype=np.float32)
        self.vis_mkpts1 = np.zeros((0, 2), dtype=np.float32)
        self.vis_title = ""
        self.vis_has_data = False

        # subs
        self.sub_image1 = self.create_subscription(Image, "/camera_test", self.image1_cb, 10)
        self.sub_image0 = self.create_subscription(Image, "/camera_test_d", self.image0_cb, 10)
        self.sub_joy = self.create_subscription(Joy, "/joy", self.joy_cb, 10)

        # pubs (keep same topics you already used)
        self.pub_annotated = self.create_publisher(Image, "/annotated_image", 10)
        self.pub_matched = self.create_publisher(Float32MultiArray, "/ALoFTR/matched_points", 10)

        # processing time topic (kept name from your LoFTR node; you can rename)
        self.pub_proc_time = self.create_publisher(Float32, "/LoFTR/processing_time_ms", 10)

        # timers
        self.process_timer = self.create_timer(0.001, self.process_timer_cb)
        self.vis_timer = self.create_timer(0.02, self.vis_timer_cb)

        self.get_logger().info(
            f"SPSG fixed node initialized: fixed input {self.fixed_w}x{self.fixed_h}."
        )

    # ---------------- Callbacks ----------------
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

        # set desired = current
        self.image0_np = self.image1_np.copy()

        # save desired image (BGR)
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

        # IMPORTANT: reset cached SuperPoint for desired
        self.desired_data = None
        self.desired_gray_u8 = None

    # ---------------- Processing loop ----------------
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

            self.process_pair_full()
        except Exception as e:
            self.get_logger().error(f"Error in process_pair_full: {e}")
        finally:
            self.processing = False

    def _ensure_desired_cached(self):
        """
        Cache desired SuperPoint outputs once (like your old self.last_data).
        """
        if self.desired_data is not None and self.desired_gray_u8 is not None:
            return

        # fixed-size gray desired
        des_gray = to_fixed_gray_np(self.image0_np, out_w=self.fixed_w, out_h=self.fixed_h)
        des_t = self.frame2tensor(des_gray, self.device)

        with torch.no_grad():
            with self.amp_ctx():
                sp0 = self.matching.superpoint({"image": des_t})

        self.desired_data = {k + "0": sp0[k] for k in self.keys}  # keypoints0/scores0/descriptors0
        self.desired_data["image0"] = des_t
        self.desired_gray_u8 = des_gray

    def publish_matched_points(self, mkpts0, mkpts1, mconf=None):
        """
        Publish on /ALoFTR/matched_points (normalized):
          [x0_n, y0_n, x1_n, y1_n, conf]
        """
        msg = Float32MultiArray()

        if mkpts0 is None or mkpts1 is None or mkpts0.shape[0] == 0 or mkpts1.shape[0] == 0:
            msg.layout.dim = []
            msg.data = []
            self.pub_matched.publish(msg)
            return

        W = float(self.curr_width)
        H = float(self.curr_height)

        N = min(mkpts0.shape[0], mkpts1.shape[0])
        mk0 = mkpts0[:N].astype(np.float32)
        mk1 = mkpts1[:N].astype(np.float32)

        if mconf is not None:
            conf = mconf[:N].astype(np.float32)
        else:
            conf = np.ones((N,), dtype=np.float32)

        arr = np.zeros((N, 5), dtype=np.float32)
        arr[:, 0] = mk0[:, 0] / W
        arr[:, 1] = mk0[:, 1] / H
        arr[:, 2] = mk1[:, 0] / W
        arr[:, 3] = mk1[:, 1] / H
        arr[:, 4] = conf

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

    def process_pair_full(self):
        """
        Full-image SuperPoint/SuperGlue, fixed input size:
          - desired SuperPoint cached
          - current SuperPoint each frame
          - SuperGlue matches
        """
        t0 = time.perf_counter()

        # ensure desired cached (SuperPoint once)
        self._ensure_desired_cached()
        if self.desired_data is None:
            return

        # fixed-size current gray
        cur_gray = to_fixed_gray_np(self.image1_np, out_w=self.fixed_w, out_h=self.fixed_h)
        cur_t = self.frame2tensor(cur_gray, self.device)

        mkpts0 = np.zeros((0, 2), dtype=np.float32)
        mkpts1 = np.zeros((0, 2), dtype=np.float32)
        confs = None

        with torch.no_grad():
            with self.amp_ctx():
                pred = self.matching({**self.desired_data, "image1": cur_t})

        # Extract matches exactly like your old code
        kpts0 = self.desired_data["keypoints0"][0].detach().cpu().numpy()
        kpts1 = pred["keypoints1"][0].detach().cpu().numpy()
        matches0 = pred["matches0"][0].detach().cpu().numpy().astype(np.int32)
        scores0 = pred["matching_scores0"][0].detach().cpu().numpy().astype(np.float32)

        valid = matches0 > -1
        if np.any(valid):
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches0[valid]]
            confs = scores0[valid]

        self.last_n_matches = int(mkpts0.shape[0])

        if self.last_n_matches == 0:
            self.publish_matched_points(
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                None
            )
            self.vis_title = "SPSG (fixed) - No matches"
            self.vis_mkpts0 = np.zeros((0, 2), dtype=np.float32)
            self.vis_mkpts1 = np.zeros((0, 2), dtype=np.float32)
        else:
            self.publish_matched_points(mkpts0, mkpts1, confs)
            self.vis_title = f"SPSG (fixed), matches={self.last_n_matches}"
            self.vis_mkpts0 = mkpts0
            self.vis_mkpts1 = mkpts1

        self.vis_img1_gray_u8 = cur_gray
        self.vis_has_data = True

        t1 = time.perf_counter()
        self.last_proc_ms = (t1 - t0) * 1000.0

        # publish processing time (ms)
        msg = Float32()
        msg.data = float(self.last_proc_ms)
        self.pub_proc_time.publish(msg)

    # ---------------- Visualization loop ----------------
    def vis_timer_cb(self):
        if not self.enable_visualization:
            return
        if not self.vis_has_data or self.vis_img1_gray_u8 is None:
            return

        vis = draw_points_on_image1_return(
            self.vis_img1_gray_u8,
            self.vis_mkpts0,
            self.vis_mkpts1,
            self.vis_title,
        )
        if vis is None:
            return

        cv2.putText(
            vis,
            f"Proc time: {self.last_proc_ms:.2f} ms",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Fixed Res: {self.fixed_w}x{self.fixed_h}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        self.publish_image(vis)

    def publish_image(self, img_bgr):
        try:
            if img_bgr.dtype != np.uint8:
                img_bgr = np.clip(img_bgr, 0.0, 1.0)
                img_bgr = (img_bgr * 255.0).astype(np.uint8)
            msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            self.pub_annotated.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SPSGFixedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
