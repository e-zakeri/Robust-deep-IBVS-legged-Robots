#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import time  # for timing

from contextlib import nullcontext
from torch.cuda.amp import autocast
from kornia.feature import LoFTR as KorniaLoFTR

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension  # <-- added Float32

torch.backends.cudnn.benchmark = False


# =========================
# LoFTR pipeline
# =========================
class LoFTRPipeline(nn.Module):
    def __init__(self, pretrained: str = "outdoor"):
        super().__init__()
        base = KorniaLoFTR(pretrained=pretrained)

        self.backbone = base.backbone
        self.pos_encoding = base.pos_encoding
        self.loftr_coarse = base.loftr_coarse
        self.coarse_matching = base.coarse_matching
        self.fine_preprocess = base.fine_preprocess
        self.loftr_fine = base.loftr_fine
        self.fine_matching = base.fine_matching

    def run_backbone(self, data):
        data.update({
            "bs": data["image0"].size(0),
            "hw0_i": data["image0"].shape[2:],
            "hw1_i": data["image1"].shape[2:]
        })

        if data["hw0_i"] == data["hw1_i"]:
            feats_c, feats_f = self.backbone(
                torch.cat([data["image0"], data["image1"]], dim=0)
            )
            (feat_c0, feat_c1), (feat_f0, feat_f1) = (
                feats_c.split(data["bs"]), feats_f.split(data["bs"])
            )
        else:
            (feat_c0, feat_f0) = self.backbone(data["image0"])
            (feat_c1, feat_f1) = self.backbone(data["image1"])

        data["feat_c0"], data["feat_c1"] = feat_c0, feat_c1
        data["feat_f0"], data["feat_f1"] = feat_f0, feat_f1

        data.update({
            "hw0_c": feat_c0.shape[2:], "hw1_c": feat_c1.shape[2:],
            "hw0_f": feat_f0.shape[2:], "hw1_f": feat_f1.shape[2:]
        })

    def run_coarse_matching(self, data):
        self.coarse_matching(data["feat_c0"], data["feat_c1"], data, None, None)

    def run_fine_preprocess(self, data):
        f0, f1 = data["feat_f0"], data["feat_f1"]
        c0, c1 = data["feat_c0"], data["feat_c1"]
        u0, u1 = self.fine_preprocess(f0, f1, c0, c1, data)
        data["feat_f0_unfold"], data["feat_f1_unfold"] = u0, u1

    def run_fine_transformer(self, data):
        if data["feat_f0_unfold"].size(0) > 0:
            f0, f1 = self.loftr_fine(
                data["feat_f0_unfold"], data["feat_f1_unfold"]
            )
            data["feat_f0_unfold"], data["feat_f1_unfold"] = f0, f1

    def run_fine_matching(self, data):
        self.fine_matching(data["feat_f0_unfold"], data["feat_f1_unfold"], data)


# =========================
# Helpers
# =========================
def load_gray_image_as_tensor_from_np(img, device, max_size=640):
    """Input img: uint8 BGR or GRAY numpy; output: (tensor [1,1,H,W], float32 gray [0,1])."""
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    img_f = np.ascontiguousarray(img.astype(np.float32) / 255.0)
    tensor = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device, non_blocking=True)
    return tensor, img_f  # img_f is float32 [0,1]


def compute_used_cells_from_matches(mkpts, Hc, Wc, img_h, img_w):
    if mkpts.shape[0] == 0:
        return np.zeros((Hc, Wc), dtype=bool)

    cell_w = img_w / float(Wc)
    cell_h = img_h / float(Hc)

    xs = mkpts[:, 0]
    ys = mkpts[:, 1]

    j = np.clip((xs / cell_w).astype(int), 0, Wc - 1)
    i = np.clip((ys / cell_h).astype(int), 0, Hc - 1)

    used_mask = np.zeros((Hc, Wc), dtype=bool)
    used_mask[i, j] = True
    return used_mask


def compute_coarse_bbox_indices(used_mask):
    if not used_mask.any():
        return None
    coords = np.argwhere(used_mask)
    i_min = int(coords[:, 0].min())
    i_max = int(coords[:, 0].max())
    j_min = int(coords[:, 1].min())
    j_max = int(coords[:, 1].max())
    return i_min, i_max, j_min, j_max


def draw_points_on_image1_return(
    img1,
    mkpts0,
    mkpts1,
    title,
    bbox_desired=None,   # bbox for desired image (image0)  -> white
    bbox_current=None,   # bbox for current image (image1)  -> orange
    max_points=20000
):
    if img1.dtype != np.uint8:
        img = np.clip(img1, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    else:
        img = img1.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if bbox_desired is not None:
        dx0, dy0, dx1, dy1 = bbox_desired
        cv2.rectangle(img, (int(dx0), int(dy0)), (int(dx1), int(dy1)), (255, 255, 255), 2)

    if bbox_current is not None:
        cx0, cy0, cx1, cy1 = bbox_current
        cv2.rectangle(img, (int(cx0), int(cy0)), (int(cx1), int(cy1)), (0, 165, 255), 2)

    mkpts0 = mkpts0.copy()
    mkpts1 = mkpts1.copy()

    N = len(mkpts0)
    if N > max_points:
        idx = np.linspace(0, N - 1, max_points).astype(int)
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]

    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        p0 = (int(x0), int(y0))
        p1 = (int(x1), int(y1))

        cv2.circle(img, p0, 3, (255, 0, 0), -1)
        cv2.circle(img, p1, 3, (0, 255, 0), -1)
        cv2.line(img, p0, p1, (0, 165, 255), 1)

    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img


# =========================
# ROS2 Node
# =========================
class LoFTRCase4Node(Node):
    def __init__(self):
        super().__init__("R1_go2_AEoFTR_T0_c4_node")

        # Parameters
        self.pretrained = "indoor"
        self.max_size = 800
        self.curr_frame_max_size = self.max_size

        self.curr_width = 0
        self.curr_height = 0

        self.enable_visualization = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.use_amp = True
        if self.device.type == "cuda" and self.use_amp:
            self.amp_ctx = autocast
        else:
            self.amp_ctx = nullcontext

        self.get_logger().info(f"AMP enabled: {self.use_amp}")
        self.get_logger().info(f"Visualization enabled (code flag): {self.enable_visualization}")

        # LoFTR model
        self.loftr = LoFTRPipeline(self.pretrained).to(self.device)
        self.loftr.eval()

        # ROS
        self.bridge = CvBridge()
        self.image0_np = None
        self.image1_np = None

        self.processing = False

        # Joystick save desired
        self.desired_button_index = 1  # button #2 -> index 1
        self._last_joy_buttons = []
        self._request_save_desired = False
        self.save_dir = os.path.expanduser("~/desired_images")
        os.makedirs(self.save_dir, exist_ok=True)
        self._desired_save_counter = 0

        # bbox state
        self.frame_count = 0
        self.bbox_warmup_frames = 100
        self.full_bbox_interval = 10

        self.cached_bbox0_coarse = None
        self.cached_bbox1_coarse = None
        self.cached_bbox0_norm = None
        self.cached_bbox1_norm = None

        self.last_bbox_area_ratio = 1.0

        # FPS counters still kept (optional, not shown on image)
        self.proc_fps_count = 0
        self.proc_fps_value = 0.0
        self.cam_fps_count = 0
        self.cam_fps_value = 0.0
        self.vis_fps_count = 0
        self.vis_fps_value = 0.0

        # last-frame debug
        self.last_n_matches = 0
        self.last_use_full_bbox = True

        # last measured per-frame times (ms)
        self.last_proc_ms = 0.0
        self.last_vis_ms = 0.0

        # timing accumulators (kept)
        self.time_backbone_acc = 0.0
        self.time_bbox_acc = 0.0
        self.time_coarse_crop_acc = 0.0
        self.time_coarse_posenc_acc = 0.0
        self.time_coarse_transformer_acc = 0.0
        self.time_coarse_matching_acc = 0.0
        self.time_fine_preprocess_acc = 0.0
        self.time_fine_transformer_acc = 0.0
        self.time_fine_matching_acc = 0.0
        self.time_img_convert_acc = 0.0
        self.time_draw_acc = 0.0
        self.time_publish_acc = 0.0
        self.time_total_acc = 0.0

        # visualization shared data
        self.vis_img1_gray = None
        self.vis_mkpts0_global = None
        self.vis_mkpts1_global = None
        self.vis_draw_bbox_desired = None
        self.vis_draw_bbox_current = None
        self.vis_title = ""
        self.vis_has_data = False

        # subs
        self.sub_image1 = self.create_subscription(Image, "/camera_test", self.image1_cb, 10)
        self.sub_image0 = self.create_subscription(Image, "/camera_test_d", self.image0_cb, 10)
        self.sub_joy = self.create_subscription(Joy, "/joy", self.joy_cb, 10)

        # pubs
        self.pub_annotated = self.create_publisher(Image, "/annotated_image", 10)
        self.pub_matched = self.create_publisher(Float32MultiArray, "/ALoFTR/matched_points", 10)

        # NEW: publish processing time (ms), same as other code
        self.pub_proc_time = self.create_publisher(Float32, "/LoFTR/processing_time_ms", 10)

        # timers
        self.process_timer = self.create_timer(0.001, self.process_timer_cb)
        self.vis_timer = self.create_timer(0.02, self.vis_timer_cb)
        self.fps_timer = self.create_timer(1.0, self.fps_timer_cb)

        self.get_logger().info("LoFTR Case4 node initialized.")

    # ---------------- Callbacks ----------------
    def image1_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert /camera_test: {e}")
            return
        self.image1_np = cv_img
        self.cam_fps_count += 1

    def image0_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert /camera_test_d: {e}")
            return
        self.image0_np = cv_img

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

        ok = False
        try:
            ok = cv2.imwrite(fpath, self.image0_np)
        except Exception as e:
            self.get_logger().error(f"Failed to write desired image: {e}")
            ok = False

        if ok:
            self.get_logger().info(f"Saved desired image: {fpath}")
            self.cached_bbox0_coarse = None
            self.cached_bbox1_coarse = None
            self.cached_bbox0_norm = None
            self.cached_bbox1_norm = None
            self.frame_count = 0
        else:
            self.get_logger().error(f"cv2.imwrite failed for: {fpath}")

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

            self.process_pair()
        except Exception as e:
            self.get_logger().error(f"Error in process_pair: {e}")
        finally:
            self.processing = False

    def publish_matched_points(self, mkpts0, mkpts1, mconf=None):
        msg = Float32MultiArray()

        if (
            mkpts0 is None or mkpts1 is None or
            mkpts0.shape[0] == 0 or mkpts1.shape[0] == 0
        ):
            msg.layout.dim = []
            msg.data = []
            self.pub_matched.publish(msg)
            return

        if self.curr_width <= 0 or self.curr_height <= 0:
            self.get_logger().warn("publish_matched_points called with invalid curr_width/height; skipping.")
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
            mconf = mconf[:N].astype(np.float32)
        else:
            mconf = np.ones((N,), dtype=np.float32)

        arr = np.zeros((N, 5), dtype=np.float32)
        arr[:, 0] = mk0[:, 0] / W
        arr[:, 1] = mk0[:, 1] / H
        arr[:, 2] = mk1[:, 0] / W
        arr[:, 3] = mk1[:, 1] / H
        arr[:, 4] = mconf

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

    def adjust_max_size_from_bbox(self, roi_w0, roi_h0, w0, h0, roi_w1, roi_h1, w1, h1):
        full_area0 = float(w0) * float(h0)
        full_area1 = float(w1) * float(h1)
        if full_area0 <= 0.0 or full_area1 <= 0.0:
            return

        bbox_area0 = float(max(1, roi_w0) * max(1, roi_h0))
        bbox_area1 = float(max(1, roi_w1) * max(1, roi_h1))
        r0 = bbox_area0 / full_area0
        r1 = bbox_area1 / full_area1
        ratio = min(r0, r1)

        old_max_size = self.max_size
        new_max_size = old_max_size

        up_to_960_from_800 = 0.25
        down_to_800_from_960 = 0.35
        up_to_800_from_640 = 0.55
        down_to_640_from_800 = 0.65

        if old_max_size == 960:
            if ratio > down_to_800_from_960:
                new_max_size = 800
        elif old_max_size == 800:
            if ratio < up_to_960_from_800:
                new_max_size = 960
            elif ratio > down_to_640_from_800:
                new_max_size = 640
        elif old_max_size == 640:
            if ratio < up_to_800_from_640:
                new_max_size = 800
        else:
            if ratio < 0.3:
                new_max_size = 960
            elif ratio < 0.6:
                new_max_size = 800
            else:
                new_max_size = 640

        if new_max_size != old_max_size:
            self.max_size = new_max_size
            self.cached_bbox0_coarse = None
            self.cached_bbox1_coarse = None

    def process_pair(self):
        t_total_start = time.perf_counter()

        self.proc_fps_count += 1
        self.frame_count += 1

        if self.frame_count <= self.bbox_warmup_frames:
            use_full_bbox = True
        else:
            no_cache = (self.cached_bbox0_coarse is None or self.cached_bbox1_coarse is None)
            frames_since_warmup = self.frame_count - self.bbox_warmup_frames
            periodic_full = (self.full_bbox_interval > 0 and frames_since_warmup % self.full_bbox_interval == 0)
            use_full_bbox = no_cache or periodic_full

        self.last_use_full_bbox = bool(use_full_bbox)

        frame_max_size = 800 if use_full_bbox else self.max_size
        frame_max_size = self.max_size  # (kept as in your code)
        self.curr_frame_max_size = frame_max_size

        # Image conversion
        t_conv_start = time.perf_counter()

        img0_t, img0_gray = load_gray_image_as_tensor_from_np(self.image0_np, self.device, max_size=frame_max_size)
        img1_t, img1_gray = load_gray_image_as_tensor_from_np(self.image1_np, self.device, max_size=frame_max_size)

        h1, w1 = img1_gray.shape[:2]
        h0, w0 = img0_gray.shape[:2]

        if (h0 != h1) or (w0 != w1):
            img0_gray = cv2.resize(img0_gray, (w1, h1), interpolation=cv2.INTER_AREA).astype(np.float32)
            img0_gray = np.ascontiguousarray(img0_gray)
            img0_t = torch.from_numpy(img0_gray).unsqueeze(0).unsqueeze(0).to(self.device, non_blocking=True)
            h0, w0 = img0_gray.shape[:2]

        t_conv_end = time.perf_counter()
        self.time_img_convert_acc += (t_conv_end - t_conv_start)

        self.curr_width = w1
        self.curr_height = h1

        mkpts0_bbox = np.zeros((0, 2), dtype=np.float32)
        mkpts1_bbox = np.zeros((0, 2), dtype=np.float32)
        mconf_bbox = None

        with torch.no_grad():
            with self.amp_ctx():
                # backbone
                t_backbone_start = time.perf_counter()
                data = {"image0": img0_t, "image1": img1_t}
                self.loftr.run_backbone(data)

                feat_c0_full = data["feat_c0"]
                feat_c1_full = data["feat_c1"]
                feat_f0_full = data["feat_f0"]
                feat_f1_full = data["feat_f1"]

                (Hc0, Wc0) = data["hw0_c"]
                (Hc1, Wc1) = data["hw1_c"]
                bs_val = data["bs"]

                t_backbone_end = time.perf_counter()
                self.time_backbone_acc += (t_backbone_end - t_backbone_start)

                # bbox select
                t_bbox_start = time.perf_counter()

                if use_full_bbox or self.cached_bbox0_norm is None or self.cached_bbox1_norm is None:
                    i0_min, i0_max, j0_min, j0_max = 0, Hc0 - 1, 0, Wc0 - 1
                    i1_min, i1_max, j1_min, j1_max = 0, Hc1 - 1, 0, Wc1 - 1
                else:
                    i0_min_n, i0_max_n, j0_min_n, j0_max_n = self.cached_bbox0_norm
                    i1_min_n, i1_max_n, j1_min_n, j1_max_n = self.cached_bbox1_norm

                    i0_min = max(0, min(Hc0 - 1, int(i0_min_n * Hc0)))
                    i0_max = max(i0_min, min(Hc0 - 1, int(i0_max_n * Hc0)))
                    j0_min = max(0, min(Wc0 - 1, int(j0_min_n * Wc0)))
                    j0_max = max(j0_min, min(Wc0 - 1, int(j0_max_n * Wc0)))

                    i1_min = max(0, min(Hc1 - 1, int(i1_min_n * Hc1)))
                    i1_max = max(i1_min, min(Hc1 - 1, int(i1_max_n * Hc1)))
                    j1_min = max(0, min(Wc1 - 1, int(j1_min_n * Wc1)))
                    j1_max = max(j1_min, min(Wc1 - 1, int(j1_max_n * Wc1)))

                self.cached_bbox0_coarse = (i0_min, i0_max, j0_min, j0_max)
                self.cached_bbox1_coarse = (i1_min, i1_max, j1_min, j1_max)

                t_bbox_end = time.perf_counter()
                self.time_bbox_acc += (t_bbox_end - t_bbox_start)

                # crop
                t_crop_start = time.perf_counter()

                Hf0, Wf0 = feat_f0_full.shape[2:]
                Hf1, Wf1 = feat_f1_full.shape[2:]
                scale_h0 = Hf0 // Hc0
                scale_w0 = Wf0 // Wc0
                scale_h1 = Hf1 // Hc1
                scale_w1 = Wf1 // Wc1

                fi0_top = i0_min * scale_h0
                fi0_bot = (i0_max + 1) * scale_h0
                fj0_left = j0_min * scale_w0
                fj0_right = (j0_max + 1) * scale_w0

                fi1_top = i1_min * scale_h1
                fi1_bot = (i1_max + 1) * scale_h1
                fj1_left = j1_min * scale_w1
                fj1_right = (j1_max + 1) * scale_w1

                feat_c0_sub = feat_c0_full[:, :, i0_min:i0_max + 1, j0_min:j0_max + 1]
                feat_c1_sub = feat_c1_full[:, :, i1_min:i1_max + 1, j1_min:j1_max + 1]

                feat_f0_sub = feat_f0_full[:, :, fi0_top:fi0_bot, fj0_left:fj0_right]
                feat_f1_sub = feat_f1_full[:, :, fi1_top:fi1_bot, fj1_left:fj1_right]

                Hb0, Wb0 = feat_c0_sub.shape[2:]
                Hb1, Wb1 = feat_c1_sub.shape[2:]

                t_crop_end = time.perf_counter()
                self.time_coarse_crop_acc += (t_crop_end - t_crop_start)

                # posenc + flatten
                t_posenc_start = time.perf_counter()

                feat_c0_pe = self.loftr.pos_encoding(feat_c0_sub).permute(0, 2, 3, 1)
                feat_c1_pe = self.loftr.pos_encoding(feat_c1_sub).permute(0, 2, 3, 1)

                B0b, Hb0p, Wb0p, C0p = feat_c0_pe.shape
                B1b, Hb1p, Wb1p, C1p = feat_c1_pe.shape

                feat_c0_flat = feat_c0_pe.reshape(B0b, Hb0 * Wb0, C0p)
                feat_c1_flat = feat_c1_pe.reshape(B1b, Hb1 * Wb1, C1p)

                t_posenc_end = time.perf_counter()
                self.time_coarse_posenc_acc += (t_posenc_end - t_posenc_start)

                # coarse transformer
                t_coarse_tr_start = time.perf_counter()
                feat_c0_out, feat_c1_out = self.loftr.loftr_coarse(feat_c0_flat, feat_c1_flat, None, None)
                t_coarse_tr_end = time.perf_counter()
                self.time_coarse_transformer_acc += (t_coarse_tr_end - t_coarse_tr_start)

                data_bbox = {
                    "feat_c0": feat_c0_out,
                    "feat_c1": feat_c1_out,
                    "feat_f0": feat_f0_sub,
                    "feat_f1": feat_f1_sub,
                    "bs": bs_val,
                }

                cell_h0 = h0 / float(Hc0)
                cell_w0 = w0 / float(Wc0)
                cell_h1 = h1 / float(Hc1)
                cell_w1 = w1 / float(Wc1)

                y0_0 = int(i0_min * cell_h0)
                y1_0 = int((i0_max + 1) * cell_h0)
                x0_0 = int(j0_min * cell_w0)
                x1_0 = int((j0_max + 1) * cell_w0)

                y0_1 = int(i1_min * cell_h1)
                y1_1 = int((i1_max + 1) * cell_h1)
                x0_1 = int(j1_min * cell_w1)
                x1_1 = int((j1_max + 1) * cell_w1)

                roi_h0 = max(1, y1_0 - y0_0)
                roi_w0 = max(1, x1_0 - x0_0)
                roi_h1 = max(1, y1_1 - y0_1)
                roi_w1 = max(1, x1_1 - x0_1)

                data_bbox["hw0_i"] = (roi_h0, roi_w0)
                data_bbox["hw1_i"] = (roi_h1, roi_w1)
                data_bbox["hw0_c"] = (Hb0, Wb0)
                data_bbox["hw1_c"] = (Hb1, Wb1)
                data_bbox["hw0_f"] = feat_f0_sub.shape[2:]
                data_bbox["hw1_f"] = feat_f1_sub.shape[2:]

                # coarse matching
                t_coarse_match_start = time.perf_counter()
                self.loftr.run_coarse_matching(data_bbox)
                t_coarse_match_end = time.perf_counter()
                self.time_coarse_matching_acc += (t_coarse_match_end - t_coarse_match_start)

                # fine preprocess
                t_fine_pre_start = time.perf_counter()
                self.loftr.run_fine_preprocess(data_bbox)
                t_fine_pre_end = time.perf_counter()
                self.time_fine_preprocess_acc += (t_fine_pre_end - t_fine_pre_start)

                # fine transformer
                t_fine_tr_start = time.perf_counter()
                self.loftr.run_fine_transformer(data_bbox)
                t_fine_tr_end = time.perf_counter()
                self.time_fine_transformer_acc += (t_fine_tr_end - t_fine_tr_start)

                # fine matching
                t_fine_match_start = time.perf_counter()
                self.loftr.run_fine_matching(data_bbox)
                t_fine_match_end = time.perf_counter()
                self.time_fine_matching_acc += (t_fine_match_end - t_fine_match_start)

                mkpts0_bbox = data_bbox["mkpts0_f"].cpu().numpy()
                mkpts1_bbox = data_bbox["mkpts1_f"].cpu().numpy()
                if "mconf" in data_bbox:
                    mconf_bbox = data_bbox["mconf"].cpu().numpy()

        bbox_desired_true = (x0_0, y0_0, x1_0, y1_0)
        bbox_current_true = (x0_1, y0_1, x1_1, y1_1)

        if self.last_use_full_bbox:
            draw_bbox_desired = (0, 0, w1 - 1, h1 - 1)
            draw_bbox_current = (0, 0, w1 - 1, h1 - 1)
        else:
            draw_bbox_desired = bbox_desired_true
            draw_bbox_current = bbox_current_true

        # no matches
        if mkpts0_bbox.shape[0] == 0:
            self.last_n_matches = 0

            if not self.last_use_full_bbox:
                vis_gray = img1_gray
                if vis_gray.dtype != np.uint8:
                    vis_gray = np.clip(vis_gray, 0.0, 1.0)
                    vis_gray = (vis_gray * 255.0).astype(np.uint8)

                self.vis_img1_gray = vis_gray
                self.vis_mkpts0_global = np.zeros((0, 2), dtype=np.float32)
                self.vis_mkpts1_global = np.zeros((0, 2), dtype=np.float32)
                self.vis_draw_bbox_desired = draw_bbox_desired
                self.vis_draw_bbox_current = draw_bbox_current
                self.vis_title = "No matches"
                self.vis_has_data = True

                self.publish_matched_points(
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.float32),
                    None,
                )

            t_total_end = time.perf_counter()
            dt = (t_total_end - t_total_start)
            self.time_total_acc += dt
            self.last_proc_ms = dt * 1000.0

            # NEW: publish processing time even in no-match case
            m = Float32()
            m.data = float(self.last_proc_ms)
            self.pub_proc_time.publish(m)

            return

        # matches
        mkpts0_global = mkpts0_bbox
        mkpts1_global = mkpts1_bbox
        mconf_global = mconf_bbox

        mkpts0_global[:, 0] += x0_0
        mkpts0_global[:, 1] += y0_0
        mkpts1_global[:, 0] += x0_1
        mkpts1_global[:, 1] += y0_1

        # update bbox on full-bbox frames only
        if self.last_use_full_bbox:
            used0_new = compute_used_cells_from_matches(mkpts0_global, Hc0, Wc0, h0, w0)
            used1_new = compute_used_cells_from_matches(mkpts1_global, Hc1, Wc1, h1, w1)

            bbox0_new = compute_coarse_bbox_indices(used0_new)
            bbox1_new = compute_coarse_bbox_indices(used1_new)

            if bbox0_new is not None and bbox1_new is not None:
                i0n_min, i0n_max, j0n_min, j0n_max = bbox0_new
                i1n_min, i1n_max, j1n_min, j1n_max = bbox1_new

                y0n_0 = int(i0n_min * cell_h0)
                y1n_0 = int((i0n_max + 1) * cell_h0)
                x0n_0 = int(j0n_min * cell_w0)
                x1n_0 = int((j0n_max + 1) * cell_w0)

                roi_h0_new = max(1, y1n_0 - y0n_0)
                roi_w0_new = max(1, x1n_0 - x0n_0)

                y0n_1 = int(i1n_min * cell_h1)
                y1n_1 = int((i1n_max + 1) * cell_h1)
                x0n_1 = int(j1n_min * cell_w1)
                x1n_1 = int((j1n_max + 1) * cell_w1)

                roi_h1_new = max(1, y1n_1 - y0n_1)
                roi_w1_new = max(1, x1n_1 - x0n_1)

                self.adjust_max_size_from_bbox(
                    roi_w0_new, roi_h0_new, w0, h0,
                    roi_w1_new, roi_h1_new, w1, h1,
                )

                self.cached_bbox0_norm = (
                    i0n_min / float(Hc0),
                    i0n_max / float(Hc0),
                    j0n_min / float(Wc0),
                    j0n_max / float(Wc0),
                )
                self.cached_bbox1_norm = (
                    i1n_min / float(Hc1),
                    i1n_max / float(Hc1),
                    j1n_min / float(Wc1),
                    j1n_max / float(Wc1),
                )

                self.cached_bbox0_coarse = bbox0_new
                self.cached_bbox1_coarse = bbox1_new

        n_matches = mkpts0_global.shape[0]
        self.last_n_matches = int(n_matches)
        title = f"BBOX (full tokens), matches={n_matches}"

        # only output on ROI frames (kept as in your code)
        if not self.last_use_full_bbox:
            self.vis_img1_gray = img1_gray
            self.vis_mkpts0_global = mkpts0_global
            self.vis_mkpts1_global = mkpts1_global
            self.vis_draw_bbox_desired = draw_bbox_desired
            self.vis_draw_bbox_current = draw_bbox_current
            self.vis_title = title
            self.vis_has_data = True

            self.publish_matched_points(mkpts0_global, mkpts1_global, mconf_global)

        t_total_end = time.perf_counter()
        dt = (t_total_end - t_total_start)
        self.time_total_acc += dt
        self.last_proc_ms = dt * 1000.0

        # NEW: publish processing time (ms), same as other code
        m = Float32()
        m.data = float(self.last_proc_ms)
        self.pub_proc_time.publish(m)

    # ---------------- Visualization loop ----------------
    def vis_timer_cb(self):
        if not self.enable_visualization:
            return
        if not self.vis_has_data or self.vis_img1_gray is None:
            return

        t_vis_start = time.perf_counter()

        t_draw_start = time.perf_counter()
        vis = draw_points_on_image1_return(
            self.vis_img1_gray,
            self.vis_mkpts0_global,
            self.vis_mkpts1_global,
            self.vis_title,
            bbox_desired=self.vis_draw_bbox_desired,
            bbox_current=self.vis_draw_bbox_current,
        )
        t_draw_end = time.perf_counter()
        self.time_draw_acc += (t_draw_end - t_draw_start)

        cv2.putText(
            vis,
            f"Proc frame time: {self.last_proc_ms:.2f} ms",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        h_vis, w_vis = vis.shape[:2]
        cv2.putText(
            vis,
            f"Res {w_vis}x{h_vis}, frame_max={self.curr_frame_max_size}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        t_pub_start = time.perf_counter()
        self.publish_image(vis)
        t_pub_end = time.perf_counter()
        self.time_publish_acc += (t_pub_end - t_pub_start)

        t_vis_end = time.perf_counter()
        self.last_vis_ms = (t_vis_end - t_vis_start) * 1000.0

        self.vis_fps_count += 1

    # ---------------- FPS timer ----------------
    def fps_timer_cb(self):
        self.proc_fps_value = float(self.proc_fps_count)
        self.cam_fps_value = float(self.cam_fps_count)
        self.vis_fps_value = float(self.vis_fps_count)

        self.proc_fps_count = 0
        self.cam_fps_count = 0
        self.vis_fps_count = 0

        self.time_backbone_acc = 0.0
        self.time_bbox_acc = 0.0
        self.time_coarse_crop_acc = 0.0
        self.time_coarse_posenc_acc = 0.0
        self.time_coarse_transformer_acc = 0.0
        self.time_coarse_matching_acc = 0.0
        self.time_fine_preprocess_acc = 0.0
        self.time_fine_transformer_acc = 0.0
        self.time_fine_matching_acc = 0.0
        self.time_img_convert_acc = 0.0
        self.time_draw_acc = 0.0
        self.time_publish_acc = 0.0
        self.time_total_acc = 0.0

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
    torch.set_grad_enabled(False)

    rclpy.init(args=args)
    node = LoFTRCase4Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
