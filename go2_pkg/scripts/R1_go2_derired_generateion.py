#!/usr/bin/env python3
import os
import csv
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

import torch
from kornia.feature import LoFTR as KorniaLoFTR

# Zero-phase Butterworth filter
from scipy.signal import butter, filtfilt


# ============================================================
# CONFIG
# ============================================================
TRACKING_DIR = "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/R1/tracking/TR_1"
TIMESTAMPS_FILE = "timestamps.txt"

# LoFTR: indoor
LOFTR_PRETRAINED = "indoor"

# Speed/size controls (optional)
USE_GPU_IF_AVAILABLE = True
RESIZE_MAX_SIDE = None   # e.g. 800 to speed up; None keeps original size

# Output
OUTPUT_DIR = os.path.join(TRACKING_DIR, "offline_6feat_errors")
SAVE_NPZ_PER_PAIR = False  # you do NOT want per-step files

# ---- Smoothing (Option 1: Butterworth + filtfilt) ----
BUTTER_ORDER = 3
CUTOFF_HZ = 1.0   # good starting point for ~5 Hz data
# ============================================================


EDOT_COLS = [
    "edot_cx", "edot_cy", "edot_perim", "edot_r23_14", "edot_rot", "edot_r12_34"
]


@dataclass
class FrameRec:
    fname: str
    time_ms: int
    path: str


def read_timestamps(tracking_dir: str, ts_filename: str) -> List[FrameRec]:
    ts_path = os.path.join(tracking_dir, ts_filename)
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"timestamps file not found: {ts_path}")

    frames: List[FrameRec] = []
    with open(ts_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            fname, t_str = parts[0], parts[1]
            try:
                t_ms = int(t_str)
            except ValueError:
                continue

            img_path = os.path.join(tracking_dir, fname)
            if not os.path.exists(img_path):
                continue

            frames.append(FrameRec(fname=fname, time_ms=t_ms, path=img_path))

    frames.sort(key=lambda x: x.time_ms)
    return frames


def load_bgr(path: str, resize_max_side: Optional[int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")

    if resize_max_side is not None:
        h, w = img.shape[:2]
        m = max(h, w)
        if m > resize_max_side:
            scale = float(resize_max_side) / float(m)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def bgr_to_gray_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(gray).float()[None, None, ...] / 255.0  # (1,1,H,W)
    return t.to(device)


class LoFTRMatcher:
    def __init__(self, pretrained: str, device: torch.device):
        self.device = device
        self.model = KorniaLoFTR(pretrained=pretrained).to(device).eval()

    @torch.no_grad()
    def match(self, img0: torch.Tensor, img1: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.model({"image0": img0, "image1": img1})
        mkpts0 = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
        mkpts1 = out["keypoints1"].detach().cpu().numpy().astype(np.float32)
        conf = out["confidence"].detach().cpu().numpy().astype(np.float32)
        return mkpts0, mkpts1, conf


# ============================================================
# 6-feature extraction logic
# ============================================================
def _segment_lengths(centroid_dict: Dict[str, Optional[Tuple[float, float]]]) -> Optional[Tuple[float, float, float, float]]:
    order = ["TL", "TR", "BR", "BL"]
    if any(centroid_dict[k] is None for k in order):
        return None

    TL = centroid_dict["TL"]
    TR = centroid_dict["TR"]
    BR = centroid_dict["BR"]
    BL = centroid_dict["BL"]

    d12 = float(np.hypot(TR[0] - TL[0], TR[1] - TL[1]))
    d23 = float(np.hypot(BR[0] - TR[0], BR[1] - TR[1]))
    d34 = float(np.hypot(BL[0] - BR[0], BL[1] - BR[1]))
    d41 = float(np.hypot(TL[0] - BL[0], TL[1] - BL[1]))
    return d12, d23, d34, d41


def _rotation_from_centroids(centroid_dict: Dict[str, Optional[Tuple[float, float]]]) -> float:
    """
    CONSISTENT rotation definition with your ROS/L_s:
      rotation = circular mean of angles of lines 12 and 43
        12: TL -> TR
        43: BL -> BR  (4=BL, 3=BR)
    """
    TL = centroid_dict.get("TL", None)
    TR = centroid_dict.get("TR", None)
    BR = centroid_dict.get("BR", None)
    BL = centroid_dict.get("BL", None)

    if TL is None or TR is None or BR is None or BL is None:
        return 0.0

    # 12: TL -> TR
    a12 = math.atan2(TR[1] - TL[1], TR[0] - TL[0])
    # 43: BL -> BR
    a43 = math.atan2(BR[1] - BL[1], BR[0] - BL[0])

    mean_sin = 0.5 * (math.sin(a12) + math.sin(a43))
    mean_cos = 0.5 * (math.cos(a12) + math.cos(a43))
    return math.atan2(mean_sin, mean_cos)


def extract_shape_features(x0_in, y0_in, x1_in, y1_in):
    cx0_global = float(x0_in.mean())
    cy0_global = float(y0_in.mean())
    cx1_global = float(x1_in.mean())
    cy1_global = float(y1_in.mean())

    # CONSISTENT quadrant definition with your ROS/L_s partition:
    # A=TL: u<=cx, v<=cy
    # B=TR: u> cx, v<=cy
    # C=BR: u> cx, v> cy
    # D=BL: u<=cx, v> cy
    q_tl = (x0_in <= cx0_global) & (y0_in <= cy0_global)   # TL / A
    q_tr = (x0_in >  cx0_global) & (y0_in <= cy0_global)   # TR / B
    q_br = (x0_in >  cx0_global) & (y0_in >  cy0_global)   # BR / C
    q_bl = (x0_in <= cx0_global) & (y0_in >  cy0_global)   # BL / D

    regions = [("TL", q_tl), ("TR", q_tr), ("BR", q_br), ("BL", q_bl)]
    desired_region_centroids = {name: None for name in ["TL", "TR", "BR", "BL"]}
    current_region_centroids = {name: None for name in ["TL", "TR", "BR", "BL"]}

    for name, mask_reg in regions:
        idx_reg = np.where(mask_reg)[0]
        if idx_reg.size == 0:
            continue
        cx0_reg = float(x0_in[idx_reg].mean())
        cy0_reg = float(y0_in[idx_reg].mean())
        cx1_reg = float(x1_in[idx_reg].mean())
        cy1_reg = float(y1_in[idx_reg].mean())
        desired_region_centroids[name] = (cx0_reg, cy0_reg)
        current_region_centroids[name] = (cx1_reg, cy1_reg)

    eps = 1e-6

    seg0 = _segment_lengths(desired_region_centroids)
    des_perim = des_ratio_23_14 = des_ratio_12_34 = 0.0
    des_rot = 0.0
    if seg0 is not None:
        d12, d23, d34, d41 = seg0
        des_perim = d12 + d23 + d34 + d41
        des_ratio_23_14 = d23 / (d41 + eps)
        des_ratio_12_34 = d12 / (d34 + eps)
        des_rot = _rotation_from_centroids(desired_region_centroids)

    seg1 = _segment_lengths(current_region_centroids)
    cur_perim = cur_ratio_23_14 = cur_ratio_12_34 = 0.0
    cur_rot = 0.0
    if seg1 is not None:
        d12, d23, d34, d41 = seg1
        cur_perim = d12 + d23 + d34 + d41
        cur_ratio_23_14 = d23 / (d41 + eps)
        cur_ratio_12_34 = d12 / (d34 + eps)
        cur_rot = _rotation_from_centroids(current_region_centroids)

    desired_features = np.array(
        [cx0_global, cy0_global, des_perim, des_ratio_23_14, des_rot, des_ratio_12_34],
        dtype=np.float32,
    )
    current_features = np.array(
        [cx1_global, cy1_global, cur_perim, cur_ratio_23_14, cur_rot, cur_ratio_12_34],
        dtype=np.float32,
    )
    return desired_features, current_features


def wrapped_rotation_error(cur_rot: float, des_rot: float) -> float:
    rot_err_raw = float(cur_rot - des_rot)
    return float(math.atan2(math.sin(rot_err_raw), math.cos(rot_err_raw)))


def compute_6feat_error(des: np.ndarray, cur: np.ndarray) -> np.ndarray:
    err = (cur - des).astype(np.float32)
    err[4] = wrapped_rotation_error(cur[4], des[4])
    return err


def safe_div(vec: np.ndarray, dt_s: float) -> np.ndarray:
    if dt_s <= 0.0 or not np.isfinite(dt_s):
        return np.full_like(vec, np.nan, dtype=np.float32)
    return (vec / dt_s).astype(np.float32)


# ============================================================
# Smoothing (Option 1: Butterworth + filtfilt)
# ============================================================
def estimate_fs_from_dt(dt_s_arr: np.ndarray) -> float:
    dt = dt_s_arr[np.isfinite(dt_s_arr) & (dt_s_arr > 0)]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def butter_filtfilt_zero_phase(X: np.ndarray, fs: float, fc: float, order: int) -> np.ndarray:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid fs={fs}")

    fc_eff = min(fc, 0.45 * fs)  # keep safely below Nyquist
    if fc_eff <= 0:
        raise ValueError(f"Invalid cutoff fc_eff={fc_eff}")

    nyq = 0.5 * fs
    wn = fc_eff / nyq
    b, a = butter(order, wn, btype="low")

    padlen_default = 3 * (max(len(a), len(b)) - 1)
    T = X.shape[0]
    padlen = min(padlen_default, max(0, T - 1))

    return filtfilt(b, a, X, axis=0, padlen=padlen)


def write_csv(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = read_timestamps(TRACKING_DIR, TIMESTAMPS_FILE)
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames in timestamps.txt to compute pairwise errors.")

    device = torch.device("cuda" if (USE_GPU_IF_AVAILABLE and torch.cuda.is_available()) else "cpu")
    matcher = LoFTRMatcher(LOFTR_PRETRAINED, device=device)

    out_csv_noisy = os.path.join(OUTPUT_DIR, "features_pairwise_with_derivative.csv")
    out_csv_smooth = os.path.join(OUTPUT_DIR, "features_pairwise_with_derivative_smoothed.csv")

    header = [
        "k",
        "prev_fname", "curr_fname",
        "prev_time_ms", "curr_time_ms", "dt_ms", "dt_s",
        "N_matches",

        "des_cx", "des_cy", "des_perim", "des_r23_14", "des_rot", "des_r12_34",
        "cur_cx", "cur_cy", "cur_perim", "cur_r23_14", "cur_rot", "cur_r12_34",

        "e_cx", "e_cy", "e_perim", "e_r23_14", "e_rot_wrapped", "e_r12_34",

        "edot_cx", "edot_cy", "edot_perim", "edot_r23_14", "edot_rot", "edot_r12_34",
    ]
    smooth_cols = [c + "_smooth" for c in EDOT_COLS]
    header_smooth = header + smooth_cols

    # Store rows in memory
    rows_noisy: List[dict] = []

    for k in range(1, len(frames)):
        prev = frames[k - 1]
        curr = frames[k]
        dt_ms = curr.time_ms - prev.time_ms
        dt_s = float(dt_ms) * 1e-3

        img0_bgr = load_bgr(prev.path, RESIZE_MAX_SIDE)
        img1_bgr = load_bgr(curr.path, RESIZE_MAX_SIDE)

        h0, w0 = img0_bgr.shape[:2]
        h1, w1 = img1_bgr.shape[:2]

        img0 = bgr_to_gray_tensor(img0_bgr, device)
        img1 = bgr_to_gray_tensor(img1_bgr, device)

        t0 = time.time()
        mkpts0_px, mkpts1_px, conf = matcher.match(img0, img1)
        elapsed_ms = (time.time() - t0) * 1000.0

        row = {name: "nan" for name in header}
        row["k"] = str(k)
        row["prev_fname"] = prev.fname
        row["curr_fname"] = curr.fname
        row["prev_time_ms"] = str(prev.time_ms)
        row["curr_time_ms"] = str(curr.time_ms)
        row["dt_ms"] = str(dt_ms)
        row["dt_s"] = str(dt_s)

        N = int(mkpts0_px.shape[0])
        row["N_matches"] = str(N)

        if N == 0:
            rows_noisy.append(row)
            print(f"[{k}/{len(frames)-1}] {prev.fname}->{curr.fname} | N=0 | took={elapsed_ms:.1f} ms")
            continue

        # Normalize matches to [0,1]
        x0_n = (mkpts0_px[:, 0] / max(w0, 1)).astype(np.float32)
        y0_n = (mkpts0_px[:, 1] / max(h0, 1)).astype(np.float32)
        x1_n = (mkpts1_px[:, 0] / max(w1, 1)).astype(np.float32)
        y1_n = (mkpts1_px[:, 1] / max(h1, 1)).astype(np.float32)

        des6, cur6 = extract_shape_features(x0_n, y0_n, x1_n, y1_n)
        e6 = compute_6feat_error(des6, cur6)
        edot6 = safe_div(e6, dt_s)

        # desired/current
        row["des_cx"] = f"{float(des6[0])}"
        row["des_cy"] = f"{float(des6[1])}"
        row["des_perim"] = f"{float(des6[2])}"
        row["des_r23_14"] = f"{float(des6[3])}"
        row["des_rot"] = f"{float(des6[4])}"
        row["des_r12_34"] = f"{float(des6[5])}"

        row["cur_cx"] = f"{float(cur6[0])}"
        row["cur_cy"] = f"{float(cur6[1])}"
        row["cur_perim"] = f"{float(cur6[2])}"
        row["cur_r23_14"] = f"{float(cur6[3])}"
        row["cur_rot"] = f"{float(cur6[4])}"
        row["cur_r12_34"] = f"{float(cur6[5])}"

        # error
        row["e_cx"] = f"{float(e6[0])}"
        row["e_cy"] = f"{float(e6[1])}"
        row["e_perim"] = f"{float(e6[2])}"
        row["e_r23_14"] = f"{float(e6[3])}"
        row["e_rot_wrapped"] = f"{float(e6[4])}"
        row["e_r12_34"] = f"{float(e6[5])}"

        # derivative
        row["edot_cx"] = "nan" if not np.isfinite(edot6[0]) else f"{float(edot6[0])}"
        row["edot_cy"] = "nan" if not np.isfinite(edot6[1]) else f"{float(edot6[1])}"
        row["edot_perim"] = "nan" if not np.isfinite(edot6[2]) else f"{float(edot6[2])}"
        row["edot_r23_14"] = "nan" if not np.isfinite(edot6[3]) else f"{float(edot6[3])}"
        row["edot_rot"] = "nan" if not np.isfinite(edot6[4]) else f"{float(edot6[4])}"
        row["edot_r12_34"] = "nan" if not np.isfinite(edot6[5]) else f"{float(edot6[5])}"

        rows_noisy.append(row)

        print(
            f"[{k}/{len(frames)-1}] {prev.fname}->{curr.fname} | dt={dt_ms} ms | N={N} | "
            f"edot=[{edot6[0]:+.4f},{edot6[1]:+.4f},{edot6[2]:+.4f},{edot6[3]:+.4f},{edot6[4]:+.4f},{edot6[5]:+.4f}] "
            f"| took={elapsed_ms:.1f} ms"
        )

    # ----------------- Write NOISY file (first output) -----------------
    write_csv(out_csv_noisy, header, rows_noisy)
    print(f"\nSaved noisy CSV:\n  {out_csv_noisy}")

    # ----------------- Smooth and write SMOOTHED file -----------------
    # Build arrays for smoothing
    dt_s_arr = np.array([float(r["dt_s"]) if r["dt_s"] != "nan" else np.nan for r in rows_noisy], dtype=np.float64)
    edot_arr = np.zeros((len(rows_noisy), len(EDOT_COLS)), dtype=np.float64)
    for j, c in enumerate(EDOT_COLS):
        edot_arr[:, j] = np.array([float(r[c]) if r[c] != "nan" else np.nan for r in rows_noisy], dtype=np.float64)

    valid = np.isfinite(dt_s_arr) & (dt_s_arr > 0) & np.isfinite(edot_arr).all(axis=1)

    # Prepare smoothed rows as a copy of noisy rows
    rows_smooth: List[dict] = [dict(r) for r in rows_noisy]
    for r in rows_smooth:
        for sc in smooth_cols:
            r[sc] = "nan"

    if valid.sum() >= 10:
        fs = estimate_fs_from_dt(dt_s_arr[valid])
        print(f"[SMOOTH] Estimated fs ≈ {fs:.3f} Hz. cutoff={CUTOFF_HZ:.3f} Hz, order={BUTTER_ORDER}.")
        edot_smooth_valid = butter_filtfilt_zero_phase(edot_arr[valid, :], fs=fs, fc=CUTOFF_HZ, order=BUTTER_ORDER)

        edot_smooth = np.full_like(edot_arr, np.nan, dtype=np.float64)
        edot_smooth[valid, :] = edot_smooth_valid

        for i in range(len(rows_smooth)):
            for j, c in enumerate(EDOT_COLS):
                key = c + "_smooth"
                val = edot_smooth[i, j]
                rows_smooth[i][key] = "nan" if not np.isfinite(val) else f"{float(val)}"
    else:
        print(f"[SMOOTH][WARN] Too few valid rows to smooth (valid={int(valid.sum())}). Writing NaNs for *_smooth.")

    write_csv(out_csv_smooth, header_smooth, rows_smooth)
    print(f"Saved smoothed CSV:\n  {out_csv_smooth}")

    print(f"\nDone.\nOutput dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
