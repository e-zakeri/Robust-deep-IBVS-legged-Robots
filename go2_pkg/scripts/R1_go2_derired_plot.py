#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
OUTPUT_DIR = "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/R1/tracking/TR_1/offline_6feat_errors"
NOISY_CSV = os.path.join(OUTPUT_DIR, "features_pairwise_with_derivative.csv")
SMOOTH_CSV = os.path.join(OUTPUT_DIR, "features_pairwise_with_derivative_smoothed.csv")

# Layout: set True for 2x3, False for 3x2
LAYOUT_2x3 = True

EDOT_COLS = [
    "edot_cx", "edot_cy", "edot_perim", "edot_r23_14", "edot_rot", "edot_r12_34"
]
EDOT_SMOOTH_COLS = [c + "_smooth" for c in EDOT_COLS]

X_COL = "curr_time_ms"   # or "k"
# =========================


def read_csv_to_dicts(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def col_as_float(rows, key):
    out = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        v = r.get(key, "nan")
        try:
            out[i] = float(v)
        except Exception:
            out[i] = np.nan
    return out


def main():
    noisy_rows = read_csv_to_dicts(NOISY_CSV)
    smooth_rows = read_csv_to_dicts(SMOOTH_CSV)

    # Assume same ordering/length; if not, we still plot by index.
    n = min(len(noisy_rows), len(smooth_rows))
    noisy_rows = noisy_rows[:n]
    smooth_rows = smooth_rows[:n]

    # X axis
    x = col_as_float(noisy_rows, X_COL)
    if X_COL.endswith("_ms"):
        x = x * 1e-3  # to seconds
        x_label = f"{X_COL} (s)"
    else:
        x_label = X_COL

    if LAYOUT_2x3:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, (c_noisy, c_smooth) in enumerate(zip(EDOT_COLS, EDOT_SMOOTH_COLS)):
        ax = axes[i]
        y_noisy = col_as_float(noisy_rows, c_noisy)
        y_smooth = col_as_float(smooth_rows, c_smooth)

        # Plot noisy and smoothed
        ax.plot(x, y_noisy, linewidth=1.0, label="noisy")
        ax.plot(x, y_smooth, linewidth=2.0, label="smoothed")

        ax.set_title(c_noisy)
        ax.grid(True)
        ax.set_ylabel("value")

    # Hide any unused subplots (shouldn't happen with 6 signals, but safe)
    for j in range(len(EDOT_COLS), len(axes)):
        axes[j].axis("off")

    # Labels + legend (single legend for all)
    for ax in axes[-ncols:]:
        ax.set_xlabel(x_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle("6-D edot signals (noisy vs smoothed)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()


if __name__ == "__main__":
    main()
