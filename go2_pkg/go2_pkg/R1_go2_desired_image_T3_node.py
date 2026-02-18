#!/usr/bin/env python3

import os
import time
import csv

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class DesiredImageNode(Node):
    def __init__(self):
        super().__init__("R1_go2_desired_image_T3_node")

        self.bridge = CvBridge()

        # ====================== CONFIGURATION PARAMETERS ======================
        # MODE SWITCH:
        #   True  -> publish single saved image on /camera_test_d (STATIC mode)
        #           AND publish /dS_d = 0 (6x1) ALWAYS
        #   False -> TRACKING mode:
        #              - on startup ONLY: publish FIRST tracking image periodically
        #                (if it exists)
        #              - once button 12 is pressed (first time): stop that forever
        #              - while button 12 is held: publish recorded sequence with timing
        #              - ALSO publish /dS_d using smoothed CSV keyed by curr_fname
        self.use_single_desired = True   # <- set True for static single-image mode

        # BUTTON INDICES (0-based)
        self.single_save_button = 1      # index for snapshot (button 2)
        self.track_button = 7            # index for continuous recording (button 8)
        self.play_button = 11            # index for playback (button 12)

        # TRACKING SETTINGS
        self.track_fps = 5               # Hz (images/second) while tracking button is held
        self.output_prefix = "img_"      # prefix for tracking filenames

        # PATHS
        self.single_save_path = (
            "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/"
            #"R1/regulate/saved_image2.png"
            "R1/regulate/saved_image3_def.png"
        )
        

        self.tracking_dir = (
            "/home/ehsan/ros2_ws/src/go2_pkg/data/desired_images/"
            "R1/tracking/TR_1_def"
        )
        self.tracking_log_filename = "timestamps.txt"

        # --------------------- Smoothed desired dS_d (from CSV) ---------------
        # CSV contains: prev_fname, curr_fname, ..., edot_*_smooth.
        # Use curr_fname as key and publish the 6 edot_*_smooth values on /dS_d.
        self.sdotd_dir = os.path.join(self.tracking_dir, "offline_6feat_errors")
        self.sdotd_csv = os.path.join(self.sdotd_dir, "features_pairwise_with_derivative_smoothed.csv")

        self.csv_curr_fname_col = "curr_fname"

        self.sdotd_cols = [
            "edot_cx_smooth",
            "edot_cy_smooth",
            "edot_perim_smooth",
            "edot_r23_14_smooth",
            "edot_rot_smooth",
            "edot_r12_34_smooth",
        ]

        # ROS TOPICS
        self.desired_publish_topic = "/camera_test_d"
        self.camera_topic = "/camera/image_raw"
        self.joy_topic = "/joy"
        self.sdotd_topic = "/dS_d"

        # TIMER FREQUENCIES
        self.desired_publish_hz = 1.0      # Hz for static / initial idle publishing
        self.playback_timer_hz = 500.0     # Hz for playback timing loop

        # PLAYBACK SPEED SCALE (1.0 = normal, 2.0 = 2x faster, 0.5 = half speed)
        self.playback_speed = 1.0*1.0
        # ======================================================================

        # ========================= INITIALIZATION ============================

        # ---- Load desired image once (for STATIC mode only) ----
        if not os.path.exists(self.single_save_path):
            raise RuntimeError(f"Static image not found: {self.single_save_path}")

        img = cv2.imread(self.single_save_path)
        if img is None:
            raise RuntimeError(f"Failed to load desired image: {self.single_save_path}")

        self.desired_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")

        # ---- Publishers ----
        self.pub_desired = self.create_publisher(Image, self.desired_publish_topic, 1)
        self.pub_sdotd = self.create_publisher(Float32MultiArray, self.sdotd_topic, 10)

        # ---- Load smoothed dS_d lookup table keyed by curr_fname (tracking mode) ----
        self.sdotd_loaded = False
        self.sdotd_by_curr_fname = {}  # basename(curr_fname) -> np.ndarray shape (6,)
        self.load_sdotd_table()

        # ---- Timer to publish desired image / initial idle image ----
        desired_period = 1.0 / self.desired_publish_hz if self.desired_publish_hz > 0 else 1.0
        self.timer = self.create_timer(desired_period, self.publish_desired_image)

        # ---- Live camera image storage (for saving & tracking) ----
        self.latest_image = None
        self.have_image = False

        # ---- Joystick state ----
        self.prev_buttons = None

        # ---- Tracking directory & log ----
        os.makedirs(self.tracking_dir, exist_ok=True)
        self.tracking_log_path = os.path.join(self.tracking_dir, self.tracking_log_filename)

        # ---- Tracking timing & indexes ----
        self.tracking_index = 1
        self.tracking_active = False
        self.tracking_base_offset_ms = 0
        self.tracking_session_start_time = None
        self.tracking_last_timestamp_ms = None

        # ---- Storage for ALL tracking frames (loaded + newly recorded) ----
        # Each element: {"filename": str, "time_ms": int, "image": np.ndarray}
        self.loaded_tracking_frames = []
        self.load_existing_tracking_data()

        # ---- Timer for continuous recording (e.g. 5 FPS = 0.2 s) ----
        track_period = 1.0 / self.track_fps if self.track_fps > 0 else 0.2
        self.tracking_timer = self.create_timer(track_period, self.tracking_timer_cb)

        # ---- Playback state (button 12) ----
        self.playback_active = False
        self.playback_current_index = 0
        self.playback_start_walltime = None
        self.playback_base_time_ms = 0

        # Timer for playback — configurable frequency
        playback_period = 1.0 / self.playback_timer_hz if self.playback_timer_hz > 0 else 0.01
        self.playback_timer = self.create_timer(playback_period, self.playback_timer_cb)

        # ---- INITIAL IDLE PUBLISH FLAG (tracking mode only) ----
        self.initial_idle_enabled = True

        # ---- Subscribers ----
        self.sub_image = self.create_subscription(Image, self.camera_topic, self.image_cb, 10)
        self.sub_joy = self.create_subscription(Joy, self.joy_topic, self.joy_cb, 10)

        mode_str = (
            "STATIC single-image mode (/dS_d = 0 always)"
            if self.use_single_desired
            else "TRACKING mode (idle first frame until first playback; /dS_d from CSV keyed by curr_fname)"
        )
        self.get_logger().info(
            f"R1_go2_desired_image_node started in {mode_str}. "
            f"Publishing desired image on {self.desired_publish_topic} and /dS_d on {self.sdotd_topic}. "
            f"Tracking dir: {self.tracking_dir}"
        )

        # Throttle warnings for missing curr_fname lookups
        self._last_missing_warn_time = 0.0

    # ====================== LOAD smoothed dS_d table =========================
    def load_sdotd_table(self):
        if not os.path.exists(self.sdotd_csv):
            self.get_logger().warn(
                f"Smoothed CSV not found: {self.sdotd_csv}. "
                "In tracking mode, /dS_d will not be published from CSV (missing keys)."
            )
            self.sdotd_loaded = False
            return

        try:
            with open(self.sdotd_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader]
        except Exception as e:
            self.get_logger().error(f"Failed to read smoothed CSV {self.sdotd_csv}: {e}")
            self.sdotd_loaded = False
            return

        if not rows:
            self.get_logger().error(f"Smoothed CSV is empty: {self.sdotd_csv}")
            self.sdotd_loaded = False
            return

        headers = list(rows[0].keys())

        if self.csv_curr_fname_col not in headers:
            self.get_logger().error(
                f"CSV column '{self.csv_curr_fname_col}' not found in smoothed CSV. "
                f"Available columns: {headers}"
            )
            self.sdotd_loaded = False
            return

        missing_cols = [c for c in self.sdotd_cols if c not in headers]
        if missing_cols:
            self.get_logger().error(
                f"Missing required smoothed columns in CSV: {missing_cols}. "
                f"Available columns: {headers}"
            )
            self.sdotd_loaded = False
            return

        table = {}
        bad_rows = 0

        for r in rows:
            name = r.get(self.csv_curr_fname_col, "")
            if not name:
                bad_rows += 1
                continue

            key = os.path.basename(name.strip())  # e.g., img_2.png

            try:
                vec = np.array([float(r[c]) for c in self.sdotd_cols], dtype=np.float32)
            except Exception:
                bad_rows += 1
                continue

            table[key] = vec

        self.sdotd_by_curr_fname = table
        self.sdotd_loaded = True

        self.get_logger().info(
            f"Loaded smoothed /dS_d lookup from {self.sdotd_csv}: "
            f"{len(table)} keys. Bad rows: {bad_rows}. Key column: {self.csv_curr_fname_col}."
        )

    def publish_zero_dS_d(self):
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_sdotd.publish(msg)

    def publish_sdotd_for_curr_fname(self, fname: str):
        """
        Publish smoothed desired dS_d vector corresponding to the desired image filename.
        Keyed by basename(curr_fname) in the smoothed CSV.
        """
        if not self.sdotd_loaded:
            return
        if not fname:
            return

        key = os.path.basename(fname)
        vec = self.sdotd_by_curr_fname.get(key, None)

        if vec is None:
            now = time.time()
            if (now - self._last_missing_warn_time) > 1.0:
                self.get_logger().warn(f"No /dS_d entry in smoothed CSV for curr_fname='{key}'.")
                self._last_missing_warn_time = now
            return

        msg = Float32MultiArray()
        msg.data = (self.playback_speed * vec).tolist()
        self.pub_sdotd.publish(msg)

    # ====================== PUBLISH DESIRED / IDLE IMAGE ====================
    def publish_desired_image(self):
        """
        - STATIC mode: publish saved_image2.png AND publish /dS_d = zeros always.
        - TRACKING mode:
            * initial idle only: publish first tracking frame and matching /dS_d by filename.
            * after idle disabled: this timer publishes nothing (playback timer publishes).
        """
        # STATIC single-image mode
        if self.use_single_desired:
            img = cv2.imread(self.single_save_path)
            self.desired_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.pub_desired.publish(self.desired_msg)

            # REQUIRED: dS_d must be zero vector in single phase
            self.publish_zero_dS_d()
            return

        # TRACKING mode: if we've already disabled idle publishing, do nothing
        if not self.initial_idle_enabled:
            return

        # No idle publishing if we have no tracking frames yet
        if not self.loaded_tracking_frames:
            return

        # Publish the first tracking frame
        frame0 = self.loaded_tracking_frames[0]
        out_msg = self.bridge.cv2_to_imgmsg(frame0["image"], encoding="bgr8")
        self.pub_desired.publish(out_msg)

        # Publish /dS_d by curr_fname mapping (curr_fname corresponds to desired frame filename)
        self.publish_sdotd_for_curr_fname(frame0["filename"])

    # ====================== LOAD EXISTING TRACKING DATA ======================
    def load_existing_tracking_data(self):
        if not os.path.exists(self.tracking_log_path):
            self.get_logger().info(
                f"No existing tracking log found at {self.tracking_log_path}. Starting fresh."
            )
            return

        try:
            with open(self.tracking_log_path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception as e:
            self.get_logger().error(f"Failed to read tracking log {self.tracking_log_path}: {e}")
            return

        max_index = 0
        last_time_ms = 0
        loaded_count = 0

        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                self.get_logger().warn(f"Bad line in tracking log: '{line}'")
                continue

            fname, t_str = parts[0], parts[1]
            try:
                t_ms = int(t_str)
            except ValueError:
                self.get_logger().warn(f"Bad time value in tracking log: '{line}'")
                continue

            img_path = os.path.join(self.tracking_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                self.get_logger().warn(f"Could not read tracking image {img_path}; skipping.")
                continue

            self.loaded_tracking_frames.append({"filename": fname, "time_ms": t_ms, "image": img})
            loaded_count += 1

            if t_ms > last_time_ms:
                last_time_ms = t_ms

            base = os.path.splitext(fname)[0]
            if base.startswith(self.output_prefix):
                idx_str = base[len(self.output_prefix):]
                try:
                    idx = int(idx_str)
                    if idx > max_index:
                        max_index = idx
                except ValueError:
                    pass

        self.loaded_tracking_frames.sort(key=lambda f: f["time_ms"])

        if loaded_count > 0:
            self.tracking_index = max_index + 1 if max_index > 0 else 1
            self.tracking_base_offset_ms = last_time_ms
            self.tracking_last_timestamp_ms = last_time_ms
            self.get_logger().info(
                f"Loaded {loaded_count} tracking frames from disk. "
                f"Last time = {last_time_ms} ms, next index = {self.tracking_index}."
            )
        else:
            self.get_logger().info("Tracking log exists but no valid frames were loaded.")

    # ====================== CAMERA CALLBACK =================================
    def image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.latest_image = cv_img
        self.have_image = True

    # ====================== JOYSTICK CALLBACK ================================
    def joy_cb(self, msg: Joy):
        buttons = msg.buttons

        if self.prev_buttons is None:
            self.prev_buttons = list(buttons)
            return

        max_needed_index = max(self.single_save_button, self.track_button, self.play_button)
        if len(buttons) <= max_needed_index:
            self.get_logger().warn(
                f"Joystick has only {len(buttons)} buttons, but indices up to {max_needed_index} are expected."
            )
            self.prev_buttons = list(buttons)
            return

        # --- SINGLE SAVE ---
        prev_single = self.prev_buttons[self.single_save_button]
        curr_single = buttons[self.single_save_button]
        if prev_single == 0 and curr_single == 1:
            self.save_single_image()

        # --- TRACKING RECORD ---
        prev_track = self.prev_buttons[self.track_button]
        curr_track = buttons[self.track_button]

        if prev_track == 0 and curr_track == 1:
            self.tracking_active = True
            self.tracking_session_start_time = None
            self.get_logger().info(f"Tracking started (button {self.track_button}).")

        if prev_track == 1 and curr_track == 0:
            self.tracking_active = False
            if self.tracking_last_timestamp_ms is not None:
                self.tracking_base_offset_ms = self.tracking_last_timestamp_ms
            self.tracking_session_start_time = None
            self.get_logger().info(
                f"Tracking stopped (button {self.track_button}). Base offset now = {self.tracking_base_offset_ms} ms."
            )

        # --- PLAYBACK ---
        prev_play = self.prev_buttons[self.play_button]
        curr_play = buttons[self.play_button]

        if prev_play == 0 and curr_play == 1:
            self.start_playback()

        if prev_play == 1 and curr_play == 0:
            self.stop_playback()

        self.prev_buttons = list(buttons)

    # ======================== SAVE ONE IMAGE ================================
    def save_single_image(self):
        if not self.have_image:
            self.get_logger().warn("Snapshot requested but no image available.")
            return

        ok = cv2.imwrite(self.single_save_path, self.latest_image)
        if ok:
            self.get_logger().info(f"Snapshot saved to {self.single_save_path}")
        else:
            self.get_logger().error("Failed to save snapshot")

    # ====================== TRACKING TIMER (recording) ======================
    def tracking_timer_cb(self):
        if not self.tracking_active:
            return
        if not self.have_image:
            return

        now = time.time()

        if self.tracking_session_start_time is None:
            self.tracking_session_start_time = now

        elapsed_ms_session = int(round((now - self.tracking_session_start_time) * 1000.0))
        elapsed_ms = self.tracking_base_offset_ms + elapsed_ms_session

        filename = f"{self.output_prefix}{self.tracking_index}.png"
        full_path = os.path.join(self.tracking_dir, filename)

        ok = cv2.imwrite(full_path, self.latest_image)
        if not ok:
            self.get_logger().error(f"Could not save tracking image {full_path}")
            return

        try:
            with open(self.tracking_log_path, "a") as f:
                f.write(f"{filename} {elapsed_ms}\n")
        except Exception as e:
            self.get_logger().error(f"Failed to write to tracking log {self.tracking_log_path}: {e}")

        self.loaded_tracking_frames.append(
            {"filename": filename, "time_ms": elapsed_ms, "image": self.latest_image.copy()}
        )

        self.tracking_last_timestamp_ms = elapsed_ms
        self.tracking_index += 1

    # ====================== PLAYBACK CONTROL ================================
    def start_playback(self):
        if not self.loaded_tracking_frames:
            self.get_logger().warn("Playback requested, but no tracking frames loaded.")
            return

        if self.initial_idle_enabled:
            self.get_logger().info("First playback start detected -> disabling initial idle publishing.")
        self.initial_idle_enabled = False

        self.playback_active = True
        self.playback_start_walltime = None
        self.get_logger().info(
            f"Playback started (button {self.play_button}). Starting from index {self.playback_current_index}."
        )

    def stop_playback(self):
        if not self.playback_active:
            return

        if 0 < self.playback_current_index <= len(self.loaded_tracking_frames):
            last_frame_time = self.loaded_tracking_frames[self.playback_current_index - 1]["time_ms"]
            self.playback_base_time_ms = last_frame_time

        self.playback_active = False
        self.playback_start_walltime = None

        # IMPORTANT: zero desired derivative when playback stops (manual stop)
        self.publish_zero_dS_d()

        self.get_logger().info(
            f"Playback stopped at index {self.playback_current_index}, base time = {self.playback_base_time_ms} ms."
        )

    # ====================== PLAYBACK TIMER (drop-frame) =====================
    def playback_timer_cb(self):
        """
        In TRACKING mode (use_single_desired == False) and playback active:
          - Compute target time
          - Publish ONLY the last due desired frame (drop-frame)
          - Publish matching /dS_d by using the SAME chosen desired frame filename (curr_fname key)
          - When playback reaches the end: publish /dS_d = 0
        """
        if self.use_single_desired:
            return
        if not self.playback_active:
            return
        if not self.loaded_tracking_frames:
            return

        now = time.time()

        if self.playback_start_walltime is None:
            self.playback_start_walltime = now

        # --------- MODIFIED: scale elapsed time by playback_speed ----------
        elapsed_ms_real = (now - self.playback_start_walltime) * 1000.0
        elapsed_ms = int(round(self.playback_speed * elapsed_ms_real))
        # ------------------------------------------------------------------

        target_time_ms = self.playback_base_time_ms + elapsed_ms

        # No more frames available -> playback ended -> zero /dS_d
        if self.playback_current_index >= len(self.loaded_tracking_frames):
            self.publish_zero_dS_d()
            return

        last_publishable_index = self.playback_current_index - 1

        i = self.playback_current_index
        n = len(self.loaded_tracking_frames)
        while i < n and self.loaded_tracking_frames[i]["time_ms"] <= target_time_ms:
            last_publishable_index = i
            i += 1

        if last_publishable_index < self.playback_current_index:
            return

        frame = self.loaded_tracking_frames[last_publishable_index]

        out_msg = self.bridge.cv2_to_imgmsg(frame["image"], encoding="bgr8")
        self.pub_desired.publish(out_msg)

        # Publish /dS_d corresponding to this desired frame (curr_fname == frame filename)
        self.publish_sdotd_for_curr_fname(frame["filename"])

        self.playback_current_index = last_publishable_index + 1


def main():
    rclpy.init()
    node = DesiredImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
