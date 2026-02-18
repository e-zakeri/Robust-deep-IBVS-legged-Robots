#!/usr/bin/env python3
"""
Hold joystick button 11 or 12 (indices 10 or 11) to run visual-servo control.
Release both → commands go to zero.
Logs each 6-component error vector with a timestamp to CSV.
"""

import os
import time
import atexit

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Vector3
import numpy as np


# ========================= CONFIGURATION =========================

# --- Node name ---
NODE_NAME = "R1_go2_IBVS_Test_2FQSMC_T1_node"

# --- Topics ---
JOY_TOPIC       = "/joy"
ERROR_TOPIC     = "/AN_features"   # subscribed feature/error topic
MOVE_TOPIC      = "/cmd/move"
EULER_TOPIC     = "/cmd/euler"
HEIGHT_TOPIC    = "/cmd/body_height"

# --- Joystick / control ---
# Enable VS when EITHER of these buttons is held:
BUTTON_INDEX_1      = 10      # button 11 (0-based index 10)
BUTTON_INDEX_2      = 11      # button 12 (0-based index 11)
CONTROL_RATE_HZ     = 10.0    # control loop rate [Hz]

# --- Error filtering (EMA) ---
USE_ERROR_FILTER    = True    # <--- set False to disable filtering
FILTER_ALPHA        = 0.6     # EMA filter coefficient when enabled

# --- Gains ---
KP_MOVE             = 0.05    # proportional gain for Cartesian move
KP_EULER            = 0.05    # proportional gain for Euler control

# --- Euler angle limits (per axis, asymmetric) ---
MAX_EULER_X_POS =  0.3
MAX_EULER_X_NEG = -0.3
MAX_EULER_Y_POS =  0.3
MAX_EULER_Y_NEG = -0.3
MAX_EULER_Z_POS =  0.3
MAX_EULER_Z_NEG = -0.3

# --- Height limits (dog body height) ---
MAX_H_Z_POS =  0.03
MAX_H_Z_NEG = -0.10

# --- Logging configuration ---
LOG_DIR      = "/home/ehsan/ros2_ws/src/go2_pkg/data/R1/Test_1/2FQSMC"
# LOG_DIR   = "/home/ehsan/ros2_ws/src/go2_pkg/data/R1/Test_1/FQSMC"
# LOG_DIR   = "/home/ehsan/ros2_ws/src/go2_pkg/data/R1/Test_1/SMC"
# LOG_DIR   = "/home/ehsan/ros2_ws/src/go2_pkg/data/R1/Test_1/PID"

LOG_FILENAME = "Test_3_2FQSMC_v3.csv"

# ================================================================


class VisualServoNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        # ── gains & control timing ───────────────────────────────
        self.use_filter   = USE_ERROR_FILTER
        self.filter_alpha = FILTER_ALPHA
        self.button_idx_1 = BUTTON_INDEX_1
        self.button_idx_2 = BUTTON_INDEX_2
        self.Kp_move      = KP_MOVE
        self.Kp_euler     = KP_EULER
        self.dt           = 1.0 / CONTROL_RATE_HZ

        # axis-specific bounds for Euler
        self.max_pos = Vector3(
            x=MAX_EULER_X_POS,
            y=MAX_EULER_Y_POS,
            z=MAX_EULER_Z_POS,
        )
        self.max_neg = Vector3(
            x=MAX_EULER_X_NEG,
            y=MAX_EULER_Y_NEG,
            z=MAX_EULER_Z_NEG,
        )
        self.max_H = MAX_H_Z_POS
        self.min_H = MAX_H_Z_NEG

        # ── state ────────────────────────────────────────────────
        self.servo_active   = False
        self.filtered_error = np.zeros(6, dtype=float)  # EMA state
        self.current_error  = np.zeros(6, dtype=float)  # input to control
        self.cmd_eul_angle  = Vector3()                 # accumulated angles
        self.hight_dog      = 0.0

        # ── set up error-log file ───────────────────────────────
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, LOG_FILENAME)
        self.log_file = open(self.log_path, "w")
        # CSV header
        self.log_file.write("timestamp,e0,e1,e2,e3,e4,e5\n")
        self.log_file.flush()
        atexit.register(self.log_file.close)

        # ── publishers / subscribers ────────────────────────────
        self.move_pub   = self.create_publisher(Vector3, MOVE_TOPIC,   10)
        self.euler_pub  = self.create_publisher(Vector3, EULER_TOPIC,  10)
        self.pub_height = self.create_publisher(Float32, HEIGHT_TOPIC, 10)

        self.create_subscription(Joy, JOY_TOPIC, self.joy_cb, 10)
        self.create_subscription(Float32MultiArray, ERROR_TOPIC, self.err_cb, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"Hold button #{self.button_idx_1+1} OR #{self.button_idx_2+1} "
            f"to enable visual servo."
        )
        self.get_logger().info(f"Logging errors to: {self.log_path}")
        self.get_logger().info(
            f"Error filtering: {'ON' if self.use_filter else 'OFF'} "
            f"(alpha={self.filter_alpha:.2f})"
        )

    # ============================================================
    #                        CALLBACKS
    # ============================================================

    def joy_cb(self, msg: Joy):
        """
        VS is active when EITHER button 11 or 12 is pressed (indices 10 or 11).
        It switches OFF only when BOTH are released.
        """
        b = msg.buttons

        def is_pressed(idx: int) -> bool:
            return idx < len(b) and b[idx] == 1

        pressed_1 = is_pressed(self.button_idx_1)
        pressed_2 = is_pressed(self.button_idx_2)

        pressed_any = pressed_1 or pressed_2

        if pressed_any and not self.servo_active:
            self.servo_active = True
            self.get_logger().info("VS ON (button 11 or 12 pressed)")
        elif (not pressed_any) and self.servo_active:
            self.servo_active = False
            self.get_logger().info("VS OFF (both buttons 11 and 12 released)")
            self.publish_zero()

    def err_cb(self, msg: Float32MultiArray):
        """
        Error / feature callback.

        Expected input:
          - Either 6 floats: [e0..e5] directly (error vector)
          - Or 12 floats: [des0..des5, cur0..cur5], in which case
              error = current - desired
        """
        data = np.array(msg.data, dtype=float)

        if data.size == 6:
            raw_err = data
        elif data.size == 12:
            des = data[:6]
            cur = data[6:]
            raw_err = cur - des
        else:
            self.get_logger().warn(
                f"/AN_features length {data.size} != 6 or 12; ignoring"
            )
            return

        if self.use_filter:
            # EMA filter
            self.filtered_error = (
                self.filter_alpha * raw_err
                + (1.0 - self.filter_alpha) * self.filtered_error
            )
            self.current_error = self.filtered_error
        else:
            # no filtering: use raw error directly
            self.current_error = raw_err
            self.filtered_error = raw_err  # keep consistent for logging

        # log timestamp + error vector (whatever is used for control)
        ts = time.time()
        vals = ",".join(f"{x:.6f}" for x in self.current_error)
        self.log_file.write(f"{ts:.6f},{vals}\n")
        self.log_file.flush()

        err_str = ", ".join(f"{v:.4f}" for v in self.current_error)
        self.get_logger().info(f"feature error [{err_str}]")

    # ============================================================
    #                       CONTROL LOOP
    # ============================================================

    def control_loop(self):
        if not self.servo_active:
            return

        e = self.current_error

        # ---------------------- Cartesian move ----------------------
        cmd_move = Vector3(
            x=-self.Kp_move * e[2] * (0.2)*100,
            y=-self.Kp_move * e[0] * (2.0) * (20.0),
            z=-self.Kp_move * e[3] * (60.0),
        )

        # raw angular velocity (currently zeros, same as your code)
        cmd_eul_v = Vector3(
            x=-self.Kp_euler * e[4] * (-0.1) * 5 * 20,
            y=-self.Kp_euler * e[5] * (10.0) * 1 * 3 * (0.1),
            z=-self.Kp_euler * e[3] * (-1.0) * 1 * 5*0,
        )

        # integrate ω → Δθ
        self.cmd_eul_angle.x += cmd_eul_v.x * self.dt
        self.cmd_eul_angle.y += cmd_eul_v.y * self.dt
        self.cmd_eul_angle.z += cmd_eul_v.z * self.dt

        # clamp each axis
        self.cmd_eul_angle.x = max(
            self.max_neg.x, min(self.cmd_eul_angle.x, self.max_pos.x)
        )
        self.cmd_eul_angle.y = max(
            self.max_neg.y, min(self.cmd_eul_angle.y, self.max_pos.y)
        )
        self.cmd_eul_angle.z = max(
            self.max_neg.z, min(self.cmd_eul_angle.z, self.max_pos.z)
        )

        # publish move and orientation
        self.move_pub.publish(cmd_move)
        self.euler_pub.publish(self.cmd_eul_angle)

        # height control
        h_v = -self.Kp_euler * e[1] * (5.0)
        self.hight_dog += h_v * self.dt
        self.hight_dog = max(self.min_H, min(self.hight_dog, self.max_H))
        self.pub_height.publish(Float32(data=self.hight_dog))

    # ============================================================
    #                       HELPER METHODS
    # ============================================================

    def publish_zero(self):
        # reset accumulated angles (stop rotation command)
        self.cmd_eul_angle = Vector3()


# ============================== MAIN ===============================

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
