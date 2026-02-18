#!/usr/bin/env python3
"""
R1_go2_controllers_node.py

Hold joystick button 11 or 12 (indices 10 or 11) to run visual-servo control.
Release both -> commands go to zero (and accumulated angles reset).

Topics (from your ros2 topic list):
  Subscribes:
    /joy
    /AN_features     (6: e_s  OR  12: [s_d(6), s(6)])
    /L_s             (36: 6x6 flattened row-major)
    /L_s_d           (36: 6x6 flattened row-major)
    /dS_d            (6: desired feature derivative s_dot_d)
  Publishes:
    /cmd/move        (geometry_msgs/Vector3)
    /cmd/euler       (geometry_msgs/Vector3)  accumulated angles
    /cmd/body_height (std_msgs/Float32)

Definitions:
  e_s = s - s_d
  sigma = e_s
  e_s_dot = s_dot - s_dot_d

Controller selection:
  CONTROLLER_MODE = 1 -> PI:     u = -K * e_s
  CONTROLLER_MODE = 2 -> SMC:    u = L_s^{-1} ( -Λ1^{-1} sign(σ) + s_dot_d )
  CONTROLLER_MODE = 3 -> FQSMC:  σC_dot = -Λ2 σC + Λ1 σ
                                u = L_s^{-1} ( -Λ1^{-1} sat(σC/φ) - Λ3 σC + s_dot_d )
  CONTROLLER_MODE = 4 -> FFSQMC: σC_dot = -Λ2 σC + Λ1 σ
                                u = L_s^{-1} ( -Λ1^{-1} f_fuzzy(σC) + s_dot_d )
  CONTROLLER_MODE = 5 -> FFSQSMC2: same as mode 4, but uses a DIFFERENT fuzzy function:
                                u = L_s^{-1} ( -Λ1^{-1} f_fuzzy2(σC) + s_dot_d )

Notes:
  - s_dot_d is now provided externally via /dS_d (Float32MultiArray, length 6).
  - If /dS_d is not received yet, s_dot_d remains zero (safe fallback).
  - No filtering is applied.
"""

import os
import time
import atexit
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Vector3


# ========================= CONFIGURATION =========================

NODE_NAME = "R1_go2_controllers_node"

# Topics
JOY_TOPIC       = "/joy"
FEATURES_TOPIC  = "/AN_features"
LS_TOPIC        = "/L_s"
LS_D_TOPIC      = "/L_s_d"
DS_D_TOPIC      = "/dS_d"          # external desired s_dot_d

MOVE_TOPIC      = "/cmd/move"
EULER_TOPIC     = "/cmd/euler"
HEIGHT_TOPIC    = "/cmd/body_height"

# Enable control when EITHER button is held
BUTTON_INDEX_1  = 10   # button 11 (0-based)
BUTTON_INDEX_2  = 11   # button 12 (0-based)
CONTROL_RATE_HZ = 30.0

# Controller selection
#   1 = PI, 2 = SMC, 3 = FQSMC, 4 = FFSQMC, 5 = FFSQSMC2
CONTROLLER_MODE = 4

# Choose which interaction matrix to use (for SMC/FQSMC/FFSQMC/FFSQSMC2):
#   "cur" -> /L_s
#   "des" -> /L_s_d
LS_SOURCE = "cur"

# -------- PI parameters (MODE 1): u = -K * e_s ----------
K_DIAG = np.array([2.0, 0.2, 2.0, 0.5, 1.0, 0.5], dtype=float) * 0.5

# -------- Shared SMC parameters ----------
LAMBDA1_DIAG_SMC = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0], dtype=float) * 10

# -------- Shared FQSMC/FFSQMC parameters ----------
#LAMBDA1_DIAG = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float) * 5.0
LAMBDA1_DIAG = np.array([6.5, 6.5, 6.5, 6.5, 6.5, 6.5], dtype=float)

# -------- FQSMC/FFSQMC parameters ----------
LAMBDA2_DIAG = LAMBDA1_DIAG

# FQSMC extra term: -Λ3 σC
LAMBDA3_DIAG = np.array([4.0, 0.2, 2.0, 2.0, 0.5, 0.5], dtype=float) * 0.5

# FQSMC saturation width φ (scalar or vector; elementwise)
PHI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float) * 10

Diag_des = np.diag([
    1.0,  # index 0
    0.0,  # index 1
    1.0,  # index 2
    1.0,  # index 3
    0.0,  # index 4
    0.0   # index 5
])


# ========================= FUZZY (Sugeno, 7 MFs) =========================
#Fuzzy_in_coff = np.diag([4.0, 0.2, 2.0, 0.5, 1.0, 0.5]) * 0.2
#Fuzzy_out_coff = np.diag([10.0, 0.2, 10.0, 75.0, 1.0, 0.5]) * 0.5
Fuzzy_in_coff = np.diag([2.0, 5.0, 2.0, 6.0, 2.0, 20.0]) *1.0
Fuzzy_out_coff = np.diag([10.0, 0.1, 10.0, 2.0, 4.0, 0.2])*0.05


'''
FUZZY_PARAMS = {
    "a": 1.0,
    "k": 20.0,
    "b": 0.7,
    "c": 0.33,
    "s2": 0.3,
    "s3": 0.2,
    "s4": 0.1,
}
'''
FUZZY_PARAMS = {
    "a": 1.0,
    "k": 20.0,
    "b": 0.7,
    "c": 0.33,
    "s2": 0.3,
    "s3": 0.2,
    "s4": 0.1,
}

#FUZZY_WEIGHTS = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0], dtype=float)
FUZZY_WEIGHTS = np.array([-7.42, -4.35, -3.17, 0.0, 3.17, 4.35, 7.42], dtype=float)

LS_INV_DAMPING = 1e-6

MIN_DERIV_DT = 1e-3  # seconds


# ======================== OTHER SETTINGS Fuzzy -2 =========================
# v T | > U
F2_OUT_TABLE_2 = np.array([
    [1,1,0.7,0,0,0,0],
    [1,0.7,0,0,0,0,0],
    [0.7,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0.7],
    [0,0,0,0,0,0.7,1],
    [0,0,0,0,0.7,1,1],
], dtype=float)

F2_OUT_TABLE_2=1-F2_OUT_TABLE_2

# v T | > U
'''
F2_OUT_TABLE_4 = np.array([
    [-3,-2,-1,0,1,2,3],
    [-2,-1,0,0,0,1,2],
    [-1,0,0,0,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
], dtype=float)
'''

RESET_STATES_ON_DISABLE = True

ENABLE_LOGGING = True
LOG_DIR      = "/home/ehsan/ros2_ws/src/go2_pkg/data/R1/Controllers"
LOG_FILENAME = "controller_log_T"+str(CONTROLLER_MODE)+"_v2_C3_def.csv"

MAX_EULER_X_POS =  0.3
MAX_EULER_X_NEG = -0.3
MAX_EULER_Y_POS =  0.3
MAX_EULER_Y_NEG = -0.3
MAX_EULER_Z_POS =  0.3
MAX_EULER_Z_NEG = -0.3

MAX_H_Z_POS =  0.03
MAX_H_Z_NEG = -0.10

KP_HEIGHT = 0.05
HEIGHT_SCALE = 5.0

# ================================================================


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


def reshape_Ls(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=float).reshape(-1)
    if flat.size != 36:
        raise ValueError(f"L_s must have 36 floats (6x6). Got {flat.size}.")
    return flat.reshape(6, 6)


def safe_solve_Ls(Ls: np.ndarray, rhs: np.ndarray, damping: float) -> np.ndarray:
    try:
        return np.linalg.solve(Ls, rhs)
    except np.linalg.LinAlgError:
        LtL = Ls.T @ Ls
        return np.linalg.solve(LtL + damping * np.eye(6), Ls.T @ rhs)


def sat(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


# ========================= FUZZY IMPLEMENTATION (MODE 4) =========================

def _sigmoid_left(x: float, a: float, k: float) -> float:
    return 1.0 / (1.0 + np.exp(k * (x + a)))


def _sigmoid_right(x: float, a: float, k: float) -> float:
    return 1.0 / (1.0 + np.exp(-k * (x - a)))


def _gauss(x: float, c: float, s: float) -> float:
    s = max(float(s), 1e-12)
    return float(np.exp(-0.5 * ((x - c) / s) ** 2))


def fuzzy_sugeno_7mf_siso(x: float, p: dict, w: np.ndarray, eps: float = 1e-12) -> float:
    a, k = float(p["a"]), float(p["k"])
    b, c = float(p["b"]), float(p["c"])
    s2, s3, s4 = float(p["s2"]), float(p["s3"]), float(p["s4"])

    mu1 = _sigmoid_left(x, a, k)
    mu2 = _gauss(x, -b, s2)
    mu3 = _gauss(x, -c, s3)
    mu4 = _gauss(x,  0.0, s4)
    mu5 = _gauss(x, +c, s3)  # mirror of mu3
    mu6 = _gauss(x, +b, s2)  # mirror of mu2
    mu7 = _sigmoid_right(x, a, k)

    mu = np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7], dtype=float)

    num = float(np.dot(mu, w.reshape(7,)))
    den = float(np.sum(mu)) + eps
    return num / den


def f_fuzzy(sigma_C_col: np.ndarray) -> np.ndarray:
    x = np.asarray(sigma_C_col, dtype=float).reshape(6,)
    out = np.zeros(6, dtype=float)
    for i in range(6):
        out[i] = fuzzy_sugeno_7mf_siso(x[i], FUZZY_PARAMS, FUZZY_WEIGHTS)
    return out.reshape(6, 1)


# ========================= FUZZY IMPLEMENTATION (MODE 5: EDIT HERE) =========================
def _mf7_scalar(x: float, p: dict) -> np.ndarray:
    """
    Returns mu = [mu1..mu7] for a scalar input x using the SAME MF shapes as your original.
    """
    a, k = float(p["a"]), float(p["k"])
    b, c = float(p["b"]), float(p["c"])
    s2, s3, s4 = float(p["s2"]), float(p["s3"]), float(p["s4"])

    mu1 = _sigmoid_left(x, a, k)
    mu2 = _gauss(x, -b, s2)
    mu3 = _gauss(x, -c, s3)
    mu4 = _gauss(x,  0.0, s4)
    mu5 = _gauss(x, +c, s3)
    mu6 = _gauss(x, +b, s2)
    mu7 = _sigmoid_right(x, a, k)

    return np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7], dtype=float)


def f_fuzzy2(sigma_C_col: np.ndarray) -> np.ndarray:
    x = np.asarray(sigma_C_col, dtype=float).reshape(6, 1)

    x_in = x.reshape(6,)  # (6,)
    mu_table = np.zeros((6, 7), dtype=float)
    for i in range(6):
        mu_table[i, :] = _mf7_scalar(float(x_in[i]), FUZZY_PARAMS)

    w = np.asarray(FUZZY_WEIGHTS, dtype=float).reshape(7,)
    eps = 1e-12

    def safe_wavg(mu: np.ndarray) -> float:
        den = float(np.sum(mu))
        if den <= eps:
            return 0.0
        return float(np.dot(mu, w) / den)

    y_out = np.zeros(6, dtype=float)

    # SISO parts
    y_out[0] = safe_wavg(mu_table[0, :])  # e_s_u
    y_out[1] = safe_wavg(mu_table[1, :])  # e_s_v
    y_out[2] = safe_wavg(mu_table[2, :])  # e_s_d
    y_out[3] = safe_wavg(mu_table[3, :])  # e_s_theta
    y_out[4] = safe_wavg(mu_table[4, :])  # e_s_psi
    y_out[5] = safe_wavg(mu_table[5, :])  # e_s_phi

    # 2D (u,v) modulation table
    mu_r1 = mu_table[0, :]
    mu_r2 = mu_table[1, :]

    MU_U_T = np.outer(mu_r1, mu_r2)  # (7,7)

    den_uv = float(np.sum(MU_U_T))
    if den_uv > eps:
        X_rot_coeff = float(np.sum(F2_OUT_TABLE_2 * MU_U_T) / den_uv)
    else:
        X_rot_coeff = 1.0  # or 1.0 depending on your design

    # Apply modulation to the intended channel
    y_out[3] = y_out[3] * X_rot_coeff

    return y_out.reshape(6, 1)


def ffsqsmc2_fuzzy_term(sigma_C_col: np.ndarray) -> np.ndarray:
    """
    Wrapper used by mode 5 controller. Keep mode-5 fuzzy changes isolated here.
    """
    return f_fuzzy2(sigma_C_col)

#==============================================================================
#==============================================================================
#==============================================================================

def parse_features(msg: Float32MultiArray) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    data = np.array(msg.data, dtype=float).reshape(-1)
    if data.size == 6:
        return "es", data, None
    if data.size == 12:
        s_d = data[:6]
        s   = data[6:]
        return "sd_s", s_d, s
    raise ValueError(f"{FEATURES_TOPIC} length {data.size} != 6 or 12")


class ControllersNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.dt = 1.0 / float(CONTROL_RATE_HZ)

        # Joystick gating
        self.button_idx_1 = BUTTON_INDEX_1
        self.button_idx_2 = BUTTON_INDEX_2
        self.servo_active = False

        # Latest signals
        self.have_es = False
        self.e_s = np.zeros(6, dtype=float)     # sigma

        self.have_s_sd = False
        self.s = np.zeros(6, dtype=float)
        self.s_d = np.zeros(6, dtype=float)

        # Derivatives estimated (s_dot from s; s_dot_d comes from /dS_d)
        self.s_dot = np.zeros(6, dtype=float)
        self.s_dot_d = np.zeros(6, dtype=float)
        self.e_s_dot = np.zeros(6, dtype=float)

        self.prev_s = None
        self.prev_s_d = None
        self.prev_feat_time = None

        # L_s matrices
        self.have_Ls = False
        self.Ls = np.eye(6, dtype=float)

        self.have_Ls_d = False
        self.Ls_d = np.eye(6, dtype=float)

        # FQSMC/FFSQMC/FFSQSMC2 state: sigma_C
        self.sigma_C = np.zeros(6, dtype=float)

        # Output state
        self.cmd_eul_angle = Vector3()
        self.hight_dog = 0.0

        # Limits
        self.max_pos = Vector3(x=MAX_EULER_X_POS, y=MAX_EULER_Y_POS, z=MAX_EULER_Z_POS)
        self.max_neg = Vector3(x=MAX_EULER_X_NEG, y=MAX_EULER_Y_NEG, z=MAX_EULER_Z_NEG)
        self.max_H = MAX_H_Z_POS
        self.min_H = MAX_H_Z_NEG

        # Logging
        self.log_file = None
        if ENABLE_LOGGING:
            os.makedirs(LOG_DIR, exist_ok=True)
            self.log_path = os.path.join(LOG_DIR, LOG_FILENAME)
            self.log_file = open(self.log_path, "w")
            self.log_file.write(
                "timestamp,mode,"
                "e0,e1,e2,e3,e4,e5,"
                "sdotd0,sdotd1,sdotd2,sdotd3,sdotd4,sdotd5,"
                "sigC0,sigC1,sigC2,sigC3,sigC4,sigC5,"
                "u0,u1,u2,u3,u4,u5\n"
            )
            self.log_file.flush()
            atexit.register(self._close_log)
            self.get_logger().info(f"Logging to: {self.log_path}")

        # Publishers
        self.move_pub   = self.create_publisher(Vector3, MOVE_TOPIC,   10)
        self.euler_pub  = self.create_publisher(Vector3, EULER_TOPIC,  10)
        self.height_pub = self.create_publisher(Float32, HEIGHT_TOPIC, 10)

        # Subscribers
        self.create_subscription(Joy, JOY_TOPIC, self.joy_cb, 10)
        self.create_subscription(Float32MultiArray, FEATURES_TOPIC, self.features_cb, 10)
        self.create_subscription(Float32MultiArray, LS_TOPIC, self.ls_cb, 10)
        self.create_subscription(Float32MultiArray, LS_D_TOPIC, self.ls_d_cb, 10)
        self.create_subscription(Float32MultiArray, DS_D_TOPIC, self.ds_d_cb, 10)

        # Timer
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"{NODE_NAME} started. Hold button #{self.button_idx_1+1} OR #{self.button_idx_2+1} to enable control."
        )
        self.get_logger().info(f"CONTROLLER_MODE={CONTROLLER_MODE} (1=PI,2=SMC,3=FQSMC,4=FFSQMC,5=FFSQSMC2)")
        self.get_logger().info(f"Using Ls source: {LS_SOURCE} ({LS_TOPIC if LS_SOURCE=='cur' else LS_D_TOPIC})")
        self.get_logger().info(f"Using external s_dot_d from {DS_D_TOPIC} (Float32MultiArray length 6).")
        self.get_logger().info("No filtering is applied.")

    # ========================= CALLBACKS =========================

    def joy_cb(self, msg: Joy):
        b = msg.buttons

        def is_pressed(idx: int) -> bool:
            return idx < len(b) and b[idx] == 1

        pressed_any = is_pressed(self.button_idx_1) or is_pressed(self.button_idx_2)

        if pressed_any and not self.servo_active:
            self.servo_active = True
            self.get_logger().info("CONTROL ON (button 11 or 12 pressed)")
        elif (not pressed_any) and self.servo_active:
            self.servo_active = False
            self.get_logger().info("CONTROL OFF (both buttons released)")
            self.publish_zero()
            if RESET_STATES_ON_DISABLE:
                self._reset_controller_states()

    def _reset_controller_states(self):
        self.sigma_C[:] = 0.0

    def ds_d_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=float).reshape(-1)
        if data.size < 6:
            return
        self.s_dot_d = data[:6].astype(float)

    def features_cb(self, msg: Float32MultiArray):
        t_now = time.time()

        try:
            mode, v0, v1 = parse_features(msg)
        except Exception as ex:
            self.get_logger().warn_throttle(1.0, f"Bad {FEATURES_TOPIC}: {ex}")
            return

        if mode == "es":
            self.e_s = v0.astype(float)
            self.have_es = True

            self.have_s_sd = False
            self.s_dot[:] = 0.0
            self.e_s_dot[:] = self.s_dot - self.s_dot_d

            self.prev_s = None
            self.prev_s_d = None
            self.prev_feat_time = None
            return

        s_d = v0.astype(float)
        s   = v1.astype(float)

        self.s_d = s_d
        self.s = s
        self.have_s_sd = True

        self.e_s = (self.s - self.s_d)
        self.have_es = True

        if self.prev_feat_time is not None and self.prev_s is not None:
            dt = t_now - self.prev_feat_time
            if dt >= MIN_DERIV_DT:
                self.s_dot = (self.s - self.prev_s) / dt

        self.e_s_dot = self.s_dot - self.s_dot_d

        self.prev_s = self.s.copy()
        self.prev_s_d = self.s_d.copy()
        self.prev_feat_time = t_now

    def ls_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=float)
        try:
            self.Ls = reshape_Ls(data)
            self.have_Ls = True
        except Exception as ex:
            self.get_logger().warn_throttle(1.0, f"Bad L_s on {LS_TOPIC}: {ex}")

    def ls_d_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=float)
        try:
            self.Ls_d = reshape_Ls(data)
            self.have_Ls_d = True
        except Exception as ex:
            self.get_logger().warn_throttle(1.0, f"Bad L_s_d on {LS_D_TOPIC}: {ex}")

    # ========================= INTERNALS =========================

    def _get_Ls_use(self) -> Optional[np.ndarray]:
        if LS_SOURCE == "cur":
            return self.Ls if self.have_Ls else None
        return self.Ls_d if self.have_Ls_d else None

    # ========================= CONTROLLERS =========================

    def compute_u_pi(self, e_s: np.ndarray, Ls_use: np.ndarray) -> np.ndarray:
        K = np.diag(K_DIAG.reshape(6,))
        rhs = -(K @ e_s.reshape(6, 1))
        return safe_solve_Ls(Ls_use, rhs, LS_INV_DAMPING).reshape(6,)

    def compute_u_smc(self, e_s: np.ndarray, s_dot_d: np.ndarray, Ls_use: np.ndarray) -> np.ndarray:
        inv_Lambda1 = np.diag(1.0 / np.maximum(LAMBDA1_DIAG_SMC, 1e-12))
        sigma = e_s.reshape(6, 1)
        sign_sigma = np.sign(sigma)
        rhs = (-inv_Lambda1 @ sign_sigma) + Diag_des @ s_dot_d.reshape(6, 1) * (30 / 30)
        return safe_solve_Ls(Ls_use, rhs, LS_INV_DAMPING).reshape(6,)

    def compute_u_fqsmc(self, e_s: np.ndarray, s_dot_d: np.ndarray, Ls_use: np.ndarray) -> np.ndarray:
        Lambda1 = np.diag(LAMBDA1_DIAG)
        Lambda2 = np.diag(LAMBDA2_DIAG)
        Lambda3 = np.diag(LAMBDA3_DIAG)

        sigma = e_s.reshape(6, 1)

        sigma_C_col = self.sigma_C.reshape(6, 1)
        sigma_C_dot = (-Lambda2 @ sigma_C_col) + (Lambda1 @ sigma)
        sigma_C_col = sigma_C_col + sigma_C_dot * self.dt
        self.sigma_C = sigma_C_col.reshape(6,)

        inv_Lambda1 = np.diag(1.0 / np.maximum(LAMBDA1_DIAG, 1e-12))
        phi = np.asarray(PHI, dtype=float).reshape(6, 1)
        sat_term = sat(sigma_C_col / np.maximum(phi, 1e-12))

        rhs = (-inv_Lambda1 @ sat_term) - (Lambda3 @ sigma_C_col) + Diag_des @ s_dot_d.reshape(6, 1) * (30 / 30)
        return safe_solve_Ls(Ls_use, rhs, LS_INV_DAMPING).reshape(6,)

    def compute_u_ffsqmc(self, e_s: np.ndarray, s_dot_d: np.ndarray, Ls_use: np.ndarray) -> np.ndarray:
        Lambda1 = np.diag(LAMBDA1_DIAG)
        Lambda2 = np.diag(LAMBDA2_DIAG)

        sigma = e_s.reshape(6, 1)

        sigma_C_col = self.sigma_C.reshape(6, 1)
        sigma_C_dot = (-Lambda2 @ sigma_C_col) + (Lambda1 @ sigma)
        sigma_C_col = sigma_C_col + sigma_C_dot * self.dt
        self.sigma_C = sigma_C_col.reshape(6,)

        inv_Lambda1 = np.diag(1.0 / np.maximum(LAMBDA1_DIAG, 1e-12))
        ff = Fuzzy_out_coff @ f_fuzzy(Fuzzy_in_coff @ sigma_C_col)

        rhs = (-inv_Lambda1 @ ff) + Diag_des @ s_dot_d.reshape(6, 1) * (15 / 30)*2
        return safe_solve_Ls(Ls_use, rhs, LS_INV_DAMPING).reshape(6,)

    def compute_u_ffsqsmc2(self, e_s: np.ndarray, s_dot_d: np.ndarray, Ls_use: np.ndarray) -> np.ndarray:
        """
        Mode 5: identical structure to mode 4, but uses f_fuzzy2()/ffsqsmc2_fuzzy_term().
        """
        Lambda1 = np.diag(LAMBDA1_DIAG)
        Lambda2 = np.diag(LAMBDA2_DIAG)

        sigma = e_s.reshape(6, 1)

        sigma_C_col = self.sigma_C.reshape(6, 1)
        sigma_C_dot = (-Lambda2 @ sigma_C_col) + (Lambda1 @ sigma)
        sigma_C_col = sigma_C_col + sigma_C_dot * self.dt
        self.sigma_C = sigma_C_col.reshape(6,)

        inv_Lambda1 = np.diag(1.0 / np.maximum(LAMBDA1_DIAG, 1e-12))
        ff = Fuzzy_out_coff @ f_fuzzy2(Fuzzy_in_coff @ sigma_C_col)

        rhs = (-inv_Lambda1 @ ff) + Diag_des @ s_dot_d.reshape(6, 1) * (30 / 30)
        return safe_solve_Ls(Ls_use, rhs, LS_INV_DAMPING).reshape(6,)

    # ========================= CONTROL LOOP =========================

    def control_loop(self):
        if not self.servo_active:
            return

        if not self.have_es:
            self.get_logger().warn_throttle(1.0, "Control enabled, but missing e_s. Publishing zeros.")
            self.publish_zero(publish_height=False)
            return

        e_s = self.e_s.reshape(6,)

        Ls_use = None
        if CONTROLLER_MODE in (1, 2, 3, 4, 5):
            Ls_use = self._get_Ls_use()
            if Ls_use is None:
                self.get_logger().warn_throttle(1.0, "Control enabled, but missing selected L_s matrix. Publishing zeros.")
                self.publish_zero(publish_height=False)
                return

        if CONTROLLER_MODE == 1:
            u = self.compute_u_pi(e_s, Ls_use)
        elif CONTROLLER_MODE == 2:
            u = self.compute_u_smc(e_s, self.s_dot_d, Ls_use)
        elif CONTROLLER_MODE == 3:
            u = self.compute_u_fqsmc(e_s, self.s_dot_d, Ls_use)
        elif CONTROLLER_MODE == 4:
            u = self.compute_u_ffsqmc(e_s, self.s_dot_d, Ls_use)
        elif CONTROLLER_MODE == 5:
            u = self.compute_u_ffsqsmc2(e_s, self.s_dot_d, Ls_use)
        else:
            self.get_logger().warn_throttle(1.0, f"Unknown CONTROLLER_MODE={CONTROLLER_MODE}. Publishing zeros.")
            u = np.zeros(6, dtype=float)

        cmd_move = Vector3(x=float(u[2] + 0.045 * 0), y=float(-u[0] - 0.027 * 0), z=float(-u[4] + 0.022 * 0))

        self.cmd_eul_angle.x += float(u[5] * 1*0) * self.dt
        self.cmd_eul_angle.y += float(-u[3] * 1*0) * self.dt
        self.cmd_eul_angle.z += float(u[4] * 0) * self.dt

        self.cmd_eul_angle.x = clamp(self.cmd_eul_angle.x, MAX_EULER_X_NEG, MAX_EULER_X_POS)
        self.cmd_eul_angle.y = clamp(self.cmd_eul_angle.y, MAX_EULER_Y_NEG, MAX_EULER_Y_POS)
        self.cmd_eul_angle.z = clamp(self.cmd_eul_angle.z, MAX_EULER_Z_NEG, MAX_EULER_Z_POS)

        self.move_pub.publish(cmd_move)
        self.euler_pub.publish(self.cmd_eul_angle)

        h_v = -KP_HEIGHT * float(u[1] * 1) * HEIGHT_SCALE
        self.hight_dog += h_v * self.dt
        self.hight_dog = clamp(self.hight_dog, MAX_H_Z_NEG, MAX_H_Z_POS)
        self.height_pub.publish(Float32(data=float(self.hight_dog)))

        if self.log_file is not None:
            ts = time.time()
            e_vals = ",".join(f"{x:.6f}" for x in e_s)
            d_vals = ",".join(f"{x:.6f}" for x in self.s_dot_d.reshape(6,))
            c_vals = ",".join(f"{x:.6f}" for x in self.sigma_C.reshape(6,))
            u_vals = ",".join(f"{x:.6f}" for x in u.reshape(6,))
            self.log_file.write(f"{ts:.6f},{CONTROLLER_MODE},{e_vals},{d_vals},{c_vals},{u_vals}\n")
            self.log_file.flush()

    # ========================= HELPERS =========================

    def publish_zero(self, publish_height: bool = False):
        self.cmd_eul_angle = Vector3()
        self.move_pub.publish(Vector3(x=0.0, y=0.0, z=0.0))
        self.euler_pub.publish(self.cmd_eul_angle)
        if publish_height:
            self.height_pub.publish(Float32(data=float(self.hight_dog)))

    def _close_log(self):
        try:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
        except Exception:
            pass


# ============================== MAIN ===============================

def main(args=None):
    rclpy.init(args=args)
    node = ControllersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero(publish_height=False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
