#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import numpy as np
import cv2
import math


class ANImageNode(Node):
    def __init__(self):
        super().__init__("R1_go2_deep_feature_T0_node")

        self.bridge = CvBridge()

        # Latest matches: (N,5) [x0_n, y0_n, x1_n, y1_n, conf] normalized
        self.latest_matches = None
        self.have_matches = False

        # Latest image (OpenCV) + header
        self.latest_image = None
        self.latest_header = None
        self.have_image = False

        # Joystick state (unused here, kept for completeness)
        self.prev_joy_buttons = None

        # FPS counters
        self.img_cb_count = 0
        self.match_cb_count = 0
        self.feature_loop_count = 0
        self.annot_loop_count = 0

        # Frame index for iterative behavior
        self.frame_index = 0
        self.warmup_frames = 30

        # Shared data between loops
        self.last_desired_features = None
        self.last_current_features = None
        self.last_feature_err = None

        # centroids
        self.last_global_centroids = None
        self.last_desired_region_centroids = None
        self.last_current_region_centroids = None

        # Store interaction matrices (6x6)
        self.last_Ls_desired = None
        self.last_Ls_current = None

        # Geometric inlier mask
        self.last_inlier_mask_geom = None

        # Threshold for geometric outliers in normalized space
        self.outlier_residual_thresh = 0.1

        # Previous-frame similarity model
        self.prev_similarity_valid = False
        self.prev_s = None
        self.prev_R = None
        self.prev_t = None

        # "Intrinsic-like" normalization for x,y in the interaction matrix
        self.fx = 1.0
        self.fy = 1.0
        self.cx = 0.5
        self.cy = 0.5

        # Default depth for all points
        self.default_Z = 1.0

        # OPTION: full vs simplified L_s
        self.use_full_Ls = False

        # Subscribers
        self.sub_image = self.create_subscription(Image, "/camera_test", self.image_cb, 10)
        self.sub_matches = self.create_subscription(Float32MultiArray, "/ALoFTR/matched_points", self.matches_cb, 10)

        # Publishers
        self.pub_annotated = self.create_publisher(Image, "/AN_image", 10)
        self.pub_features = self.create_publisher(Float32MultiArray, "/AN_features", 10)
        self.pub_Ls_current = self.create_publisher(Float32MultiArray, "/L_s", 10)
        self.pub_Ls_desired = self.create_publisher(Float32MultiArray, "/L_s_d", 10)

        # Timers
        self.feature_timer = self.create_timer(0.01, self.feature_timer_cb)
        self.annot_timer = self.create_timer(0.1, self.annot_timer_cb)
        self.print_timer = self.create_timer(0.05, self.print_timer_cb)
        self.fps_timer = self.create_timer(10.0, self.fps_timer_cb)

        self.get_logger().info(
            "ANImageNode: /camera/image_raw + /ALoFTR/matched_points -> "
            "/AN_image, /AN_features, /L_s, /L_s_d"
        )

    # --------------------------- Callbacks ---------------------------
    def matches_cb(self, msg: Float32MultiArray):
        self.match_cb_count += 1
        data = np.array(msg.data, dtype=np.float32)

        if data.size == 0:
            self.latest_matches = None
            self.have_matches = False
            return

        if data.size % 5 != 0:
            self.get_logger().warn(
                f"Received matched_points with data.size={data.size}, not multiple of 5; ignoring."
            )
            self.latest_matches = None
            self.have_matches = False
            return

        self.latest_matches = data.reshape(-1, 5)
        self.have_matches = True

    def image_cb(self, msg: Image):
        self.img_cb_count += 1
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.latest_image = cv_img
        self.latest_header = msg.header
        self.have_image = True

    # ----------------------- Geometry helpers -----------------------
    def _segment_lengths(self, centroid_dict):
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

    def _rotation_from_centroids(self, centroid_dict):
        """
        rotation = circular mean of angles of lines 12 and 43:
          12: TL -> TR  (A->B)
          43: BL -> BR  (D->C)
        """
        TL = centroid_dict.get("TL", None)
        TR = centroid_dict.get("TR", None)
        BR = centroid_dict.get("BR", None)
        BL = centroid_dict.get("BL", None)
        if TL is None or TR is None or BR is None or BL is None:
            return 0.0

        a12 = math.atan2(TR[1] - TL[1], TR[0] - TL[0])
        a43 = math.atan2(BR[1] - BL[1], BR[0] - BL[0])

        mean_sin = 0.5 * (math.sin(a12) + math.sin(a43))
        mean_cos = 0.5 * (math.cos(a12) + math.cos(a43))
        return math.atan2(mean_sin, mean_cos)

    def extract_shape_features(self, x0_in, y0_in, x1_in, y1_in):
        """
        Feature order:
          1) cx
          2) cy
          3) perimeter
          4) ratio_23_14
          5) rotation
          6) ratio_12_34

        Quadrants defined ONCE using desired centroid; current points inherit membership.
        """
        cx0_global = float(x0_in.mean())
        cy0_global = float(y0_in.mean())
        cx1_global = float(x1_in.mean())
        cy1_global = float(y1_in.mean())

        q_tl = (x0_in < cx0_global) & (y0_in < cy0_global)
        q_tr = (x0_in >= cx0_global) & (y0_in < cy0_global)
        q_bl = (x0_in < cx0_global) & (y0_in >= cy0_global)
        q_br = (x0_in >= cx0_global) & (y0_in >= cy0_global)

        regions = [("TL", q_tl), ("TR", q_tr), ("BR", q_br), ("BL", q_bl)]

        desired_region_centroids = {name: None for name in ["TL", "TR", "BR", "BL"]}
        current_region_centroids = {name: None for name in ["TL", "TR", "BR", "BL"]}

        for name, mask_reg in regions:
            idx_reg = np.where(mask_reg)[0]
            if idx_reg.size == 0:
                continue
            desired_region_centroids[name] = (float(x0_in[idx_reg].mean()), float(y0_in[idx_reg].mean()))
            current_region_centroids[name] = (float(x1_in[idx_reg].mean()), float(y1_in[idx_reg].mean()))

        eps = 1e-6

        seg0 = self._segment_lengths(desired_region_centroids)
        des_perim = des_ratio_23_14 = des_ratio_12_34 = des_rot = 0.0
        if seg0 is not None:
            d12, d23, d34, d41 = seg0
            des_perim = d12 + d23 + d34 + d41
            des_ratio_23_14 = d23 / (d41 + eps)
            des_ratio_12_34 = d12 / (d34 + eps)
            des_rot = self._rotation_from_centroids(desired_region_centroids)

        seg1 = self._segment_lengths(current_region_centroids)
        cur_perim = cur_ratio_23_14 = cur_ratio_12_34 = cur_rot = 0.0
        if seg1 is not None:
            d12, d23, d34, d41 = seg1
            cur_perim = d12 + d23 + d34 + d41
            cur_ratio_23_14 = d23 / (d41 + eps)
            cur_ratio_12_34 = d12 / (d34 + eps)
            cur_rot = self._rotation_from_centroids(current_region_centroids)

        desired_features = [cx0_global, cy0_global, des_perim, des_ratio_23_14, des_rot, des_ratio_12_34]
        current_features = [cx1_global, cy1_global, cur_perim, cur_ratio_23_14, cur_rot, cur_ratio_12_34]

        return (cx0_global, cy0_global,
                cx1_global, cy1_global,
                desired_region_centroids,
                current_region_centroids,
                desired_features,
                current_features)

    def estimate_similarity_from_features(self, desired_features, current_features, eps=1e-8):
        cx0, cy0, per0, _, rot0, _ = desired_features
        cx1, cy1, per1, _, rot1, _ = current_features

        s = 1.0 if abs(per0) < eps else (per1 / per0)

        rot_err_raw = rot1 - rot0
        theta = math.atan2(math.sin(rot_err_raw), math.cos(rot_err_raw))

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float32)

        c0 = np.array([cx0, cy0], dtype=np.float32)
        c1 = np.array([cx1, cy1], dtype=np.float32)
        t = c1 - s * (R @ c0)
        return s, R, t

    # ------------------- Ls with desired-based partition -------------------
    def _rotation_coeffs_circular_mean_12_43(self, uA, vA, uB, vB, uC, vC, uD, vD, eps=1e-12):
        dx12 = (uB - uA); dy12 = (vB - vA)
        dx43 = (uC - uD); dy43 = (vC - vD)

        r12 = dx12*dx12 + dy12*dy12
        r43 = dx43*dx43 + dy43*dy43
        r12 = max(r12, eps)
        r43 = max(r43, eps)

        a12 = math.atan2(dy12, dx12)
        a43 = math.atan2(dy43, dx43)

        S = 0.5 * (math.sin(a12) + math.sin(a43))
        C = 0.5 * (math.cos(a12) + math.cos(a43))
        den = S*S + C*C
        den = max(den, eps)

        w12 = 0.5 * (C*math.cos(a12) + S*math.sin(a12)) / den
        w43 = 0.5 * (C*math.cos(a43) + S*math.sin(a43)) / den

        da12_duA =  dy12 / r12
        da12_duB = -dy12 / r12
        da12_dvA = -dx12 / r12
        da12_dvB =  dx12 / r12

        da43_duD =  dy43 / r43
        da43_duC = -dy43 / r43
        da43_dvD = -dx43 / r43
        da43_dvC =  dx43 / r43

        dth_duA = w12 * da12_duA
        dth_dvA = w12 * da12_dvA

        dth_duB = w12 * da12_duB
        dth_dvB = w12 * da12_dvB

        dth_duC = w43 * da43_duC
        dth_dvC = w43 * da43_dvC

        dth_duD = w43 * da43_duD
        dth_dvD = w43 * da43_dvD

        return {
            "A": (dth_duA, dth_dvA),
            "B": (dth_duB, dth_dvB),
            "C": (dth_duC, dth_dvC),
            "D": (dth_duD, dth_dvD),
        }

    def _compute_Ls_given_region_indices(self, u_norm, v_norm, region_indices):
        L_full = np.zeros((6, 6), dtype=np.float32)
        if u_norm is None or v_norm is None:
            return L_full
        if len(u_norm) == 0:
            return L_full

        u = np.asarray(u_norm, dtype=np.float32)
        v = np.asarray(v_norm, dtype=np.float32)

        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        Z = np.full_like(x, self.default_Z, dtype=np.float32)
        Z = np.where(np.abs(Z) < 1e-6, 1.0, Z)
        invZ = 1.0 / Z

        reg_data = {}
        for name, idx in region_indices.items():
            idx = np.asarray(idx, dtype=np.int64)
            N_I = int(idx.size)
            if N_I == 0:
                reg_data[name] = {"N": 0}
                continue

            u_I = u[idx]; v_I = v[idx]
            x_I = x[idx]; y_I = y[idx]
            invZ_I = invZ[idx]

            reg_data[name] = {
                "N": N_I,
                "uc": float(u_I.mean()),
                "vc": float(v_I.mean()),
                "sum_invZ": float(invZ_I.sum()),
                "sum_x_invZ": float((x_I * invZ_I).sum()),
                "sum_y_invZ": float((y_I * invZ_I).sum()),
                "sum_xy": float((x_I * y_I).sum()),
                "sum_x2": float((x_I * x_I).sum()),
                "sum_y2": float((y_I * y_I).sum()),
                "sum_x": float(x_I.sum()),
                "sum_y": float(y_I.sum()),
            }

        region_names = ["A", "B", "C", "D"]

        sum11 = sum13 = sum14 = sum15 = sum16 = 0.0
        sum23 = sum24 = sum25 = sum26 = 0.0

        for name in region_names:
            N_I = reg_data[name]["N"]
            if N_I == 0:
                continue

            invZ_sum = reg_data[name]["sum_invZ"]
            x_invZ_sum = reg_data[name]["sum_x_invZ"]
            y_invZ_sum = reg_data[name]["sum_y_invZ"]
            xy_sum = reg_data[name]["sum_xy"]
            x2_sum = reg_data[name]["sum_x2"]
            y2_sum = reg_data[name]["sum_y2"]
            x_sum = reg_data[name]["sum_x"]
            y_sum = reg_data[name]["sum_y"]

            sum11 += invZ_sum / N_I
            sum13 += x_invZ_sum / N_I
            sum14 += xy_sum / N_I
            sum15 += (N_I + x2_sum) / N_I
            sum16 += y_sum / N_I

            sum23 += y_invZ_sum / N_I
            sum24 += (N_I + y2_sum) / N_I
            sum25 += xy_sum / N_I
            sum26 += x_sum / N_I

        L11 = -0.25 * sum11
        L13 = 0.25 * sum13
        L14 = 0.25 * sum14
        L15 = -0.25 * sum15
        L16 = 0.25 * sum16

        L22 = -0.25 * sum11
        L23 = 0.25 * sum23
        L24 = 0.25 * sum24
        L25 = -0.25 * sum25
        L26 = -0.25 * sum26

        L_full[0, 0] = L11
        L_full[0, 2] = L13
        L_full[0, 3] = L14
        L_full[0, 4] = L15
        L_full[0, 5] = L16

        L_full[1, 1] = L22
        L_full[1, 2] = L23
        L_full[1, 3] = L24
        L_full[1, 4] = L25
        L_full[1, 5] = L26

        needed = all(reg_data[nm]["N"] > 0 for nm in region_names)
        if not needed:
            return L_full if self.use_full_Ls else self._simplify_Ls(L_full)

        uA, vA = reg_data["A"]["uc"], reg_data["A"]["vc"]
        uB, vB = reg_data["B"]["uc"], reg_data["B"]["vc"]
        uC, vC = reg_data["C"]["uc"], reg_data["C"]["vc"]
        uD, vD = reg_data["D"]["uc"], reg_data["D"]["vc"]

        def dist(ux, vx, uy, vy):
            return math.hypot(ux - uy, vx - vy)

        l_AB = dist(uA, vA, uB, vB)
        l_BC = dist(uB, vB, uC, vC)
        l_CD = dist(uC, vC, uD, vD)
        l_DA = dist(uD, vD, uA, vA)

        eps = 1e-6
        l_AB = max(l_AB, eps)
        l_BC = max(l_BC, eps)
        l_CD = max(l_CD, eps)
        l_DA = max(l_DA, eps)

        a_d_A = 0.25 * ((uA - uB) / l_AB + (uA - uD) / l_DA)
        b_d_A = 0.25 * ((vA - vB) / l_AB + (vA - vD) / l_DA)
        a_d_B = 0.25 * ((uB - uA) / l_AB + (uB - uC) / l_BC)
        b_d_B = 0.25 * ((vB - vA) / l_AB + (vB - vC) / l_BC)
        a_d_C = 0.25 * ((uC - uB) / l_BC + (uC - uD) / l_CD)
        b_d_C = 0.25 * ((vC - vB) / l_BC + (vC - vD) / l_CD)
        a_d_D = 0.25 * ((uD - uC) / l_CD + (uD - uA) / l_DA)
        b_d_D = 0.25 * ((vD - vC) / l_CD + (vD - vA) / l_DA)

        a_t_A = -(l_BC * (uA - uD)) / (l_DA ** 3)
        b_t_A = -(l_BC * (vA - vD)) / (l_DA ** 3)
        a_t_B = (uB - uC) / (l_BC * l_DA)
        b_t_B = (vB - vC) / (l_BC * l_DA)
        a_t_C = (uC - uB) / (l_BC * l_DA)
        b_t_C = (vC - vB) / (l_BC * l_DA)
        a_t_D = -(l_BC * (uD - uA)) / (l_DA ** 3)
        b_t_D = -(l_BC * (vD - vA)) / (l_DA ** 3)

        a_p_A = (uA - uB) / (l_AB * l_CD)
        b_p_A = (vA - vB) / (l_AB * l_CD)
        a_p_B = (uB - uA) / (l_AB * l_CD)
        b_p_B = (vB - vA) / (l_AB * l_CD)
        a_p_C = -(l_AB * (uC - uD)) / (l_CD ** 3)
        b_p_C = -(l_AB * (vC - vD)) / (l_CD ** 3)
        a_p_D = -(l_AB * (uD - uC)) / (l_CD ** 3)
        b_p_D = -(l_AB * (vD - vC)) / (l_CD ** 3)

        coeff_rot = self._rotation_coeffs_circular_mean_12_43(uA, vA, uB, vB, uC, vC, uD, vD)

        coeff_d = {"A": (a_d_A, b_d_A), "B": (a_d_B, b_d_B), "C": (a_d_C, b_d_C), "D": (a_d_D, b_d_D)}
        coeff_t = {"A": (a_t_A, b_t_A), "B": (a_t_B, b_t_B), "C": (a_t_C, b_t_C), "D": (a_t_D, b_t_D)}
        coeff_p = {"A": (a_p_A, b_p_A), "B": (a_p_B, b_p_B), "C": (a_p_C, b_p_C), "D": (a_p_D, b_p_D)}

        def compute_row_from_coeff(coeff_dict):
            s31 = s32 = s33 = s34 = s35 = s36 = 0.0
            for nm in region_names:
                N_I = reg_data[nm]["N"]
                if N_I == 0:
                    continue
                aI, bI = coeff_dict[nm]
                invZ_sum = reg_data[nm]["sum_invZ"]
                x_invZ_sum = reg_data[nm]["sum_x_invZ"]
                y_invZ_sum = reg_data[nm]["sum_y_invZ"]
                xy_sum = reg_data[nm]["sum_xy"]
                x2_sum = reg_data[nm]["sum_x2"]
                y2_sum = reg_data[nm]["sum_y2"]
                x_sum = reg_data[nm]["sum_x"]
                y_sum = reg_data[nm]["sum_y"]

                s31 += aI * invZ_sum / N_I
                s32 += bI * invZ_sum / N_I
                s33 += (aI * x_invZ_sum + bI * y_invZ_sum) / N_I
                s34 += (aI * xy_sum + bI * (N_I + y2_sum)) / N_I
                s35 += (bI * xy_sum + aI * (N_I + x2_sum)) / N_I
                s36 += (aI * y_sum - bI * x_sum) / N_I
            return s31, s32, s33, s34, s35, s36

        r31, r32, r33, r34, r35, r36 = compute_row_from_coeff(coeff_d)
        L_full[2, 0] = -r31; L_full[2, 1] = -r32; L_full[2, 2] = r33
        L_full[2, 3] =  r34; L_full[2, 4] = -r35; L_full[2, 5] = r36
        L_full[2, :] *= 4.0

        r41, r42, r43, r44, r45, r46 = compute_row_from_coeff(coeff_t)
        L_full[3, 0] = -r41; L_full[3, 1] = -r42; L_full[3, 2] = r43
        L_full[3, 3] =  r44; L_full[3, 4] = -r45; L_full[3, 5] = r46

        r51, r52, r53, r54, r55, r56 = compute_row_from_coeff(coeff_p)
        L_full[4, 0] = -r51; L_full[4, 1] = -r52; L_full[4, 2] = r53
        L_full[4, 3] =  r54; L_full[4, 4] = -r55; L_full[4, 5] = r56

        r61, r62, r63, r64, r65, r66 = compute_row_from_coeff(coeff_rot)
        L_full[5, 0] = -r61; L_full[5, 1] = -r62; L_full[5, 2] = r63
        L_full[5, 3] =  r64; L_full[5, 4] = -r65; L_full[5, 5] = r66

        L_full[[4, 5], :] = L_full[[5, 4], :]

        return L_full if self.use_full_Ls else self._simplify_Ls(L_full)

    def _simplify_Ls(self, L_full):
        L_simple = np.zeros_like(L_full)

        L_simple[0, 0] = L_full[0, 0]
        L_simple[0, 4] = L_full[0, 4]

        L_simple[1, 1] = L_full[1, 1]
        L_simple[1, 3] = L_full[1, 3]

        L_simple[2, 2] = L_full[2, 2]

        L_simple[3, 4] = L_full[3, 4]

        L_simple[4, 5] = L_full[4, 5]

        L_simple[5, 3] = L_full[5, 3]

        return L_simple

    def compute_Ls_from_points_with_desired_partition(self, x0_in, y0_in, x1_in, y1_in):
        cx0 = float(x0_in.mean())
        cy0 = float(y0_in.mean())

        mask_A = (x0_in <= cx0) & (y0_in <= cy0)  # TL
        mask_B = (x0_in >  cx0) & (y0_in <= cy0)  # TR
        mask_C = (x0_in >  cx0) & (y0_in >  cy0)  # BR
        mask_D = (x0_in <= cx0) & (y0_in >  cy0)  # BL

        region_indices = {
            "A": np.where(mask_A)[0],
            "B": np.where(mask_B)[0],
            "C": np.where(mask_C)[0],
            "D": np.where(mask_D)[0],
        }

        Ls_des = self._compute_Ls_given_region_indices(x0_in, y0_in, region_indices)
        Ls_cur = self._compute_Ls_given_region_indices(x1_in, y1_in, region_indices)
        return Ls_des, Ls_cur

    # ----------------------- FEATURE EXTRACTION LOOP ------------------------
    def feature_timer_cb(self):
        self.feature_loop_count += 1
        self.frame_index += 1

        if not (self.have_image and self.have_matches):
            return
        if self.latest_image is None or self.latest_matches is None:
            return

        matches = self.latest_matches
        N = matches.shape[0]
        if N == 0:
            self.last_desired_features = None
            self.last_current_features = None
            self.last_feature_err = None
            self.last_global_centroids = None
            self.last_desired_region_centroids = None
            self.last_current_region_centroids = None
            self.last_inlier_mask_geom = None
            self.last_Ls_desired = None
            self.last_Ls_current = None
            return

        x0 = matches[:, 0]
        y0 = matches[:, 1]
        x1 = matches[:, 2]
        y1 = matches[:, 3]

        use_prev_model = (self.frame_index > self.warmup_frames and self.prev_similarity_valid)
        if use_prev_model:
            p0 = matches[:, 0:2]
            p1 = matches[:, 2:4]
            p0_rot = p0 @ self.prev_R.T
            p1_hat = self.prev_s * p0_rot + self.prev_t
            residuals = np.linalg.norm(p1 - p1_hat, axis=1)
            inlier_mask = residuals <= self.outlier_residual_thresh
            if inlier_mask.sum() < 4:
                inlier_mask[:] = True
        else:
            inlier_mask = np.ones(N, dtype=bool)

        if not inlier_mask.any():
            return

        x0_in = x0[inlier_mask]; y0_in = y0[inlier_mask]
        x1_in = x1[inlier_mask]; y1_in = y1[inlier_mask]

        (
            cx0_global, cy0_global, cx1_global, cy1_global,
            desired_region_centroids, current_region_centroids,
            desired_features, current_features
        ) = self.extract_shape_features(x0_in, y0_in, x1_in, y1_in)

        des = desired_features
        cur = current_features

        rot_err_raw = cur[4] - des[4]
        rot_err = math.atan2(math.sin(rot_err_raw), math.cos(rot_err_raw))

        err = [
            cur[0] - des[0],
            cur[1] - des[1],
            cur[2] - des[2],
            cur[3] - des[3],
            rot_err,
            cur[5] - des[5],
        ]

        self.last_desired_features = des
        self.last_current_features = cur
        self.last_feature_err = err

        self.last_global_centroids = (cx0_global, cy0_global, cx1_global, cy1_global)
        self.last_desired_region_centroids = desired_region_centroids
        self.last_current_region_centroids = current_region_centroids
        self.last_inlier_mask_geom = inlier_mask

        try:
            s_cur, R_cur, t_cur = self.estimate_similarity_from_features(des, cur)
            self.prev_s = s_cur
            self.prev_R = R_cur
            self.prev_t = t_cur
            self.prev_similarity_valid = True
        except Exception as e:
            self.get_logger().warn(f"Failed to estimate similarity model: {e}")

        feat_msg = Float32MultiArray()
        feat_msg.data = des + cur
        self.pub_features.publish(feat_msg)

        Ls_des, Ls_cur = self.compute_Ls_from_points_with_desired_partition(x0_in, y0_in, x1_in, y1_in)

        self.last_Ls_desired = Ls_des
        self.last_Ls_current = Ls_cur

        msg_Ls_d = Float32MultiArray()
        msg_Ls_d.data = Ls_des.reshape(-1).tolist()
        self.pub_Ls_desired.publish(msg_Ls_d)

        msg_Ls = Float32MultiArray()
        msg_Ls.data = Ls_cur.reshape(-1).tolist()
        self.pub_Ls_current.publish(msg_Ls)

    # ----------------------- ANNOTATION LOOP ------------------------
    def annot_timer_cb(self):
        """
        Periodic loop that:
        - draws matches with confidence visualization
        - highlights geometric outliers (based on the SAME inlier mask used in feature_timer_cb) with BOLD PURPLE lines
        - draws desired partition centroid as intersection of two WHITE lines (region boundaries)
        - draws global centroids + region centroids (TL/TR/BR/BL) with BIG markers + labels
        - draws polygons connecting TL/TR/BR/BL for desired and current
        - overlays text
        - publishes /AN_image

        Color convention (consistent):
        - desired items: GREEN family
        - current items: BLUE family
        """
        self.annot_loop_count += 1

        if not (self.have_image and self.have_matches):
            return
        if self.latest_image is None or self.latest_matches is None:
            return

        img = self.latest_image.copy()
        matches = self.latest_matches  # (N,5) [x0_n, y0_n, x1_n, y1_n, conf]
        h, w = img.shape[:2]

        M = matches.shape[0]
        if M == 0:
            return

        # Geometric outliers: complement of last_inlier_mask_geom (only if aligned in length)
        geom_outlier_mask = None
        if self.last_inlier_mask_geom is not None and self.last_inlier_mask_geom.shape[0] == M:
            geom_outlier_mask = ~self.last_inlier_mask_geom

        # Denormalize for drawing
        x0_n = matches[:, 0]
        y0_n = matches[:, 1]
        x1_n = matches[:, 2]
        y1_n = matches[:, 3]
        conf = matches[:, 4]

        x0 = x0_n * w
        y0 = y0_n * h
        x1 = x1_n * w
        y1 = y1_n * h

        # Colors (BGR)
        desired_color = (0, 255, 0)          # green
        current_color = (255, 0, 0)          # blue
        high_conf_color = (0, 165, 255)      # orange
        low_conf_color = (0, 0, 255)         # red
        geom_outlier_color = (255, 0, 255)   # purple
        white = (255, 255, 255)

        # -------------------- Draw matches --------------------
        # (Option) slightly bigger match points for visibility
        match_pt_r = 4
        match_pt_r_out = 6  # when geom outlier
        for i, (px0, py0, px1, py1, cf) in enumerate(zip(x0, y0, x1, y1, conf)):
            cx0 = int(round(px0))
            cy0 = int(round(py0))
            cx1 = int(round(px1))
            cy1 = int(round(py1))

            # base line color from confidence
            if cf < 0.8:
                line_color = low_conf_color
                thickness = 1
            else:
                line_color = high_conf_color
                thickness = 1

            # if geometric outlier, override
            outlier_here = (geom_outlier_mask is not None and geom_outlier_mask[i])
            if outlier_here:
                line_color = geom_outlier_color
                thickness = 3

            # points
            r0 = match_pt_r_out if outlier_here else match_pt_r
            r1 = match_pt_r_out if outlier_here else match_pt_r

            if 0 <= cx0 < w and 0 <= cy0 < h:
                cv2.circle(img, (cx0, cy0), r0, desired_color, -1)
            if 0 <= cx1 < w and 0 <= cy1 < h:
                cv2.circle(img, (cx1, cy1), r1, current_color, -1)

            # line
            cv2.line(img, (cx0, cy0), (cx1, cy1), line_color, thickness)

        # -------------------- Draw partition centroid cross (desired centroid) --------------------
        # This is the centroid used for dividing matched points into 4 regions in feature_timer_cb.
        if self.last_global_centroids is not None:
            cx0_global_n, cy0_global_n, cx1_global_n, cy1_global_n = self.last_global_centroids

            cxg_i = int(round(float(cx0_global_n) * w))
            cyg_i = int(round(float(cy0_global_n) * h))
            cxg_i = max(0, min(w - 1, cxg_i))
            cyg_i = max(0, min(h - 1, cyg_i))

            # White cross = region boundaries
            cv2.line(img, (0, cyg_i), (w - 1, cyg_i), white, 2)
            cv2.line(img, (cxg_i, 0), (cxg_i, h - 1), white, 2)
            cv2.circle(img, (cxg_i, cyg_i), 6, white, -1)

            # Also draw current global centroid (for reference)
            cx1_i = int(round(float(cx1_global_n) * w))
            cy1_i = int(round(float(cy1_global_n) * h))
            cx1_i = max(0, min(w - 1, cx1_i))
            cy1_i = max(0, min(h - 1, cy1_i))

            # Global centroid markers (bigger squares)
            r = 10
            # desired: green fill + white border
            cv2.rectangle(img, (cxg_i - r, cyg_i - r), (cxg_i + r, cyg_i + r), (200, 255, 200), -1)
            cv2.rectangle(img, (cxg_i - r, cyg_i - r), (cxg_i + r, cyg_i + r), white, 2)
            cv2.putText(img, "D-C", (cxg_i + r + 5, cyg_i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, desired_color, 2, cv2.LINE_AA)

            # current: blue fill + white border
            cv2.rectangle(img, (cx1_i - r, cy1_i - r), (cx1_i + r, cy1_i + r), (255, 200, 200), -1)
            cv2.rectangle(img, (cx1_i - r, cy1_i - r), (cx1_i + r, cy1_i + r), white, 2)
            cv2.putText(img, "C-C", (cx1_i + r + 5, cy1_i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2, cv2.LINE_AA)

        # -------------------- Draw region centroids + polygon (big points) --------------------
        def _draw_centroids_and_polygon(img_in, centroids_norm, point_color, edge_color, label_prefix=""):
            if centroids_norm is None:
                return
            keys = ["TL", "TR", "BR", "BL"]
            if any(centroids_norm.get(k, None) is None for k in keys):
                return

            pts = []
            for k in keys:
                u, v = centroids_norm[k]
                px = int(round(float(u) * w))
                py = int(round(float(v) * h))
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))
                pts.append((px, py))

                # BIG markers for the 4 points
                cv2.circle(img_in, (px, py), 12, point_color, -1)
                cv2.circle(img_in, (px, py), 14, white, 2)

                cv2.putText(
                    img_in,
                    f"{label_prefix}{k}",
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    point_color,
                    2,
                    cv2.LINE_AA,
                )

            # polygon edges
            for i in range(4):
                p0 = pts[i]
                p1 = pts[(i + 1) % 4]
                cv2.line(img_in, p0, p1, edge_color, 3)

        # Desired region centroids (green) + polygon
        _draw_centroids_and_polygon(
            img,
            self.last_desired_region_centroids,
            point_color=desired_color,
            edge_color=desired_color,
            label_prefix="D-",
        )

        # Current region centroids (blue) + polygon
        _draw_centroids_and_polygon(
            img,
            self.last_current_region_centroids,
            point_color=current_color,
            edge_color=current_color,
            label_prefix="C-",
        )

        # -------------------- Text overlay --------------------
        txt = f"Matches: {matches.shape[0]}  (frame {self.frame_index})"
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, high_conf_color, 2, cv2.LINE_AA)

        # -------------------- Resize then publish --------------------
        new_w, new_h = max(1, w // 2), max(1, h // 2)
        vis = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            if self.latest_header is not None:
                out_msg.header = self.latest_header
            self.pub_annotated.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish /AN_image: {e}")


    def print_timer_cb(self):
        return

    def fps_timer_cb(self):
        self.img_cb_count = 0
        self.match_cb_count = 0
        self.feature_loop_count = 0
        self.annot_loop_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = ANImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
