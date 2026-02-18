#!/usr/bin/env python3
import numpy as np


class DeepFeatureLs:
    """
    Compute the 6x6 image interaction matrix L_s for your deep-shape features,
    using your own feature function.

    L_s is defined so that (locally):
        f_cur_dot = L_s @ V
    where:
        - f_cur is your 6D *current* feature vector
        - V = [Vx, Vy, Vz, Wx, Wy, Wz]^T is the camera twist

    This class does NOT implement feature formulas; it calls a user-provided
    feature_fn(x0_in, y0_in, x1_in, y1_in) -> (f_des, f_cur).
    """

    def __init__(self, fx, fy, cx, cy, width, height, feature_fn):
        """
        Args:
            fx, fy, cx, cy : camera intrinsics
            width, height  : image size in pixels
            feature_fn     : callable(x0_in, y0_in, x1_in, y1_in) -> (f_des, f_cur)
                             It must implement the SAME feature definition you use
                             in your ANImageNode.
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.feature_fn = feature_fn  # external feature calculator

    # ---------------------------- Public API ----------------------------

    def compute_Ls(
        self,
        matches,
        current_features,
        Z_const,
        inlier_mask=None,
        eps=1e-4,
    ):
        """
        Compute the 6x6 image interaction matrix L_s numerically
        around the current state.

        Args:
            matches: (N,5) array [x0_n, y0_n, x1_n, y1_n, conf] (normalized)
            current_features: iterable of length 6 (your current feature vector)
                              [f1, f2, f3, f4, f5, f6]
            Z_const: scalar depth used for ALL current points (approximation)
            inlier_mask: (N,) bool array selecting which matches to use.
                         If None -> use all matches.
            eps: small finite-difference step.

        Returns:
            L_s: (6,6) numpy array
        """
        matches = np.asarray(matches, dtype=np.float32)
        N = matches.shape[0]
        if N == 0:
            raise ValueError("compute_Ls: no matches provided.")

        # Inliers
        if inlier_mask is None:
            inlier_mask = np.ones(N, dtype=bool)
        else:
            inlier_mask = np.asarray(inlier_mask, dtype=bool)
            if inlier_mask.shape[0] != N:
                raise ValueError("compute_Ls: inlier_mask has wrong length.")

        if not np.any(inlier_mask):
            raise ValueError("compute_Ls: no inliers selected.")

        # Basic sanity for features
        f0 = np.asarray(current_features, dtype=np.float64)
        if f0.shape[0] != 6:
            raise ValueError("compute_Ls: current_features must have length 6.")

        # Unpack matches and select inliers
        x0_n = matches[:, 0]
        y0_n = matches[:, 1]
        x1_n = matches[:, 2]
        y1_n = matches[:, 3]

        x0_in = x0_n[inlier_mask]
        y0_in = y0_n[inlier_mask]
        x1_in = x1_n[inlier_mask]
        y1_in = y1_n[inlier_mask]

        if x0_in.size < 4:
            raise ValueError("compute_Ls: fewer than 4 inliers for shape features.")

        # Precompute per-point normalized Jacobians d[x1_norm, y1_norm]/dV
        L_norm_list = self._compute_point_jacobians_normalized(
            x1_in, y1_in, Z_const
        )

        # Numerical differentiation: for each basis twist e_j
        L_s = np.zeros((6, 6), dtype=np.float64)

        for j in range(6):
            # basis vector e_j in twist space
            e_j = np.zeros(6, dtype=np.float64)
            e_j[j] = 1.0

            # Perturb current normalized coords according to IBVS model
            x1_pert = x1_in.copy()
            y1_pert = y1_in.copy()

            for idx, L_norm in enumerate(L_norm_list):
                # [dx_norm, dy_norm] = L_norm_i @ e_j
                dx_norm, dy_norm = L_norm @ e_j
                x1_pert[idx] += eps * dx_norm
                y1_pert[idx] += eps * dy_norm

            # Recompute features for perturbed CURRENT points
            # Desired (x0_in,y0_in) is fixed
            _, f_cur_pert = self.feature_fn(x0_in, y0_in, x1_pert, y1_pert)
            f1 = np.asarray(f_cur_pert, dtype=np.float64)
            if f1.shape[0] != 6:
                raise ValueError("feature_fn must return 6D current features.")

            # Finite-difference column j
            L_s[:, j] = (f1 - f0) / eps

        return L_s

    # ------------------------- Internal helpers -------------------------

    def _compute_point_jacobians_normalized(self, x1_in, y1_in, Z_const):
        """
        For each CURRENT inlier point (normalized x1,y1), compute the 2x6
        Jacobian mapping camera twist V -> [x_norm_dot, y_norm_dot]^T.
        """
        Z = float(Z_const)
        if Z <= 1e-6:
            Z = 1e-6

        L_norm_list = []
        for xn, yn in zip(x1_in, y1_in):
            # Normalized [0,1] -> pixel coordinates
            u = xn * self.width
            v = yn * self.height

            # Camera-plane coordinates (x_img, y_img) = (u - cx)/fx, (v - cy)/fy
            x_img = (u - self.cx) / self.fx
            y_img = (v - self.cy) / self.fy

            # Standard IBVS point Jacobian for (x_img, y_img, Z)
            L_xy = self._point_jacobian_xy(x_img, y_img, Z)  # (2,6)

            # Map to normalized image coordinates:
            # u = fx * x_img + cx,  x_norm = u / width
            # => dx_norm/dV = (fx / width) * dx_img/dV
            # similarly for y_norm
            scale_x = self.fx / float(self.width)
            scale_y = self.fy / float(self.height)

            S = np.diag([scale_x, scale_y])  # (2,2)
            L_norm = S @ L_xy                 # (2,6)
            L_norm_list.append(L_norm)

        return L_norm_list

    def _point_jacobian_xy(self, x, y, Z):
        """
        Standard IBVS interaction matrix for a point given normalized
        camera-plane coordinates x = X/Z, y = Y/Z and depth Z.

        Returns:
            L_xy: (2,6) mapping [Vx,Vy,Vz,Wx,Wy,Wz] -> [x_dot, y_dot].
        """
        if Z <= 1e-6:
            Z = 1e-6

        L = np.zeros((2, 6), dtype=np.float64)

        # Row for x_dot
        L[0, 0] = -1.0 / Z
        L[0, 1] = 0.0
        L[0, 2] = x / Z
        L[0, 3] = x * y
        L[0, 4] = -(1.0 + x * x)
        L[0, 5] = y

        # Row for y_dot
        L[1, 0] = 0.0
        L[1, 1] = -1.0 / Z
        L[1, 2] = y / Z
        L[1, 3] = 1.0 + y * y
        L[1, 4] = -x * y
        L[1, 5] = -x

        return L
