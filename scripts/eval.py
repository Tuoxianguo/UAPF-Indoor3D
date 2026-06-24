#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory and reconstruction evaluation helpers.

Implements the metrics reported in the paper so that results can be reproduced from
the predicted poses (``all_poses.npz``) and reconstructed point clouds:

  * ATE  (Absolute Trajectory Error, RMSE after Sim(3)/SE(3) alignment)
  * RPE  (Relative Pose Error)
  * Chamfer distance and reconstruction F1 (point cloud vs. ground truth)

For ATE/RPE on standard benchmarks we recommend the well-tested `evo` package; the
implementation below is self-contained and dependency-light for quick checks.
"""
from __future__ import annotations

import argparse

import numpy as np


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares Sim(3)/SE(3) alignment of two trajectories (N x 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = (D * np.diag(S)).sum() / (s0 ** 2).sum() * src.shape[0] if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return R, t, scale


def ate(pred_xyz: np.ndarray, gt_xyz: np.ndarray, with_scale: bool = True) -> float:
    """Absolute Trajectory Error (RMSE) after alignment."""
    R, t, s = umeyama_alignment(pred_xyz, gt_xyz, with_scale)
    aligned = (s * (R @ pred_xyz.T)).T + t
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def rpe(pred_xyz: np.ndarray, gt_xyz: np.ndarray, delta: int = 1) -> float:
    """Relative Pose Error (translation RMSE over a fixed step)."""
    dp = pred_xyz[delta:] - pred_xyz[:-delta]
    dg = gt_xyz[delta:] - gt_xyz[:-delta]
    err = np.linalg.norm(dp - dg, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def chamfer_distance(pc_a: np.ndarray, pc_b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds (uses scikit-learn KD-tree)."""
    from sklearn.neighbors import KDTree
    ta, tb = KDTree(pc_a), KDTree(pc_b)
    da, _ = tb.query(pc_a, k=1)
    db, _ = ta.query(pc_b, k=1)
    return float(da.mean() + db.mean())


def reconstruction_f1(pred: np.ndarray, gt: np.ndarray, thr: float = 0.05) -> float:
    """Reconstruction F1 at a distance threshold (precision/recall on point clouds)."""
    from sklearn.neighbors import KDTree
    tg, tp = KDTree(gt), KDTree(pred)
    dp, _ = tg.query(pred, k=1)
    dr, _ = tp.query(gt, k=1)
    precision = float((dp < thr).mean())
    recall = float((dr < thr).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _load_xyz(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    for key in ("poses", "c2w", "all_poses", "traj"):
        if key in data:
            arr = data[key]
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                return arr[:, :3, 3]
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]
    raise KeyError(f"No pose array found in {npz_path}; keys = {list(data.keys())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="predicted poses .npz (all_poses.npz)")
    ap.add_argument("--gt", required=True, help="ground-truth poses .npz")
    ap.add_argument("--no_scale", action="store_true", help="disable scale in alignment")
    args = ap.parse_args()

    pred = _load_xyz(args.pred)
    gt = _load_xyz(args.gt)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    print(f"frames: {n}")
    print(f"ATE (m): {ate(pred, gt, with_scale=not args.no_scale):.4f}")
    print(f"RPE (m): {rpe(pred, gt):.4f}")


if __name__ == "__main__":
    main()
