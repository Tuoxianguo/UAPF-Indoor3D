#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time
import os
import hashlib
from datetime import datetime
from tqdm import tqdm
import logging as log
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d

# 修复：正确导入Open3D可视化模块
try:
    # 适用于Open3D >= 0.13.0
    from open3d.visualization import gui, rendering
except ImportError:
    # 回退方案：适用于旧版本Open3D
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

from must3r.model.blocks.attention import has_xformers, toggle_memory_efficient_attention
from must3r.slam.data import AutoMultiLoader
from must3r.slam.model import SLAM_MUSt3R

try:
    o3d.cuda
except AttributeError:
    print('Fallback to open3d.cpu')
    o3d.cuda = o3d.cpu  # workaround for module open3d has no attribute cuda

MB = 1024. ** 2
camcols = [  # different frustrum colors for each agent
    [.1, .1, .9],  # blue
    [1., .5, 0.],  # orange
    [.5, 0., .5],  # purple
    [0., 1., 1.],  # cyan
]

SKIP_EVERY = 1


def grab_frame(camera):
    read = camera.read()
    frame = read[1]
    camid = 0 if len(read) != 3 else read[2]

    for _ in range(SKIP_EVERY - 1):
        camera.grab()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    return img, camid


def _is_video_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def _probe_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        frame_count = n
    cap.release()
    return frame_count, fps


def _extract_uniform_frames(video_path: str, out_dir: str, target_frames: int, jpg_quality: int = 95):
    os.makedirs(out_dir, exist_ok=True)
    frame_count, _ = _probe_video(video_path)
    if frame_count <= 0:
        raise RuntimeError(f"Empty video or cannot decode frames: {video_path}")

    target_n = max(1, min(int(target_frames), int(frame_count)))
    idx = np.linspace(0, frame_count - 1, num=target_n, dtype=int)
    idx = np.unique(idx).astype(int)
    target_n = int(idx.size)
    digits = max(3, len(str(target_n)))

    wanted = set(int(x) for x in idx.tolist())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = 0
    frame_id = 0
    while saved < target_n:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_id in wanted:
            saved += 1
            name = f"{saved:0{digits}d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        frame_id += 1

    cap.release()
    if saved == 0:
        raise RuntimeError("No frames were saved. Check video decoding and input path.")
    return out_dir


def img2o3d(im):
    res = o3d.cuda.pybind.geometry.Image(im.astype(np.uint8))
    return res


def colorize_depth(depth, mode='grayscale'):
    if depth is None:
        return depth
    colored_depth = None
    if mode == 'grayscale':
        mind, maxd = depth.min(), depth.max()
        depth = 255. * (depth - mind) / (maxd - mind + 1e-9)
        colored_depth = torch.stack([depth, depth, depth], dim=-1)
    elif mode == 'conf':
        colored_depth = depth - 1.0
    else:
        raise ValueError(f"Unknown colorization mode {mode}.")
    return colored_depth.cpu().numpy()


def _so3_rotation_angle(R, eps=1e-7):
    tr = torch.diagonal(R, 0, -2, -1).sum(-1)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_theta)


def _robust_scale_median(values, eps=1e-9):
    if not values:
        return 1.0
    med = float(np.median(np.asarray(values, dtype=np.float64)))
    return max(med, eps)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _compute_overlap_and_gain(pts3d, conf, depth, overlap_tree, cam_center, min_conf, overlap_mode, percentile):
    if overlap_tree is None:
        return 0.0, 0.0
    conf_np = conf.detach().cpu().numpy()
    depth_np = depth.detach().cpu().numpy() if depth is not None else None
    msk = conf_np > float(min_conf)
    if msk.sum() == 0:
        return 0.0, 0.0
    dists = overlap_tree.query(pts3d[msk], cam_center=cam_center)
    if "norm" in str(overlap_mode) and depth_np is not None:
        dists = dists / (depth_np[msk] + 1e-9)
    dists[np.isposinf(dists)] = np.finfo(dists.dtype).max
    raw_dist = float(np.percentile(dists, float(percentile)))
    overlap_score = float(np.exp(-raw_dist))
    dist_med = float(np.median(dists))
    dist_scale = max(dist_med, 1e-9)
    info_gain = float(np.mean(_sigmoid((dists - dist_scale) / dist_scale)))
    return overlap_score, info_gain


# ==================== 创新性改进模块：概率图模型驱动的重叠度估计 ====================

class ProbabilisticOverlapModel:
    """
    概率图模型驱动的重叠度计算
    将重叠度估计建模为贝叶斯推断问题
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def compute_posterior_overlap(self, geo_dist, app_sim, motion_res):
        """
        计算重叠度的后验分布 P(O|G,A,M)

        Args:
            geo_dist: 几何距离 (越小越好)
            app_sim: 外观相似度 [0,1] (越大越好)
            motion_res: 运动残差 (越小越好)

        Returns:
            posterior_mean: 后验期望重叠度 [0,1]
        """
        # 先验：Beta分布，假设初始重叠度适中
        alpha_prior, beta_prior = 2.0, 2.0

        # 几何似然：距离越小 => 重叠度越高
        # 建模为指数分布：P(G|O) ∝ exp(-λ*G), λ与O正相关
        geo_precision = 10.0  # 假设重叠度越高，几何精度越高
        geo_likelihood_alpha = geo_precision * np.exp(-geo_dist)

        # 外观似然：相似度直接提供证据
        # Beta分布：P(A|O) ∝ A^(α-1)*(1-A)^(β-1)
        app_weight = 5.0
        app_likelihood_alpha = alpha_prior + app_sim * app_weight
        app_likelihood_beta = beta_prior + (1 - app_sim) * app_weight

        # 运动似然：残差越小 => 运动越一致 => 重叠度越高
        motion_precision = 5.0
        motion_likelihood_alpha = motion_precision / (motion_res + self.eps)

        # 后验参数（简化乘积近似）
        posterior_alpha = geo_likelihood_alpha * app_likelihood_alpha * motion_likelihood_alpha
        posterior_beta = beta_prior + 1.0  # 归一化项

        # 返回后验期望
        return posterior_alpha / (posterior_alpha + posterior_beta + self.eps)


# 新增：改进的重叠度计算和关键帧选择类
class KeyframeDecisionFactors:
    """关键帧决策因子容器"""

    def __init__(self):
        self.overlap_score = 0.0
        self.confidence_score = 0.0
        self.motion_magnitude = 0.0
        self.time_since_last_kf = 0
        self.information_gain = 0.0
        self.tracking_quality = 1.0
        self.point_density = 0.0
        self.overlap_mode = "nn-norm"  # 默认重叠度模式


class DecisionHistory:
    """决策历史管理器"""

    def __init__(self, window_size=50):
        self.overlap_scores = []
        self.frame_ids = []
        self.keyframe_decisions = []  # 记录关键帧决策结果
        self.motion_metrics = []
        self.tracking_qualities = []
        self.window_size = window_size

    def update(self, overlap_score, frame_id, was_selected, motion_metric=None, tracking_quality=None):
        self.overlap_scores.append(overlap_score)
        self.frame_ids.append(frame_id)
        self.keyframe_decisions.append(was_selected)
        if motion_metric is not None:
            self.motion_metrics.append(motion_metric)
        if tracking_quality is not None:
            self.tracking_qualities.append(tracking_quality)

        # 保持滑动窗口
        if len(self.overlap_scores) > self.window_size:
            self.overlap_scores.pop(0)
            self.frame_ids.pop(0)
            self.keyframe_decisions.pop(0)
            if self.motion_metrics:
                self.motion_metrics.pop(0)
            if self.tracking_qualities:
                self.tracking_qualities.pop(0)

    def get_recent_overlap_scores(self, lookback=20):
        if len(self.overlap_scores) > lookback:
            return self.overlap_scores[-lookback:]
        else:
            return self.overlap_scores

    def get_time_since_last_keyframe(self, current_frame_id):
        """计算距离上一个关键帧的时间间隔"""
        if not self.keyframe_decisions:
            return current_frame_id + 1  # 第一个帧

        # 从后往前找最近的关键帧
        for i in range(len(self.keyframe_decisions) - 1, -1, -1):
            if self.keyframe_decisions[i]:
                return current_frame_id - self.frame_ids[i]
        return current_frame_id + 1  # 如果没有关键帧


def get_overlap_score_improved(res,
                               overlap_tree,
                               cam_center,
                               global_descriptor=None,
                               memory_global_descriptors=None,
                               mode='fusion',
                               kf_x_subsamp=None,
                               min_conf_keyframe=1.5,
                               percentile=70,
                               eps=1e-9):
    """
    改进的重叠度计算：融合几何和外观信息
    """
    # --- 1. 几何重叠度 (保留原算法核心) ---
    geometric_score = 0.0
    pts3d = res['pts3d'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp] if kf_x_subsamp else res['pts3d']
    msk = res['conf'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp] if kf_x_subsamp else res['conf']
    msk = msk > min_conf_keyframe

    raw_geometric_dist = 10.0  # 默认大距离
    if msk.sum() > 0:
        # 计算当前帧3D点到内存地图的最近邻距离
        dists = overlap_tree.query(pts3d[msk], cam_center=cam_center)
        if 'norm' in mode:
            depths = res['pts3d_local'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp, -1]
            dists /= depths[msk].cpu().numpy() + eps
        dists[np.isposinf(dists)] = np.finfo(dists.dtype).max
        # 保存原始几何距离
        raw_geometric_dist = np.percentile(dists, percentile)
        # 几何得分：距离越小，重叠度越高，所以用指数衰减
        geometric_score = np.exp(-raw_geometric_dist)

    # --- 2. 外观相似度 (新增) ---
    appearance_score = 0.0
    if global_descriptor is not None and memory_global_descriptors is not None:
        # 计算当前帧与所有内存关键帧的视觉相似度
        similarities = []
        for mem_desc in memory_global_descriptors:
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(global_descriptor, mem_desc, dim=1)
            similarities.append(cos_sim.item())
        # 外观得分取与最相似关键帧的相似度
        appearance_score = max(similarities) if similarities else 0.0

    # --- 3. 运动一致性 (新增) ---
    # 如果相机运动剧烈，即使区域重叠，特征匹配也会困难
    # 这里用一个与运动速度负相关的因子，运动越快，该因子越小
    motion_consistency_factor = 1.0  # 默认值
    motion_residual = 0.1  # 默认小残差
    # 在实际应用中，这里应该计算真实的运动残差
    # 为简化，我们假设motion_consistency_factor = exp(-motion_residual)
    # 因此 motion_residual = -log(motion_consistency_factor)

    # --- 4. 多因子融合：使用概率图模型替代简单加权 ---
    if mode == 'fusion':
        # 将几何得分转换回距离
        geo_dist = -np.log(geometric_score + 1e-9) if geometric_score > 0 else 10.0
        app_sim = appearance_score
        # 将运动一致性因子转换为残差
        motion_res = -np.log(motion_consistency_factor + 1e-9) if motion_consistency_factor > 0 else 1.0

        # 使用概率图模型进行贝叶斯融合
        prob_model = ProbabilisticOverlapModel()
        overlap_score = prob_model.compute_posterior_overlap(geo_dist, app_sim, motion_res)

    elif mode == 'geo_only':
        overlap_score = geometric_score
    elif mode == 'app_only':
        overlap_score = appearance_score
    else:
        raise ValueError(f"Unknown overlap score method {mode}")

    return overlap_score


def adaptive_keyframe_selection(decision_factors,
                                history,
                                current_frame_id,
                                min_conf_keyframe=1.5):
    """
    自适应关键帧选择策略
    """

    # --- 自适应阈值计算 ---
    def compute_adaptive_threshold(overlap_scores_history, current_overlap):
        if len(overlap_scores_history) < 10:
            return 0.15  # 初始默认阈值

        # 方法1：使用历史数据的统计量
        scores_array = np.array(overlap_scores_history)
        # 动态阈值 = 历史均值 - α * 历史标准差
        # 当场景变化大（标准差大）时，阈值更宽松
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        adaptive_thr = mean_score - 0.5 * std_score

        # 方法2：基于当前场景类型
        # 如果历史重叠度普遍较高，说明场景狭窄，应提高阈值
        # 如果历史重叠度普遍较低，说明场景开阔，应降低阈值
        if mean_score > 0.7:
            # 狭窄场景，需要更少的关键帧
            adaptive_thr = max(adaptive_thr, 0.2)  # 狭窄场景，阈值更高
        elif mean_score < 0.3:
            # 开阔场景，阈值更低
            adaptive_thr = min(adaptive_thr, 0.1)
        else:
            adaptive_thr = max(adaptive_thr, 0.15)  # 中等场景

        return adaptive_thr

    # --- 多因素决策 ---
    decision = False
    reasons = []  # 用于调试和日志记录

    # 必要条件：置信度必须达标
    if decision_factors.confidence_score < min_conf_keyframe:
        return False, ["置信度过低"]

    # 因子1：重叠度 (主要因素)
    adaptive_thr = compute_adaptive_threshold(history.get_recent_overlap_scores(), decision_factors.overlap_score)

    # 改进的重叠度判断
    if 'nn' in decision_factors.overlap_mode:
        geo_decision = decision_factors.overlap_score < adaptive_thr  # 几何重叠度低
    else:
        geo_decision = decision_factors.overlap_score > adaptive_thr  # 其他模式

    # 因子2：距离上一个关键帧的时间
    time_since_last_kf = history.get_time_since_last_keyframe(current_frame_id)
    time_decision = time_since_last_kf > 10  # 至少间隔10帧

    # 因子3：信息增益 (当前帧是否观测到新的区域)
    info_decision = decision_factors.information_gain > 0.1

    # 因子4：运动幅度 - 运动太小（几乎静止）或太大（运动模糊）都不适合作为关键帧
    if len(history.motion_metrics) >= 10:
        recent_motion = np.asarray(history.motion_metrics[-20:], dtype=np.float64)
        med_motion = float(np.median(recent_motion))
        mad_motion = float(np.median(np.abs(recent_motion - med_motion)))
        low_motion = max(0.1, med_motion - 1.5 * mad_motion)
        high_motion = med_motion + 1.5 * mad_motion
    else:
        low_motion, high_motion = 0.3, 3.0
    motion_decision = (decision_factors.motion_magnitude > low_motion and
                       decision_factors.motion_magnitude < high_motion)

    # 因子5：追踪质量 - 如果追踪不稳定，需要插入关键帧
    tracking_decision = (decision_factors.tracking_quality < 0.7)

    # 综合决策逻辑
    # 情况A：强制插入 - 追踪质量差或距离上一个关键帧太远
    force_insertion = (decision_factors.tracking_quality < 0.5 or
                       time_since_last_kf > 30)

    if force_insertion:
        decision = True
        reasons.append("强制插入：追踪质量差或时间间隔过长")

    # 情况B：基于重叠度和信息增益的常规插入
    elif geo_decision and (info_decision or motion_decision):
        decision = True
        reasons.append("常规插入：低重叠度且高信息增益/合理运动")

    # 情况C：时间基线插入 - 确保即使场景静止，也有足够的时间基线进行优化
    elif time_since_last_kf > 15:
        decision = True
        reasons.append("时间基线插入")

    # 情况D：如果所有条件都不满足，但置信度特别高且信息增益显著
    elif (decision_factors.confidence_score > min_conf_keyframe * 2 and
          decision_factors.information_gain > 0.2):
        decision = True
        reasons.append("高置信度高信息增益插入")

    return decision, reasons, time_since_last_kf


# 新增：改进的SLAM模型包装器
class ImprovedSLAMWrapper:
    """改进的SLAM模型包装器，集成新的关键帧选择算法"""

    def __init__(self, original_model, use_improved_kf_selection=True):
        self.original_model = original_model
        self.use_improved_kf_selection = use_improved_kf_selection
        self.decision_history = DecisionHistory()
        self.last_keyframe_id = -1
        self.last_c2w = None
        self.motion_trans_hist = []
        self.motion_rot_hist = []
        self.motion_metric_ema = None
        self.latest_motion = None
        self.prev_motion_metric = None

    def __call__(self, image, frame_id, camid):
        # 调用原始模型获取结果
        pts3d, colors, depth, conf, focal, w2c, HW, iskeyframe = self.original_model(image, frame_id, camid)

        c2w = w2c.inverse()
        cam_center = c2w[:3, -1].detach().cpu().numpy()

        motion_translation = 0.0
        motion_rotation = 0.0
        motion_metric = 0.0
        if self.last_c2w is not None:
            delta = torch.matmul(torch.inverse(self.last_c2w), c2w)
            R = delta[:3, :3]
            t = delta[:3, 3]
            motion_translation = float(torch.linalg.norm(t).detach().cpu())
            motion_rotation = float(_so3_rotation_angle(R).detach().cpu())

            self.motion_trans_hist.append(motion_translation)
            self.motion_rot_hist.append(motion_rotation)
            if len(self.motion_trans_hist) > 50:
                self.motion_trans_hist.pop(0)
            if len(self.motion_rot_hist) > 50:
                self.motion_rot_hist.pop(0)

            t_ref = _robust_scale_median(self.motion_trans_hist)
            r_ref = _robust_scale_median(self.motion_rot_hist)
            motion_metric = float(np.sqrt((motion_translation / t_ref) ** 2 + (motion_rotation / r_ref) ** 2))

            if self.motion_metric_ema is None:
                self.motion_metric_ema = motion_metric
            else:
                alpha = 0.2
                self.motion_metric_ema = float(alpha * motion_metric + (1.0 - alpha) * self.motion_metric_ema)

        self.latest_motion = {
            "translation": motion_translation,
            "rotation": motion_rotation,
            "metric": self.motion_metric_ema if self.motion_metric_ema is not None else motion_metric,
        }

        self.last_c2w = c2w.detach()

        if self.use_improved_kf_selection and frame_id > 0:  # 从第二帧开始使用改进算法
            # 收集决策因子
            decision_factors = KeyframeDecisionFactors()
            decision_factors.confidence_score = conf.mean().item()
            decision_factors.overlap_mode = self.original_model.overlap_mode
            decision_factors.motion_magnitude = self.latest_motion["metric"]

            overlap_score = 0.0
            info_gain = 0.0
            min_conf = getattr(self.original_model, "min_conf_keyframe", 1.5)
            try:
                overlap_tree = getattr(self.original_model, "overlap_tree", None)
                subsamp = getattr(self.original_model, "kf_x_subsamp", None)
                percentile = getattr(self.original_model, "overlap_percentile", 70)
                overlap_mode = getattr(self.original_model, "overlap_mode", "nn-norm")

                if overlap_tree is not None and getattr(self.original_model, "num_mem_frames", 0) > 0:
                    if subsamp and int(subsamp) > 1:
                        pts = pts3d[::subsamp, ::subsamp]
                        conf_s = conf[::subsamp, ::subsamp]
                        depth_s = depth[::subsamp, ::subsamp]
                    else:
                        pts = pts3d
                        conf_s = conf
                        depth_s = depth

                    overlap_score, info_gain = _compute_overlap_and_gain(
                        pts, conf_s, depth_s, overlap_tree, cam_center, min_conf, overlap_mode, percentile
                    )
            except Exception:
                overlap_score = 0.0
                info_gain = 0.0

            decision_factors.overlap_score = overlap_score
            decision_factors.information_gain = info_gain

            conf_mean = float(conf.mean().item())
            conf_std = float(conf.std().item())
            conf_term = _sigmoid((conf_mean - (min_conf + 0.2)) / 0.2)
            var_term = _sigmoid((0.6 - conf_std) / 0.2)
            motion_delta = 0.0 if self.prev_motion_metric is None else abs(self.latest_motion["metric"] - self.prev_motion_metric)
            smooth_term = float(np.exp(-motion_delta))
            decision_factors.tracking_quality = float(np.clip(conf_term * var_term * smooth_term, 0.0, 1.0))
            self.prev_motion_metric = self.latest_motion["metric"]

            # 计算信息增益（简化版本，使用点云密度）
            decision_factors.point_density = pts3d.shape[0] / (HW[0] * HW[1]) if pts3d.shape[0] > 0 else 0

            # 使用改进的关键帧选择
            improved_decision, reasons, time_since_last = adaptive_keyframe_selection(
                decision_factors, self.decision_history, frame_id)

            # 更新历史记录
            self.decision_history.update(
                decision_factors.overlap_score,
                frame_id,
                improved_decision,
                motion_metric=decision_factors.motion_magnitude,
                tracking_quality=decision_factors.tracking_quality
            )

            # 如果改进算法决定插入关键帧，但原算法没有，则强制插入
            if improved_decision and not iskeyframe:
                iskeyframe = True
                print(f"帧 {frame_id}: 改进算法插入关键帧 - {', '.join(reasons)}")
            elif not improved_decision and iskeyframe:
                print(f"帧 {frame_id}: 改进算法否决关键帧插入")

            if improved_decision:
                self.last_keyframe_id = frame_id

        return pts3d, colors, depth, conf, focal, w2c, HW, iskeyframe

    def __getattr__(self, name):
        # 代理所有其他方法和属性到原始模型
        return getattr(self.original_model, name)


# Open3D classes
# Processing
class PipelineModel:
    """Controls IO. Methods run in worker threads."""

    def __init__(self,
                 model,
                 camera,
                 update_view,
                 device=None,
                 res=512,
                 show_cameras=True,
                 chunk=-1,  # -1 means no chunking
                 chunking_overlap=4,
                 viz_conf=2.5,  # conf thresh for pts3d viz
                 use_improved_kf_selection=True  # 新增：是否使用改进的关键帧选择
                 ):
        """Initialize.
        Args:
            update_view (callback): Callback to update display elements for a
                frame.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
            res: maxdim of the images in pixels
            show_camera: display camera locations with the 3D model
            chunk: chunk size for keyframe chunking (split sequence memory to
                 redefine origin as the frame number augments since MUSt3R can hardly go above 50 keyframes)
            chunking_overlap : when creating a new memory chunk, how many images of the previous one should be used
            use_improved_kf_selection: 是否使用改进的关键帧选择算法
        """
        self.chunk = chunk
        self.chunking_overlap = chunking_overlap
        self.res = res
        self.show_cameras = show_cameras
        self.viz_conf = viz_conf
        self.update_view = update_view
        self.use_improved_kf_selection = use_improved_kf_selection

        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = o3d.core.Device(self.device)

        self.cv_capture = threading.Condition()  # condition variable
        self.query_view = None
        self.camera = camera
        self.depth_in_color = None

        # 使用改进的SLAM包装器
        self.must3r = ImprovedSLAMWrapper(model, use_improved_kf_selection)

        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.flag_start = False

        self.keyframes_data = []
        self.keyframe_focals = []
        self.keyframe_confs = []

        self.pcd_frame = None
        self.rgbd_frame = None
        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Process')
        self.flag_exit = False

        self.cache = {}

        # 新增：决策历史管理器
        self.decision_history = DecisionHistory(window_size=50)

    @property
    def max_points(self):
        return 10 * self.res ** 2

    def run(self):
        """Run pipeline."""
        frame_id = 0
        t1 = time.perf_counter()
        cam_centers = []
        memory_map = None
        improved_kf_count = 0  # 统计改进算法插入的关键帧数量

        print(f"使用改进的关键帧选择算法: {self.use_improved_kf_selection}")

        while not self.flag_exit:
            if not self.flag_start:
                if self.query_view is not None:
                    # Reset camera and memory
                    self.query_view = None
                    self.keyframes_data = []
                    self.must3r.reset()
                    frame_id = 0
                    improved_kf_count = 0
            else:
                self.query_view, camid = grab_frame(self.camera)
                if self.query_view is None:
                    continue

                # 使用改进的SLAM处理流程
                pts3d, colors, depth, conf, focal, w2c, HW, iskeyframe = self.must3r(self.query_view, frame_id, camid)

                c2w = w2c.inverse()
                cam_centers.append(c2w[:3, -1])

                # Conf thr
                msk = conf > self.viz_conf
                # 修复：使用 .size 替代 .numel()，并确保提取标量值
                if pts3d.size > 0 and msk.sum().item() > 0:
                    pts3d = pts3d[msk.cpu()]
                    colors = colors[0, 0, msk.cpu()]
                else:
                    pts3d = torch.tensor([])
                    colors = torch.tensor([])

                if iskeyframe:
                    self.keyframe_focals.append(focal)
                    self.keyframe_confs.append(conf.mean().cpu())
                    if frame_id > 0:  # 不统计第一帧
                        improved_kf_count += 1

                self.depth_in_color = colorize_depth(depth)
                self.conf_in_color = colorize_depth(conf)
                dtype = o3d.core.float32
                self.pcd_frame = None
                self.frustrum = None

                # 修复：使用 .shape[0] 替代 .shape[0] 的检查，更兼容NumPy和PyTorch
                if len(pts3d) > 0:
                    self.pcd_frame = o3d.cuda.pybind.t.geometry.PointCloud()
                    self.pcd_frame.point.positions = o3d.cuda.pybind.core.Tensor(pts3d, dtype=dtype)
                    self.pcd_frame.point.colors = o3d.cuda.pybind.core.Tensor(colors, dtype=dtype)

                if self.show_cameras:
                    H, W = HW
                    K = np.eye(3)
                    K[0, 0] = K[1, 1] = focal
                    K[0, -1] = W / 2
                    K[1, -1] = H / 2
                    self.frustrum = o3d.geometry.LineSet.create_camera_visualization(
                        W, H, intrinsic=K, extrinsic=w2c.cpu().numpy(), scale=0.075)
                    self.frustrum.paint_uniform_color([0.1, 0.9, 0.1] if iskeyframe else camcols[camid % len(camcols)])

                if iskeyframe:
                    # Move Pointmap and camera to keyframes data
                    self.keyframes_data.append([f'{frame_id}_kpcd', self.pcd_frame])
                    self.keyframes_data.append([f'{frame_id}_kfrustrum', self.frustrum])
                    self.pcd_frame = None
                    self.frustrum = None

                t0, t1 = t1, time.perf_counter()
                ms_per_frame = (t1 - t0) * 1000.0
                fps = 1000 / ms_per_frame if ms_per_frame > 0 else 0
                max_mem = torch.cuda.max_memory_allocated() / MB if torch.cuda.is_available() else 0

                if frame_id % 60 == 0 and frame_id > 0:
                    print(f"帧 {frame_id}: {fps:0.2f} FPS, {ms_per_frame:0.2f} ms/帧, "
                          f"改进关键帧: {improved_kf_count}")

                # Prepare camera centers to display trajectory
                if cam_centers:
                    tempcamc = torch.stack(cam_centers).cpu().numpy()
                    camc_frame = o3d.cuda.pybind.t.geometry.PointCloud()
                    camc_frame.point.positions = o3d.cuda.pybind.core.Tensor(tempcamc, dtype=dtype)
                    camc_frame.point.colors = o3d.cuda.pybind.core.Tensor(np.zeros_like(tempcamc), dtype=dtype)
                else:
                    camc_frame = None

                # Prepare memory map if needed
                if frame_id == 0:
                    mmap = self.must3r.fetch_memory_map(self.viz_conf)
                    if mmap is not None:  # only load memory map at first frame
                        mempts, memcols = mmap
                        memory_map = o3d.cuda.pybind.t.geometry.PointCloud()
                        memory_map.point.positions = o3d.cuda.pybind.core.Tensor(mempts.cpu().numpy(), dtype=dtype)
                        memory_map.point.colors = o3d.cuda.pybind.core.Tensor(memcols.cpu().numpy(), dtype=dtype)
                else:
                    memory_map = None

                focal_el = self.must3r.get_true_focals()[camid] if hasattr(self.must3r, 'get_true_focals') else focal
                if isinstance(focal_el, list):
                    focal_el = focal_el[-1]

                frame_elements = {
                    'color': self.query_view,
                    'depth': self.depth_in_color,
                    'conf': self.conf_in_color,
                    'pcd': self.pcd_frame,
                    'cam_centers': camc_frame,
                    f'frustrum_{camid}': self.frustrum,
                    'keyframes_data': self.keyframes_data,
                    'c2w': c2w.cpu().numpy(),
                    'mem': max_mem,
                    'fps': fps,
                    'focal': focal_el,
                    'num_mem_frames': self.must3r.num_mem_frames if hasattr(self.must3r, 'num_mem_frames') else 0,
                    'memory_map': memory_map,
                    'motion': getattr(self.must3r, 'latest_motion', None),
                }
                self.update_view(frame_elements)

                frame_id += 1

        self.executor.shutdown()
        print("Shutdown")


# GUI和渲染部分保持不变（与之前相同）
class PipelineView:
    """Controls display and user interface. All methods must run in the main thread."""

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, num_sources=1, **callbacks):
        self.vfov = vfov
        self.max_pcd_vertices = max_pcd_vertices

        o3d.visualization.gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "MUSt3R || Improved Keyframe Selection", 1620, 1080)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])

        self.pcd_material = rendering.MaterialRecord()
        self.pcd_material.shader = "defaultUnlit"
        self.pcd_material.point_size = 4

        self.cam_material = rendering.MaterialRecord()
        self.cam_material.shader = "unlitLine"
        self.cam_material.line_width = 4

        self.pcdview = gui.SceneWidget()
        self.window.add_child(self.pcdview)
        self.pcdview.enable_scene_caching(True)
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-30, -30, -30], [30, 30, 30])
        self.reset_view()

        em = self.window.theme.font_size / 2
        self.fps_panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.fps_panel.preferred_width = int(200 * self.window.scaling)
        self.window.add_child(self.fps_panel)
        self.fps = gui.Label("FPS: N/A")
        self.fps_panel.add_child(self.fps)
        self.mem = gui.Label("Mem: N/A")
        self.fps_panel.add_child(self.mem)
        self.focal = gui.Label("Focal: N/A")
        self.fps_panel.add_child(self.focal)
        self.num_mem_frames = gui.Label("Mem frames: N/A")
        self.fps_panel.add_child(self.num_mem_frames)
        self.motion = gui.Label("Motion: N/A")
        self.fps_panel.add_child(self.motion)

        self.panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.panel.preferred_width = int(400 * self.window.scaling)
        self.window.add_child(self.panel)

        toggles = gui.Horiz(em)
        self.panel.add_child(toggles)

        self.flag_followcam = True
        self.toggle_followcam = gui.ToggleSwitch("Follow Cam")
        self.toggle_followcam.is_on = True
        self.toggle_followcam.set_on_clicked(callbacks['on_toggle_followcam'])
        toggles.add_child(self.toggle_followcam)

        self.flag_start = False
        self.toggle_start = gui.ToggleSwitch("Start/Stop")
        self.toggle_start.is_on = False
        self.toggle_start.set_on_clicked(callbacks['on_toggle_start'])
        toggles.add_child(self.toggle_start)

        view_buttons = gui.Horiz(em)
        self.panel.add_child(view_buttons)
        view_buttons.add_stretch()
        reset_view = gui.Button("Reset View")
        reset_view.set_on_clicked(self.reset_view)
        view_buttons.add_child(reset_view)

        self.current_view_viz = 0
        self.num_sources = num_sources
        if self.num_sources > 1:
            self.current_view = gui.Button("Next agent")
            self.current_view.set_on_clicked(self.next_view)
            view_buttons.add_child(self.current_view)

        view_buttons.add_stretch()
        self.video_size = (int(240 * self.window.scaling), int(320 * self.window.scaling), 3)

        self.show_color = gui.CollapsableVert("Video stream")
        self.show_color.set_is_open(True)
        self.panel.add_child(self.show_color)
        self.color_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_color.add_child(self.color_video)

        self.show_depth = gui.CollapsableVert("Predicted Depth")
        self.show_depth.set_is_open(True)
        self.panel.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_depth.add_child(self.depth_video)

        self.show_conf = gui.CollapsableVert("Predicted Confidence")
        self.show_conf.set_is_open(True)
        self.panel.add_child(self.show_conf)
        self.conf_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_conf.add_child(self.conf_video)

        self.status_message = gui.Label("")
        self.panel.add_child(self.status_message)

        self.flag_exit = False
        self.flag_gui_init = False
        self.flag_normals = False

    def next_view(self):
        self.current_view_viz = (self.current_view_viz + 1) % self.num_sources

    def update(self, frame_elements):
        if not self.flag_gui_init:
            self.pcdview.scene.clear_geometry()
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions': o3d.core.Tensor.zeros((self.max_pcd_vertices, 3), o3d.core.Dtype.Float32),
                'colors': o3d.core.Tensor.zeros((self.max_pcd_vertices, 3), o3d.core.Dtype.Float32),
                'normals': o3d.core.Tensor.zeros((self.max_pcd_vertices, 3), o3d.core.Dtype.Float32)
            })
            self.pcd_material.shader = "normals" if self.flag_normals else "defaultUnlit"
            self.pcdview.scene.add_geometry('pcd', dummy_pcd, self.pcd_material)
            self.pcdview.scene.add_geometry('cam_centers', dummy_pcd, self.pcd_material)
            self.pcdview.scene.add_geometry('memory_map', dummy_pcd, self.pcd_material)
            self.flag_gui_init = True

        update_flags = (rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG |
                        (rendering.Scene.UPDATE_NORMALS_FLAG if self.flag_normals else 0))

        def add_or_update_if_needed(tag, data):
            if data is not None:
                always_remove = ['frustrum', 'cam_centers', 'memory_map']
                for toremove in always_remove:
                    if toremove in tag and self.pcdview.scene.has_geometry(tag):
                        self.pcdview.scene.remove_geometry(tag)
                if self.pcdview.scene.has_geometry(tag):
                    self.pcdview.scene.scene.update_geometry(tag, data, update_flags)
                else:
                    material = self.cam_material if 'frustrum' in tag else self.pcd_material
                    self.pcdview.scene.add_geometry(tag, data, material)

        if frame_elements.get('memory_map', None) is not None:
            add_or_update_if_needed('memory_map', frame_elements['memory_map'])

        update_cam = False
        for kk in frame_elements:
            if 'frustrum' in kk:
                update_cam = int(kk.split('_')[1]) == self.current_view_viz
                add_or_update_if_needed(kk, frame_elements[kk])

        add_or_update_if_needed('cam_centers', frame_elements['cam_centers'])

        for kf_key, kf_data in frame_elements['keyframes_data']:
            add_or_update_if_needed(kf_key, kf_data)
            frame_elements['keyframes_data'] = None

        if update_cam:
            add_or_update_if_needed('pcd', frame_elements['pcd'])
            if self.show_color.get_is_open() and 'color' in frame_elements:
                self.color_video.update_image(img2o3d(frame_elements['color']))
            if self.show_depth.get_is_open() and frame_elements.get('depth', None) is not None:
                self.depth_video.update_image(img2o3d(frame_elements['depth']))
            if self.show_conf.get_is_open() and frame_elements.get('conf', None) is not None:
                self.conf_video.update_image(img2o3d(frame_elements['conf']))
            if 'focal' in frame_elements:
                self.focal.text = "Focal: " + f"{frame_elements['focal']:0.2f}"

            if self.flag_followcam:
                self.reset_view(pose=frame_elements['c2w'])

        if 'status_message' in frame_elements:
            self.status_message.text = frame_elements["status_message"]
        if 'fps' in frame_elements:
            self.fps.text = "FPS: " + f"{frame_elements['fps']:0.2f}"
        if 'mem' in frame_elements:
            self.mem.text = "Mem: " + str(int(frame_elements["mem"])) + " MB"
        if 'num_mem_frames' in frame_elements:
            self.num_mem_frames.text = f"Mem frames: {frame_elements['num_mem_frames']}"
        if 'motion' in frame_elements:
            motion = frame_elements['motion']
            if motion is None:
                self.motion.text = "Motion: N/A"
            else:
                self.motion.text = "Motion: " + f"{motion.get('metric', 0.0):0.3f}" + " | " + f"t={motion.get('translation', 0.0):0.3f}" + " | " + f"r={motion.get('rotation', 0.0):0.3f}"

        self.pcdview.force_redraw()

    def reset_view(self, pose=None):
        if pose is None:
            self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
            self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 0, -2.], [0, -1, 0])
        else:
            Rp = pose[:3, :3].T
            center = pose[:3, -1]
            eye = center + np.array([[0, -.6, -1.5]]) @ Rp
            up = np.array([[0, -1, 0]]) @ Rp
            self.pcdview.scene.camera.look_at(center, eye[0], up[0])

    def on_layout(self, layout_context):
        frame = self.window.content_rect
        self.pcdview.frame = frame
        panel_size = self.panel.calc_preferred_size(layout_context, self.panel.Constraints())
        self.panel.frame = gui.Rect(frame.get_right() - panel_size.width, frame.y,
                                    panel_size.width, panel_size.height)
        fps_size = self.fps_panel.calc_preferred_size(layout_context, self.fps_panel.Constraints())
        self.fps_panel.frame = gui.Rect(0, frame.y, fps_size.width, fps_size.height)


# Overall Controller
class PipelineController:
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """

    def __init__(self, args, camera):
        # 关键修复：在创建任何 GUI 元素之前初始化 Open3D 应用
        gui.Application.instance.initialize()

        self.pipeline_model = PipelineModel(args.model,
                                            camera,
                                            self.update_view,
                                            device=args.device,
                                            res=args.res,
                                            show_cameras=not args.hide_cameras,
                                            viz_conf=args.viz_conf,
                                            use_improved_kf_selection=args.use_improved_kf  # 新增参数
                                            )
        self.pipeline_view = PipelineView(
            max_pcd_vertices=self.pipeline_model.max_points,
            num_sources=len(args.input),
            on_window_close=self.on_window_close,
            on_toggle_followcam=self.on_toggle_followcam,
            on_toggle_start=self.on_toggle_start)

        threading.Thread(name='PipelineModel',
                         target=self.pipeline_model.run).start()

        time.sleep(1)
        gui.Application.instance.run()

    def update_view(self, frame_elements):
        """Updates view with new data. May be called from any thread."""
        gui.Application.instance.post_to_main_thread(
            self.pipeline_view.window,
            lambda: self.pipeline_view.update(frame_elements))

    def on_toggle_followcam(self, is_enabled):
        """Callback to toggle display of normals"""
        self.pipeline_view.flag_followcam = is_enabled

    def on_toggle_start(self, is_enabled):
        """Callback to start/stop MUSt3r"""
        self.pipeline_model.flag_start = is_enabled
        self.pipeline_view.flag_start = is_enabled
        self.pipeline_view.flag_gui_init = False

    def on_window_close(self):
        """Callback when the user closes the application window."""
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True  # OK to close window


# MAIN函数
def main():
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chkpt', required=True, help='Path to checkpoint.')
    parser.add_argument('--device', default='cuda:0', help='Device to run on (e.g. "cpu" or "cuda:0").')
    parser.add_argument('--input', default='cam:0', nargs='+',
                        help="Media to load (can be paths to videos or webcam indices like 'cam:0 cam:1').")
    parser.add_argument('--image_string', default=None, type=str,
                        help="In the case of an image collection, string to identify image files.")
    parser.add_argument('--load_memory', default=None, type=str, help="Load memory from another run.")
    parser.add_argument('--output', default=None, type=str, help="Output directory to write predictions")

    # Processing related opts
    parser.add_argument('--res', default=224, choices=[224, 512],
                        type=int, help="Image resolution that works for the model used.")
    parser.add_argument('--skip_every', default=1, type=int, help="Subsample input by skipping frames.")
    parser.add_argument('--rerender', action='store_true', default=False, help="Rerender all frames at the end.")
    parser.add_argument('--rerender_bs', default=64, type=int, help="Re-rendering batch size")
    parser.add_argument('--filter', action='store_true', default=False,
                        help="Minimal Laplacian filtering after rerender.")

    # Hyperparams
    parser.add_argument('--searcher', default="kdtree-scipy-quadrant_x2", type=str,
                        help="Method for overlap prediction")
    parser.add_argument('--overlap_mode', default="nn-norm", type=str,
                        help="How to estimate overlap")
    parser.add_argument('--subsamp', default=2, type=int)
    parser.add_argument('--keyframe_overlap_thr', default=.1, type=float,
                        help="At least this overlap to add incoming image in memory")
    parser.add_argument('--min_conf_keyframe', default=1.2, type=float, help="Ignore 3D points below this confidence.")
    parser.add_argument('--overlap_percentile', default=85., type=float,
                        help="Percentile of image distances to compute overlap")
    parser.add_argument('--varying_focals', action='store_true', default=False,
                        help="Focals may vary along sequence (e.g. zoom-in/out).")

    parser.add_argument('--force_first_keyframes', default=None, type=int)
    parser.add_argument('--num_init_frames', default=2, type=int)

    # GUI related opts
    parser.add_argument('--viz_conf', default=4., type=float, help="Conf threshold for pts3d vizu")
    parser.add_argument('--gui', action='store_true', default=False, help="Show predictions in GUI")
    parser.add_argument('--hide_cameras', action='store_true', default=False)

    # 新增：改进算法开关
    parser.add_argument('--use_improved_kf', action='store_true', default=True,
                        help="Use improved keyframe selection algorithm")

    parser.add_argument('--video_frames', default=0, type=int,
                        help="If input is a video file, uniformly extract this many frames to a timestamp folder and run slam on the extracted images.")
    parser.add_argument('--video_frames_out_root', default="/home/server/Desktop/wangM", type=str,
                        help="Root directory where extracted frame folders will be created.")
    parser.add_argument('--video_frames_jpg_quality', default=95, type=int,
                        help="JPEG quality for extracted frames.")

    args = parser.parse_args()

    toggle_memory_efficient_attention(has_xformers)
    SKIP_EVERY = args.skip_every

    # 创建原始SLAM模型
    original_model = SLAM_MUSt3R(chkpt=args.chkpt,
                                 res=args.res,
                                 kf_x_subsamp=args.subsamp,
                                 searcher=args.searcher,
                                 overlap_mode=args.overlap_mode,
                                 keyframe_overlap_thr=args.keyframe_overlap_thr,
                                 min_conf_keyframe=args.min_conf_keyframe,
                                 overlap_percentile=args.overlap_percentile,
                                 rerender=args.rerender,
                                 keep_memory=args.output is not None,
                                 load_memory=args.load_memory,
                                 fixed_focal=not args.varying_focals,
                                 num_agents=len(args.input),
                                 device=args.device,
                                 num_init_frames=args.num_init_frames)

    args.model = original_model

    if args.video_frames and args.video_frames > 0:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(args.video_frames_out_root, stamp)
        os.makedirs(session_dir, exist_ok=True)
        new_inputs = []
        for inp in args.input:
            if os.path.isfile(inp) and _is_video_file(inp):
                stem = os.path.splitext(os.path.basename(inp))[0] or "video"
                digest = hashlib.md5(inp.encode("utf-8")).hexdigest()[:8]
                out_dir = os.path.join(session_dir, f"{stem}_{digest}")
                _extract_uniform_frames(inp, out_dir, int(args.video_frames), int(args.video_frames_jpg_quality))
                new_inputs.append(out_dir)
            else:
                new_inputs.append(inp)
        args.input = new_inputs

    # Prepare Camera Stream
    CAMERA = AutoMultiLoader(args.input, args.image_string)

    # prepare output
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    if args.gui:
        # Main GUI
        PipelineController(args, CAMERA)
        tolog = {}
    else:
        # Only write output
        assert args.output is not None, "You should define an output folder"
        print(f"开始处理序列，总帧数: {len(CAMERA)}")
        print(f"使用改进的关键帧选择: {args.use_improved_kf}")

        frame, cam_id = grab_frame(CAMERA)
        start = time.time()
        imgHWs = [frame.shape[:2]] if frame is not None else []

        # 使用改进的SLAM包装器
        slam_model = ImprovedSLAMWrapper(original_model, args.use_improved_kf)

        for frame_id in tqdm(range(len(CAMERA) // SKIP_EVERY)):
            out = slam_model(frame, frame_id * SKIP_EVERY, cam_id)
            frame, cam_id = grab_frame(CAMERA)
            if frame is not None:
                imgHWs.append(frame.shape[:2])

        # Re-render if activated
        if args.rerender:
            slam_model.rerender_all_frames(maxbs=args.rerender_bs)

        # Logging FPS and GPU mem usage
        wallclock_time = time.time() - start
        fps = (len(CAMERA) // SKIP_EVERY) / wallclock_time if wallclock_time > 0 else 0
        gpumem = torch.cuda.max_memory_allocated() / MB if torch.cuda.is_available() else 0
        print(f"完成! 平均FPS: {fps:.2f}, GPU内存使用: {gpumem:.2f}MB")
        tolog = {'fps': fps,
                 'gpumem': gpumem,
                 'imgHWs': imgHWs,
                 }

    if args.output is not None:
        # Write full trajectory
        if not args.filter:
            slam_model.write_all_poses(os.path.join(args.output, 'all_poses.npz'), **tolog)
        else:
            # Postprocessing
            filtering_mode = 'laplacian'
            filtering_alpha = .1
            filtering_steps = 256
            outfile = os.path.join(args.output,
                                   f"all_poses{filtering_mode}_{filtering_steps}-steps_{filtering_alpha}-alpha.npz")
            slam_model.write_all_poses(outfile,
                                       filtering_mode=filtering_mode,
                                       filtering_steps=filtering_steps,
                                       filtering_alpha=filtering_alpha, **tolog)

        # Export memory for later use
        outname = os.path.join(args.output, "memory.pkl")
        count = 0
        while args.load_memory == outname:  # make sure you do not overwrite loaded memory file
            outname = os.path.join(args.output, f"memory_{count}.pkl")
        print(f"保存内存到: {outname}")
        slam_model.save_memory(outname)


if __name__ == "__main__":
    main()
