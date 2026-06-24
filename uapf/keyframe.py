#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty-aware adaptive keyframe decision.

Given the posterior overlap (mean + variance) from :mod:`uapf.overlap` and a set of
auxiliary factors (information gain, motion plausibility, tracking quality, temporal
baseline), this module decides whether the current frame should become a keyframe.

The threshold is *adaptive*: it tracks the recent overlap statistics and is relaxed
when the perceptual uncertainty is high, so that noisy overlap estimates in
low-texture regions do not trigger spurious decisions. The module is pure NumPy and
runs without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class KeyframeDecisionFactors:
    """Container of the per-frame factors used by the decision rule."""

    overlap_score: float = 0.0       # posterior mean of overlap
    overlap_uncertainty: float = 0.0  # posterior variance (uncertainty)
    confidence_score: float = 0.0    # mean point-cloud confidence
    motion_magnitude: float = 0.0    # normalised motion metric
    information_gain: float = 0.0
    tracking_quality: float = 1.0
    point_density: float = 0.0
    overlap_mode: str = "nn-norm"


class DecisionHistory:
    """Sliding-window buffer of past decisions (Algorithm 3 in the paper)."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.overlap_scores: List[float] = []
        self.frame_ids: List[int] = []
        self.keyframe_decisions: List[bool] = []
        self.motion_metrics: List[float] = []
        self.tracking_qualities: List[float] = []

    def update(self, overlap_score: float, frame_id: int, was_selected: bool,
               motion_metric: Optional[float] = None,
               tracking_quality: Optional[float] = None) -> None:
        self.overlap_scores.append(overlap_score)
        self.frame_ids.append(frame_id)
        self.keyframe_decisions.append(was_selected)
        if motion_metric is not None:
            self.motion_metrics.append(motion_metric)
        if tracking_quality is not None:
            self.tracking_qualities.append(tracking_quality)
        if len(self.overlap_scores) > self.window_size:
            self.overlap_scores.pop(0)
            self.frame_ids.pop(0)
            self.keyframe_decisions.pop(0)
            if self.motion_metrics:
                self.motion_metrics.pop(0)
            if self.tracking_qualities:
                self.tracking_qualities.pop(0)

    def get_recent_overlap_scores(self, lookback: int = 20) -> List[float]:
        return self.overlap_scores[-lookback:] if len(self.overlap_scores) > lookback \
            else list(self.overlap_scores)

    def get_time_since_last_keyframe(self, current_frame_id: int) -> int:
        if not self.keyframe_decisions:
            return current_frame_id + 1
        for i in range(len(self.keyframe_decisions) - 1, -1, -1):
            if self.keyframe_decisions[i]:
                return current_frame_id - self.frame_ids[i]
        return current_frame_id + 1


def adaptive_threshold(overlap_history: List[float],
                       uncertainty: float = 0.0,
                       k_std: float = 0.5,
                       uncertainty_gain: float = 0.5,
                       default_thr: float = 0.15) -> float:
    """Compute the uncertainty-aware adaptive overlap threshold.

    thr = mean(history) - k_std * std(history) - uncertainty_gain * uncertainty

    The threshold is clamped according to the scene type implied by the mean overlap
    (narrow vs. open scenes), and is *relaxed* (lowered) as perceptual uncertainty
    rises, preventing the decision from being misled by noisy overlap estimates.
    """
    if len(overlap_history) < 10:
        return default_thr
    scores = np.asarray(overlap_history, dtype=np.float64)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    thr = mean_score - k_std * std_score - uncertainty_gain * float(uncertainty)
    if mean_score > 0.7:        # narrow scene -> fewer keyframes
        thr = max(thr, 0.2)
    elif mean_score < 0.3:      # open scene -> more keyframes
        thr = min(thr, 0.1)
    else:
        thr = max(thr, 0.15)
    return float(thr)


def adaptive_keyframe_selection(factors: KeyframeDecisionFactors,
                                history: DecisionHistory,
                                current_frame_id: int,
                                min_conf_keyframe: float = 1.5,
                                info_gain_thr: float = 0.1,
                                force_track_thr: float = 0.5,
                                regular_track_thr: float = 0.7,
                                t_force: int = 30,
                                t_baseline: int = 15,
                                t_min: int = 10) -> Tuple[bool, List[str], int]:
    """Hierarchical, uncertainty-aware keyframe decision (Algorithm 2 in the paper).

    Three prioritised insertion modes are evaluated, from urgent to routine:
      * forced       : tracking quality collapses or the keyframe gap is too large;
      * regular      : low overlap *and* (high information gain or plausible motion);
      * time-baseline: a fixed maximum interval since the last keyframe.

    Returns ``(is_keyframe, reasons, time_since_last_keyframe)``.
    """
    reasons: List[str] = []

    if factors.confidence_score < min_conf_keyframe:
        return False, ["low confidence"], history.get_time_since_last_keyframe(current_frame_id)

    thr = adaptive_threshold(history.get_recent_overlap_scores(),
                             uncertainty=factors.overlap_uncertainty)

    if 'nn' in factors.overlap_mode:
        geo_decision = factors.overlap_score < thr
    else:
        geo_decision = factors.overlap_score > thr

    time_since_last = history.get_time_since_last_keyframe(current_frame_id)
    info_decision = factors.information_gain > info_gain_thr

    if len(history.motion_metrics) >= 10:
        recent = np.asarray(history.motion_metrics[-20:], dtype=np.float64)
        med = float(np.median(recent))
        mad = float(np.median(np.abs(recent - med)))
        low, high = max(0.1, med - 1.5 * mad), med + 1.5 * mad
    else:
        low, high = 0.3, 3.0
    motion_decision = low < factors.motion_magnitude < high

    decision = False
    if factors.tracking_quality < force_track_thr or time_since_last > t_force:
        decision = True
        reasons.append("forced (tracking/time)")
    elif geo_decision and (info_decision or motion_decision):
        decision = True
        reasons.append("regular (low overlap + gain/motion)")
    elif time_since_last > t_baseline:
        decision = True
        reasons.append("time baseline")
    elif factors.confidence_score > 2 * min_conf_keyframe and factors.information_gain > 0.2:
        decision = True
        reasons.append("high confidence + high gain")

    return decision, reasons, time_since_last
