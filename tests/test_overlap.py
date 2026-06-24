#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-free unit tests for the core scientific contribution.

Run with:  pytest -q   (or)   python tests/test_overlap.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uapf import (  # noqa: E402
    BayesianOverlapModel,
    DecisionHistory,
    KeyframeDecisionFactors,
    adaptive_keyframe_selection,
    adaptive_threshold,
    compute_overlap_and_info_gain,
)


def test_posterior_in_range():
    m = BayesianOverlapModel()
    p = m.posterior(geo_dist=0.1, app_sim=0.9, motion_res=0.05)
    assert 0.0 <= p.mean <= 1.0
    assert p.variance >= 0.0


def test_high_overlap_when_all_cues_agree():
    m = BayesianOverlapModel()
    high = m.posterior(geo_dist=0.05, app_sim=0.95, motion_res=0.02)   # all say "overlap"
    low = m.posterior(geo_dist=5.0, app_sim=0.05, motion_res=3.0)      # all say "no overlap"
    assert high.mean > 0.7
    assert low.mean < 0.3


def test_uncertainty_grows_under_contradiction():
    m = BayesianOverlapModel()
    agree = m.posterior(geo_dist=0.05, app_sim=0.95, motion_res=0.02)
    conflict = m.posterior(geo_dist=5.0, app_sim=0.95, motion_res=0.02)  # geometry disagrees
    # contradictory evidence -> higher posterior variance (uncertainty)
    assert conflict.variance > agree.variance


def test_info_gain_detects_new_region():
    # info_gain (normalised by the median) fires when a *minority* of the frame is
    # much farther than a familiar background -- e.g. a corner appearing in a corridor.
    uniform = np.full(100, 0.05)                                   # nothing new
    new_corner = np.concatenate([np.full(90, 0.05), np.full(10, 2.0)])  # small new region
    _, g_uniform = compute_overlap_and_info_gain(uniform)
    _, g_corner = compute_overlap_and_info_gain(new_corner)
    assert g_corner > g_uniform


def test_adaptive_threshold_relaxes_with_uncertainty():
    hist = [0.5] * 20
    thr_low_unc = adaptive_threshold(hist, uncertainty=0.0)
    thr_high_unc = adaptive_threshold(hist, uncertainty=0.2)
    assert thr_high_unc <= thr_low_unc


def test_forced_insertion_on_tracking_loss():
    hist = DecisionHistory()
    f = KeyframeDecisionFactors(overlap_score=0.9, overlap_uncertainty=0.01,
                                confidence_score=2.0, motion_magnitude=1.0,
                                information_gain=0.0, tracking_quality=0.2)  # tracking lost
    is_kf, reasons, _ = adaptive_keyframe_selection(f, hist, current_frame_id=5)
    assert is_kf is True
    assert any("forced" in r for r in reasons)


def test_low_confidence_rejected():
    hist = DecisionHistory()
    f = KeyframeDecisionFactors(confidence_score=0.1)  # below min_conf_keyframe
    is_kf, reasons, _ = adaptive_keyframe_selection(f, hist, current_frame_id=5)
    assert is_kf is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
