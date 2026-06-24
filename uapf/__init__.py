#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAPF-Indoor3D: Uncertainty-Aware Probabilistic Fusion for indoor 3D reconstruction.

Public API:
    BayesianOverlapModel, OverlapPosterior, compute_overlap_and_info_gain
    KeyframeDecisionFactors, DecisionHistory, adaptive_keyframe_selection,
    adaptive_threshold
"""
from .overlap import (
    BayesianOverlapModel,
    OverlapPosterior,
    compute_overlap_and_info_gain,
    sigmoid,
)
from .keyframe import (
    KeyframeDecisionFactors,
    DecisionHistory,
    adaptive_keyframe_selection,
    adaptive_threshold,
)

__version__ = "1.0.0"
__all__ = [
    "BayesianOverlapModel",
    "OverlapPosterior",
    "compute_overlap_and_info_gain",
    "sigmoid",
    "KeyframeDecisionFactors",
    "DecisionHistory",
    "adaptive_keyframe_selection",
    "adaptive_threshold",
]
