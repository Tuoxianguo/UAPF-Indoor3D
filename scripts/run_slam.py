#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for the full UAPF reconstruction pipeline.

This is a thin wrapper around :func:`uapf.slam_runner.main`, which integrates the
uncertainty-aware overlap estimation and adaptive keyframe decision into the dense
reconstruction backbone.

Example
-------
    python scripts/run_slam.py \
        --chkpt /path/to/MUSt3R_512.pth \
        --input /path/to/image_folder \
        --output ./results \
        --res 512 --overlap_mode nn-norm --use_improved_kf
"""
import os
import sys

# allow `python scripts/run_slam.py` to find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uapf.slam_runner import main  # noqa: E402

if __name__ == "__main__":
    main()
