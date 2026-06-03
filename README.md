# UAPF-Indoor3D

**Uncertainty-Aware Probabilistic Fusion for Robust Indoor 3D Reconstruction in Low-Texture Environments**

1. **Multi-factor Bayesian overlap estimation** — inter-frame overlap is modelled as a latent random variable; geometric distance, appearance similarity and motion consistency are fused through a Beta posterior that returns both a **point estimate** (posterior mean) and an **uncertainty** (posterior variance).
2. **Uncertainty-aware adaptive keyframe decision** — the posterior overlap and its uncertainty drive an adaptive threshold and a hierarchical, multi-criteria keyframe policy that suppresses long-sequence drift while reducing redundant keyframes.

The two core modules (`uapf/overlap.py`, `uapf/keyframe.py`) depend only on **NumPy** and run **without a GPU**, so the scientific contribution can be reproduced, cited and extended independently of the full SLAM stack.

---

## Problem formulation (reusable)

| Symbol | Meaning |
| --- | --- |
| Input | monocular RGB image stream |
| Latent variable | inter-frame overlap `O ∈ [0, 1]` |
| Evidence | geometric distance `G`, appearance similarity `A`, motion residual `M` |
| Output | posterior mean `E[O]` and posterior variance `Var[O]` |
| Decision variables | per-frame binary keyframe insertion / removal |
| Objective | maximise the ELBO of the Bayesian overlap model + minimise the global pose-graph residual |
| Assumptions | predominantly rigid static scene, calibrated pinhole camera, bounded inter-frame motion, dense geometry inferred (not measured) |

This formulation is **dataset- and sensor-agnostic**.

---

## Repository layout

```
UAPF-Indoor3D/
├── uapf/
│   ├── overlap.py        # Bayesian overlap model (NumPy only)  — Algorithm 1
│   ├── keyframe.py       # adaptive keyframe decision (NumPy only) — Algorithms 2 & 3
│   ├── slam_runner.py    # full runnable pipeline (needs torch/open3d/must3r)
│   └── __init__.py
├── scripts/
│   ├── run_slam.py       # CLI entry point for the full pipeline
│   ├── eval.py           # trajectory / reconstruction evaluation helpers
│   └── download_weights.sh
├── configs/
│   └── default.yaml      # all hyper-parameters used in the paper
├── docs/
│   ├── ALGORITHMS.md     # pseudocode for Algorithms 1–3
│   └── DATASET.md        # Weak Texture-Indoor acquisition & benchmark protocol
├── tests/
│   └── test_overlap.py   # GPU-free unit tests for the core contribution
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── NOTICE
```

---

## Installation

```bash
# Python >= 3.9
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The lightweight core (`uapf.overlap`, `uapf.keyframe`) only needs `numpy` (and `scipy`
for the optional ELBO diagnostic). The full pipeline (`uapf/slam_runner.py`)
additionally requires `torch`, `open3d` and the [MUSt3R](https://github.com/naver/must3r)
package, from which the dense geometry backbone and pretrained weights are obtained.

---

## Quick start — core contribution (no GPU)

```python
from uapf import BayesianOverlapModel, KeyframeDecisionFactors, DecisionHistory, adaptive_keyframe_selection

model = BayesianOverlapModel()

# A smooth wall: clear appearance but noisy geometry -> the cues disagree
post = model.posterior(geo_dist=2.0, app_sim=0.9, motion_res=0.1)
print(post.mean, post.std)        # point estimate + uncertainty

# Feed the posterior into the adaptive keyframe decision
hist = DecisionHistory(window_size=50)
f = KeyframeDecisionFactors(overlap_score=post.mean, overlap_uncertainty=post.variance,
                            confidence_score=2.0, motion_magnitude=1.0,
                            information_gain=0.3, tracking_quality=0.9)
is_kf, reasons, gap = adaptive_keyframe_selection(f, hist, current_frame_id=42)
print(is_kf, reasons)
```

Run the unit tests:

```bash
pytest -q            # or: python -m pytest tests/
```

## Full reconstruction pipeline

```bash
# download the MUSt3R/DUSt3R backbone weights first
bash scripts/download_weights.sh

python scripts/run_slam.py \
    --chkpt /path/to/MUSt3R_512.pth \
    --input /path/to/image_folder \
    --output ./results \
    --res 512 --overlap_mode nn-norm --use_improved_kf
```

See `configs/default.yaml` for the full hyper-parameter list used in the paper.

---

## Dataset

The custom **Weak Texture-Indoor** benchmark (12 sequences, 18,720 frames, 6 low-texture
scene types) and its acquisition / evaluation protocol are documented in
[`docs/DATASET.md`](docs/DATASET.md). Raw data is released after anonymisation
(face blurring, de-identification) and ethical review; the benchmark protocol and
metadata are released with this repository.

---

## Citation

If you use this code or the benchmark, please cite (see `CITATION.cff`):

```bibtex
@article{wang2026uapf,
  title   = {Uncertainty-Aware Probabilistic Fusion for Robust Indoor 3D Reconstruction in Low-Texture Environments},
  author  = {Wang, Min and Wang, Xiaogang and Zhang, Liuhong and Li, Haijun},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Code: https://github.com/<anonymised>/UAPF-Indoor3D}
}
```

Upon acceptance, an archived release with a Zenodo **DOI** and a permanent link will be
provided.

---

## License

This project is released under a **Non-Commercial License** inherited from MUSt3R
(Copyright © 2025 NAVER Corporation), because the dense-geometry backbone and pretrained
weights are derived from MUSt3R/DUSt3R. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
For commercial use, please contact the original rights holders.

## Acknowledgements

This work builds on [MUSt3R](https://github.com/naver/must3r) and
[DUSt3R](https://github.com/naver/dust3r). We thank the authors for releasing their code
and models.
