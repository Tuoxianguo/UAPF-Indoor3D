# Weak Texture-Indoor benchmark

A custom dataset targeting the failure modes of indoor 3D reconstruction in
low-texture environments.

## Acquisition protocol

| Item | Specification |
| --- | --- |
| Sensor | Intel RealSense D435i (RGB-D) |
| Resolution | 640 × 480 |
| Frame rate | 30 FPS |
| Calibration | factory + hand-eye calibration with the **Kalibr** toolbox (intrinsics + RGB↔depth extrinsics) |
| Synchronisation | hardware-synchronised RGB and depth streams |

> Note on modality: depth is recorded for **ground-truth / evaluation and metric scale
> alignment only**. The proposed pipeline consumes **RGB only** at online inference
> (front-end reconstruction, overlap estimation and keyframe selection).

## Scene statistics

12 sequences, **18,720 frames** in total, covering 6 typical low-texture scenarios:

| # | Scenario | Sequences | Frames |
| --- | --- | --- | --- |
| 1 | Pure white-wall corridor | 3 | 4,560 |
| 2 | Repetitive tile-floor room | 2 | 3,120 |
| 3 | Smooth concrete stairwell | 2 | 2,880 |
| 4 | Empty office with plain desks | 2 | 3,240 |
| 5 | Reflective glass-wall corridor | 2 | 2,760 |
| 6 | Untextured warehouse storage | 1 | 2,160 |

## Texture-level annotation criteria

Each frame region is labelled into three texture levels using the local gradient /
feature density:

* **High** — rich keypoints, dense matchable descriptors;
* **Medium** — sparse but usable features;
* **Low** — near-featureless surfaces (plain walls, smooth floors, glass).

Sequences are curated so that low-texture regions dominate (≈ 2:1 over textured
regions) to focus on the core research problem.

## Ground-truth generation

* Trajectory ground truth: RGB-D fused reconstruction refined with COLMAP /
  motion-capture where available.
* Surface ground truth: TSDF fusion of the depth stream at high confidence.

## Train / test split

* **Train / validation**: 8 sequences (used only for hyper-parameter selection).
* **Test**: 4 held-out sequences (one per dominant scenario), reported in the paper.
* The exact split file (`splits.json`) is shipped with the release.

## Ethics & privacy

* Sequences are recorded in non-private indoor spaces with permission.
* Any incidental persons are removed or face-blurred; all metadata is de-identified.
* Raw data is released **after** anonymisation and ethical review; the benchmark
  protocol, calibration parameters, sample frames and evaluation scripts are released
  with this repository.

## Evaluation metrics

ATE, RPE, Chamfer distance, reconstruction F1, low-texture reconstruction coverage
(LTRC), tracking success rate, long-sequence drift, and loop-closure precision/recall.
See `scripts/eval.py`.
