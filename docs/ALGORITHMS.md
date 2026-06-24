# Algorithms

Pseudocode for the three core algorithms. They correspond one-to-one to the equations
in Section 3 of the paper and to the implementation in `uapf/overlap.py` and
`uapf/keyframe.py`.

## Symbol table

| Symbol | Meaning |
| --- | --- |
| `O` | latent inter-frame overlap, `O ∈ [0, 1]` |
| `G`, `A`, `M` | geometric distance / appearance similarity / motion residual |
| `α, β` | Beta posterior success / failure pseudo-counts |
| `E[O]`, `Var[O]` | posterior mean (point estimate) / variance (uncertainty) |
| `thr` | adaptive overlap threshold |
| `Δt` | frames since the last keyframe |

---

## Algorithm 1 — Multi-factor Bayesian overlap estimation

```
Input : pts3d, conf, depth, memory_tree, cam_center, gdesc, memdescs,
        mode, min_conf, percentile, eps
Output: overlap_mean, overlap_variance, info_gain

function MF_BAYESIAN_OVERLAP(...):
    msk  <- conf > min_conf
    geo_score, info_gain <- 0, 0
    if sum(msk) > 0:
        dists <- memory_tree.query(pts3d[msk], cam_center)
        if "norm" in mode: dists <- dists / (depth[msk] + eps)
        dists[dists == +inf] <- max_finite(dists)
        raw_dist  <- percentile(dists, percentile)
        geo_score <- exp(-raw_dist)                         # geometry cue
        med       <- median(dists)
        info_gain <- mean(sigmoid((dists - med) / (med + eps)))

    app_score <- max_j cosine(gdesc, memdescs[j])  if memdescs else 0   # appearance cue
    motion_score <- exp(-motion_residual)                                # motion cue

    # Beta posterior (conjugate update); w = obs_weight
    α <- α_prior + w*geo_score + w*app_score + w*motion_score
    β <- β_prior + w*(1-geo_score) + w*(1-app_score) + w*(1-motion_score)
    s <- α + β
    overlap_mean     <- α / s
    overlap_variance <- α*β / (s^2 * (s + 1))               # uncertainty
    return overlap_mean, overlap_variance, info_gain
```

When evidence is **contradictory** (e.g. high appearance similarity but large
geometric distance) the pseudo-counts split between `α` and `β`, so `Var[O]` grows —
this rising uncertainty is what relaxes the keyframe threshold in Algorithm 2.

---

## Algorithm 2 — Uncertainty-aware adaptive keyframe decision

```
Input : factors (overlap_mean, overlap_variance, conf, motion, info_gain, track_q),
        history, frame_id, min_conf
Output: is_keyframe, reasons, Δt

function ADAPTIVE_KF(factors, history, frame_id):
    if factors.conf < min_conf: return False, ["low conf"], history.gap(frame_id)

    function ADAPTIVE_THR(hist, unc):
        if len(hist) < 10: return 0.15
        μ, σ <- mean(hist), std(hist)
        thr  <- μ - 0.5σ - 0.5*unc                 # relax under high uncertainty
        if   μ > 0.7: thr <- max(thr, 0.2)         # narrow scene
        elif μ < 0.3: thr <- min(thr, 0.1)         # open scene
        else:         thr <- max(thr, 0.15)
        return thr

    thr   <- ADAPTIVE_THR(history.recent_overlap(), factors.overlap_variance)
    Δt    <- history.gap(frame_id)
    geo   <- factors.overlap_mean < thr            # (nn mode)
    info  <- factors.info_gain > 0.1
    [low, high] <- robust_motion_band(history.motion)   # median ± 1.5*MAD
    mot   <- low < factors.motion < high

    if   factors.track_q < 0.5 or Δt > 30: is_kf <- True; reasons += "forced"
    elif geo and (info or mot):            is_kf <- True; reasons += "regular"
    elif Δt > 15:                          is_kf <- True; reasons += "time baseline"
    elif factors.conf > 2*min_conf and factors.info_gain > 0.2:
                                           is_kf <- True; reasons += "high conf"
    else:                                  is_kf <- False

    history.update(factors.overlap_mean, frame_id, is_kf, factors.motion, factors.track_q)
    return is_kf, reasons, Δt
```

---

## Algorithm 3 — Decision history buffer

```
class DecisionHistory(window_size = 50):
    overlaps, frame_ids, kf_flags, motions, tracking_q = [], [], [], [], []

    UPDATE(overlap, fid, is_kf, motion, track_q):
        append all; pop(0) from each list when len > window_size

    GET_RECENT_OVERLAP(n = 20): return overlaps[-n:]

    TIME_SINCE_LAST_KF(curr_fid):
        scan kf_flags backwards; return curr_fid - fid of last True (else curr_fid+1)
```
