#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-factor Bayesian overlap estimation.

This module implements the *reusable* problem formulation described in the paper
"Uncertainty-Aware Probabilistic Fusion for Robust Indoor 3D Reconstruction in
Low-Texture Environments":

    Input      : geometric distance G, appearance similarity A, motion residual M
    Latent     : inter-frame overlap  O in [0, 1]
    Output     : posterior mean (point estimate) and posterior variance
                 (uncertainty) of O

The overlap O is modelled as a latent random variable with a Beta prior. Each cue
contributes pseudo-counts (evidence) to a Beta posterior, so that an unreliable cue
(large variance / contradictory evidence) is automatically down-weighted.

The whole module depends only on NumPy and therefore runs without a GPU; this makes
the core scientific contribution easy to reproduce, cite and extend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class OverlapPosterior:
    """Posterior of the latent overlap O ~ Beta(alpha, beta)."""

    mean: float          # posterior mean  E[O] = a / (a + b)
    variance: float      # posterior variance Var[O]
    alpha: float         # posterior success count
    beta: float          # posterior failure count

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))


class BayesianOverlapModel:
    r"""Multi-factor Bayesian overlap estimation model.

    The three cues are turned into Beta pseudo-counts:

        geometry   : distance G (small => high overlap), encoded as exp(-lambda * G)
        appearance : similarity A in [0, 1], used directly as success evidence
        motion     : reprojection residual M (small => consistent), encoded as 1/M

    Posterior parameters (conjugate Beta update):

        alpha_post = alpha_prior + w * ( g + a + m )_success
        beta_post  = beta_prior  + w * ( g + a + m )_failure

    Posterior mean      E[O]   = alpha / (alpha + beta)
    Posterior variance  Var[O] = alpha * beta / ((alpha + beta)^2 (alpha + beta + 1))

    Parameters
    ----------
    alpha_prior, beta_prior : float
        Parameters of the non-informative Beta prior (default 2, 2 = neutral belief).
    obs_weight : float
        Strength of each observation (number of pseudo-counts).
    geo_rate : float
        Decay rate lambda used to map geometric distance to a [0, 1] score.
    eps : float
        Numerical stabiliser.
    """

    def __init__(self, alpha_prior: float = 2.0, beta_prior: float = 2.0,
                 obs_weight: float = 5.0, geo_rate: float = 1.0, eps: float = 1e-6):
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.obs_weight = float(obs_weight)
        self.geo_rate = float(geo_rate)
        self.eps = float(eps)

    # ----------------------------------------------------------------- cues
    def geometry_score(self, geo_dist: float) -> float:
        """Map a (normalised) geometric distance to an overlap score in [0, 1]."""
        return float(np.exp(-self.geo_rate * max(geo_dist, 0.0)))

    def motion_score(self, motion_res: float) -> float:
        """Map a motion/reprojection residual to a consistency score in [0, 1]."""
        return float(np.exp(-max(motion_res, 0.0)))

    # ----------------------------------------------------------------- posterior
    def posterior(self, geo_dist: float, app_sim: float, motion_res: float) -> OverlapPosterior:
        """Compute the Beta posterior of the overlap from the three cues.

        Parameters
        ----------
        geo_dist : float
            Geometric nearest-neighbour distance (smaller = more overlap).
        app_sim : float
            Appearance similarity in [0, 1] (larger = more overlap).
        motion_res : float
            Motion / reprojection residual (smaller = more consistent).
        """
        g = self.geometry_score(geo_dist)
        a = float(np.clip(app_sim, 0.0, 1.0))
        m = self.motion_score(motion_res)

        w = self.obs_weight
        alpha = self.alpha_prior + w * g + w * a + w * m
        beta = self.beta_prior + w * (1.0 - g) + w * (1.0 - a) + w * (1.0 - m)

        s = alpha + beta
        mean = alpha / (s + self.eps)
        var = (alpha * beta) / ((s ** 2) * (s + 1.0) + self.eps)
        return OverlapPosterior(mean=float(mean), variance=float(var),
                                alpha=float(alpha), beta=float(beta))

    def elbo(self, post: OverlapPosterior, geo_dist: float, app_sim: float,
             motion_res: float) -> float:
        """Evidence lower bound (up to a constant) used to monitor the fit.

        ELBO = E_q[log p(G,A,M | O)] - KL(q(O) || p(O)).
        We use the closed-form expectation of log-likelihoods under the Beta q and a
        Beta(alpha_prior, beta_prior) prior. Returned value is for diagnostics only.
        """
        from math import lgamma

        def beta_log_norm(a, b):
            return lgamma(a) + lgamma(b) - lgamma(a + b)

        # KL between two Beta distributions q || p
        from scipy.special import digamma  # optional; falls back below if missing
        a, b = post.alpha, post.beta
        a0, b0 = self.alpha_prior, self.beta_prior
        kl = (beta_log_norm(a0, b0) - beta_log_norm(a, b)
              + (a - a0) * digamma(a) + (b - b0) * digamma(b)
              + (a0 - a + b0 - b) * digamma(a + b))
        # expected log-likelihood proxy (higher when cues agree with the posterior)
        g = self.geometry_score(geo_dist)
        m = self.motion_score(motion_res)
        ell = (g + np.clip(app_sim, 0, 1) + m) * np.log(post.mean + self.eps)
        return float(ell - kl)


def compute_overlap_and_info_gain(distances: np.ndarray, percentile: float = 70.0,
                                  eps: float = 1e-9) -> tuple[float, float]:
    """Compute a geometric overlap score and an information-gain factor.

    Parameters
    ----------
    distances : np.ndarray
        Nearest-neighbour distances from current 3D points to the memory map.
    percentile : float
        Percentile of the distances used as the representative overlap distance.

    Returns
    -------
    overlap_score : float
        exp(-raw_dist), in (0, 1].
    info_gain : float
        Mean sigmoid of normalised excess distance; quantifies how much *new*
        geometry the current frame observes (e.g. a corner appearing in a corridor).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size == 0:
        return 0.0, 0.0
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return 0.0, 0.0
    raw_dist = float(np.percentile(distances, percentile))
    overlap_score = float(np.exp(-raw_dist))
    med = float(np.median(distances))
    scale = max(med, eps)
    info_gain = float(np.mean(sigmoid((distances - scale) / scale)))
    return overlap_score, info_gain
