"""Confidence calibration and conformal prediction sets.

A classifier's scores are only useful as probabilities if "0.9" means right
about 90% of the time. Two tools here, both fitted on labelled examples the
model was not trained on:

* temperature scaling rescales the whole score distribution with one number
  `T` (`p_i ** (1/T)`, renormalised): T > 1 softens an over-confident model,
  T < 1 sharpens an under-confident one. It never changes which class wins.
* split conformal prediction turns the calibrated scores into a *set* of
  labels that contains the true one with probability at least `1 - alpha`
  (for data exchangeable with the calibration set).

Everything here is a pure function of arrays so it can be tested on its own.
"""

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-12


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale a probability vector (or rows of a matrix) by `p ** (1/T)`, renormalised."""
    probs = np.clip(np.asarray(probs, dtype=float), EPS, None)
    scaled = np.exp(np.log(probs) / temperature)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, targets: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(targets)), targets])))


def fit_temperature(probs: np.ndarray, targets: Sequence[int],
                    bounds=(0.05, 20.0)) -> float:
    """The temperature minimising the negative log-likelihood of `targets`."""
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if probs.ndim != 2 or len(probs) != len(targets):
        raise ValueError("probs must be (n_samples, n_classes) and match targets")
    if len(targets) < 2:
        raise ValueError("Need at least 2 calibration examples")

    def loss(log_t: float) -> float:
        return negative_log_likelihood(apply_temperature(probs, math.exp(log_t)), targets)

    result = minimize_scalar(loss, bounds=(math.log(bounds[0]), math.log(bounds[1])),
                             method="bounded", options={"xatol": 1e-4})
    return float(math.exp(result.x))


def nonconformity_scores(probs: np.ndarray, targets: Sequence[int]) -> np.ndarray:
    """1 minus the probability given to the true class (higher = more surprising)."""
    probs = np.asarray(probs, dtype=float)
    return 1.0 - probs[np.arange(len(targets)), np.asarray(targets, dtype=int)]


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    """The score quantile that gives `1 - alpha` coverage; infinity when there are too few scores."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1; got {alpha}")
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))        # 1-based rank of the quantile
    if n == 0 or rank > n:
        return float("inf")
    return float(scores[rank - 1])


def expected_calibration_error(probs: np.ndarray, targets: Sequence[int], bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by how many predictions fall in each bin."""
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def summarise(probs: np.ndarray, targets: Sequence[int], bins: int = 10) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    return {
        "n": int(len(targets)),
        "accuracy": float((probs.argmax(axis=1) == targets).mean()),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "ece": expected_calibration_error(probs, targets, bins),
        "nll": negative_log_likelihood(probs, targets),
    }
