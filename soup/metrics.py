from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class MetricBundle:
    raw: Dict[str, float]
    zscores: Dict[str, float] | None = None


def compute_occ_metrics(nll_values: np.ndarray) -> Dict[str, float]:
    nll_values = np.asarray(nll_values)
    nll_mean = float(np.mean(nll_values))
    nll_var = float(np.var(nll_values))
    return {
        "nll_mean": nll_mean,
        "nll_var": nll_var,
        "neg_nll_mean": -nll_mean,
        "neg_nll_var": -nll_var,
    }


def compute_topk_margin(scores: np.ndarray, k: int) -> float:
    scores = np.asarray(scores)
    if scores.ndim == 1:
        sorted_scores = np.sort(scores)
        topk = sorted_scores[-k:]
        bottomk = sorted_scores[:k]
    else:
        sorted_scores = np.sort(scores, axis=-1)
        topk = sorted_scores[..., -k:]
        bottomk = sorted_scores[..., :k]
    return float(np.mean(topk) - np.mean(bottomk))


def compute_entropy(scores: np.ndarray) -> float:
    scores = np.clip(scores, 1e-6, 1 - 1e-6)
    entropy = -scores * np.log(scores) - (1 - scores) * np.log(1 - scores)
    return float(np.mean(entropy))


def compute_weakly_metrics(scores: np.ndarray, topk: int) -> Dict[str, float]:
    margin = compute_topk_margin(scores, topk=topk)
    entropy = compute_entropy(scores)
    return {
        "topk_margin": margin,
        "entropy": entropy,
    }


def compute_consistency_metrics(
    prev_scores: np.ndarray | None,
    current_scores: np.ndarray,
    fixed_index_ratio: float | None = None,
) -> Dict[str, float]:
    if prev_scores is None:
        delta = 0.0
    else:
        prev_scores = np.asarray(prev_scores)
        current_scores = np.asarray(current_scores)
        delta = float(np.mean(np.abs(current_scores - prev_scores)))
    metrics = {"delta_score": delta}
    if fixed_index_ratio is not None:
        metrics["index_retention"] = fixed_index_ratio
    return metrics


def compute_zscores(metric_list: List[Dict[str, float]], keys: Sequence[str]) -> List[Dict[str, float]]:
    stacked = {key: np.array([metrics[key] for metrics in metric_list]) for key in keys}
    zscore_list: List[Dict[str, float]] = []
    for metrics in metric_list:
        z_scores = {}
        for key in keys:
            values = stacked[key]
            mean = values.mean()
            std = values.std()
            if std == 0:
                z = 0.0
            else:
                z = (metrics[key] - mean) / std
            z_scores[key] = float(z)
        zscore_list.append(z_scores)
    return zscore_list
