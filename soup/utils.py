from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


TensorDict = Dict[str, torch.Tensor]


@dataclass
class CandidateState:
    """Container holding candidate metadata for soup building."""

    name: str
    state_dict: TensorDict
    fisher: TensorDict
    metrics: Dict[str, float]
    path: str
    prune_masks: TensorDict | None = None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict(state_dict: TensorDict) -> TensorDict:
    return {k: v.clone() for k, v in state_dict.items()}


def combine_state_dicts(
    candidates: Sequence[CandidateState],
    alphas: Sequence[float],
) -> TensorDict:
    """
    Linearly combine candidate parameters using weights alphas.

    Args:
        candidates: CandidateState sequence.
        alphas: Weight vector (must sum to 1).

    Returns:
        Combined state_dict.
    """
    if not math.isclose(sum(alphas), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Soup weights must sum to 1.")
    combined: TensorDict | None = None
    for candidate, alpha in zip(candidates, alphas):
        if combined is None:
            combined = {k: alpha * v for k, v in candidate.state_dict.items()}
        else:
            for key, value in candidate.state_dict.items():
                combined[key] += alpha * value
    return combined or {}


def apply_prune_mask(
    state_dict: TensorDict, prune_masks: TensorDict | None
) -> TensorDict:
    """Zero-out parameters according to stored prune masks."""
    if prune_masks is None:
        return state_dict
    masked = {}
    for key, tensor in state_dict.items():
        mask = prune_masks.get(key)
        if mask is None:
            masked[key] = tensor
        else:
            masked[key] = tensor * mask
    return masked


def fisher_inner_product(fisher_a: TensorDict, fisher_b: TensorDict) -> float:
    product = 0.0
    for key in fisher_a.keys():
        if key not in fisher_b:
            continue
        product += torch.sum(fisher_a[key] * fisher_b[key]).item()
    return product


def normalize_fisher(fisher: TensorDict) -> TensorDict:
    normalized: TensorDict = {}
    for key, tensor in fisher.items():
        denom = torch.norm(tensor)
        if denom > 0:
            normalized[key] = tensor / denom
        else:
            normalized[key] = tensor
    return normalized


def dirichlet_weights(num_candidates: int) -> List[float]:
    sample = np.random.dirichlet(np.ones(num_candidates))
    return sample.tolist()


def uniform_simplex_grid(num_candidates: int, resolution: int) -> Iterable[Tuple[float, ...]]:
    """
    Generate a coarse grid over the probability simplex.

    Args:
        num_candidates: Number of soup components.
        resolution: Grid resolution (larger => finer grid).
    """
    if num_candidates == 1:
        yield (1.0,)
        return

    def recurse(prefix: List[float], remaining: int, total: int) -> Iterable[Tuple[float, ...]]:
        if remaining == 1:
            yield tuple(prefix + [total])
            return
        for i in range(total + 1):
            yield from recurse(prefix + [i], remaining - 1, total - i)

    for raw in recurse([], num_candidates, resolution):
        weights = [value / resolution for value in raw]
        yield tuple(weights)
