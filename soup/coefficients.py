from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .utils import dirichlet_weights, uniform_simplex_grid


@dataclass
class CoefficientCandidate:
    weights: Tuple[float, ...]
    description: str


def generate_coefficients(
    strategy: str,
    num_candidates: int,
    pool_size: int,
    resolution: int | None = None,
    seed: int | None = None,
) -> List[CoefficientCandidate]:
    """
    Sample coefficient vectors based on requested strategy.

    Args:
        strategy: "random", "grid", or "anchor".
        num_candidates: Number of available soup components.
        pool_size: Number of coefficient samples to return.
        resolution: Optional grid resolution (used for grid strategy).
        seed: Optional RNG seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    strategy = strategy.lower()
    samples: Iterable[Tuple[float, ...]]
    if strategy == "random":
        samples = (tuple(dirichlet_weights(num_candidates)) for _ in range(pool_size))
    elif strategy == "grid":
        if resolution is None:
            resolution = max(4, pool_size // 4)
        grid = list(uniform_simplex_grid(num_candidates, resolution))
        if len(grid) > pool_size:
            indices = np.linspace(0, len(grid) - 1, pool_size).astype(int)
            samples = (grid[idx] for idx in indices)
        else:
            samples = (tuple(val) for val in grid)
    elif strategy == "anchor":
        # Anchor strategy keeps one candidate dominant while exploring the rest.
        anchors: List[Tuple[float, ...]] = []
        anchor_weight = 0.6 if num_candidates > 1 else 1.0
        remaining = 1.0 - anchor_weight
        for anchor_idx in range(num_candidates):
            base = [remaining / (num_candidates - 1) if num_candidates > 1 else 0.0] * num_candidates
            base[anchor_idx] = anchor_weight
            anchors.append(tuple(base))
        samples = anchors
    else:
        raise ValueError(f"Unsupported coefficient strategy: {strategy}")

    results: List[CoefficientCandidate] = []
    for idx, weights in enumerate(samples):
        if len(results) >= pool_size:
            break
        desc = f"{strategy}_{idx}"
        results.append(CoefficientCandidate(weights=tuple(weights), description=desc))
    # Always include uniform averaging as fallback.
    uniform = tuple([1.0 / num_candidates] * num_candidates)
    results.append(CoefficientCandidate(weights=uniform, description="uniform"))
    return results
