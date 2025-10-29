"""
Utilities for building Fisher-weighted soups across the Occ (STG-NF) and Weakly (RTFM) models.

This package exposes the `FisherSoupManager` entry point which orchestrates candidate
collection, Fisher-information estimation, coefficient search, pruning, and soup materialisation.
"""

from .manager import FisherSoupManager
from .config import load_soup_config

__all__ = ["FisherSoupManager", "load_soup_config"]
