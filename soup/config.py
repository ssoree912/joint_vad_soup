import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "defaults", "fisher_soup.json"
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge that favors values from `override`."""
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_soup_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the Fisher-soup configuration.

    Args:
        path: Optional path to a JSON file. If omitted, the default config is used.

    Returns:
        Dictionary containing configuration values.
    """
    base = _load_json(DEFAULT_CONFIG_PATH)
    if path is None:
        return base
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Soup config file not found: {path}")
    override = _load_json(path)
    return merge_dicts(base, override)
