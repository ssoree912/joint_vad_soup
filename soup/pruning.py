from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import prune

PRUNABLE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d)


def _collect_prunable_modules(model: torch.nn.Module) -> Tuple[Tuple[torch.nn.Module, str], ...]:
    module_to_name = {module: name for name, module in model.named_modules()}
    prunable = []
    for module, name in module_to_name.items():
        if isinstance(module, PRUNABLE_TYPES):
            prunable.append((module, name))
    return tuple(prunable)


def magnitude_prune_model(
    model: torch.nn.Module,
    amount: float,
    structured: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Apply magnitude pruning to model and return binary masks.

    Args:
        model: Model to prune in-place.
        amount: Fraction of parameters to prune.
        structured: If True, apply structured pruning (per-channel).

    Returns:
        Dictionary of prune masks keyed by parameter name.
    """
    masks: Dict[str, torch.Tensor] = {}
    if amount <= 0.0:
        return masks

    prunable = _collect_prunable_modules(model)
    module_to_name = {module: name for module, name in prunable}
    parameters_to_prune = [(module, "weight") for module, _ in prunable]

    pruning_method = prune.LnStructured if structured else prune.L1Unstructured
    kwargs = {"amount": amount}
    if structured:
        kwargs["n"] = 2
        kwargs["dim"] = 0

    for module, param_name in parameters_to_prune:
        pruning_method.apply(module, name=param_name, **kwargs)
        mask = getattr(module, f"{param_name}_mask")
        module_name = module_to_name.get(module, module._get_name())
        key = f"{module_name}.{param_name}" if module_name else param_name
        masks[key] = mask.detach().clone()
    return masks


def random_prune_model(model: torch.nn.Module, amount: float) -> Dict[str, torch.Tensor]:
    """Apply random unstructured pruning and return masks."""
    masks: Dict[str, torch.Tensor] = {}
    if amount <= 0.0:
        return masks

    prunable = _collect_prunable_modules(model)
    module_to_name = {module: name for module, name in prunable}
    for module, _ in prunable:
        prune.RandomUnstructured.apply(module, name="weight", amount=amount)
        mask = getattr(module, "weight_mask")
        module_name = module_to_name.get(module, module._get_name())
        key = f"{module_name}.weight" if module_name else "weight"
        masks[key] = mask.detach().clone()
    return masks


def pack_prune_masks(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Collect prune masks registered on a model."""
    masks: Dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        for param_name, tensor in module.named_parameters(recurse=False):
            mask_name = f"{param_name}_mask"
            if hasattr(module, mask_name):
                mask = getattr(module, mask_name)
                masks[f"{module_name}.{param_name}"] = mask.detach().clone()
    return masks


def remove_pruning(model: torch.nn.Module) -> None:
    """Remove pruning re-parametrisation and restore raw parameters."""
    for module in model.modules():
        if isinstance(module, PRUNABLE_TYPES):
            for param_name in ["weight", "bias"]:
                if hasattr(module, f"{param_name}_orig"):
                    prune.remove(module, param_name)
