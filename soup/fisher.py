from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader

from Occ.models.training import compute_loss
from Weakly.train import RTFM_loss, smooth, sparsity


def _zero_fisher_like(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(param, device=param.device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compute_occ_fisher(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    fisher_floor: float,
    normalize: bool,
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Fisher information for the STG-NF model.
    """
    model.eval()
    fisher = _zero_fisher_like(model)
    total_batches = 0
    for batch_idx, data_arr in enumerate(loader):
        if batch_idx >= max_batches:
            break
        data = [d.to(device, non_blocking=True) for d in data_arr]
        score = data[-3].amin(dim=-1)
        label = data[-2]
        weight = data[-1].amin(dim=-1)
        if hasattr(model, "args") and getattr(model.args, "model_confidence", False):
            samp = data[0]
        else:
            samp = data[0][:, :2]

        model.zero_grad(set_to_none=True)
        _, nll = model(samp.float(), label=label, score=score)
        if nll is None:
            continue
        nll = nll * weight
        nll = nll[nll != 0]
        loss = compute_loss(nll, reduction="mean")["total_loss"]
        loss.backward()
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            fisher[name] += param.grad.detach() ** 2
        total_batches += 1

    total_batches = max(total_batches, 1)
    for name in fisher:
        fisher[name] = torch.clamp(fisher[name] / total_batches, min=fisher_floor)
        if normalize:
            norm = torch.norm(fisher[name])
            if norm > 0:
                fisher[name] = fisher[name] / norm
    return fisher


def _iterate_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def compute_weakly_fisher(
    model: torch.nn.Module,
    normal_loader: DataLoader,
    abnormal_loader: DataLoader,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    fisher_floor: float,
    normalize: bool,
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Fisher information for the RTFM model.
    """
    model.train()
    fisher = _zero_fisher_like(model)
    total_batches = 0
    normal_iter = _iterate_loader(normal_loader)
    abnormal_iter = _iterate_loader(abnormal_loader)
    loss_fn = RTFM_loss(alpha=0.0001, margin=100)

    for batch_idx in range(max_batches):
        ninput, nlabel = next(normal_iter)
        ainput, alabel = next(abnormal_iter)
        input_tensor = torch.cat((ninput, ainput), 0).to(device)
        nlabel = nlabel[:batch_size].to(device)
        alabel = alabel[:batch_size].to(device)

        (
            score_abnormal,
            score_normal,
            feat_select_abn,
            feat_select_normal,
            feat_abn_bottom,
            feat_normal_bottom,
            scores,
            scores_nor_bottom,
            scores_nor_abn_bag,
            _,
        ) = model(input_tensor)

        scores = scores.view(batch_size * 16 * 2, -1).squeeze()
        abn_scores = scores[batch_size * 16 :]

        model.zero_grad(set_to_none=True)
        loss_sparse = sparsity(abn_scores, batch_size, lamda2=8e-3)
        loss_smooth = smooth(abn_scores, lamda1=8e-4)
        loss_cls = loss_fn(
            score_normal,
            score_abnormal,
            nlabel,
            alabel,
            feat_select_normal,
            feat_select_abn,
            feat_abn_bottom,
            device=device,
        )
        loss = loss_cls + loss_smooth + loss_sparse
        loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            fisher[name] += param.grad.detach() ** 2
        total_batches += 1

    total_batches = max(total_batches, 1)
    for name in fisher:
        fisher[name] = torch.clamp(fisher[name] / total_batches, min=fisher_floor)
        if normalize:
            norm = torch.norm(fisher[name])
            if norm > 0:
                fisher[name] = fisher[name] / norm
    return fisher
