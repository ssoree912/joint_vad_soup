from __future__ import annotations

import os
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_curve

from Occ.train_eval import Occ, prepare_occ_trainer
from Occ.utils.scoring_utils import score_dataset
from Occ.utils.training_scoring_utils import training_score_dataset
from Weakly.train_eval import Weakly, Weakly_update, prepare_weakly_environment
from Weakly.test_10crop import test as weakly_test

from .coefficients import CoefficientCandidate, generate_coefficients
from .fisher import compute_occ_fisher, compute_weakly_fisher
from .metrics import (
    MetricBundle,
    compute_consistency_metrics,
    compute_occ_metrics,
    compute_weakly_metrics,
    compute_zscores,
)
from .pruning import magnitude_prune_model, random_prune_model, pack_prune_masks, remove_pruning
from .utils import CandidateState, apply_prune_mask, combine_state_dicts, set_random_seed


@dataclass
class SoupSelection:
    coefficients: CoefficientCandidate
    metrics: MetricBundle
    state_dict: Dict[str, torch.Tensor]
    training_scores: np.ndarray
    nll_values: np.ndarray | None = None
    weakly_scores: np.ndarray | None = None
    roc_scores: np.ndarray | None = None
    roc_gt: np.ndarray | None = None


class FisherSoupManager:
    """
    Orchestrates Fisher-weighted soup construction for Occ and Weakly models.
    """

    def __init__(self, config: Dict, writer=None):
        self.config = config or {}
        self.writer = writer
        self.enabled = bool(self.config)
        self.prev_occ_scores: np.ndarray | None = None
        self.prev_weakly_scores: np.ndarray | None = None
        self.round_index = 0

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def run_occ_round(
        self,
        args,
        dataset,
        pseudo_weight,
        dir_best_record: str,
        load: bool,
        writer,
        steps: int,
        round_idx: int,
    ):
        if not self.enabled:
            scores, auc, info = Occ(
                Occ_args=args,
                dataset=dataset,
                pseudo_weight=pseudo_weight,
                dir_best_record=dir_best_record,
                load=load,
                writer=writer,
                steps=steps,
            )
            self.prev_occ_scores = scores
            return scores, auc, info

        occ_cfg = self.config.get("occ", {})
        seeds = occ_cfg.get("candidate_seeds", [args.seed])
        prune_ratio = self._resolve_prune_ratio(model_type="occ", round_idx=round_idx)

        candidate_states: List[CandidateState] = []
        candidate_dirs: List[str] = []
        candidate_metrics: List[Dict[str, float]] = []

        for candidate_idx, seed in enumerate(seeds):
            candidate_dir = self._candidate_dir(dir_best_record, "occ", seed)
            os.makedirs(candidate_dir, exist_ok=True)

            candidate_args = deepcopy(args)
            candidate_args.seed = seed
            set_random_seed(seed)

            _, _, details = Occ(
                Occ_args=candidate_args,
                dataset=dataset,
                pseudo_weight=pseudo_weight,
                dir_best_record=candidate_dir,
                load=False,
                writer=writer,
                steps=steps,
            )

            state = self._prepare_occ_candidate(
                args=candidate_args,
                dataset=dataset,
                candidate_dir=candidate_dir,
                prune_ratio=prune_ratio,
                candidate_idx=candidate_idx,
                pseudo_weight=pseudo_weight,
            )
            candidate_states.append(state)
            candidate_dirs.append(candidate_dir)
            candidate_metrics.append(state.metrics)

        selection = self._select_occ_soup(
            args=args,
            dataset=dataset,
            candidates=candidate_states,
            prune_ratio=prune_ratio,
            round_idx=round_idx,
        )

        final_checkpoint = os.path.join(dir_best_record, "Occ_epoch_final_checkpoint.pth.tar")
        self._persist_occ_state(selection.state_dict, candidate_dirs[0], final_checkpoint)

        # Final evaluation with soup weights
        metrics, eval_scores, nll_values, roc_info = self._evaluate_occ_state(
            args=args,
            dataset=dataset,
            state_dict=selection.state_dict,
            collect_roc=True,
        )
        self.prev_occ_scores = eval_scores
        auc = roc_info.get("auc", 0.0)

        info = {
            "nll_values": nll_values,
            "roc_scores": roc_info["scores"],
            "roc_gt": roc_info["gt"],
            "checkpoint_path": final_checkpoint,
            "metrics": metrics.raw,
        }
        self._log_metrics("occ", metrics, round_idx)
        self._maybe_save_roc("occ", round_idx, roc_info)
        return eval_scores, auc, info

    def run_weakly_round(
        self,
        args,
        pseudo_idx,
        dir_best_record: str,
        load: bool,
        writer,
        steps: int,
        round_idx: int,
    ):
        if not self.enabled:
            scores, auc, info = Weakly(
                Weakly_args=args,
                pseudo_idx=pseudo_idx,
                dir_best_record=dir_best_record,
                load=load,
                writer=writer,
                steps=steps,
            )
            self.prev_weakly_scores = scores
            return scores, auc, info

        weakly_cfg = self.config.get("weakly", {})
        seeds = weakly_cfg.get("candidate_seeds", [args.seed])
        prune_ratio = self._resolve_prune_ratio(model_type="weakly", round_idx=round_idx)

        candidate_states: List[CandidateState] = []
        candidate_dirs: List[str] = []

        for candidate_idx, seed in enumerate(seeds):
            candidate_dir = self._candidate_dir(dir_best_record, "weakly", seed)
            os.makedirs(candidate_dir, exist_ok=True)

            candidate_args = deepcopy(args)
            candidate_args.seed = seed
            set_random_seed(seed)

            scores, auc, details = Weakly(
                Weakly_args=candidate_args,
                pseudo_idx=pseudo_idx,
                dir_best_record=candidate_dir,
                load=False,
                writer=writer,
                steps=steps,
            )

            state = self._prepare_weakly_candidate(
                args=candidate_args,
                candidate_dir=candidate_dir,
                prune_ratio=prune_ratio,
                candidate_idx=candidate_idx,
                pseudo_idx=pseudo_idx,
            )
            candidate_states.append(state)
            candidate_dirs.append(candidate_dir)

        selection = self._select_weakly_soup(
            args=args,
            pseudo_idx=pseudo_idx,
            candidates=candidate_states,
            prune_ratio=prune_ratio,
            round_idx=round_idx,
        )

        final_checkpoint = os.path.join(dir_best_record, "Weakly_epoch_final_checkpoint.pkl")
        self._persist_weakly_state(selection.state_dict, candidate_dirs[0], final_checkpoint)

        config, device, model, loaders = prepare_weakly_environment(args, pseudo_idx=pseudo_idx)
        state_dict = torch.load(final_checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(device)

        auc, fpr, tpr = weakly_test(loaders["test"], model, args, device, return_curve=True)
        scores = Weakly_update("", args, device, update=True, state_dict=state_dict)

        prev_scores = self.prev_weakly_scores
        consistency = compute_consistency_metrics(prev_scores, scores)
        final_metrics_raw = compute_weakly_metrics(scores, topk=self.config.get("weakly", {}).get("topk", 8))
        final_metrics_raw.update(consistency)
        final_metrics_raw["neg_entropy"] = -final_metrics_raw["entropy"]
        metrics_bundle = MetricBundle(raw=final_metrics_raw)

        details = {
            "checkpoint_path": final_checkpoint,
            "metrics": final_metrics_raw,
            "auc_curve": {"auc": auc, "fpr": fpr, "tpr": tpr},
        }
        self._log_metrics("weakly", metrics_bundle, round_idx)
        self.prev_weakly_scores = scores
        self._maybe_save_roc("weakly", round_idx, details["auc_curve"])
        return scores, auc, details

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _resolve_prune_ratio(self, model_type: str, round_idx: int) -> float:
        variant = self.config.get("experiment_variant", "baseline").lower()
        if variant == "baseline":
            return 0.0
        if variant == "pruning":
            pruning_cfg = self.config.get(model_type, {}).get("pruning", {})
            mag = pruning_cfg.get("magnitude_ratio") or 0.0
            rand = pruning_cfg.get("random_ratio", 0.0)
            return max(mag, rand)
        if variant == "mixed":
            schedule = self.config.get("mixed_schedule", {}).get(model_type, [])
            if not schedule:
                return 0.0
            if round_idx < len(schedule):
                return schedule[round_idx]
            return schedule[-1]
        return 0.0

    def _candidate_dir(self, base_dir: str, model_type: str, seed: int) -> str:
        return os.path.join(base_dir, "soup_candidates", model_type, f"seed_{seed}")

    def _prepare_occ_candidate(
        self,
        args,
        dataset,
        candidate_dir: str,
        prune_ratio: float,
        candidate_idx: int,
        pseudo_weight,
    ) -> CandidateState:
        occ_cfg = self.config.get("occ", {})
        args_clone = deepcopy(args)
        args_clone.seed = getattr(args, "seed", 0)
        _, trainer, _ = prepare_occ_trainer(
            Occ_args=args_clone,
            dataset=dataset,
            pseudo_weight=pseudo_weight,
            dir_best_record=candidate_dir,
            load=True,
        )

        model = trainer.model
        prune_cfg = occ_cfg.get("pruning", {})
        mag_ratio_cfg = prune_cfg.get("magnitude_ratio")
        rand_ratio = prune_cfg.get("random_ratio", 0.0)
        if prune_ratio is None or prune_ratio <= 0:
            mag_ratio = 0.0
            rand_ratio = 0.0
        else:
            mag_ratio = mag_ratio_cfg if mag_ratio_cfg is not None else prune_ratio
        structured = prune_cfg.get("mode", "unstructured") != "unstructured"
        prune_masks = None
        pruning_applied = False
        if mag_ratio and mag_ratio > 0:
            mag_seed = self._resolve_prune_seed(
                pruning_cfg=prune_cfg,
                base_seed=args_clone.seed,
                candidate_idx=candidate_idx,
                kind="magnitude",
            )
            if mag_seed is not None:
                set_random_seed(mag_seed)
            magnitude_prune_model(model, amount=mag_ratio, structured=structured)
            pruning_applied = True
        if rand_ratio and rand_ratio > 0:
            rand_seed = self._resolve_prune_seed(
                pruning_cfg=prune_cfg,
                base_seed=args_clone.seed,
                candidate_idx=candidate_idx,
                kind="random",
            )
            if rand_seed is not None:
                set_random_seed(rand_seed)
            random_prune_model(model, amount=rand_ratio)
            pruning_applied = True

        if pruning_applied:
            prune_masks = {k: v.detach().cpu() for k, v in pack_prune_masks(model).items()}
            remove_pruning(model)
            set_random_seed(args_clone.seed)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        fisher = compute_occ_fisher(
            model=model,
            loader=trainer.train_loader,
            device=device,
            max_batches=occ_cfg.get("max_batches", 80),
            fisher_floor=occ_cfg.get("fisher_floor", 1e-6),
            normalize=occ_cfg.get("normalize_fisher", True),
        )
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        metrics = compute_occ_metrics(trainer.collect_nll_distribution())
        return CandidateState(
            name=os.path.basename(candidate_dir),
            state_dict=state_dict,
            fisher={k: v.detach().cpu() for k, v in fisher.items()},
            metrics=metrics,
            path=os.path.join(candidate_dir, "Occ_epoch_final_checkpoint.pth.tar"),
            prune_masks={k: v.detach().cpu() for k, v in (prune_masks or {}).items()},
        )

    def _prepare_weakly_candidate(
        self,
        args,
        candidate_dir: str,
        prune_ratio: float,
        candidate_idx: int,
        pseudo_idx,
    ) -> CandidateState:
        weakly_cfg = self.config.get("weakly", {})
        config, device, model, loaders = prepare_weakly_environment(args, pseudo_idx=pseudo_idx)

        checkpoint = torch.load(os.path.join(candidate_dir, "Weakly_epoch_final_checkpoint.pkl"))
        model.load_state_dict(checkpoint)
        prune_cfg = weakly_cfg.get("pruning", {})
        mag_ratio_cfg = prune_cfg.get("magnitude_ratio")
        rand_ratio = prune_cfg.get("random_ratio", 0.0)
        if prune_ratio is None or prune_ratio <= 0:
            mag_ratio = 0.0
            rand_ratio = 0.0
        else:
            mag_ratio = mag_ratio_cfg if mag_ratio_cfg is not None else prune_ratio
        structured = prune_cfg.get("mode", "unstructured") != "unstructured"
        prune_masks = None
        pruning_applied = False
        if mag_ratio and mag_ratio > 0:
            mag_seed = self._resolve_prune_seed(
                pruning_cfg=prune_cfg,
                base_seed=args.seed,
                candidate_idx=candidate_idx,
                kind="magnitude",
            )
            if mag_seed is not None:
                set_random_seed(mag_seed)
            magnitude_prune_model(model, amount=mag_ratio, structured=structured)
            pruning_applied = True
        if rand_ratio and rand_ratio > 0:
            rand_seed = self._resolve_prune_seed(
                pruning_cfg=prune_cfg,
                base_seed=args.seed,
                candidate_idx=candidate_idx,
                kind="random",
            )
            if rand_seed is not None:
                set_random_seed(rand_seed)
            random_prune_model(model, amount=rand_ratio)
            pruning_applied = True

        if pruning_applied:
            prune_masks = {k: v.detach().cpu() for k, v in pack_prune_masks(model).items()}
            remove_pruning(model)
            set_random_seed(args.seed)

        model = model.to(device)
        fisher = compute_weakly_fisher(
            model=model,
            normal_loader=loaders["train_normal"],
            abnormal_loader=loaders["train_abnormal"],
            batch_size=args.Weakly_batch_size,
            device=device,
            max_batches=weakly_cfg.get("max_batches", 80),
            fisher_floor=weakly_cfg.get("fisher_floor", 1e-6),
            normalize=weakly_cfg.get("normalize_fisher", True),
        )

        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        pred_scores = Weakly_update(
            checkpoint_path="",
            args=args,
            device=device,
            state_dict=state_dict,
        )
        metrics = compute_weakly_metrics(
            pred_scores,
            topk=weakly_cfg.get("topk", 8),
        )
        return CandidateState(
            name=os.path.basename(candidate_dir),
            state_dict=state_dict,
            fisher={k: v.detach().cpu() for k, v in fisher.items()},
            metrics=metrics,
            path=os.path.join(candidate_dir, "Weakly_epoch_final_checkpoint.pkl"),
            prune_masks={k: v.detach().cpu() for k, v in (prune_masks or {}).items()},
        )

    def _select_occ_soup(
        self,
        args,
        dataset,
        candidates: Sequence[CandidateState],
        prune_ratio: float,
        round_idx: int,
    ) -> SoupSelection:
        occ_cfg = self.config.get("occ", {})
        coeff_candidates = generate_coefficients(
            strategy=occ_cfg.get("coefficient_strategy", "random"),
            num_candidates=len(candidates),
            pool_size=occ_cfg.get("n_coefficient_candidates", 40),
            resolution=occ_cfg.get("grid_resolution"),
            seed=self.config.get("seed"),
        )

        metric_list: List[Dict[str, float]] = []
        selections: List[SoupSelection] = []

        for coeff in coeff_candidates:
            soup_state = combine_state_dicts(candidates, coeff.weights)
            if prune_ratio > 0:
                soup_state = apply_prune_mask(
                    soup_state,
                    self._blend_masks(candidates, coeff.weights, model_type="occ"),
                )

            metrics, training_scores, nll_values, roc_info = self._evaluate_occ_state(
                args=args,
                dataset=dataset,
                state_dict=soup_state,
                collect_roc=False,
            )
            metric_list.append(metrics.raw)
            selections.append(
                SoupSelection(
                    coefficients=coeff,
                    metrics=metrics,
                    state_dict=soup_state,
                    training_scores=training_scores,
                    nll_values=nll_values,
                )
            )

        metric_keys = ["neg_nll_mean", "neg_nll_var", "delta_score"]
        zscores = compute_zscores(metric_list, metric_keys)
        lambdas = self.config.get("objective", {}).get(
            "lambdas", {"margin": 0.7, "entropy": 0.7, "consistency": 0.7}
        )
        lambda_consistency = lambdas.get("consistency", 0.7)

        best_idx = 0
        best_value = -float("inf")
        for idx, selection in enumerate(selections):
            selection.metrics.zscores = zscores[idx]
            z = selection.metrics.zscores
            score = z["neg_nll_mean"] + z["neg_nll_var"] - lambda_consistency * z["delta_score"]
            if score > best_value:
                best_value = score
                best_idx = idx

        chosen = selections[best_idx]
        # recompute metrics with roc info for chosen soup
        metrics, training_scores, nll_values, roc_info = self._evaluate_occ_state(
            args=args,
            dataset=dataset,
            state_dict=chosen.state_dict,
            collect_roc=True,
        )
        chosen.metrics = metrics
        chosen.training_scores = training_scores
        chosen.nll_values = nll_values
        chosen.roc_scores = roc_info["scores"]
        chosen.roc_gt = roc_info["gt"]
        return chosen

    def _select_weakly_soup(
        self,
        args,
        pseudo_idx,
        candidates: Sequence[CandidateState],
        prune_ratio: float,
        round_idx: int,
    ) -> SoupSelection:
        weakly_cfg = self.config.get("weakly", {})
        coeff_candidates = generate_coefficients(
            strategy=weakly_cfg.get("coefficient_strategy", "random"),
            num_candidates=len(candidates),
            pool_size=weakly_cfg.get("n_coefficient_candidates", 40),
            resolution=weakly_cfg.get("grid_resolution"),
            seed=self.config.get("seed"),
        )

        metric_list: List[Dict[str, float]] = []
        selections: List[SoupSelection] = []

        for coeff in coeff_candidates:
            soup_state = combine_state_dicts(candidates, coeff.weights)
            if prune_ratio > 0:
                soup_state = apply_prune_mask(
                    soup_state,
                    self._blend_masks(candidates, coeff.weights, model_type="weakly"),
                )

            metrics, scores = self._evaluate_weakly_state(
                args=args,
                pseudo_idx=pseudo_idx,
                state_dict=soup_state,
            )
            metric_list.append(metrics.raw)
            selections.append(
                SoupSelection(
                    coefficients=coeff,
                    metrics=metrics,
                    state_dict=soup_state,
                    training_scores=np.empty(0),
                    weakly_scores=scores,
                )
            )

        metric_keys = ["topk_margin", "neg_entropy", "delta_score"]
        zscores = compute_zscores(metric_list, metric_keys)
        lambdas = self.config.get("objective", {}).get(
            "lambdas", {"margin": 0.7, "entropy": 0.7, "consistency": 0.7}
        )

        best_idx = 0
        best_value = -float("inf")
        for idx, selection in enumerate(selections):
            selection.metrics.zscores = zscores[idx]
            z = selection.metrics.zscores
            score = z["topk_margin"] - lambdas.get("entropy", 0.7) * z["neg_entropy"] - lambdas.get(
                "consistency", 0.7
            ) * z["delta_score"]
            if score > best_value:
                best_value = score
                best_idx = idx

        return selections[best_idx]

    def _evaluate_occ_state(self, args, dataset, state_dict, collect_roc: bool = False):
        args_clone = deepcopy(args)
        pseudo_weight = None
        _, trainer, _ = prepare_occ_trainer(
            Occ_args=args_clone,
            dataset=dataset,
            pseudo_weight=pseudo_weight,
            dir_best_record="",
            load=False,
        )
        trainer.model.load_state_dict(state_dict, strict=False)
        normality_scores, nll_values = trainer.training_scores_producer()
        training_scores = training_score_dataset(normality_scores, dataset["train_test"].metadata, args=args_clone)

        consistency = compute_consistency_metrics(self.prev_occ_scores, training_scores)
        ab_ratio = getattr(args_clone, "ab_ratio", 0.15)
        if self.prev_occ_scores is not None:
            k = max(1, int(len(training_scores) * ab_ratio))
            prev_top = np.argpartition(self.prev_occ_scores, -k)[-k:]
            curr_top = np.argpartition(training_scores, -k)[-k:]
            retention = len(np.intersect1d(prev_top, curr_top)) / k
        else:
            retention = 1.0
        consistency["index_retention"] = retention

        metrics = compute_occ_metrics(nll_values)
        metrics.update(consistency)
        metrics["neg_entropy"] = metrics.get("neg_entropy", 0.0)
        bundle = MetricBundle(raw=metrics)

        roc_info = {"scores": None, "gt": None}
        auc = 0.0
        if collect_roc:
            test_scores = trainer.test()
            auc, roc_scores, roc_gt = score_dataset(test_scores, dataset["test"].metadata, args=args_clone)
            roc_info["scores"] = roc_scores
            roc_info["gt"] = roc_gt
        return bundle, training_scores, nll_values, roc_info | {"auc": auc}

    def _evaluate_weakly_state(self, args, pseudo_idx, state_dict):
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        pred_scores = Weakly_update(
            checkpoint_path="",
            args=args,
            device=device,
            state_dict=state_dict,
        )
        metrics = compute_weakly_metrics(pred_scores, topk=self.config.get("weakly", {}).get("topk", 8))
        consistency = compute_consistency_metrics(self.prev_weakly_scores, pred_scores)
        metrics.update(consistency)
        metrics["neg_entropy"] = -metrics["entropy"]
        bundle = MetricBundle(raw=metrics)
        return bundle, pred_scores

    def _resolve_prune_seed(
        self,
        pruning_cfg: Dict,
        base_seed: int,
        candidate_idx: int,
        kind: str = "random",
    ) -> Optional[int]:
        if not pruning_cfg:
            return None
        if kind == "magnitude":
            mag_seed = pruning_cfg.get("magnitude_seed")
            return base_seed if mag_seed is None else mag_seed

        mode = pruning_cfg.get("random_seed_mode", "aligned")
        if mode == "aligned":
            return base_seed
        if mode == "offset":
            offset = pruning_cfg.get("random_seed_offset", 0)
            return base_seed + offset
        if mode == "pool":
            pool = pruning_cfg.get("random_seed_pool", [])
            if not pool:
                return base_seed
            return pool[candidate_idx % len(pool)]
        if mode == "random":
            return random.randint(0, 2**31 - 1)
        return base_seed

    def _blend_masks(self, candidates: Sequence[CandidateState], weights: Sequence[float], model_type: str):
        aggregated: Dict[str, torch.Tensor] = {}
        for candidate, weight in zip(candidates, weights):
            if candidate.prune_masks is None:
                continue
            for key, mask in candidate.prune_masks.items():
                aggregated.setdefault(key, torch.zeros_like(mask, dtype=torch.float32))
                aggregated[key] += weight * mask.float()

        if not aggregated:
            return None

        keep_fraction = (
            self.config.get(model_type, {})
            .get("pruning", {})
            .get("keep_mask_fraction", 1.0)
        )
        blended = {}
        for key, tensor in aggregated.items():
            flat = tensor.flatten()
            k = int(len(flat) * keep_fraction)
            if k <= 0:
                threshold = float("inf")
            else:
                threshold = torch.topk(flat, k, sorted=False).values.min()
            blended[key] = (tensor >= threshold).float()
        return blended

    def _persist_occ_state(self, state_dict, template_dir: str, output_path: str):
        template_path = os.path.join(template_dir, "Occ_epoch_final_checkpoint.pth.tar")
        checkpoint = torch.load(template_path, map_location="cpu")
        checkpoint["state_dict"] = state_dict
        torch.save(checkpoint, output_path)

    def _persist_weakly_state(self, state_dict, template_dir: str, output_path: str):
        template_path = os.path.join(template_dir, "Weakly_epoch_final_checkpoint.pkl")
        torch.save(state_dict, output_path)

    def _maybe_save_roc(self, model_type: str, round_idx: int, payload: Dict):
        roc_cfg = self.config.get("roc", {})
        output_dir = roc_cfg.get("output_dir")
        if not output_dir:
            return
        filename_key = "occ_filename" if model_type == "occ" else "weakly_filename"
        filename_template = roc_cfg.get(filename_key)
        if not filename_template:
            return

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename_template.format(round=round_idx + 1))

        data = {"auc": payload.get("auc")}
        if model_type == "occ":
            scores = payload.get("scores")
            gt = payload.get("gt")
            if scores is None or gt is None:
                return
            fpr, tpr, _ = roc_curve(gt, scores)
            data.update(
                {
                    "scores": scores.tolist(),
                    "gt": gt.tolist(),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }
            )
        else:
            fpr = payload.get("fpr")
            tpr = payload.get("tpr")
            if fpr is None or tpr is None:
                return
            data.update({"fpr": np.asarray(fpr).tolist(), "tpr": np.asarray(tpr).tolist()})

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def _log_metrics(self, model_type: str, metrics: MetricBundle, step: int):
        if self.writer is None:
            return
        logging_cfg = self.config.get("logging", {})
        model_metrics_key = f"{model_type}_metrics"
        model_logging = logging_cfg.get(model_metrics_key, {})
        for key, tb_key in model_logging.items():
            value = metrics.raw.get(key)
            if value is None:
                continue
            self.writer.add_scalar(tb_key, value, step)

        consistency_cfg = logging_cfg.get("consistency_metrics", {})
        for key, tb_key in consistency_cfg.items():
            value = metrics.raw.get(key)
            if value is None:
                continue
            self.writer.add_scalar(tb_key, value, step)
