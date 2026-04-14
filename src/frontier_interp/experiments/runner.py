"""Main experiment runner.

This module is deliberately explicit because it is the orchestration layer the
paper will depend on. It handles:
- config-driven model sweeps,
- dataset loading,
- behavior/mechanism matched comparisons,
- cross-family generalization,
- dataset-size scaling,
- controls and counterfactuals,
- summaries, plots, and artifact exports.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import random
import numpy as np

import torch

from frontier_interp.config_schema import ExperimentConfig
from frontier_interp.data.dataset_factory import load_examples_from_spec
from frontier_interp.modeling.target_model import FrozenTargetModel
from frontier_interp.modeling.interpreter import build_interpreter
from frontier_interp.experiments.behavior import train_behavior_interpreter, eval_behavior_interpreter
from frontier_interp.experiments.mechanism import train_attention_mechanism_interpreter, eval_attention_mechanism_interpreter
from frontier_interp.analysis.stats import summarize_raw_rows, summarize_by_seed, summarize_seed_aggregates, signed_gap_test
from frontier_interp.analysis.plots import plot_mean_gap, plot_frac_hard, plot_head_heatmap, plot_p90_gap, plot_dataset_scaling
from frontier_interp.utils.seeds import set_seed
from frontier_interp.utils.io import ensure_dir, save_csv, save_yaml_snapshot, save_jsonl
from frontier_interp.utils.logging import RunLogger
from frontier_interp.utils.time import ProgressTracker
from frontier_interp.registries.models import resolve_model_spec


def _bucket_examples(examples, max_prompt_len: int):
    by_bucket = defaultdict(list)
    for ex in examples:
        by_bucket[(ex.family, ex.source)].append(ex)

    batches = []
    for (family, source), items in by_bucket.items():
        batches.append({
            "family": family,
            "source": source,
            "examples": items,
            "max_prompt_len": max_prompt_len,
        })
    return batches



def _split_examples_by_family_seed(all_examples, train_frac: float):
    family_to_items = defaultdict(list)
    for ex in all_examples:
        family_to_items[ex.family].append(ex)

    train_examples = []
    val_examples = []
    for family, items in family_to_items.items():
        items = list(items)
        random.shuffle(items)
        split_idx = max(1, int(len(items) * train_frac)) if len(items) > 1 else len(items)
        train_examples.extend(items[:split_idx])
        val_examples.extend(items[split_idx:])
    return train_examples, val_examples



def _family_match(family: str, allow: List[str]) -> bool:
    return "ALL" in allow or family in allow



def _filter_examples_by_families(examples, allow: List[str]):
    return [ex for ex in examples if _family_match(ex.family, allow)]



def _subsample_examples(examples, train_size, stratify_by_family: bool):
    if train_size == "full" or train_size is None:
        return list(examples)
    train_size = int(train_size)
    if train_size >= len(examples):
        return list(examples)
    if not stratify_by_family:
        items = list(examples)
        random.shuffle(items)
        return items[:train_size]

    family_to_items = defaultdict(list)
    for ex in examples:
        family_to_items[ex.family].append(ex)
    fams = list(family_to_items.keys())
    base = max(1, train_size // max(len(fams), 1))
    chosen = []
    for fam in fams:
        items = list(family_to_items[fam])
        random.shuffle(items)
        chosen.extend(items[: min(base, len(items))])
    if len(chosen) < train_size:
        remaining = [ex for ex in examples if ex not in chosen]
        random.shuffle(remaining)
        chosen.extend(remaining[: train_size - len(chosen)])
    random.shuffle(chosen)
    return chosen[:train_size]



def _aggregate_behavior_rows(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["family"]].append(r)
    out = {}
    for fam, items in grouped.items():
        def mean(name):
            vals = [x[name] for x in items if name in x and not np.isnan(x[name])]
            return float(np.mean(vals)) if vals else float("nan")
        out[fam] = {
            "behavior_normed_loss": mean("normed_loss"),
            "behavior_top1_match": mean("top1_match"),
            "behavior_top5_overlap": mean("top5_overlap"),
            "choice_match_to_target": mean("choice_match_to_target"),
            "choice_rank_corr": mean("choice_rank_corr"),
            "choice_acc_vs_gold": mean("choice_acc_vs_gold"),
            "target_choice_acc_vs_gold": mean("target_choice_acc_vs_gold"),
        }
    if rows:
        out["ALL"] = {
            k: float(np.mean([v[k] for v in out.values() if not np.isnan(v[k])])) if any(not np.isnan(v[k]) for v in out.values()) else float("nan")
            for k in next(iter(out.values())).keys()
        }
    return out



def _aggregate_mech_rows(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["family"]].append(r)
    out = {}
    for fam, items in grouped.items():
        def mean(name):
            vals = [x[name] for x in items if name in x and not np.isnan(x[name])]
            return float(np.mean(vals)) if vals else float("nan")
        out[fam] = {
            "mechanism_normed_loss": mean("normed_loss"),
            "mechanism_top1_match": mean("top1_match"),
        }
    if rows:
        out["ALL"] = {
            k: float(np.mean([v[k] for v in out.values() if not np.isnan(v[k])])) if any(not np.isnan(v[k]) for v in out.values()) else float("nan")
            for k in next(iter(out.values())).keys()
        }
    return out



def _dataset_size_options(config: ExperimentConfig):
    if not config.sweep.dataset_scaling.enabled:
        return ["full"]
    vals = ["full"] + list(config.sweep.dataset_scaling.train_sizes)
    dedup = []
    for v in vals:
        if v not in dedup:
            dedup.append(v)
    return dedup



def run_experiment(config: ExperimentConfig):
    run_dir = ensure_dir(Path(config.outputs.root_dir) / config.outputs.run_name)
    logger = RunLogger(str(run_dir / "logs" / "run.log"))
    save_yaml_snapshot(run_dir / "config_snapshot.yaml", config)

    raw_rows = []
    enabled_model_specs = [m for m in config.models if m.enabled]
    enabled_dataset_specs = [d for d in config.datasets if d.enabled]
    enabled_splits = [s for s in config.sweep.cross_family_splits if s.enabled]
    train_size_options = _dataset_size_options(config)

    rough_total_jobs = len(config.sweep.seeds) * len(enabled_model_specs) * len(enabled_splits) * len(train_size_options) * len(config.sweep.interpreter_sizes) * max(1, len(config.sweep.interpreter_arches)) * (1 + max(1, config.sweep.limit_layers if config.sweep.limit_layers > 0 else 1) * max(1, config.sweep.limit_heads if config.sweep.limit_heads > 0 else 1) * max(1, len(config.sweep.mechanism_types)) * max(1, len(config.sweep.controls)))
    progress = ProgressTracker(max(1, rough_total_jobs))

    for model_spec in enabled_model_specs:
        registry_info = resolve_model_spec(model_spec.registry_key)
        logger.log(f"Loading frozen target model: {model_spec.registry_key} -> {registry_info['hf_name']}")
        target_model = FrozenTargetModel(model_spec.registry_key, config.runtime, model_spec)

        if config.outputs.save_model_cards:
            save_yaml_snapshot(run_dir / "model_cards" / f"{model_spec.registry_key}.yaml", registry_info)

        num_layers = target_model.num_layers
        num_heads = target_model.num_heads
        if config.sweep.limit_layers > 0:
            num_layers = min(num_layers, config.sweep.limit_layers)
        if config.sweep.limit_heads > 0:
            num_heads = min(num_heads, config.sweep.limit_heads)
        logger.log(f"Using layers={num_layers}, heads={num_heads}")

        all_examples = []
        for dataset_spec in enabled_dataset_specs:
            examples = load_examples_from_spec(dataset_spec)
            all_examples.extend(examples)
            logger.log(f"Loaded {len(examples)} examples from {dataset_spec.registry_key}")

        for seed in config.sweep.seeds:
            set_seed(seed)
            logger.log(f"{'='*90}\nSEED {seed}\n{'='*90}")
            base_train_examples, base_val_examples = _split_examples_by_family_seed(all_examples, train_frac=0.8)

            for split_spec in enabled_splits:
                logger.log(f"Split {split_spec.name}: train={split_spec.train_families} eval={split_spec.eval_families}")
                split_train_examples = _filter_examples_by_families(base_train_examples, split_spec.train_families)
                split_val_examples = _filter_examples_by_families(base_val_examples, split_spec.eval_families)
                if not split_train_examples or not split_val_examples:
                    logger.log(f"Skipping split {split_spec.name} due to empty train/val examples.")
                    continue

                for train_size in train_size_options:
                    curr_train_examples = _subsample_examples(split_train_examples, train_size, config.sweep.dataset_scaling.stratify_by_family)
                    curr_val_examples = split_val_examples if config.sweep.dataset_scaling.eval_uses_full_validation else _subsample_examples(split_val_examples, train_size, config.sweep.dataset_scaling.stratify_by_family)
                    train_batches = _bucket_examples(curr_train_examples, config.training.max_prompt_len)
                    val_batches = _bucket_examples(curr_val_examples, config.training.max_prompt_len)
                    logger.log(f"Train size setting={train_size} -> train_examples={len(curr_train_examples)} val_examples={len(curr_val_examples)}")

                    for arch in config.sweep.interpreter_arches:
                        for size in config.sweep.interpreter_sizes:
                            label = f"[{model_spec.registry_key}][seed={seed}][split={split_spec.name}][train={train_size}][{arch}][size={size}] "
                            logger.log(label)

                            behavior_model = build_interpreter(
                                arch=arch,
                                vocab_size=target_model.model.config.vocab_size,
                                d_model=size,
                                n_heads=config.training.interpreter_heads,
                                n_layers=config.training.interpreter_layers,
                                max_len=config.training.max_prompt_len,
                                dropout=config.training.dropout,
                            ).to(target_model.device)

                            behavior_baseline = train_behavior_interpreter(
                                behavior_model, target_model, train_batches,
                                steps=config.training.behavior_steps,
                                logger=logger, run_label=label,
                                lr=config.training.lr, weight_decay=config.training.weight_decay,
                            )
                            behavior_rows, behavior_artifacts = eval_behavior_interpreter(
                                behavior_model, target_model, val_batches, behavior_baseline, config.training.max_prompt_len
                            )
                            behavior_by_family = _aggregate_behavior_rows(behavior_rows)

                            if config.outputs.save_per_example_artifacts and behavior_artifacts:
                                save_jsonl(
                                    run_dir / "artifacts" / "behavior" / f"behavior_examples_{model_spec.registry_key}_seed{seed}_{split_spec.name}_train{train_size}_{arch}_size{size}.jsonl",
                                    behavior_artifacts[:32],
                                )

                            progress.update()
                            logger.log_progress(prefix=label, progress_status=progress.status())

                            for mechanism_type in config.sweep.mechanism_types:
                                for control_name in config.sweep.controls:
                                    for layer_idx in range(num_layers):
                                        for head_idx in range(num_heads):
                                            mech_label = f"{label}[{mechanism_type}][{control_name}][L={layer_idx}][H={head_idx}] "
                                            mech_model = build_interpreter(
                                                arch=arch,
                                                vocab_size=target_model.model.config.vocab_size,
                                                d_model=size,
                                                n_heads=config.training.interpreter_heads,
                                                n_layers=config.training.interpreter_layers,
                                                max_len=config.training.max_prompt_len,
                                                dropout=config.training.dropout,
                                            ).to(target_model.device)

                                            mech_baseline, complexity = train_attention_mechanism_interpreter(
                                                mech_model, target_model, train_batches,
                                                layer_idx=layer_idx, head_idx=head_idx,
                                                control_name=control_name, mechanism_type=mechanism_type,
                                                steps=config.training.mechanism_steps,
                                                logger=logger, run_label=mech_label,
                                                lr=config.training.lr, weight_decay=config.training.weight_decay,
                                            )
                                            mech_rows, mech_artifacts = eval_attention_mechanism_interpreter(
                                                mech_model, target_model, val_batches,
                                                layer_idx=layer_idx, head_idx=head_idx,
                                                control_name=control_name, mechanism_type=mechanism_type,
                                                baseline_loss=mech_baseline,
                                            )
                                            mech_by_family = _aggregate_mech_rows(mech_rows)

                                            if config.outputs.save_per_example_artifacts and control_name == "none" and layer_idx < 2 and head_idx < 2:
                                                artifact_rows = []
                                                for a in mech_artifacts[:8]:
                                                    artifact_rows.append({
                                                        "family": a["family"],
                                                        "source": a["source"],
                                                        "text": a["text"],
                                                        "target_top1_indices": np.argmax(a["target_attn"], axis=-1).tolist(),
                                                        "pred_top1_indices": np.argmax(a["pred_attn"], axis=-1).tolist(),
                                                    })
                                                save_jsonl(
                                                    run_dir / "artifacts" / "mechanism" / f"mech_examples_{model_spec.registry_key}_seed{seed}_{split_spec.name}_train{train_size}_{arch}_size{size}_{mechanism_type}_L{layer_idx}H{head_idx}.jsonl",
                                                    artifact_rows,
                                                )

                                            families_to_emit = sorted(set(list(behavior_by_family.keys()) + list(mech_by_family.keys())))
                                            for family in families_to_emit:
                                                if family not in behavior_by_family or family not in mech_by_family:
                                                    continue
                                                raw_rows.append({
                                                    "model_key": model_spec.registry_key,
                                                    "seed": seed,
                                                    "size": size,
                                                    "family": family,
                                                    "mechanism_type": mechanism_type,
                                                    "interpreter_arch": arch,
                                                    "control": control_name,
                                                    "layer": layer_idx,
                                                    "head": head_idx,
                                                    "split_name": split_spec.name,
                                                    "train_size": train_size,
                                                    "behavior_normed_loss": behavior_by_family[family]["behavior_normed_loss"],
                                                    "behavior_top1_match": behavior_by_family[family]["behavior_top1_match"],
                                                    "behavior_top5_overlap": behavior_by_family[family]["behavior_top5_overlap"],
                                                    "choice_match_to_target": behavior_by_family[family]["choice_match_to_target"],
                                                    "choice_rank_corr": behavior_by_family[family]["choice_rank_corr"],
                                                    "choice_acc_vs_gold": behavior_by_family[family]["choice_acc_vs_gold"],
                                                    "target_choice_acc_vs_gold": behavior_by_family[family]["target_choice_acc_vs_gold"],
                                                    "mechanism_normed_loss": mech_by_family[family]["mechanism_normed_loss"],
                                                    "mechanism_top1_match": mech_by_family[family]["mechanism_top1_match"],
                                                    "gap": mech_by_family[family]["mechanism_normed_loss"] - behavior_by_family[family]["behavior_normed_loss"],
                                                    **complexity,
                                                })

                                            del mech_model
                                            torch.cuda.empty_cache()
                                            progress.update()
                                            logger.log_progress(prefix=mech_label, progress_status=progress.status())

                            del behavior_model
                            torch.cuda.empty_cache()

                    if config.outputs.save_checkpoints:
                        save_csv(run_dir / f"checkpoints/raw_rows_{model_spec.registry_key}_seed{seed}_{split_spec.name}_train{train_size}.csv", raw_rows)

        del target_model
        torch.cuda.empty_cache()

    summary_rows = summarize_raw_rows(raw_rows)
    seed_rows = summarize_by_seed(raw_rows)
    seed_agg_rows = summarize_seed_aggregates(
        seed_rows,
        confidence_level=config.stats.confidence_level,
        bootstrap_iterations=config.stats.bootstrap_iterations,
    )
    test_rows = signed_gap_test(seed_rows, fdr_method=config.stats.fdr_method)

    save_csv(run_dir / "raw_rows.csv", raw_rows)
    save_csv(run_dir / "summary_rows.csv", summary_rows)
    save_csv(run_dir / "seed_level_summary.csv", seed_rows)
    save_csv(run_dir / "seed_aggregated_summary.csv", seed_agg_rows)
    save_csv(run_dir / "signed_gap_tests.csv", test_rows)

    plot_mean_gap(summary_rows, run_dir / "plots")
    plot_frac_hard(summary_rows, run_dir / "plots")
    plot_p90_gap(summary_rows, run_dir / "plots")
    plot_dataset_scaling(seed_agg_rows, run_dir / "plots")
    for model_spec in enabled_model_specs:
        for mechanism_type in config.sweep.mechanism_types:
            plot_head_heatmap(raw_rows, run_dir / "plots", mechanism_type=mechanism_type, model_key=model_spec.registry_key)

    logger.log("Run complete.")
    logger.log(f"Outputs written to: {run_dir}")
