"""Main experiment runner.

This module is deliberately explicit because it is the orchestration layer the
paper will depend on. It handles:
- config-driven model sweeps,
- dataset loading,
- behavior/mechanism matched comparisons,
- controls and counterfactuals,
- summaries, plots, and artifact exports.
"""

from __future__ import annotations

from dataclasses import asdict
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
from frontier_interp.analysis.plots import plot_mean_gap, plot_frac_hard, plot_head_heatmap, plot_p90_gap
from frontier_interp.utils.seeds import set_seed
from frontier_interp.utils.io import ensure_dir, save_csv, save_yaml_snapshot, save_jsonl
from frontier_interp.utils.logging import RunLogger
from frontier_interp.utils.time import ProgressTracker
from frontier_interp.registries.models import resolve_model_spec



def _bucket_examples(examples, max_prompt_len: int):
    """Group examples into family/source-specific mini-batches.

    Keeping source and family labels separate is important because later analysis
    often wants to slice by prompt family while artifact exports still need to
    know which open dataset produced the example.
    """
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
        split_idx = max(1, int(len(items) * train_frac))
        train_examples.extend(items[:split_idx])
        val_examples.extend(items[split_idx:])
    return train_examples, val_examples



def run_experiment(config: ExperimentConfig):
    run_dir = ensure_dir(Path(config.outputs.root_dir) / config.outputs.run_name)
    logger = RunLogger(str(run_dir / "logs" / "run.log"))
    save_yaml_snapshot(run_dir / "config_snapshot.yaml", config)

    raw_rows = []

    enabled_model_specs = [m for m in config.models if m.enabled]
    enabled_dataset_specs = [d for d in config.datasets if d.enabled]

    # Behavior model counts as one job per (seed, model, arch, size).
    # Each mechanism objective is one job per (seed, model, arch, size, layer, head, mechanism, control).
    rough_total_jobs = len(config.sweep.seeds) * len(enabled_model_specs) * len(config.sweep.interpreter_sizes) * max(1, len(config.sweep.interpreter_arches))
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

            train_examples, val_examples = _split_examples_by_family_seed(all_examples, train_frac=0.8)
            train_batches = _bucket_examples(train_examples, config.training.max_prompt_len)
            val_batches = _bucket_examples(val_examples, config.training.max_prompt_len)

            for arch in config.sweep.interpreter_arches:
                for size in config.sweep.interpreter_sizes:
                    logger.log(f"--- model={model_spec.registry_key} seed={seed} arch={arch} size={size} ---")

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
                        behavior_model,
                        target_model,
                        train_batches,
                        steps=config.training.behavior_steps,
                        logger=logger,
                        run_label=f"[{model_spec.registry_key}][seed={seed}][{arch}][size={size}] ",
                        lr=config.training.lr,
                        weight_decay=config.training.weight_decay,
                    )
                    behavior_rows, behavior_artifacts = eval_behavior_interpreter(
                        behavior_model,
                        target_model,
                        val_batches,
                        baseline_loss=behavior_baseline,
                        max_prompt_len=config.training.max_prompt_len,
                    )

                    if config.outputs.save_per_example_artifacts:
                        save_jsonl(
                            run_dir / "artifacts" / "behavior" / f"behavior_examples_{model_spec.registry_key}_seed{seed}_{arch}_size{size}.jsonl",
                            behavior_artifacts,
                        )

                    behavior_by_family = defaultdict(dict)
                    for family in set(r["family"] for r in behavior_rows):
                        sub = [r for r in behavior_rows if r["family"] == family]
                        behavior_by_family[family] = {
                            "behavior_normed_loss": float(sum(r["normed_loss"] for r in sub) / len(sub)),
                            "behavior_top1_match": float(sum(r["top1_match"] for r in sub) / len(sub)),
                            "behavior_top5_overlap": float(sum(r["top5_overlap"] for r in sub) / len(sub)),
                            "choice_match_to_target": float(sum(r["choice_match_to_target"] for r in sub if str(r["choice_match_to_target"]) != 'nan') / max(1, sum(1 for r in sub if str(r["choice_match_to_target"]) != 'nan'))),
                            "choice_rank_corr": float(sum(r["choice_rank_corr"] for r in sub if str(r["choice_rank_corr"]) != 'nan') / max(1, sum(1 for r in sub if str(r["choice_rank_corr"]) != 'nan'))),
                            "choice_acc_vs_gold": float(sum(r["choice_acc_vs_gold"] for r in sub if str(r["choice_acc_vs_gold"]) != 'nan') / max(1, sum(1 for r in sub if str(r["choice_acc_vs_gold"]) != 'nan'))),
                            "target_choice_acc_vs_gold": float(sum(r["target_choice_acc_vs_gold"] for r in sub if str(r["target_choice_acc_vs_gold"]) != 'nan') / max(1, sum(1 for r in sub if str(r["target_choice_acc_vs_gold"]) != 'nan'))),
                        }

                    for mechanism_type in config.sweep.mechanism_types:
                        for layer_idx in range(num_layers):
                            for head_idx in range(num_heads):
                                for control_name in (config.sweep.controls or ["none"]):
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
                                        mech_model,
                                        target_model,
                                        train_batches,
                                        layer_idx=layer_idx,
                                        head_idx=head_idx,
                                        control_name=control_name,
                                        mechanism_type=mechanism_type,
                                        steps=config.training.mechanism_steps,
                                        logger=logger,
                                        run_label=(
                                            f"[{model_spec.registry_key}]"
                                            f"[seed={seed}]"
                                            f"[{arch}]"
                                            f"[size={size}]"
                                            f"[{mechanism_type}]"
                                            f"[L{layer_idx}H{head_idx}]"
                                            f"[{control_name}] "
                                        ),
                                        lr=config.training.lr,
                                        weight_decay=config.training.weight_decay,
                                    )
                                    mech_rows, mech_artifacts = eval_attention_mechanism_interpreter(
                                        mech_model,
                                        target_model,
                                        val_batches,
                                        layer_idx=layer_idx,
                                        head_idx=head_idx,
                                        control_name=control_name,
                                        mechanism_type=mechanism_type,
                                        baseline_loss=mech_baseline,
                                    )

                                    if config.outputs.save_per_example_artifacts and control_name == "none" and layer_idx < 2 and head_idx < 2:
                                        artifact_rows = []
                                        for a in mech_artifacts[:4]:
                                            artifact_rows.append({
                                                "family": a["family"],
                                                "source": a["source"],
                                                "text": a["text"],
                                                "target_top1_indices": np.argmax(a["target_attn"], axis=-1).tolist() if 'np' in globals() else None,
                                                "pred_top1_indices": np.argmax(a["pred_attn"], axis=-1).tolist() if 'np' in globals() else None,
                                            })
                                        save_jsonl(
                                            run_dir / "artifacts" / "mechanism" / f"mech_examples_{model_spec.registry_key}_seed{seed}_{arch}_size{size}_{mechanism_type}_L{layer_idx}H{head_idx}.jsonl",
                                            artifact_rows,
                                        )

                                    for family in behavior_by_family.keys():
                                        family_mech = [r for r in mech_rows if r["family"] == family]
                                        if not family_mech:
                                            continue
                                        mech_normed = float(sum(r["normed_loss"] for r in family_mech) / len(family_mech))
                                        mech_top1 = float(sum(r["top1_match"] for r in family_mech) / len(family_mech))
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
                                            "behavior_normed_loss": behavior_by_family[family]["behavior_normed_loss"],
                                            "behavior_top1_match": behavior_by_family[family]["behavior_top1_match"],
                                            "behavior_top5_overlap": behavior_by_family[family]["behavior_top5_overlap"],
                                            "choice_match_to_target": behavior_by_family[family]["choice_match_to_target"],
                                            "choice_rank_corr": behavior_by_family[family]["choice_rank_corr"],
                                            "choice_acc_vs_gold": behavior_by_family[family]["choice_acc_vs_gold"],
                                            "target_choice_acc_vs_gold": behavior_by_family[family]["target_choice_acc_vs_gold"],
                                            "mechanism_normed_loss": mech_normed,
                                            "mechanism_top1_match": mech_top1,
                                            "gap": mech_normed - behavior_by_family[family]["behavior_normed_loss"],
                                            **complexity,
                                        })

                                    del mech_model
                                    torch.cuda.empty_cache()

                    del behavior_model
                    torch.cuda.empty_cache()
                    progress.update()
                    logger.log_progress(prefix=f"[{model_spec.registry_key}][seed={seed}][{arch}][size={size}] ", progress_status=progress.status())

            if config.outputs.save_checkpoints:
                save_csv(run_dir / f"checkpoints/raw_rows_{model_spec.registry_key}_seed{seed}.csv", raw_rows)

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
    for model_spec in enabled_model_specs:
        for mechanism_type in config.sweep.mechanism_types:
            plot_head_heatmap(raw_rows, run_dir / "plots", mechanism_type=mechanism_type, model_key=model_spec.registry_key)

    logger.log("Run complete.")
    logger.log(f"Outputs written to: {run_dir}")
