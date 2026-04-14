
"""Main experiment runner.

This module is intentionally verbose and heavily commented because it is the core
orchestration point for the paper. It handles:
- config-driven model sweeps,
- dataset loading,
- behavior/mechanism matched comparisons,
- controls,
- summaries and plots.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch

from frontier_interp.config_schema import ExperimentConfig
from frontier_interp.data.dataset_factory import load_examples_from_spec
from frontier_interp.modeling.target_model import FrozenTargetModel
from frontier_interp.modeling.interpreter import build_interpreter
from frontier_interp.experiments.behavior import train_behavior_interpreter, eval_behavior_interpreter
from frontier_interp.experiments.mechanism import train_attention_mechanism_interpreter, eval_attention_mechanism_interpreter
from frontier_interp.analysis.stats import summarize_raw_rows, summarize_by_seed, summarize_seed_aggregates, signed_gap_test
from frontier_interp.analysis.plots import plot_mean_gap, plot_frac_hard, plot_head_heatmap
from frontier_interp.utils.seeds import set_seed
from frontier_interp.utils.io import ensure_dir, save_csv, save_yaml_snapshot
from frontier_interp.utils.logging import RunLogger
from frontier_interp.utils.time import ProgressTracker


def _bucket_examples(examples, max_prompt_len: int):
    """Group examples into family-specific mini-batches.

    We keep family labels explicit because many plots and summary tables are
    stratified by prompt family.
    """
    by_family = defaultdict(list)
    for ex in examples:
        by_family[ex.family].append(ex)

    # Each batch is a simple dictionary; the model wrappers tokenize later.
    batches = []
    for family, items in by_family.items():
        texts = [x.text for x in items]
        batches.append({
            "family": family,
            "texts": texts,
            "max_prompt_len": max_prompt_len,
        })
    return batches


def run_experiment(config: ExperimentConfig):
    run_dir = ensure_dir(Path(config.outputs.root_dir) / config.outputs.run_name)
    logger = RunLogger(str(run_dir / "logs" / "run.log"))
    save_yaml_snapshot(run_dir / "config_snapshot.yaml", config)

    raw_rows = []

    enabled_model_specs = [m for m in config.models if m.enabled]
    enabled_dataset_specs = [d for d in config.datasets if d.enabled]

    total_jobs = (
        len(config.sweep.seeds)
        * len(enabled_model_specs)
        * len(config.sweep.interpreter_sizes)
        * max(1, len(config.sweep.interpreter_arches))
    )
    progress = ProgressTracker(total_jobs)

    for model_spec in enabled_model_specs:
        logger.log(f"Loading frozen target model: {model_spec.registry_key}")
        target_model = FrozenTargetModel(model_spec.registry_key, config.runtime, model_spec)

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

        # Per-family split. Keep the split simple and deterministic per seed.
        family_to_items = defaultdict(list)
        for ex in all_examples:
            family_to_items[ex.family].append(ex)

        for seed in config.sweep.seeds:
            set_seed(seed)
            logger.log(f"{'='*90}\nSEED {seed}\n{'='*90}")

            train_examples = []
            val_examples = []
            for family, items in family_to_items.items():
                split_idx = max(1, int(len(items) * 0.8))
                train_examples.extend(items[:split_idx])
                val_examples.extend(items[split_idx:])

            train_batches = _bucket_examples(train_examples, config.training.max_prompt_len)
            val_batches = _bucket_examples(val_examples, config.training.max_prompt_len)

            for arch in config.sweep.interpreter_arches:
                for size in config.sweep.interpreter_sizes:
                    logger.log(f"{'-'*90}\nmodel={model_spec.registry_key} seed={seed} arch={arch} size={size}\n{'-'*90}")

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
                    behavior_rows = eval_behavior_interpreter(
                        behavior_model,
                        target_model,
                        val_batches,
                        baseline_loss=behavior_baseline,
                    )

                    behavior_by_family = {}
                    for family in {r["family"] for r in behavior_rows}:
                        sub = [r for r in behavior_rows if r["family"] == family]
                        behavior_by_family[family] = {
                            "behavior_normed_loss": float(sum(r["normed_loss"] for r in sub) / len(sub)),
                            "behavior_top1_match": float(sum(r["top1_match"] for r in sub) / len(sub)),
                            "behavior_top5_overlap": float(sum(r["top5_overlap"] for r in sub) / len(sub)),
                        }

                    for mechanism_type in config.sweep.mechanism_types:
                        if mechanism_type != "attention_probs":
                            logger.log(f"Skipping unsupported mechanism type for now: {mechanism_type}")
                            continue

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
                                        steps=config.training.mechanism_steps,
                                        logger=logger,
                                        run_label=f"[{model_spec.registry_key}][seed={seed}][{arch}][size={size}][L{layer_idx}H{head_idx}][{control_name}] ",
                                        lr=config.training.lr,
                                        weight_decay=config.training.weight_decay,
                                    )
                                    mech_rows = eval_attention_mechanism_interpreter(
                                        mech_model,
                                        target_model,
                                        val_batches,
                                        layer_idx=layer_idx,
                                        head_idx=head_idx,
                                        control_name=control_name,
                                        baseline_loss=mech_baseline,
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

            # checkpoint per seed / model
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
    test_rows = signed_gap_test(seed_rows)

    save_csv(run_dir / "raw_rows.csv", raw_rows)
    save_csv(run_dir / "summary_rows.csv", summary_rows)
    save_csv(run_dir / "seed_level_summary.csv", seed_rows)
    save_csv(run_dir / "seed_aggregated_summary.csv", seed_agg_rows)
    save_csv(run_dir / "signed_gap_tests.csv", test_rows)

    plot_mean_gap(summary_rows, run_dir / "plots")
    plot_frac_hard(summary_rows, run_dir / "plots")
    for model_spec in enabled_model_specs:
        plot_head_heatmap(raw_rows, run_dir / "plots", mechanism_type="attention_probs", model_key=model_spec.registry_key)

    logger.log("Run complete.")
    logger.log(f"Outputs written to: {run_dir}")
