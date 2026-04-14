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
from frontier_interp.modeling.self_reflection import build_self_reflection_probe
from frontier_interp.experiments.behavior import train_behavior_interpreter, eval_behavior_interpreter
from frontier_interp.experiments.mechanism import train_attention_mechanism_interpreter, eval_attention_mechanism_interpreter
from frontier_interp.experiments.self_reflection import (
    train_self_probe_behavior,
    eval_self_probe_behavior,
    train_self_probe_mechanism,
    eval_self_probe_mechanism,
    eval_prompted_self_report_behavior,
    eval_prompted_self_report_mechanism,
)
from frontier_interp.analysis.stats import summarize_raw_rows, summarize_by_seed, summarize_seed_aggregates, signed_gap_test
from frontier_interp.analysis.plots import plot_mean_gap, plot_frac_hard, plot_head_heatmap, plot_p90_gap, plot_dataset_scaling
from frontier_interp.analysis.reporting import write_markdown_report, write_latex_tables, build_submission_bundle
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



def _aggregate_behavior_rows(rows: List[Dict]) -> Dict[tuple, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["family"], r.get("source", "unknown"))].append(r)
    out = {}
    for key, items in grouped.items():
        def mean(name):
            vals = [x[name] for x in items if name in x and not np.isnan(x[name])]
            return float(np.mean(vals)) if vals else float("nan")
        out[key] = {
            "behavior_normed_loss": mean("normed_loss"),
            "behavior_top1_match": mean("top1_match"),
            "behavior_top5_overlap": mean("top5_overlap"),
            "choice_match_to_target": mean("choice_match_to_target"),
            "choice_rank_corr": mean("choice_rank_corr"),
            "choice_acc_vs_gold": mean("choice_acc_vs_gold"),
            "target_choice_acc_vs_gold": mean("target_choice_acc_vs_gold"),
        }
    if rows and out:
        out[("ALL", "ALL")] = {
            k: float(np.mean([v[k] for v in out.values() if not np.isnan(v[k])])) if any(not np.isnan(v[k]) for v in out.values()) else float("nan")
            for k in next(iter(out.values())).keys()
        }
    return out


def _aggregate_mech_rows(rows: List[Dict]) -> Dict[tuple, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["family"], r.get("source", "unknown"))].append(r)
    out = {}
    for key, items in grouped.items():
        def mean(name):
            vals = [x[name] for x in items if name in x and not np.isnan(x[name])]
            return float(np.mean(vals)) if vals else float("nan")
        out[key] = {
            "mechanism_normed_loss": mean("normed_loss"),
            "mechanism_top1_match": mean("top1_match"),
        }
    if rows and out:
        out[("ALL", "ALL")] = {
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





def _summarize_self_reflection_rows(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        key = (r.get("model_key"), r.get("seed"), r.get("split_name"), r.get("train_size"), r.get("source", "ALL"), r.get("family"), r.get("self_reflection_mode"), r.get("probe_arch", "none"), r.get("probe_size", -1), r.get("layer", -1), r.get("head", -1))
        grouped[key].append(r)

    out = []
    for key, items in grouped.items():
        metric_keys = [k for k in items[0].keys() if k.endswith("_mean") or k in {"normed_loss", "top1_match", "exact_match", "parsed_ok", "abs_error"}]
        row = {
            "model_key": key[0],
            "seed": key[1],
            "split_name": key[2],
            "train_size": key[3],
            "source": key[4],
            "family": key[5],
            "self_reflection_mode": key[6],
            "probe_arch": key[7],
            "probe_size": key[8],
            "layer": key[9],
            "head": key[10],
            "num_rows": len(items),
        }
        for mk in metric_keys:
            vals = [x[mk] for x in items if mk in x and not np.isnan(x[mk])]
            if vals:
                row[f"avg_{mk}"] = float(np.mean(vals))
        out.append(row)
    return out


def _aggregate_probe_rows(rows: List[Dict], prefix: str) -> Dict[tuple, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["family"], r.get("source", "unknown"))].append(r)
    out = {}
    for key, items in grouped.items():
        def mean(name):
            vals = [x[name] for x in items if name in x and not np.isnan(x[name])]
            return float(np.mean(vals)) if vals else float("nan")
        out[key] = {
            f"{prefix}_normed_loss": mean("normed_loss"),
            f"{prefix}_top1_match": mean("top1_match"),
        }
    if rows and out:
        out[("ALL", "ALL")] = {
            k: float(np.mean([v[k] for v in out.values() if not np.isnan(v[k])])) if any(not np.isnan(v[k]) for v in out.values()) else float("nan")
            for k in next(iter(out.values())).keys()
        }
    return out


def _aggregate_report_rows(rows: List[Dict], mode_prefix: str) -> Dict[tuple, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["family"], r.get("source", "unknown"))].append(r)
    out = {}
    for key, items in grouped.items():
        def mean(name):
            vals = [x[name] for x in items if name in x and not np.isnan(x[name])]
            return float(np.mean(vals)) if vals else float("nan")
        out[key] = {
            f"{mode_prefix}_exact_match": mean("exact_match"),
            f"{mode_prefix}_exact_match_to_target": mean("exact_match_to_target"),
            f"{mode_prefix}_exact_match_to_gold": mean("exact_match_to_gold"),
            f"{mode_prefix}_parsed_ok": mean("parsed_ok"),
            f"{mode_prefix}_within_one": mean("within_one"),
            f"{mode_prefix}_abs_error": mean("abs_error"),
        }
    if rows and out:
        out[("ALL", "ALL")] = {
            k: float(np.mean([v[k] for v in out.values() if not np.isnan(v[k])])) if any(not np.isnan(v[k]) for v in out.values()) else float("nan")
            for k in next(iter(out.values())).keys()
        }
    return out




def _evaluate_self_reflection_validation(summary_rows: List[Dict], validation_cfg) -> List[Dict]:
    evaluations = []
    for row in summary_rows:
        mode = row.get('self_reflection_mode', '')
        promoted = True
        reasons = []
        num_rows = int(row.get('num_rows', 0))
        if num_rows < validation_cfg.min_num_rows:
            promoted = False
            reasons.append(f'num_rows<{validation_cfg.min_num_rows}')

        parsed_keys = [k for k in row if 'parsed_ok' in k]
        parsed_val = max([row[k] for k in parsed_keys if isinstance(row.get(k), (int, float)) and not np.isnan(row[k])] or [float('nan')])
        if mode.startswith('self_report'):
            if np.isnan(parsed_val) or parsed_val < validation_cfg.self_report_min_parsed_ok:
                promoted = False
                reasons.append('parsed_ok_below_threshold')
            if mode == 'self_report_behavior':
                target_match_keys = [k for k in row if 'exact_match_to_target' in k]
                gold_match_keys = [k for k in row if 'exact_match_to_gold' in k]
                target_match = max([row[k] for k in target_match_keys if isinstance(row.get(k), (int, float)) and not np.isnan(row[k])] or [float('nan')])
                gold_match = max([row[k] for k in gold_match_keys if isinstance(row.get(k), (int, float)) and not np.isnan(row[k])] or [float('nan')])
                if np.isnan(target_match) or target_match < validation_cfg.self_report_min_exact_match_to_target:
                    promoted = False
                    reasons.append('target_match_below_threshold')
                # Only enforce gold if a finite value exists for this grouping
                if not np.isnan(gold_match) and gold_match < validation_cfg.self_report_min_exact_match_to_gold:
                    promoted = False
                    reasons.append('gold_match_below_threshold')
            elif mode == 'self_report_mechanism':
                within_one_keys = [k for k in row if 'within_one' in k]
                within_one = max([row[k] for k in within_one_keys if isinstance(row.get(k), (int, float)) and not np.isnan(row[k])] or [float('nan')])
                if np.isnan(within_one) or within_one < validation_cfg.self_report_mechanism_min_within_one:
                    promoted = False
                    reasons.append('within_one_below_threshold')

        evaluations.append({
            'model_key': row.get('model_key'),
            'family': row.get('family'),
            'source': row.get('source'),
            'self_reflection_mode': mode,
            'probe_arch': row.get('probe_arch'),
            'probe_size': row.get('probe_size'),
            'layer': row.get('layer'),
            'head': row.get('head'),
            'num_rows': num_rows,
            'promoted': bool(promoted),
            'reasons': ';'.join(reasons) if reasons else 'passed',
        })
    return evaluations


def _write_validation_checklist(run_dir: Path, validation_rows: List[Dict]):
    path = run_dir / 'VALIDATION_CHECKLIST.md'
    promoted = [r for r in validation_rows if r['promoted']]
    rejected = [r for r in validation_rows if not r['promoted']]
    lines = ['# Validation Checklist\n']
    lines.append(f'Total self-reflection groupings checked: {len(validation_rows)}\n')
    lines.append(f'Promoted optional pathways: {len(promoted)}\n')
    lines.append(f'Not promoted: {len(rejected)}\n')
    lines.append('## Rejected groupings\n')
    lines.append('| model | family | source | mode | arch | size | layer | head | reasons |\n|---|---|---|---|---|---:|---:|---:|---|')
    for r in rejected[:200]:
        lines.append(f"| {r.get('model_key','')} | {r.get('family','')} | {r.get('source','')} | {r.get('self_reflection_mode','')} | {r.get('probe_arch','')} | {r.get('probe_size','')} | {r.get('layer','')} | {r.get('head','')} | {r.get('reasons','')} |")
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

def run_experiment(config: ExperimentConfig):
    run_dir = ensure_dir(Path(config.outputs.root_dir) / config.outputs.run_name)
    logger = RunLogger(str(run_dir / "logs" / "run.log"))
    save_yaml_snapshot(run_dir / "config_snapshot.yaml", config)

    raw_rows = []
    self_reflection_rows = []
    enabled_model_specs = [m for m in config.models if m.enabled]
    enabled_dataset_specs = [d for d in config.datasets if d.enabled]
    enabled_splits = [s for s in config.sweep.cross_family_splits if s.enabled]
    train_size_options = _dataset_size_options(config)

    rough_total_jobs = len(config.sweep.seeds) * len(enabled_model_specs) * len(enabled_splits) * len(train_size_options) * len(config.sweep.interpreter_sizes) * max(1, len(config.sweep.interpreter_arches)) * (1 + max(1, config.sweep.limit_layers if config.sweep.limit_layers > 0 else 1) * max(1, config.sweep.limit_heads if config.sweep.limit_heads > 0 else 1) * max(1, len(config.sweep.mechanism_types)) * max(1, len(config.sweep.controls)))
    if config.self_reflection.enabled:
        rough_total_jobs += len(config.sweep.seeds) * len(enabled_model_specs) * len(enabled_splits) * len(train_size_options) * (
            len(config.self_reflection.probe_arches) * len(config.self_reflection.probe_sizes) * (1 + max(1, config.sweep.limit_layers if config.sweep.limit_layers > 0 else 1) * max(1, config.sweep.limit_heads if config.sweep.limit_heads > 0 else 1) * max(1, len(config.sweep.mechanism_types)))
            + max(1, len(config.self_reflection.modes))
        )
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
                                            for family_key in families_to_emit:
                                                if family_key not in behavior_by_family or family_key not in mech_by_family:
                                                    continue
                                                family, source = family_key
                                                raw_rows.append({
                                                    "model_key": model_spec.registry_key,
                                                    "seed": seed,
                                                    "size": size,
                                                    "family": family,
                                                    "source": source,
                                                    "mechanism_type": mechanism_type,
                                                    "interpreter_arch": arch,
                                                    "control": control_name,
                                                    "layer": layer_idx,
                                                    "head": head_idx,
                                                    "split_name": split_spec.name,
                                                    "train_size": train_size,
                                                    "behavior_normed_loss": behavior_by_family[family_key]["behavior_normed_loss"],
                                                    "behavior_top1_match": behavior_by_family[family_key]["behavior_top1_match"],
                                                    "behavior_top5_overlap": behavior_by_family[family_key]["behavior_top5_overlap"],
                                                    "choice_match_to_target": behavior_by_family[family_key]["choice_match_to_target"],
                                                    "choice_rank_corr": behavior_by_family[family_key]["choice_rank_corr"],
                                                    "choice_acc_vs_gold": behavior_by_family[family_key]["choice_acc_vs_gold"],
                                                    "target_choice_acc_vs_gold": behavior_by_family[family_key]["target_choice_acc_vs_gold"],
                                                    "mechanism_normed_loss": mech_by_family[family_key]["mechanism_normed_loss"],
                                                    "mechanism_top1_match": mech_by_family[family_key]["mechanism_top1_match"],
                                                    "gap": mech_by_family[family_key]["mechanism_normed_loss"] - behavior_by_family[family_key]["behavior_normed_loss"],
                                                    **complexity,
                                                })

                                            del mech_model
                                            torch.cuda.empty_cache()
                                            progress.update()
                                            logger.log_progress(prefix=mech_label, progress_status=progress.status())

                            del behavior_model
                            torch.cuda.empty_cache()

                            if config.self_reflection.enabled:
                                sr_examples = curr_val_examples[: config.self_reflection.report_num_examples]
                                for probe_arch in config.self_reflection.probe_arches:
                                    for probe_size in config.self_reflection.probe_sizes:
                                        sr_label = f"[{model_spec.registry_key}][seed={seed}][split={split_spec.name}][train={train_size}][self-reflect][{probe_arch}][size={probe_size}] "

                                        if "self_probe_behavior" in config.self_reflection.modes:
                                            probe = build_self_reflection_probe(
                                                arch=probe_arch,
                                                hidden_dim=int(target_model.model.config.hidden_size),
                                                vocab_size=target_model.model.config.vocab_size,
                                                probe_dim=probe_size,
                                            ).to(target_model.device)
                                            sr_baseline = train_self_probe_behavior(
                                                probe, target_model, train_batches,
                                                steps=config.self_reflection.probe_steps,
                                                logger=logger, run_label=sr_label,
                                                lr=config.training.lr, weight_decay=config.training.weight_decay,
                                            )
                                            sr_rows = eval_self_probe_behavior(probe, target_model, val_batches, sr_baseline)
                                            sr_by_family = _aggregate_probe_rows(sr_rows, prefix="self_probe_behavior")
                                            for family_key, stats in sr_by_family.items():
                                                family, source = family_key
                                                self_reflection_rows.append({
                                                    "model_key": model_spec.registry_key,
                                                    "seed": seed,
                                                    "split_name": split_spec.name,
                                                    "train_size": train_size,
                                                    "family": family,
                                                    "source": source,
                                                    "self_reflection_mode": "self_probe_behavior",
                                                    "probe_arch": probe_arch,
                                                    "probe_size": probe_size,
                                                    "layer": -1,
                                                    "head": -1,
                                                    **stats,
                                                })
                                            del probe
                                            torch.cuda.empty_cache()
                                            progress.update()
                                            logger.log_progress(prefix=sr_label, progress_status=progress.status())

                                        if "self_probe_mechanism" in config.self_reflection.modes:
                                            for mechanism_type in config.sweep.mechanism_types:
                                                for layer_idx in range(num_layers):
                                                    for head_idx in range(num_heads):
                                                        probe = build_self_reflection_probe(
                                                            arch=probe_arch,
                                                            hidden_dim=int(target_model.model.config.hidden_size),
                                                            vocab_size=target_model.model.config.vocab_size,
                                                            probe_dim=probe_size,
                                                        ).to(target_model.device)
                                                        mech_sr_label = f"{sr_label}[{mechanism_type}][L={layer_idx}][H={head_idx}] "
                                                        sr_baseline = train_self_probe_mechanism(
                                                            probe, target_model, train_batches,
                                                            layer_idx=layer_idx, head_idx=head_idx,
                                                            mechanism_type=mechanism_type,
                                                            steps=config.self_reflection.probe_steps,
                                                            logger=logger, run_label=mech_sr_label,
                                                            lr=config.training.lr, weight_decay=config.training.weight_decay,
                                                        )
                                                        sr_rows = eval_self_probe_mechanism(
                                                            probe, target_model, val_batches,
                                                            layer_idx=layer_idx, head_idx=head_idx,
                                                            mechanism_type=mechanism_type,
                                                            baseline_loss=sr_baseline,
                                                        )
                                                        sr_by_family = _aggregate_probe_rows(sr_rows, prefix="self_probe_mechanism")
                                                        for family_key, stats in sr_by_family.items():
                                                            family, source = family_key
                                                            self_reflection_rows.append({
                                                                "model_key": model_spec.registry_key,
                                                                "seed": seed,
                                                                "split_name": split_spec.name,
                                                                "train_size": train_size,
                                                                "family": family,
                                                                "source": source,
                                                                "self_reflection_mode": "self_probe_mechanism",
                                                                "probe_arch": probe_arch,
                                                                "probe_size": probe_size,
                                                                "mechanism_type": mechanism_type,
                                                                "layer": layer_idx,
                                                                "head": head_idx,
                                                                **stats,
                                                            })
                                                        del probe
                                                        torch.cuda.empty_cache()
                                                        progress.update()
                                                        logger.log_progress(prefix=mech_sr_label, progress_status=progress.status())

                                if "self_report_behavior" in config.self_reflection.modes and sr_examples:
                                    report_rows = eval_prompted_self_report_behavior(
                                        target_model, sr_examples,
                                        max_prompt_len=config.training.max_prompt_len,
                                        max_new_tokens=config.self_reflection.report_max_new_tokens,
                                        temperature=config.self_reflection.report_temperature,
                                    )
                                    report_by_family = _aggregate_report_rows(report_rows, mode_prefix="self_report_behavior")
                                    for family_key, stats in report_by_family.items():
                                        family, source = family_key
                                        self_reflection_rows.append({
                                            "model_key": model_spec.registry_key,
                                            "seed": seed,
                                            "split_name": split_spec.name,
                                            "train_size": train_size,
                                            "family": family,
                                            "source": source,
                                            "self_reflection_mode": "self_report_behavior",
                                            "probe_arch": "none",
                                            "probe_size": -1,
                                            "layer": -1,
                                            "head": -1,
                                            **stats,
                                        })

                                if "self_report_mechanism" in config.self_reflection.modes and sr_examples:
                                    for layer_idx in range(num_layers):
                                        for head_idx in range(num_heads):
                                            report_rows = eval_prompted_self_report_mechanism(
                                                target_model, sr_examples,
                                                layer_idx=layer_idx, head_idx=head_idx,
                                                max_prompt_len=config.training.max_prompt_len,
                                                max_new_tokens=config.self_reflection.report_max_new_tokens,
                                                temperature=config.self_reflection.report_temperature,
                                            )
                                            report_by_family = _aggregate_report_rows(report_rows, mode_prefix="self_report_mechanism")
                                            for family_key, stats in report_by_family.items():
                                                family, source = family_key
                                                self_reflection_rows.append({
                                                    "model_key": model_spec.registry_key,
                                                    "seed": seed,
                                                    "split_name": split_spec.name,
                                                    "train_size": train_size,
                                                    "family": family,
                                                    "source": source,
                                                    "self_reflection_mode": "self_report_mechanism",
                                                    "probe_arch": "none",
                                                    "probe_size": -1,
                                                    "layer": layer_idx,
                                                    "head": head_idx,
                                                    **stats,
                                                })

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

    self_reflection_summary = []
    promoted_self_reflection_summary = []
    validation_rows = []
    validation_path = None
    if self_reflection_rows:
        self_reflection_summary = _summarize_self_reflection_rows(self_reflection_rows)
        validation_rows = _evaluate_self_reflection_validation(self_reflection_summary, config.validation)
        promoted_keys = {
            (r['model_key'], r['family'], r['source'], r['self_reflection_mode'], r['probe_arch'], r['probe_size'], r['layer'], r['head'])
            for r in validation_rows if r['promoted']
        }
        promoted_self_reflection_summary = [
            r for r in self_reflection_summary
            if (r.get('model_key'), r.get('family'), r.get('source'), r.get('self_reflection_mode'), r.get('probe_arch'), r.get('probe_size'), r.get('layer'), r.get('head')) in promoted_keys
        ]
        save_csv(run_dir / "self_reflection_rows.csv", self_reflection_rows)
        save_csv(run_dir / "self_reflection_summary.csv", self_reflection_summary)
        save_csv(run_dir / "self_reflection_validation.csv", validation_rows)
        validation_path = _write_validation_checklist(run_dir, validation_rows)

    plot_mean_gap(summary_rows, run_dir / "plots")
    plot_frac_hard(summary_rows, run_dir / "plots")
    plot_p90_gap(summary_rows, run_dir / "plots")
    plot_dataset_scaling(seed_agg_rows, run_dir / "plots")
    for model_spec in enabled_model_specs:
        for mechanism_type in config.sweep.mechanism_types:
            plot_head_heatmap(raw_rows, run_dir / "plots", mechanism_type=mechanism_type, model_key=model_spec.registry_key)

    report_path = None
    if config.outputs.save_markdown_report:
        report_path = write_markdown_report(run_dir, config, summary_rows, seed_agg_rows, test_rows, promoted_self_reflection_summary if config.validation.promote_only_validated_optional else self_reflection_summary, validation_rows=validation_rows)
    if config.outputs.save_latex_tables:
        write_latex_tables(run_dir, seed_agg_rows, test_rows, promoted_self_reflection_summary if config.validation.promote_only_validated_optional else self_reflection_summary, validation_rows=validation_rows)
    if config.outputs.save_submission_bundle:
        extras = []
        if (run_dir / "plots").exists():
            extras.extend(sorted((run_dir / "plots").glob("*.png"))[:12])
        if validation_path is not None:
            extras.append(validation_path)
        build_submission_bundle(run_dir, report_path=report_path, extra_files=extras, bundle_name=config.outputs.submission_bundle_name)

    logger.log("Run complete.")
    logger.log(f"Outputs written to: {run_dir}")
