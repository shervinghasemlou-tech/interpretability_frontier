"""Plot generation for paper figures."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def _group_summary(summary_rows: List[Dict], key_name: str):
    families = sorted(set(r["family"] for r in summary_rows))
    sizes = sorted(set(r["size"] for r in summary_rows))
    split_names = sorted(set(r.get("split_name", "all_to_all") for r in summary_rows))
    grouped = {}
    for split in split_names:
        grouped[split] = {}
        for fam in families:
            grouped[split][fam] = []
            for s in sizes:
                match = [
                    r for r in summary_rows
                    if r["family"] == fam and r["size"] == s and r["control"] == "none" and r.get("split_name", "all_to_all") == split and str(r.get("train_size", "full")) == 'full'
                ]
                grouped[split][fam].append(match[0][key_name] if match else np.nan)
    return families, sizes, split_names, grouped


def plot_mean_gap(summary_rows: List[Dict], out_dir: str, title_prefix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families, sizes, split_names, grouped = _group_summary(summary_rows, "mean_gap")

    for split in split_names:
        plt.figure(figsize=(9, 5))
        for fam in families:
            plt.plot(sizes, grouped[split][fam], marker="o", label=fam)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("Interpreter size")
        plt.ylabel("Mean gap (mechanism - behavior)")
        plt.title(f"{title_prefix}Average gap across heads | split={split}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"mean_gap_by_family_{split}.png", dpi=150)
        plt.close()


def plot_frac_hard(summary_rows: List[Dict], out_dir: str, title_prefix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families, sizes, split_names, grouped = _group_summary(summary_rows, "frac_hard_heads")

    for split in split_names:
        plt.figure(figsize=(9, 5))
        for fam in families:
            plt.plot(sizes, grouped[split][fam], marker="o", label=fam)
        plt.xlabel("Interpreter size")
        plt.ylabel("Fraction of heads with positive gap")
        plt.title(f"{title_prefix}Fraction of heads harder than behavior | split={split}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"frac_hard_heads_by_family_{split}.png", dpi=150)
        plt.close()


def plot_p90_gap(summary_rows: List[Dict], out_dir: str, title_prefix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families, sizes, split_names, grouped = _group_summary(summary_rows, "p90_gap")

    for split in split_names:
        plt.figure(figsize=(9, 5))
        for fam in families:
            plt.plot(sizes, grouped[split][fam], marker="o", label=fam)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("Interpreter size")
        plt.ylabel("P90 gap")
        plt.title(f"{title_prefix}Upper-tail gap across heads | split={split}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"p90_gap_by_family_{split}.png", dpi=150)
        plt.close()


def plot_dataset_scaling(seed_agg_rows: List[Dict], out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in seed_agg_rows if str(r.get("train_size", "full")) != "full" and r.get("control") == "none"]
    if not rows:
        return
    families = sorted(set(r["family"] for r in rows))
    sizes = sorted(set(r["size"] for r in rows))
    split_names = sorted(set(r.get("split_name", "all_to_all") for r in rows))
    train_sizes = sorted(set(int(r["train_size"]) for r in rows))

    for split in split_names:
        for size in sizes:
            plt.figure(figsize=(9, 5))
            for fam in families:
                ys = []
                for ts in train_sizes:
                    match = [r for r in rows if r["family"] == fam and r["size"] == size and r.get("split_name", "all_to_all") == split and int(r["train_size"]) == ts]
                    ys.append(match[0]["mean_gap_over_seeds"] if match else np.nan)
                plt.plot(train_sizes, ys, marker="o", label=fam)
            plt.xlabel("Train set size")
            plt.ylabel("Mean gap over seeds")
            plt.title(f"Dataset-size scaling | split={split} | size={size}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"dataset_scaling_split_{split}_size_{size}.png", dpi=150)
            plt.close()


def plot_head_heatmap(raw_rows: List[Dict], out_dir: str, mechanism_type: str, model_key: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        r for r in raw_rows
        if r["family"] == "ALL" and r["mechanism_type"] == mechanism_type and r["model_key"] == model_key and r["control"] == "none" and str(r.get("train_size", "full")) == 'full'
    ]
    if not rows:
        return

    sizes = sorted(set(r["size"] for r in rows))
    split_names = sorted(set(r.get("split_name", "all_to_all") for r in rows))
    for split in split_names:
        split_rows = [r for r in rows if r.get("split_name", "all_to_all") == split]
        if not split_rows:
            continue
        for size in sizes:
            sub = [r for r in split_rows if r["size"] == size]
            if not sub:
                continue
            n_layers = max(r["layer"] for r in sub) + 1
            n_heads = max(r["head"] for r in sub) + 1
            mat = np.zeros((n_layers, n_heads), dtype=np.float32)
            cnt = np.zeros((n_layers, n_heads), dtype=np.float32)
            for r in sub:
                mat[r["layer"], r["head"]] += r["gap"]
                cnt[r["layer"], r["head"]] += 1.0
            mat = mat / np.maximum(cnt, 1.0)

            plt.figure(figsize=(10, 6))
            plt.imshow(mat, aspect="auto")
            plt.colorbar(label="Mean gap")
            plt.xlabel("Head")
            plt.ylabel("Layer")
            plt.title(f"Headwise gap heatmap | model={model_key} | size={size} | split={split} | mechanism={mechanism_type}")
            plt.tight_layout()
            plt.savefig(out_dir / f"headwise_gap_heatmap_{model_key}_{mechanism_type}_{split}_size_{size}.png", dpi=150)
            plt.close()
