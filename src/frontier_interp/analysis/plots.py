"""Plot generation for paper figures."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def _group_summary(summary_rows: List[Dict], key_name: str):
    families = sorted(set(r["family"] for r in summary_rows))
    sizes = sorted(set(r["size"] for r in summary_rows))
    grouped = {}
    for fam in families:
        grouped[fam] = []
        for s in sizes:
            match = [r for r in summary_rows if r["family"] == fam and r["size"] == s and r["control"] == "none"]
            grouped[fam].append(match[0][key_name] if match else np.nan)
    return families, sizes, grouped


def plot_mean_gap(summary_rows: List[Dict], out_dir: str, title_prefix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families, sizes, grouped = _group_summary(summary_rows, "mean_gap")

    plt.figure(figsize=(9, 5))
    for fam in families:
        plt.plot(sizes, grouped[fam], marker="o", label=fam)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Interpreter size")
    plt.ylabel("Mean gap (mechanism - behavior)")
    plt.title(f"{title_prefix}Average gap across heads")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_gap_by_family.png", dpi=150)
    plt.close()


def plot_frac_hard(summary_rows: List[Dict], out_dir: str, title_prefix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families, sizes, grouped = _group_summary(summary_rows, "frac_hard_heads")

    plt.figure(figsize=(9, 5))
    for fam in families:
        plt.plot(sizes, grouped[fam], marker="o", label=fam)
    plt.xlabel("Interpreter size")
    plt.ylabel("Fraction of heads with positive gap")
    plt.title(f"{title_prefix}Fraction of heads harder than behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "frac_hard_heads_by_family.png", dpi=150)
    plt.close()


def plot_p90_gap(summary_rows: List[Dict], out_dir: str, title_prefix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families, sizes, grouped = _group_summary(summary_rows, "p90_gap")

    plt.figure(figsize=(9, 5))
    for fam in families:
        plt.plot(sizes, grouped[fam], marker="o", label=fam)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Interpreter size")
    plt.ylabel("P90 gap")
    plt.title(f"{title_prefix}Upper-tail gap across heads")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "p90_gap_by_family.png", dpi=150)
    plt.close()


def plot_head_heatmap(raw_rows: List[Dict], out_dir: str, mechanism_type: str, model_key: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        r for r in raw_rows
        if r["family"] == "ALL" and r["mechanism_type"] == mechanism_type and r["model_key"] == model_key and r["control"] == "none"
    ]
    if not rows:
        return

    sizes = sorted(set(r["size"] for r in rows))
    for size in sizes:
        sub = [r for r in rows if r["size"] == size]
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
        plt.title(f"Headwise gap heatmap | model={model_key} | size={size} | mechanism={mechanism_type}")
        plt.tight_layout()
        plt.savefig(out_dir / f"headwise_gap_heatmap_{model_key}_{mechanism_type}_size_{size}.png", dpi=150)
        plt.close()
