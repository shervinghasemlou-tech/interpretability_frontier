"""Statistical summaries, bootstrap intervals, and multiple-testing correction."""

from __future__ import annotations

from typing import Iterable, Dict, List
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values: Iterable[float], confidence_level: float = 0.95, num_bootstrap: int = 1000, seed: int = 0):
    values = np.asarray(list(values), dtype=np.float32)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(num_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    alpha = 1.0 - confidence_level
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction.

    Returns q-values aligned to the input order.
    """
    arr = np.asarray(p_values, dtype=np.float64)
    n = len(arr)
    order = np.argsort(arr)
    ranked = arr[order]
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.tolist()


def summarize_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in raw_rows:
        grouped[(r["model_key"], r["size"], r["family"], r["mechanism_type"], r["interpreter_arch"], r["control"])].append(r)

    summary = []
    for key, rows in grouped.items():
        gaps = np.array([r["gap"] for r in rows], dtype=np.float32)
        p90_gap = float(np.quantile(gaps, 0.9)) if len(gaps) else float("nan")
        lo, hi = bootstrap_ci(gaps)
        summary.append({
            "model_key": key[0],
            "size": key[1],
            "family": key[2],
            "mechanism_type": key[3],
            "interpreter_arch": key[4],
            "control": key[5],
            "num_rows": len(rows),
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "p90_gap": p90_gap,
            "std_gap": float(np.std(gaps)),
            "frac_hard_heads": float(np.mean(gaps > 0.0)),
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
            "max_gap": float(np.max(gaps)),
            "min_gap": float(np.min(gaps)),
        })
    return summary


def summarize_by_seed(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in raw_rows:
        grouped[(r["model_key"], r["seed"], r["size"], r["family"], r["mechanism_type"], r["interpreter_arch"], r["control"])].append(r)

    rows = []
    for key, items in grouped.items():
        gaps = np.array([r["gap"] for r in items], dtype=np.float32)
        rows.append({
            "model_key": key[0],
            "seed": key[1],
            "size": key[2],
            "family": key[3],
            "mechanism_type": key[4],
            "interpreter_arch": key[5],
            "control": key[6],
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "p90_gap": float(np.quantile(gaps, 0.9)),
            "std_gap": float(np.std(gaps)),
            "frac_hard_heads": float(np.mean(gaps > 0.0)),
        })
    return rows


def summarize_seed_aggregates(seed_rows: List[Dict], confidence_level: float = 0.95, bootstrap_iterations: int = 1000) -> List[Dict]:
    grouped = defaultdict(list)
    for r in seed_rows:
        grouped[(r["model_key"], r["size"], r["family"], r["mechanism_type"], r["interpreter_arch"], r["control"])].append(r)

    out = []
    for key, rows in grouped.items():
        mean_gaps = np.array([r["mean_gap"] for r in rows], dtype=np.float32)
        frac_hard = np.array([r["frac_hard_heads"] for r in rows], dtype=np.float32)
        lo, hi = bootstrap_ci(mean_gaps, confidence_level=confidence_level, num_bootstrap=bootstrap_iterations)
        out.append({
            "model_key": key[0],
            "size": key[1],
            "family": key[2],
            "mechanism_type": key[3],
            "interpreter_arch": key[4],
            "control": key[5],
            "num_seeds": len(rows),
            "mean_gap_over_seeds": float(np.mean(mean_gaps)),
            "std_gap_over_seeds": float(np.std(mean_gaps)),
            "mean_frac_hard_over_seeds": float(np.mean(frac_hard)),
            "std_frac_hard_over_seeds": float(np.std(frac_hard)),
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
        })
    return out


def signed_gap_test(seed_rows: List[Dict], fdr_method: str = "bh") -> List[Dict]:
    """Perform a Wilcoxon signed-rank test on seed-level mean gaps against zero."""
    grouped = defaultdict(list)
    for r in seed_rows:
        grouped[(r["model_key"], r["size"], r["family"], r["mechanism_type"], r["interpreter_arch"], r["control"])].append(r["mean_gap"])

    out = []
    for key, vals in grouped.items():
        if len(vals) < 2:
            p = float("nan")
        else:
            try:
                p = float(wilcoxon(vals).pvalue)
            except ValueError:
                p = float("nan")
        out.append({
            "model_key": key[0],
            "size": key[1],
            "family": key[2],
            "mechanism_type": key[3],
            "interpreter_arch": key[4],
            "control": key[5],
            "p_value": p,
        })

    finite_p = [r["p_value"] for r in out if not np.isnan(r["p_value"])]
    q_values = benjamini_hochberg(finite_p) if finite_p else []
    j = 0
    for r in out:
        if np.isnan(r["p_value"]):
            r["q_value"] = float("nan")
        else:
            r["q_value"] = q_values[j]
            j += 1
    return out
