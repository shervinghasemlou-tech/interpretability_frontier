"""Submission-facing report generation helpers.

These helpers turn raw experiment outputs into concise artifacts the paper can
use directly: a Markdown report, LaTeX tables, and a lightweight submission
bundle manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json
import shutil


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _top_rows(rows: List[Dict], *, key: str, n: int = 10, reverse: bool = True):
    rows = [r for r in rows if key in r]
    rows.sort(key=lambda r: r[key], reverse=reverse)
    return rows[:n]


def write_markdown_report(run_dir: str | Path, config, summary_rows: List[Dict], seed_agg_rows: List[Dict], test_rows: List[Dict], self_reflection_summary: Optional[List[Dict]] = None, validation_rows: Optional[List[Dict]] = None):
    run_dir = Path(run_dir)
    report_path = run_dir / "REPORT.md"

    top_positive = _top_rows(summary_rows, key="mean_gap", n=12, reverse=True)
    top_negative = _top_rows(summary_rows, key="mean_gap", n=12, reverse=False)
    strongest_tests = _top_rows([r for r in test_rows if str(r.get("q_value", "nan")) != "nan"], key="q_value", n=12, reverse=False)

    lines = []
    lines.append("# Frontier Interpretability Report\n")
    lines.append(f"Run name: `{config.outputs.run_name}`\n")
    lines.append(f"Description: {config.description}\n")
    lines.append("## Primary metrics\n")
    lines.append("Primary metrics are configured in the YAML and should be treated as confirmatory. Everything else is secondary or exploratory.\n")

    lines.append("## Top positive mean-gap settings\n")
    lines.append("| model | family | source | split | train_size | mech | arch | control | size | mean_gap | frac_hard |\n|---|---|---|---|---:|---|---|---|---:|---:|---:|")
    for r in top_positive:
        lines.append(f"| {r.get('model_key','')} | {r.get('family','')} | {r.get('source','')} | {r.get('split_name','')} | {r.get('train_size','')} | {r.get('mechanism_type','')} | {r.get('interpreter_arch','')} | {r.get('control','')} | {r.get('size','')} | {_fmt(r.get('mean_gap'))} | {_fmt(r.get('frac_hard_heads'))} |")

    lines.append("\n## Most negative mean-gap settings\n")
    lines.append("| model | family | source | split | train_size | mech | arch | control | size | mean_gap | frac_hard |\n|---|---|---|---|---:|---|---|---|---:|---:|---:|")
    for r in top_negative:
        lines.append(f"| {r.get('model_key','')} | {r.get('family','')} | {r.get('source','')} | {r.get('split_name','')} | {r.get('train_size','')} | {r.get('mechanism_type','')} | {r.get('interpreter_arch','')} | {r.get('control','')} | {r.get('size','')} | {_fmt(r.get('mean_gap'))} | {_fmt(r.get('frac_hard_heads'))} |")

    lines.append("\n## Strongest signed-gap tests\n")
    lines.append("| model | family | source | split | train_size | mech | arch | control | size | p | q |\n|---|---|---|---|---:|---|---|---|---:|---:|---:|")
    for r in strongest_tests:
        lines.append(f"| {r.get('model_key','')} | {r.get('family','')} | {r.get('source','')} | {r.get('split_name','')} | {r.get('train_size','')} | {r.get('mechanism_type','')} | {r.get('interpreter_arch','')} | {r.get('control','')} | {r.get('size','')} | {_fmt(r.get('p_value'))} | {_fmt(r.get('q_value'))} |")

    if seed_agg_rows:
        lines.append("\n## Seed-aggregated summary preview\n")
        lines.append("| model | family | source | split | train_size | mech | size | mean_gap_over_seeds | std_gap_over_seeds | mean_frac_hard |\n|---|---|---|---|---:|---|---:|---:|---:|---:|")
        for r in _top_rows(seed_agg_rows, key="mean_gap_over_seeds", n=12, reverse=True):
            lines.append(f"| {r.get('model_key','')} | {r.get('family','')} | {r.get('source','')} | {r.get('split_name','')} | {r.get('train_size','')} | {r.get('mechanism_type','')} | {r.get('size','')} | {_fmt(r.get('mean_gap_over_seeds'))} | {_fmt(r.get('std_gap_over_seeds'))} | {_fmt(r.get('mean_frac_hard_over_seeds'))} |")

    if validation_rows:
        promoted = sum(1 for r in validation_rows if r.get("promoted"))
        rejected = sum(1 for r in validation_rows if not r.get("promoted"))
        lines.append("\n## Optional-pathway validation\n")
        lines.append(f"Promoted optional self-reflection groupings: **{promoted}**\n")
        lines.append(f"Rejected optional self-reflection groupings: **{rejected}**\n")

    if self_reflection_summary:
        lines.append("\n## Self-reflection summary preview\n")
        lines.append("| model | family | source | mode | arch | size | layer | head | metric_1 | metric_2 |\n|---|---|---|---|---|---:|---:|---:|---:|---:|")
        for r in self_reflection_summary[:20]:
            metric_keys = [k for k in r.keys() if k.startswith('avg_')]
            m1 = r.get(metric_keys[0], '') if metric_keys else ''
            m2 = r.get(metric_keys[1], '') if len(metric_keys) > 1 else ''
            lines.append(f"| {r.get('model_key','')} | {r.get('family','')} | {r.get('source','')} | {r.get('self_reflection_mode','')} | {r.get('probe_arch','')} | {r.get('probe_size','')} | {r.get('layer','')} | {r.get('head','')} | {_fmt(m1)} | {_fmt(m2)} |")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_latex_tables(run_dir: str | Path, seed_agg_rows: List[Dict], test_rows: List[Dict], self_reflection_summary: Optional[List[Dict]] = None, validation_rows: Optional[List[Dict]] = None):
    run_dir = Path(run_dir)
    table_dir = run_dir / "paper_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    top_rows = sorted(seed_agg_rows, key=lambda r: r.get('mean_gap_over_seeds', float('-inf')), reverse=True)[:12]
    lines = []
    lines.append(r"% Auto-generated table: top mean-gap settings")
    lines.append(r"\begin{tabular}{l l l r r r}")
    lines.append(r"\toprule")
    lines.append(r"Model & Family & Mechanism & Size & Mean gap & Frac. hard \\")
    lines.append(r"\midrule")
    for r in top_rows:
        lines.append(f"{r.get('model_key','')} & {r.get('family','')} & {r.get('mechanism_type','')} & {r.get('size','')} & {_fmt(r.get('mean_gap_over_seeds'))} & {_fmt(r.get('mean_frac_hard_over_seeds'))} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (table_dir / "top_mean_gap_table.tex").write_text("\n".join(lines), encoding="utf-8")

    sig_rows = [r for r in test_rows if not isinstance(r.get('q_value'), str)]
    sig_rows = sorted(sig_rows, key=lambda r: float('inf') if r.get('q_value') != r.get('q_value') else r.get('q_value'))[:12]
    lines = []
    lines.append(r"% Auto-generated table: strongest signed-gap tests")
    lines.append(r"\begin{tabular}{l l l r r}")
    lines.append(r"\toprule")
    lines.append(r"Model & Family & Mechanism & p & q \\")
    lines.append(r"\midrule")
    for r in sig_rows:
        lines.append(f"{r.get('model_key','')} & {r.get('family','')} & {r.get('mechanism_type','')} & {_fmt(r.get('p_value'))} & {_fmt(r.get('q_value'))} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (table_dir / "signed_gap_tests_table.tex").write_text("\n".join(lines), encoding="utf-8")

    if self_reflection_summary:
        lines = []
        lines.append(r"% Auto-generated table: self-reflection preview")
        lines.append(r"\begin{tabular}{l l l r r}")
        lines.append(r"\toprule")
        lines.append(r"Model & Family & Mode & Layer & Head \\")
        lines.append(r"\midrule")
        for r in self_reflection_summary[:12]:
            lines.append(f"{r.get('model_key','')} & {r.get('family','')} & {r.get('self_reflection_mode','')} & {r.get('layer','')} & {r.get('head','')} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        (table_dir / "self_reflection_preview.tex").write_text("\n".join(lines), encoding="utf-8")


    if validation_rows:
        lines = []
        lines.append(r"% Auto-generated table: validation gate summary")
        lines.append(r"\begin{tabular}{l l r r}")
        lines.append(r"\toprule")
        lines.append("Mode & Probe arch & Promoted & Rejected \\")
        lines.append(r"\midrule")
        grouped = {}
        for r in validation_rows:
            key = (r.get('self_reflection_mode',''), r.get('probe_arch',''))
            grouped.setdefault(key, {'promoted': 0, 'rejected': 0})
            grouped[key]['promoted' if r.get('promoted') else 'rejected'] += 1
        for (mode, arch), stats in grouped.items():
            lines.append(f"{mode} & {arch} & {stats['promoted']} & {stats['rejected']} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        (table_dir / "validation_gate_summary.tex").write_text("\n".join(lines), encoding="utf-8")

    return table_dir


def build_submission_bundle(run_dir: str | Path, report_path: Optional[Path] = None, extra_files: Optional[List[Path]] = None, bundle_name: str = 'submission_bundle'):
    run_dir = Path(run_dir)
    bundle_dir = run_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    core = [
        run_dir / 'config_snapshot.yaml',
        run_dir / 'summary_rows.csv',
        run_dir / 'seed_aggregated_summary.csv',
        run_dir / 'signed_gap_tests.csv',
    ]
    if report_path is not None:
        core.append(report_path)
    if extra_files:
        core.extend(extra_files)

    manifest = []
    for path in core:
        if path.exists():
            dest = bundle_dir / path.name
            if path.resolve() != dest.resolve():
                shutil.copy2(path, dest)
            manifest.append({"file": path.name, "source": str(path)})

    (bundle_dir / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return bundle_dir
