# Frontier Mechanistic Compressibility

A config-driven research codebase for studying **mechanistic compressibility** and **self-interpretability gaps** in sub-1B open-weight language models.

This repository is designed to be the experimental backbone for a paper on the distribution of internal compressibility across models, datasets, mechanism types, prompt families, interpreter capacities, and reviewer-facing controls.

## Design goals

- **One-command execution**: pass a single config file; everything else lives in YAML.
- **Model-agnostic**: supports a curated registry of sub-1B OSS base/instruct or base/chat pairs.
- **Dataset-agnostic**: supports handwritten diagnostics plus open datasets from Hugging Face.
- **Ablation-first**: counterarguments, controls, and counterfactuals are first-class config items.
- **Paper-ready outputs**: raw CSVs, seed-level summaries, FDR-corrected tests, plots, logs, config snapshots, per-example artifacts, Markdown reports, LaTeX tables, and a submission bundle.

## What the repository can do

It now supports both **external compression** experiments and **self-reflection** experiments. The latter include:
- self-probes trained on the target model's own hidden states,
- prompted self-reports where the target model is asked to predict its own behavior or head-level attention target.


- Distill behavior from a frozen target model into a smaller interpreter.
- Predict internal mechanisms from the same frozen target model using the **same interpreter family**.
- Sweep over:
  - models
  - datasets / prompt families
  - interpreter sizes
  - random seeds
  - layers / heads
  - mechanism types
  - controls and counterfactuals
  - architecture controls
- Compute:
  - mean / median gap
  - 90th percentile gap
  - easiest / hardest mechanism
  - fraction of mechanisms harder than behavior
  - layerwise and headwise summaries
  - bootstrap confidence intervals and FDR-corrected nonparametric tests
- Generate plots for paper figures.

## Mechanism views currently implemented

The main predictor still outputs an attention matrix, but the code now supports several **mechanism views** over that object:

- `attention_probs`: full attention matrix matching
- `attention_top1`: top-attended-token prediction
- `attention_entropy`: row-wise entropy prediction

This lets the paper separate “full probability reconstruction” from simpler derived mechanism targets.

## Behavioral evaluation currently implemented

- **KL distillation** over the frozen target model's next-token distribution
- **Proper restricted-choice evaluation** for datasets such as HellaSwag, PIQA, and ARC
  - target/interpreter choice agreement
  - rank correlation
  - interpreter MCQ accuracy against gold
  - target-model MCQ accuracy against gold
- The summaries now surface MCQ accuracy explicitly instead of leaving it buried in per-example artifacts.

## Repository layout

```text
frontier_interp_repo/
├── configs/                  # YAML configs; one command points here
├── paper/                    # LaTeX scaffold for the manuscript
├── src/frontier_interp/      # Python package
│   ├── analysis/             # metrics, stats, plots
│   ├── data/                 # prompt suites and dataset loaders
│   ├── experiments/          # runners, controls, mechanism complexity
│   ├── modeling/             # interpreters and target-model wrappers
│   ├── registries/           # curated model and dataset registries
│   └── utils/                # logging, io, seeds, timing
├── tests/                    # lightweight smoke tests
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## One-command usage

```bash
python -m frontier_interp.cli --config configs/debug_qwen.yaml
```

Everything about the run lives in the config, including:
- model sweep
- datasets
- full-split vs subsample loading
- ablations
- controls
- seeds
- budgets
- plots
- statistics
- report and submission-bundle generation


## Low-memory runs for 8 GB GPUs

For smaller GPUs, the frozen target model should use either:
- a smaller registry model such as `smollm2_360m_base`, or
- bitsandbytes quantization with `load_in_8bit: true` or `load_in_4bit: true`, ideally with `device_map: auto`.

Useful starter configs:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m frontier_interp.cli --config configs/tiny_smoke.yaml
python -m frontier_interp.cli --config configs/debug_qwen_8bit.yaml
```

`expandable_segments` can help reduce fragmentation in PyTorch's CUDA allocator, but it does not replace actual memory reduction from smaller models or quantization.

## Recommended execution order

### 1. Fast smoke test
```bash
python -m frontier_interp.cli --config configs/debug_qwen.yaml
```

### 2. Restricted-choice sanity test
```bash
python -m frontier_interp.cli --config configs/debug_choice_eval.yaml
```

### 3. Mechanism-view ablation
```bash
python -m frontier_interp.cli --config configs/ablate_mechanism_views.yaml
```

### 4. Architecture and control ablations
```bash
python -m frontier_interp.cli --config configs/counterfactual_controls.yaml
```

### 5. Cross-model debugging
```bash
python -m frontier_interp.cli --config configs/debug_multimodel.yaml
```

### 6. Tier-1 paper core
```bash
python -m frontier_interp.cli --config configs/tier1_core.yaml
```

### 7. Submission-ready balanced run
```bash
python -m frontier_interp.cli --config configs/submit_ready_balanced.yaml
```

## Included model registry

The registry includes a deliberate mix of base / instruct or base / chat pairs under 1B parameters:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `apple/OpenELM-450M`
- `apple/OpenELM-450M-Instruct`
- `h2oai/h2o-danube3-500m-base`
- `h2oai/h2o-danube3-500m-chat`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-410m-deduped`

## Included dataset families

### Handwritten diagnostics
- factual continuations
- repeated-span prompts
- pattern continuation prompts
- reasoning-like prompts

### OSS datasets
- WikiText
- HellaSwag
- PIQA
- AI2 ARC
- GSM8K
- Alpaca
- UltraChat 200k

## Counterarguments and controls supported

- label shuffle
- random target matrices
- diagonal / causal attention baseline
- architecture controls (`transformer`, `mlp`, `position_only`)
- restricted-choice behavioral evaluation
- mechanism complexity covariates
- multiple mechanism views over attention
- cross-family generalization splits
- dataset-size scaling sweeps
- FDR-corrected signed-gap tests

## Outputs

Each run writes to `outputs/<run_name>/`:
- `raw_rows.csv`
- `summary_rows.csv`
- `seed_level_summary.csv`
- `seed_aggregated_summary.csv`
- `signed_gap_tests.csv`
- `REPORT.md`
- `paper_tables/*.tex`
- `submission_bundle/*`
- `plots/*.png`
- `artifacts/*.jsonl`
- `logs/run.log`
- config snapshot
- model-card snapshots

## Caveat

This repository is intended to be **near-final research code**, not a toy notebook. Some larger reviewer-facing experiments remain computationally expensive, so the configs are split into:
- smoke tests
- restricted-choice debugging
- mechanism-view ablations
- architecture/control ablations
- cross-model debugging
- tier-1 core experiments
- full-paper template

## New configs for reviewer-preemptive checks

- `configs/debug_cross_family.yaml`: train on one family and evaluate on another
- `configs/debug_dataset_scaling.yaml`: measure how the gap changes as train size scales

These are designed to answer two likely reviewer questions early:
1. does the effect survive out-of-family evaluation?
2. is the result just a small-data artifact?


## Optional-pathway validation gates

Self-report experiments are kept as first-class raw outputs, but the submission-facing report can automatically **promote** only those self-report groupings that pass validation thresholds from the config. This is intended to prevent weak optional pathways from contaminating the confirmatory story while still preserving all exploratory artifacts.

Key knobs live under `validation:` in the YAML config. The runner emits:
- `self_reflection_validation.csv`
- `VALIDATION_CHECKLIST.md`

These files document which optional pathways were promoted and which remained exploratory.
