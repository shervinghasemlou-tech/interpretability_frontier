# Frontier Mechanistic Compressibility

A config-driven codebase for studying **mechanistic compressibility** and **self-interpretability gaps** in sub-1B open-weight language models.

This repository is designed to be the experimental backbone for a paper on the distribution of internal compressibility across models, prompt families, mechanism types, and interpreter capacities.

## Key design goals

- **One-command execution**: pass a single config file; everything else is in the config.
- **Model-agnostic**: supports a registry of sub-1B OSS models and base/instruct pairs.
- **Dataset-agnostic**: supports handwritten diagnostics plus open datasets from Hugging Face.
- **Ablation-first**: controls, counterfactuals, and reviewer-facing objections are first-class configuration items.
- **Paper-ready outputs**: raw CSVs, summary CSVs, plots, logs, and a minimal paper scaffold.

## What this repository can do

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
- Compute:
  - mean / median gap
  - 90th percentile gap
  - easiest / hardest mechanism
  - fraction of mechanisms harder than behavior
  - layerwise and headwise summaries
  - bootstrap confidence intervals and nonparametric tests
- Generate plots for paper figures.

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
- ablations
- controls
- seeds
- budgets
- plots
- statistics

## Recommended execution order

### 1. Fast smoke test
```bash
python -m frontier_interp.cli --config configs/debug_qwen.yaml
```

### 2. Cross-model debugging
```bash
python -m frontier_interp.cli --config configs/debug_multimodel.yaml
```

### 3. Tier-1 paper core
```bash
python -m frontier_interp.cli --config configs/tier1_core.yaml
```

### 4. Controls / counterfactuals
```bash
python -m frontier_interp.cli --config configs/counterfactual_controls.yaml
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

These choices emphasize:
- modern compact base / instruct pairs,
- architectural diversity,
- and at least one family explicitly designed for interpretability research.

## Included dataset families

The repository supports both a handwritten diagnostic suite and OSS datasets.

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

## Controls and counterfactuals supported in config

- label shuffle
- head shuffle
- layer shuffle
- random target matrices
- diagonal / causal attention baseline
- position-only interpreter
- architecture controls (linear / MLP / transformer)
- train-on-family / test-on-family transfer
- mechanism complexity covariates

## Notes on philosophy

This codebase is built around a **distributional** view of self-interpretability. It does **not** assume that all internal mechanisms are uniformly harder than behavior. Instead, it measures a spectrum of compressibility and focuses especially on:

- average-case gap,
- upper-tail gap,
- and the subset of mechanisms that remain hard under matched interpreter budgets.

## Outputs

Each run writes to `outputs/<run_name>/`:
- `raw_rows.csv`
- `summary_rows.csv`
- `seed_level_summary.csv`
- `seed_aggregated_summary.csv`
- `plots/*.png`
- `logs/run.log`
- config snapshot

## Paper scaffold

The `paper/` folder contains a simple LaTeX skeleton that mirrors the intended narrative:
- abstract
- introduction
- related work
- methods
- results
- limitations
- appendix

## Caveat

This repository aims to be **near-final research code**, not a toy notebook. It is intentionally modular and ablation-heavy. Full runs over many models, datasets, and heads can be expensive. The provided configs are split into:
- debugging
- tier-1 core experiments
- counterfactuals
- full paper sweeps

