
"""CLI entry point.

Usage
-----
python -m frontier_interp.cli --config configs/debug_qwen.yaml
"""

from __future__ import annotations

import argparse

from frontier_interp.config_schema import load_config
from frontier_interp.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run config-driven frontier interpretability experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
