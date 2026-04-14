"""Output directory and serialization helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def save_csv(path: str | Path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def save_jsonl(path: str | Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def save_yaml_snapshot(path: str | Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
