"""Small helpers shared by the pipeline stages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def add_project_root_to_path() -> None:
    """Let ``scripts/*.py`` import ``lib`` when run directly."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def input_files(input_dir: Path, pattern: str = "*.parquet") -> list[Path]:
    files = sorted(Path(input_dir).glob(pattern))
    if not files:
        raise SystemExit(f"no files matching {pattern!r} under {input_dir}")
    return files


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_summary(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the earlier pipeline stage first")
    return json.loads(path.read_text())


def progress(iterable, total: int, desc: str):
    """tqdm when it is installed, a terse line-per-item fallback otherwise."""
    try:
        from tqdm import tqdm
    except ImportError:
        def _plain():
            for i, item in enumerate(iterable, 1):
                print(f"[{desc}] {i}/{total}", flush=True)
                yield item
        return _plain()
    return tqdm(iterable, total=total, desc=desc, unit="file")


def row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows
