"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable, TypeVar

import jax

T = TypeVar("T")


def get_gpu_name() -> str:
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        if out:
            return out
    except Exception:
        pass
    try:
        devices = jax.devices("gpu")
        if devices:
            return devices[0].device_kind
    except Exception:
        pass
    return "unknown_GPU"


def gpu_slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2))


def output_paths(script_dir: Path, gpu_name: str) -> tuple[Path, Path]:
    slug = gpu_slug(gpu_name)
    return script_dir / f"results-{slug}.csv", script_dir / f"plot-{slug}.png"


def time_blocked(run: Callable[[], T], n_runs: int) -> tuple[float, T]:
    result = run()
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = run()
        jax.block_until_ready(result)
    return (time.perf_counter() - t0) / n_runs * 1000, result
