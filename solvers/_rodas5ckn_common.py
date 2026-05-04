"""Shared utilities for numba-cuda Rodas5 custom kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import cuda
from nvmath.device import LUPivotSolver


def as_launch_block_dim(block_dim):
    if isinstance(block_dim, int):
        return block_dim
    if isinstance(block_dim, (tuple, list)):
        return tuple(int(x) for x in block_dim)
    x = getattr(block_dim, "x", None)
    y = getattr(block_dim, "y", 1)
    z = getattr(block_dim, "z", 1)
    if x is not None:
        return (int(x), int(y), int(z))
    return 128


def block_threads_x(block_dim) -> int:
    launch = as_launch_block_dim(block_dim)
    if isinstance(launch, int):
        return launch
    return int(launch[0])


@dataclass
class Workspace:
    y0_dev: Any
    times_dev: Any
    params_dev: Any
    hist_dev: Any
    accepted_dev: Any
    rejected_dev: Any
    loop_dev: Any
    work: list[Any]
    jac_dev: Any


@dataclass(frozen=True)
class PreparedSolve:
    kernel: Any
    lu_solver: Any
    workspace: Workspace
    dt0: np.float64
    rtol: np.float64
    atol: np.float64
    max_steps: np.int32
    blocks: int
    threads: Any


def make_lu_solver(
    n_vars: int,
    *,
    batches_per_block="suggested",
    block_dim="suggested",
):
    return LUPivotSolver(
        size=(n_vars, n_vars, 1),
        precision=np.float32,
        execution="Block",
        arrangement=("row_major", "row_major"),
        batches_per_block=batches_per_block,
        block_dim=block_dim,
    )


def get_workspace(
    cache: dict, n: int, n_vars: int, n_save: int, n_params: int
) -> Workspace:
    key = (n, n_vars, n_save, n_params)
    workspace = cache.get(key)
    if workspace is not None:
        return workspace

    workspace = Workspace(
        y0_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        times_dev=cuda.device_array(n_save, dtype=np.float64),
        params_dev=cuda.device_array((n, n_params), dtype=np.float64),
        hist_dev=cuda.device_array((n, n_save, n_vars), dtype=np.float64),
        accepted_dev=cuda.device_array(n, dtype=np.int32),
        rejected_dev=cuda.device_array(n, dtype=np.int32),
        loop_dev=cuda.device_array(n, dtype=np.int32),
        work=[cuda.device_array((n, n_vars), dtype=np.float64) for _ in range(10)],
        jac_dev=cuda.device_array((n, n_vars, n_vars), dtype=np.float64),
    )
    cache[key] = workspace
    return workspace
