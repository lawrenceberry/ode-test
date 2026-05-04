"""Dispatcher for numba-cuda Rodas5 custom kernels."""

from __future__ import annotations

from solvers._rodas5ckn_common import make_lu_solver
from solvers.rodas5cknp import (
    prepare_solve as prepare_solve_packed,
)
from solvers.rodas5cknp import (
    run_prepared as run_prepared_packed,
)
from solvers.rodas5cknp import (
    solve as solve_packed,
)
from solvers.rodas5ckns import (
    prepare_solve as prepare_solve_single,
)
from solvers.rodas5ckns import (
    run_prepared as run_prepared_single,
)
from solvers.rodas5ckns import (
    solve as solve_single,
)


def _use_single(n_vars: int) -> bool:
    return int(make_lu_solver(n_vars).batches_per_block) == 1


def prepare_solve(ode_fn, jac_fn, y0, t_span, params, **kwargs):
    n_vars = y0.shape[1] if y0.ndim == 2 else y0.shape[0]
    if _use_single(int(n_vars)):
        return prepare_solve_single(ode_fn, jac_fn, y0, t_span, params, **kwargs)
    return prepare_solve_packed(ode_fn, jac_fn, y0, t_span, params, **kwargs)


def run_prepared(prepared, **kwargs):
    if int(prepared.lu_solver.batches_per_block) == 1:
        return run_prepared_single(prepared, **kwargs)
    return run_prepared_packed(prepared, **kwargs)


def solve(ode_fn, jac_fn, y0, t_span, params, **kwargs):
    n_vars = y0.shape[1] if y0.ndim == 2 else y0.shape[0]
    if _use_single(int(n_vars)):
        return solve_single(ode_fn, jac_fn, y0, t_span, params, **kwargs)
    return solve_packed(ode_fn, jac_fn, y0, t_span, params, **kwargs)
