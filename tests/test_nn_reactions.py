"""Tests for synthetic stiff nearest-neighbor mass-conserving ODE systems."""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest
from scipy.linalg import expm

from solvers.rodas5 import make_solver as make_rodas5_v2_linear_solver
from tests.reference_solvers.python.scalar_rodas5 import (
    make_solver as make_rodas5_solver,
)

_T_SPAN = (0.0, 1.0)
_V6_MULTI_SAVE_TIMES = jnp.array((0.0, 0.125, 0.25, 0.5, 1.0), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]
_LINEAR_SOLVER_PRECISIONS = ("fp64", "fp32")


def _time_linear_scale(t):
    """Scalar multiplier for the time-dependent linear systems."""
    return 1.0 + 0.2 * t


def _time_linear_scale_integral(t0, tf):
    """Integral of ``_time_linear_scale`` over ``[t0, tf]``."""
    return (tf - t0) + 0.1 * (tf**2 - t0**2)


def _build_linear_chain_matrix(n_vars, kf, kb):
    """Build matrix M for the D-dimensional linear chain dy/dt = M y."""
    M = np.zeros((n_vars, n_vars), dtype=np.float64)
    M[0, 0] = -kf[0]
    M[0, 1] = kb[0]

    for i in range(1, n_vars - 1):
        M[i, i - 1] = kf[i - 1]
        M[i, i] = -(kb[i - 1] + kf[i])
        M[i, i + 1] = kb[i]

    M[n_vars - 1, n_vars - 2] = kf[n_vars - 2]
    M[n_vars - 1, n_vars - 1] = -kb[n_vars - 2]
    return M


def _make_nn_reaction_system(n_vars):
    """Construct a configurable D-dimensional nearest-neighbor reaction system."""
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    edge_count = n_vars - 1
    kf = tuple(10.0 ** (-2.0 + 8.0 * i / (edge_count - 1)) for i in range(edge_count))
    kb = tuple(10.0 ** (6.0 - 8.0 * i / (edge_count - 1)) for i in range(edge_count))
    M_np = _build_linear_chain_matrix(n_vars, kf, kb)
    M = jnp.array(M_np, dtype=jnp.float64)
    y0 = jnp.array([1.0] + [0.0] * (n_vars - 1), dtype=jnp.float64)
    y0_np = np.asarray(y0, dtype=np.float64)

    def ode(y, p):
        """D-dimensional stiff linear chain with conserved total mass."""
        s = p[0]
        dy = [None] * n_vars
        dy[0] = -s * kf[0] * y[0] + s * kb[0] * y[1]

        for i in range(1, n_vars - 1):
            dy[i] = (
                s * kf[i - 1] * y[i - 1]
                - s * (kb[i - 1] + kf[i]) * y[i]
                + s * kb[i] * y[i + 1]
            )

        dy[n_vars - 1] = (
            s * kf[n_vars - 2] * y[n_vars - 2] - s * kb[n_vars - 2] * y[n_vars - 1]
        )
        return tuple(dy)

    @jax.jit
    def array(y, p):
        """Array-returning adapter for the non-Pallas reference solver."""
        return jnp.array(ode(y, p))

    @jax.jit
    def ode_array(y, t, p):
        """Time-aware adapter for the nonlinear path of rodas5_v2_linear."""
        del t
        return jnp.array(ode(y, p))

    def jac(y, p):
        """Explicit Jacobian for the D-dimensional stiff linear chain."""
        del y
        s = p[0]
        return tuple(
            tuple(s * M_np[i, j] for j in range(n_vars)) for i in range(n_vars)
        )

    def time_jac(t):
        """Time-dependent matrix callback for the v6 solver."""
        s = _time_linear_scale(t)
        return tuple(
            tuple(s * M_np[i, j] for j in range(n_vars)) for i in range(n_vars)
        )

    def jac_array(t, p):
        """Jacobian as a JAX array, function of time and params."""
        s = p[0]
        return s * M

    def time_exact(y_init, t_span):
        """Closed-form solution for M(t) = a(t) * M with commuting matrices."""
        scale_int = _time_linear_scale_integral(*t_span)
        return jnp.array(expm(scale_int * M_np) @ np.asarray(y_init), dtype=jnp.float64)

    return {
        "n_vars": n_vars,
        "matrix": M,
        "matrix_np": M_np,
        "ode": ode,
        "array": array,
        "ode_array": ode_array,
        "jac": jac,
        "time_jac": time_jac,
        "jac_array": jac_array,
        "time_exact": time_exact,
        "y0": y0,
        "y0_np": y0_np,
    }


def _dim_id(n_vars):
    return f"{n_vars}d"


def _linear_solver_id(linear_solver):
    return linear_solver


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _broadcast_y0(y0, size):
    return jnp.broadcast_to(y0, (size, y0.shape[0]))


def _time_exact_trajectory(system, save_times):
    save_times_np = np.asarray(save_times, dtype=np.float64)
    t0 = float(save_times_np[0])
    return np.stack(
        [
            np.asarray(
                system["time_exact"](system["y0"], (t0, float(tf))), dtype=np.float64
            )
            for tf in save_times_np
        ],
        axis=0,
    )


def _reference_trajectory(system, params_batch, save_times):
    solve_ref = make_rodas5_solver(system["array"])
    return np.asarray(
        solve_ref(
            y0=system["y0"],
            t_span=save_times,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready()
    )


@pytest.fixture
def nn_reaction_system(request):
    """Configurable nearest-neighbor reaction system parameterized by dimension."""
    return _make_nn_reaction_system(request.param)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 vmap ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_solver(system["array"])
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_make_solver_matches_solve_ensemble(nn_reaction_system):
    """Validate separate Rodas5 solver instances agree on the same problem."""
    N = 100
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=123)

    solve_a = make_rodas5_solver(system["array"])
    solve_b = make_rodas5_solver(system["array"])
    y_a = solve_a(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_b = solve_b(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    np.testing.assert_allclose(y_a, y_b, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Rodas5 — ode_fn path (nonlinear, AD Jacobian)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("batch_size", [1, 7, None], ids=["bs1", "bs7", "bsN"])
def test_rodas5_ode_fn_matches_reference(nn_reaction_system, batch_size):
    """Validate ode_fn (nonlinear) path against vmap reference."""
    N = 100
    system = nn_reaction_system
    params = _make_params_batch(N, seed=0)

    solve_v2 = make_rodas5_v2_linear_solver(
        ode_fn=system["ode_array"], batch_size=batch_size
    )
    solve_ref = make_rodas5_solver(system["array"])
    y_v2 = solve_v2(
        y0=system["y0"],
        t_span=_T_SPAN,
        params=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = solve_ref(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v2.shape == (N, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(
        y_v2[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0
    )
    np.testing.assert_allclose(y_v2.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v2, y_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_ode_fn_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 ode_fn ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve_v2 = make_rodas5_v2_linear_solver(ode_fn=system["ode_array"])
    results = benchmark.pedantic(
        lambda: solve_v2(
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


# ---------------------------------------------------------------------------
# Rodas5 — jac_fn path (linear, no AD)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_jac_fn_matches_reference(nn_reaction_system):
    """Validate jac_fn (linear) path against the ode_fn (AD) path."""
    N = 256
    system = nn_reaction_system
    params = _make_params_batch(N, seed=0)

    solve_linear = make_rodas5_v2_linear_solver(jac_fn=system["jac_array"])
    solve_ref = make_rodas5_solver(system["array"])
    y_jac = solve_linear(
        y0=system["y0"],
        t_span=_T_SPAN,
        params=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = solve_ref(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_jac.shape == (N, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(
        y_jac[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0
    )
    np.testing.assert_allclose(y_jac.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_jac[:, -1, :], y_ref[:, -1, :], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_jac_fn_fp32_matches_fp64(nn_reaction_system):
    """Validate FP32 LU solves against the FP64 baseline on the jac_fn path."""
    N = 256
    system = nn_reaction_system
    params = _make_params_batch(N, seed=0)

    solve_fp64 = make_rodas5_v2_linear_solver(
        jac_fn=system["jac_array"], lu_precision="fp64"
    )
    y_fp64 = solve_fp64(
        y0=system["y0"], t_span=_T_SPAN, params=params, first_step=1e-6, rtol=1e-6, atol=1e-8
    ).block_until_ready()
    solve_fp32 = make_rodas5_v2_linear_solver(
        jac_fn=system["jac_array"], lu_precision="fp32"
    )
    y_fp32 = solve_fp32(
        y0=system["y0"], t_span=_T_SPAN, params=params, first_step=1e-6, rtol=1e-6, atol=1e-8
    ).block_until_ready()

    np.testing.assert_allclose(
        np.asarray(y_fp32), np.asarray(y_fp64), rtol=2e-4, atol=3e-8
    )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_jac_fn_saves_multiple_times(nn_reaction_system):
    """Validate multiple save times on the jac_fn path."""
    N = 64
    system = nn_reaction_system
    params = _make_params_batch(N, seed=0)

    solve_linear = make_rodas5_v2_linear_solver(
        jac_fn=system["jac_array"], lu_precision="fp32"
    )
    y_hist = solve_linear(
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_ref = _reference_trajectory(system, params, _V6_MULTI_SAVE_TIMES)

    assert y_hist.shape == (N, len(_V6_MULTI_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(
        y_hist[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0
    )
    np.testing.assert_allclose(y_hist.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(y_hist), y_ref, rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_jac_fn_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 jac_fn (linear, no AD) ensemble benchmark."""
    system = nn_reaction_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve_linear = make_rodas5_v2_linear_solver(
        jac_fn=system["jac_array"],
        lu_precision="fp32",
        mv_precision="fp32",
    )
    results = benchmark.pedantic(
        lambda: solve_linear(
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
@pytest.mark.parametrize("lu_precision", _LINEAR_SOLVER_PRECISIONS, ids=_linear_solver_id)
def test_rodas5_jac_fn_precision_timing(benchmark, nn_reaction_system, lu_precision):
    """Benchmark the jac_fn path with FP64 vs FP32 LU precision."""
    ensemble_size = 10_000
    system = nn_reaction_system
    params = _make_params_batch(ensemble_size, seed=42)

    solve_linear = make_rodas5_v2_linear_solver(
        jac_fn=system["jac_array"],
        lu_precision=lu_precision,
    )
    results = benchmark.pedantic(
        lambda: solve_linear(
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    benchmark.extra_info["backend"] = jax.default_backend()

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)
