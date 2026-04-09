"""Tests for 35 coupled van der Pol oscillators (70D stiff ODE system).

Each oscillator has position x_i and velocity v_i with damping mu_i,
giving 70 state variables total.  Damping coefficients span 0.5 to 100
on a log scale so the system has a wide stiffness range.
"""

import json
import shutil
import subprocess

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from tests.reference_solvers.python.scalar_rodas5 import (
    make_solver as make_rodas5_solver,
)

_N_OSC = 35
_N_VARS = 2 * _N_OSC  # 70: (x_i, v_i) for each oscillator
_T_SPAN = (0.0, 1.0)

# Damping coefficients: 35 values from ~0.5 to 100 on a log scale.
_MU = tuple(10.0 ** (-0.3 + 2.3 * i / (_N_OSC - 1)) for i in range(_N_OSC))

_JULIA_SCRIPT = "benchmarks/vdp_julia.jl"
_HAS_JULIA = shutil.which("julia") is not None


def _vdp_70d_ode(y, p):
    """35 van der Pol oscillators with different damping strengths.

    State: y = (x_0, v_0, x_1, v_1, ..., x_34, v_34)
    Each oscillator i:
        dx_i/dt = v_i
        dv_i/dt = p[0] * mu_i * (1 - x_i^2) * v_i - x_i

    p[0] scales all damping coefficients per trajectory for ensemble variability.
    """
    s = p[0]
    dy = [None] * _N_VARS
    for i in range(_N_OSC):
        x = y[2 * i]
        v = y[2 * i + 1]
        dy[2 * i] = v
        dy[2 * i + 1] = s * _MU[i] * (1.0 - x * x) * v - x
    return tuple(dy)


@jax.jit
def _vdp_70d_array(y, p):
    """Array-returning adapter for non-Pallas ensemble reference solver."""
    return jnp.array(_vdp_70d_ode(y, p))


def _run_julia(n=2, rtol=1e-6, atol=1e-8, timeout=600):
    """Run the Julia GPU ensemble VdP benchmark and return parsed JSON result."""
    cmd = ["julia", _JULIA_SCRIPT, str(n), str(rtol), str(atol)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Julia failed:\n{result.stderr}")
    return json.loads(result.stdout.strip())


@pytest.fixture
def params_batch_vdp(request):
    """VdP damping-scale parameter sets with +/-10% uniform perturbation."""
    N = request.param
    rng = np.random.default_rng(42)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((N, 1)) - 1.0),
        dtype=jnp.float64,
    )


@pytest.mark.parametrize(
    "params_batch_vdp", [2, 100, 1000, 10000, 100_000], indirect=True
)
def test_rodas5_vdp_ensemble_N(benchmark, params_batch_vdp):
    """Rodas5 vmap ensemble benchmark on the 70D van der Pol system."""
    y0 = jnp.array([2.0, 0.0] * _N_OSC, dtype=jnp.float64)
    solve = make_rodas5_solver(_vdp_70d_array)
    results = benchmark.pedantic(
        lambda: solve(
            y0=y0,
            t_span=_T_SPAN,
            params_batch=params_batch_vdp,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (params_batch_vdp.shape[0], len(_T_SPAN), _N_VARS)
    assert np.all(np.isfinite(results))


# ---------------------------------------------------------------------------
# Julia / DiffEqGPU.jl GPU ensemble benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JULIA, reason="Julia not installed")
@pytest.mark.parametrize("N", [2, 100, 1000, 10000, 100_000])
def test_julia_gpu_vdp_ensemble_N(benchmark, N):
    """Julia GPURosenbrock23 + EnsembleGPUKernel (CUDA, Float64) for 70D VdP."""
    data = benchmark.pedantic(
        lambda: _run_julia(N, rtol=1e-6, atol=1e-8),
        warmup_rounds=0,
        rounds=1,
    )

    # Override benchmark-reported time with Julia's internal solve timing
    from pytest_benchmark.stats import Stats

    stats = Stats()
    stats.update(data["adaptive_dt"]["min_time_ms"] / 1000)
    benchmark.stats.stats = stats
