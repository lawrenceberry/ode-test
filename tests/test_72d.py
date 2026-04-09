"""Benchmarks for 72D CMB-inspired Boltzmann ODE system.

72-dimensional ODE with time-varying stiffness mimicking CMB Einstein-Boltzmann
equations. The system consists of coupled damped oscillators with sinusoidal
solutions at varying frequencies:

- y[0]: clock variable (dy/dt = 1, so y[0] = t)
- y[1..36]: 18 non-stiff oscillator pairs (constant damping 0.01)
- y[37..70]: 17 stiff oscillator pairs (damping = p[0] * clock, grows with time)
- y[71]: single stiff exponential decay mode (rate = p[1] * clock)

At t=0 all modes are non-stiff. As clock advances, the stiff modes acquire
large damping (stiffness ratio ~10,000 by t=10), mimicking the tight-coupling
regime in CMB Boltzmann solvers.
"""

import json
import math
import shutil
import subprocess

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.rodas5 import make_solver as make_rodas5_solver

_JULIA_SCRIPT = "benchmarks/boltzmann_72d_julia.jl"
_HAS_JULIA = shutil.which("julia") is not None

_N_VARS = 72
_N_NONSTIFF_PAIRS = 18
_N_STIFF_PAIRS = 17
_NONSTIFF_DAMPING = 0.01


def _boltzmann_72d_ode(y, p):
    """72D CMB-inspired ODE with time-varying stiffness.

    Works with both tuples (Pallas kernel) and JAX arrays (vmap solver).
    Uses only element-wise ops and Python-level indexing.
    """
    clock = y[0]
    ones = clock * 0.0 + 1.0

    dy = [None] * _N_VARS
    dy[0] = ones  # clock

    # 18 non-stiff oscillator pairs (k=0..17)
    for k in range(_N_NONSTIFF_PAIRS):
        i1, i2 = 2 * k + 1, 2 * k + 2
        omega = 2.0 * math.pi * (k + 1)
        dy[i1] = -_NONSTIFF_DAMPING * y[i1] + omega * y[i2]
        dy[i2] = -omega * y[i1] - _NONSTIFF_DAMPING * y[i2]

    # 17 stiff oscillator pairs (k=18..34)
    for k in range(_N_NONSTIFF_PAIRS, _N_NONSTIFF_PAIRS + _N_STIFF_PAIRS):
        i1, i2 = 2 * k + 1, 2 * k + 2
        damping = p[0] * clock
        omega = 2.0 * math.pi * (k + 1)
        dy[i1] = -damping * y[i1] + omega * y[i2]
        dy[i2] = -omega * y[i1] - damping * y[i2]

    # Single stiff decay mode
    dy[71] = -p[1] * clock * y[71]

    return tuple(dy)
