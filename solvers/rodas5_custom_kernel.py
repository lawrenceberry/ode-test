"""Rodas5 solver with Cramer's-rule inverse and Pallas GPU kernel.

Two ensemble strategies:
1. solve_ensemble: vmap + Cramer's rule (generic, works for any 3-component ODE)
2. solve_ensemble_pallas: Pallas/Triton custom kernel with fully scalar arithmetic
   (Robertson-specific, runs entire solver loop in a single GPU kernel)

The Pallas kernel mirrors DiffEqGPU.jl's EnsembleGPUKernel architecture:
one GPU thread per trajectory, independent adaptive timestepping, all state
in registers. It is Robertson-specific because Triton requires all tensor
sizes to be powers of 2 and lacks slice/concatenate primitives, forcing
fully scalar arithmetic with a hardcoded ODE function and Jacobian.
"""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402
from jax._src.pallas.triton import core as pltriton
from jax.experimental import pallas as pl

# fmt: off
# Rodas5 W-transformed coefficients (identical to rodas5.py)
_gamma = 0.19

_a21 = 2.0
_a31 = 3.040894194418781;  _a32 = 1.041747909077569
_a41 = 2.576417536461461;  _a42 = 1.622083060776640;  _a43 = -0.9089668560264532
_a51 = 2.760842080225597;  _a52 = 1.446624659844071;  _a53 = -0.3036980084553738;  _a54 = 0.2877498600325443
_a61 = -14.09640773051259; _a62 = 6.925207756232704;  _a63 = -41.47510893210728;   _a64 = 2.343771018586405;  _a65 = 24.13215229196062
_a76 = 1.0
_a86 = 1.0; _a87 = 1.0

_C21 = -10.31323885133993
_C31 = -21.04823117650003; _C32 = -7.234992135176716
_C41 = 32.22751541853323;  _C42 = -4.943732386540191;  _C43 = 19.44922031041879
_C51 = -20.69865579590063; _C52 = -8.816374604402768;  _C53 = 1.260436877740897;   _C54 = -0.7495647613787146
_C61 = -46.22004352711257; _C62 = -17.49534862857472;  _C63 = -289.6389582892057;  _C64 = 93.60855400400906;  _C65 = 318.3822534212147
_C71 = 34.20013733472935;  _C72 = -14.15535402717690;  _C73 = 57.82335640988400;   _C74 = 25.83362985412365;  _C75 = 1.408950972071624;  _C76 = -6.551835421242162
_C81 = 42.57076742291101;  _C82 = -13.80770672017997;  _C83 = 93.98938432427124;   _C84 = 18.77919633714503;  _C85 = -31.58359187223370;  _C86 = -6.685968952921985;  _C87 = -5.810979938412932
# fmt: on


# ---------------------------------------------------------------------------
# Array-based helpers (for solve / solve_ensemble via vmap)
# ---------------------------------------------------------------------------


def _build_W(J, dtgamma):
    d = 1.0 / dtgamma
    return jnp.array(
        [
            [d - J[0, 0], -J[0, 1], -J[0, 2]],
            [-J[1, 0], d - J[1, 1], -J[1, 2]],
            [-J[2, 0], -J[2, 1], d - J[2, 2]],
        ]
    )


def _inv3x3(W):
    w00, w01, w02 = W[0, 0], W[0, 1], W[0, 2]
    w10, w11, w12 = W[1, 0], W[1, 1], W[1, 2]
    w20, w21, w22 = W[2, 0], W[2, 1], W[2, 2]
    a00 = w11 * w22 - w12 * w21
    a01 = w02 * w21 - w01 * w22
    a02 = w01 * w12 - w02 * w11
    a10 = w12 * w20 - w10 * w22
    a11 = w00 * w22 - w02 * w20
    a12 = w02 * w10 - w00 * w12
    a20 = w10 * w21 - w11 * w20
    a21 = w01 * w20 - w00 * w21
    a22 = w00 * w11 - w01 * w10
    inv_det = 1.0 / (w00 * a00 + w01 * a10 + w02 * a20)
    return jnp.array(
        [
            [a00 * inv_det, a01 * inv_det, a02 * inv_det],
            [a10 * inv_det, a11 * inv_det, a12 * inv_det],
            [a20 * inv_det, a21 * inv_det, a22 * inv_det],
        ]
    )


def _step(f_fn, jac_fn, y, dt):
    J = jac_fn(y)
    W = _build_W(J, dt * _gamma)
    Wi = _inv3x3(W)
    inv_dt = 1.0 / dt

    dy = f_fn(y)
    k1 = Wi @ dy
    u = y + _a21 * k1
    du = f_fn(u)
    k2 = Wi @ (du + (_C21 * k1) * inv_dt)
    u = y + _a31 * k1 + _a32 * k2
    du = f_fn(u)
    k3 = Wi @ (du + (_C31 * k1 + _C32 * k2) * inv_dt)
    u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
    du = f_fn(u)
    k4 = Wi @ (du + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt)
    u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
    du = f_fn(u)
    k5 = Wi @ (du + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt)
    u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
    du = f_fn(u)
    k6 = Wi @ (
        du + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5) * inv_dt
    )
    u = u + k6
    du = f_fn(u)
    k7 = Wi @ (
        du
        + (_C71 * k1 + _C72 * k2 + _C73 * k3 + _C74 * k4 + _C75 * k5 + _C76 * k6)
        * inv_dt
    )
    u = u + k7
    du = f_fn(u)
    k8 = Wi @ (
        du
        + (
            _C81 * k1
            + _C82 * k2
            + _C83 * k3
            + _C84 * k4
            + _C85 * k5
            + _C86 * k6
            + _C87 * k7
        )
        * inv_dt
    )

    return u + k8, k8


# ---------------------------------------------------------------------------
# Public API: single solve + vmap ensemble (generic)
# ---------------------------------------------------------------------------


def solve(f, y0, t_span, *, rtol=1e-8, atol=1e-10, first_step=None, max_steps=100000):
    """Solve a stiff autonomous ODE using Rodas5 with Cramer's rule."""
    y0_arr = jnp.asarray(y0, dtype=jnp.float64)
    t0, tf = t_span
    dt0 = jnp.float64(first_step if first_step is not None else (tf - t0) * 1e-6)
    jac_fn = jax.jacfwd(f)

    def cond_fn(state):
        t, _, _, n = state
        return (t < tf) & (n < max_steps)

    def body_fn(state):
        t, y, dt, n = state
        dt = jnp.minimum(dt, tf - t)
        y_new, err_est = _step(f, jac_fn, y, dt)
        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
        err_norm = jnp.sqrt(jnp.mean((err_est / scale) ** 2))
        accept = (err_norm <= 1.0) & ~jnp.isnan(err_norm)
        t_new = jnp.where(accept, t + dt, t)
        y_out = jnp.where(accept, y_new, y)
        safe_err = jnp.where(
            jnp.isnan(err_norm) | (err_norm > 1e18),
            1e18,
            jnp.where(err_norm == 0.0, 1e-18, err_norm),
        )
        factor = jnp.clip(0.9 * safe_err ** (-1.0 / 6.0), 0.2, 6.0)
        return (t_new, y_out, dt * factor, n + 1)

    _, final_y, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (jnp.float64(t0), y0_arr, dt0, jnp.int32(0))
    )
    return final_y


def solve_ensemble(
    f,
    y0,
    t_span,
    params_batch,
    *,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
):
    """Solve ensemble using vmap + Cramer's-rule inverse (generic)."""

    def _solve_one(params):
        return solve(
            lambda y: f(y, params),
            y0,
            t_span,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
        )

    return jax.jit(jax.vmap(_solve_one))(params_batch)


# ---------------------------------------------------------------------------
# Pallas/Triton custom kernel (Robertson-specific, fully scalar arithmetic)
# ---------------------------------------------------------------------------


def _robertson_step_scalar(y0, y1, y2, p0, p1, p2, dt):
    """Single Rodas5 step for Robertson, fully scalar. Returns (y0_new, y1_new, y2_new, e0, e1, e2)."""
    # Analytical Jacobian of Robertson system
    j00 = -p0
    j01 = p1 * y2
    j02 = p1 * y1
    j10 = p0
    j11 = -p1 * y2 - 2.0 * p2 * y1
    j12 = -p1 * y1
    j20 = 0.0 * p0
    j21 = 2.0 * p2 * y1
    j22 = 0.0 * p0

    # W = I/(dt*gamma) - J
    d = 1.0 / (dt * _gamma)
    w00 = d - j00
    w01 = -j01
    w02 = -j02
    w10 = -j10
    w11 = d - j11
    w12 = -j12
    w20 = -j20
    w21 = -j21
    w22 = d - j22

    # Adjugate -> inverse
    a00 = w11 * w22 - w12 * w21
    a01 = w02 * w21 - w01 * w22
    a02 = w01 * w12 - w02 * w11
    a10 = w12 * w20 - w10 * w22
    a11 = w00 * w22 - w02 * w20
    a12 = w02 * w10 - w00 * w12
    a20 = w10 * w21 - w11 * w20
    a21 = w01 * w20 - w00 * w21
    a22 = w00 * w11 - w01 * w10
    id = 1.0 / (w00 * a00 + w01 * a10 + w02 * a20)
    i00 = a00 * id
    i01 = a01 * id
    i02 = a02 * id
    i10 = a10 * id
    i11 = a11 * id
    i12 = a12 * id
    i20 = a20 * id
    i21 = a21 * id
    i22 = a22 * id

    inv_dt = 1.0 / dt

    # Robertson RHS
    def F(s0, s1, s2):
        return (
            -p0 * s0 + p1 * s1 * s2,
            p0 * s0 - p1 * s1 * s2 - p2 * s1 * s1,
            p2 * s1 * s1,
        )

    # Wi @ v
    def S(v0, v1, v2):
        return (
            i00 * v0 + i01 * v1 + i02 * v2,
            i10 * v0 + i11 * v1 + i12 * v2,
            i20 * v0 + i21 * v1 + i22 * v2,
        )

    # Stage 1
    f0, f1, f2 = F(y0, y1, y2)
    k10, k11, k12 = S(f0, f1, f2)

    # Stage 2
    u0 = y0 + _a21 * k10
    u1 = y1 + _a21 * k11
    u2 = y2 + _a21 * k12
    f0, f1, f2 = F(u0, u1, u2)
    k20, k21, k22 = S(
        f0 + _C21 * k10 * inv_dt, f1 + _C21 * k11 * inv_dt, f2 + _C21 * k12 * inv_dt
    )

    # Stage 3
    u0 = y0 + _a31 * k10 + _a32 * k20
    u1 = y1 + _a31 * k11 + _a32 * k21
    u2 = y2 + _a31 * k12 + _a32 * k22
    f0, f1, f2 = F(u0, u1, u2)
    k30, k31, k32 = S(
        f0 + (_C31 * k10 + _C32 * k20) * inv_dt,
        f1 + (_C31 * k11 + _C32 * k21) * inv_dt,
        f2 + (_C31 * k12 + _C32 * k22) * inv_dt,
    )

    # Stage 4
    u0 = y0 + _a41 * k10 + _a42 * k20 + _a43 * k30
    u1 = y1 + _a41 * k11 + _a42 * k21 + _a43 * k31
    u2 = y2 + _a41 * k12 + _a42 * k22 + _a43 * k32
    f0, f1, f2 = F(u0, u1, u2)
    c0 = (_C41 * k10 + _C42 * k20 + _C43 * k30) * inv_dt
    c1 = (_C41 * k11 + _C42 * k21 + _C43 * k31) * inv_dt
    c2 = (_C41 * k12 + _C42 * k22 + _C43 * k32) * inv_dt
    k40, k41, k42 = S(f0 + c0, f1 + c1, f2 + c2)

    # Stage 5
    u0 = y0 + _a51 * k10 + _a52 * k20 + _a53 * k30 + _a54 * k40
    u1 = y1 + _a51 * k11 + _a52 * k21 + _a53 * k31 + _a54 * k41
    u2 = y2 + _a51 * k12 + _a52 * k22 + _a53 * k32 + _a54 * k42
    f0, f1, f2 = F(u0, u1, u2)
    c0 = (_C51 * k10 + _C52 * k20 + _C53 * k30 + _C54 * k40) * inv_dt
    c1 = (_C51 * k11 + _C52 * k21 + _C53 * k31 + _C54 * k41) * inv_dt
    c2 = (_C51 * k12 + _C52 * k22 + _C53 * k32 + _C54 * k42) * inv_dt
    k50, k51, k52 = S(f0 + c0, f1 + c1, f2 + c2)

    # Stage 6
    u0 = y0 + _a61 * k10 + _a62 * k20 + _a63 * k30 + _a64 * k40 + _a65 * k50
    u1 = y1 + _a61 * k11 + _a62 * k21 + _a63 * k31 + _a64 * k41 + _a65 * k51
    u2 = y2 + _a61 * k12 + _a62 * k22 + _a63 * k32 + _a64 * k42 + _a65 * k52
    f0, f1, f2 = F(u0, u1, u2)
    c0 = (_C61 * k10 + _C62 * k20 + _C63 * k30 + _C64 * k40 + _C65 * k50) * inv_dt
    c1 = (_C61 * k11 + _C62 * k21 + _C63 * k31 + _C64 * k41 + _C65 * k51) * inv_dt
    c2 = (_C61 * k12 + _C62 * k22 + _C63 * k32 + _C64 * k42 + _C65 * k52) * inv_dt
    k60, k61, k62 = S(f0 + c0, f1 + c1, f2 + c2)

    # Stage 7 (u = u_stage6 + k6)
    u0 = u0 + k60
    u1 = u1 + k61
    u2 = u2 + k62
    f0, f1, f2 = F(u0, u1, u2)
    c0 = (
        _C71 * k10 + _C72 * k20 + _C73 * k30 + _C74 * k40 + _C75 * k50 + _C76 * k60
    ) * inv_dt
    c1 = (
        _C71 * k11 + _C72 * k21 + _C73 * k31 + _C74 * k41 + _C75 * k51 + _C76 * k61
    ) * inv_dt
    c2 = (
        _C71 * k12 + _C72 * k22 + _C73 * k32 + _C74 * k42 + _C75 * k52 + _C76 * k62
    ) * inv_dt
    k70, k71, k72 = S(f0 + c0, f1 + c1, f2 + c2)

    # Stage 8 (u = u_stage7 + k7)
    u0 = u0 + k70
    u1 = u1 + k71
    u2 = u2 + k72
    f0, f1, f2 = F(u0, u1, u2)
    c0 = (
        _C81 * k10
        + _C82 * k20
        + _C83 * k30
        + _C84 * k40
        + _C85 * k50
        + _C86 * k60
        + _C87 * k70
    ) * inv_dt
    c1 = (
        _C81 * k11
        + _C82 * k21
        + _C83 * k31
        + _C84 * k41
        + _C85 * k51
        + _C86 * k61
        + _C87 * k71
    ) * inv_dt
    c2 = (
        _C81 * k12
        + _C82 * k22
        + _C83 * k32
        + _C84 * k42
        + _C85 * k52
        + _C86 * k62
        + _C87 * k72
    ) * inv_dt
    k80, k81, k82 = S(f0 + c0, f1 + c1, f2 + c2)

    # y_new = u + k8, error = k8
    return u0 + k80, u1 + k81, u2 + k82, k80, k81, k82


_BLOCK = 32


def solve_ensemble_pallas(
    y0,
    t_span,
    params_batch,
    *,
    rtol=1e-6,
    atol=1e-8,
    first_step=None,
    max_steps=100000,
):
    """Solve Robertson ensemble using a Pallas/Triton custom GPU kernel.

    Processes 32 trajectories per Triton block (one warp). Each trajectory
    has independent adaptive timestepping; finished trajectories are masked
    so the block continues until all 32 are done.
    """
    N = params_batch.shape[0]
    N_pad = ((N + _BLOCK - 1) // _BLOCK) * _BLOCK
    t0 = float(t_span[0])
    tf = float(t_span[1])
    dt0 = float(first_step if first_step is not None else (tf - t0) * 1e-6)
    _rtol = float(rtol)
    _atol = float(atol)
    _max_steps = int(max_steps)
    _y00, _y01, _y02 = float(y0[0]), float(y0[1]), float(y0[2])

    # Split params into separate 1D arrays, pad to N_pad
    p0_arr = jnp.pad(params_batch[:, 0], (0, N_pad - N))
    p1_arr = jnp.pad(params_batch[:, 1], (0, N_pad - N))
    p2_arr = jnp.pad(params_batch[:, 2], (0, N_pad - N))

    def kernel_body(p0_ref, p1_ref, p2_ref, y0_ref, y1_ref, y2_ref):
        # Full-block reads: each is (_BLOCK,)
        p0 = p0_ref[...]
        p1 = p1_ref[...]
        p2 = p2_ref[...]

        # Traced (_BLOCK,) vectors from params (avoids captured constants)
        z = p0 * 0.0
        t = z + t0
        s0 = z + _y00
        s1 = z + _y01
        s2 = z + _y02
        dt_v = z + dt0

        def cond_fn(state):
            t, _, _, _, _, n = state
            return (jnp.min(t) < tf) & (n < _max_steps)

        def body_fn(state):
            t, s0, s1, s2, dt_v, n = state
            active = t < tf
            # Clamp dt positive so inactive trajectories don't cause div-by-zero
            dt_use = jnp.maximum(jnp.minimum(dt_v, tf - t), 1e-30)

            n0, n1, n2, e0, e1, e2 = _robertson_step_scalar(
                s0, s1, s2, p0, p1, p2, dt_use
            )

            sc0 = _atol + _rtol * jnp.maximum(jnp.abs(s0), jnp.abs(n0))
            sc1 = _atol + _rtol * jnp.maximum(jnp.abs(s1), jnp.abs(n1))
            sc2 = _atol + _rtol * jnp.maximum(jnp.abs(s2), jnp.abs(n2))
            err_sq = (e0 / sc0) ** 2 + (e1 / sc1) ** 2 + (e2 / sc2) ** 2
            err_norm = jnp.sqrt(err_sq / 3.0)

            accept = (err_norm <= 1.0) & ~jnp.isnan(err_norm)
            mask = active & accept

            t_new = jnp.where(mask, t + dt_use, t)
            o0 = jnp.where(mask, n0, s0)
            o1 = jnp.where(mask, n1, s1)
            o2 = jnp.where(mask, n2, s2)

            safe = jnp.where(
                jnp.isnan(err_norm) | (err_norm > 1e18),
                1e18,
                jnp.where(err_norm == 0.0, 1e-18, err_norm),
            )
            factor = jnp.clip(0.9 * safe ** (-1.0 / 6.0), 0.2, 6.0)
            new_dt = jnp.where(active, dt_use * factor, dt_v)

            return (t_new, o0, o1, o2, new_dt, n + 1)

        _, r0, r1, r2, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (t, s0, s1, s2, dt_v, jnp.int32(0))
        )

        y0_ref[...] = r0
        y1_ref[...] = r1
        y2_ref[...] = r2

    bs = pl.BlockSpec((_BLOCK,), lambda i: (i,))
    y0_out, y1_out, y2_out = pl.pallas_call(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((N_pad,), jnp.float64),
            jax.ShapeDtypeStruct((N_pad,), jnp.float64),
            jax.ShapeDtypeStruct((N_pad,), jnp.float64),
        ],
        grid=(N_pad // _BLOCK,),
        in_specs=[bs, bs, bs],
        out_specs=[bs, bs, bs],
        compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=1),
    )(p0_arr, p1_arr, p2_arr)

    return jnp.stack([y0_out[:N], y1_out[:N], y2_out[:N]], axis=1)
