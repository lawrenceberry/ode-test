"""Generic Rodas5 custom kernel using JAX Pallas lowered through Triton."""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas.triton as pl_triton
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from solvers import _rodas5ck_common as ck

_BLOCK_SIZE = 32


class _PallasMatrix:
    """Tiny matrix facade that lowers y[:, j] to component-vector access."""

    def __init__(self, columns):
        self._columns = columns

    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise NotImplementedError("Pallas RHS matrices support y[:, j] indexing")
        rows, col = idx
        if rows != slice(None):
            raise NotImplementedError("Pallas RHS matrices support y[:, j] indexing")
        if not isinstance(col, int):
            raise NotImplementedError("Pallas RHS matrices support integer columns")
        return self._columns[col]


def _tuple_add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _tuple_scale(s, a):
    return tuple(s * x for x in a)


def _tuple_axpy(base, h, *terms):
    out = []
    for j, yj in enumerate(base):
        acc = yj
        for coeff, k in terms:
            acc = acc + h * coeff * k[j]
        out.append(acc)
    return tuple(out)


def _lu_factor(mat, n_vars):
    """In-place Doolittle LU factorization on a list-of-lists of block_size vectors.

    mat[i][j] is the (i,j) matrix element as a (block_size,) JAX array.
    Returns the same structure with the combined LU packed in-place (L below
    diagonal, U on and above).  No pivoting — the W-transformed matrix
    (1/(γ·dt))·I − J is well-conditioned for Rodas5 step sizes.
    """
    lu = [list(row) for row in mat]
    for k in range(n_vars):
        inv_diag = 1.0 / lu[k][k]
        for i in range(k + 1, n_vars):
            factor = lu[i][k] * inv_diag
            lu[i][k] = factor
            for j in range(k + 1, n_vars):
                lu[i][j] = lu[i][j] - factor * lu[k][j]
    return lu


def _lu_solve(lu, rhs, n_vars):
    """Solve LU·x = rhs in-place. lu from _lu_factor, rhs a list of block_size vectors."""
    x = list(rhs)
    # Forward substitution: L·x = rhs
    for i in range(1, n_vars):
        for j in range(i):
            x[i] = x[i] - lu[i][j] * x[j]
    # Back substitution: U·x = x
    for i in range(n_vars - 1, -1, -1):
        for j in range(i + 1, n_vars):
            x[i] = x[i] - lu[i][j] * x[j]
        x[i] = x[i] / lu[i][i]
    return tuple(x)


@functools.partial(
    jax.jit,
    static_argnames=(
        "ode_fn",
        "n_save",
        "n_vars",
        "n_vars_work",
        "n_params",
        "max_steps",
        "block_size",
    ),
)
def _solve_kernel(
    ode_fn,
    y0,
    times,
    params,
    n_actual,
    dt0,
    rtol,
    atol,
    *,
    n_save,
    n_vars,
    n_vars_work,
    n_params,
    max_steps,
    block_size,
):
    n_padded = y0.shape[0]
    n_blocks = n_padded // block_size

    def kernel(
        y0_ref,
        times_ref,
        params_ref,
        n_actual_ref,
        dt0_ref,
        rtol_ref,
        atol_ref,
        hist_ref,
        acc_ref,
        rej_ref,
        loop_ref,
    ):
        block_start = pl.program_id(0) * block_size
        offsets = block_start + jnp.arange(block_size)
        valid = offsets < n_actual_ref[()]

        y = tuple(y0_ref[offsets, j] for j in range(n_vars_work))
        p = tuple(params_ref[offsets, j] for j in range(n_params))
        for j in range(n_vars_work):
            hist_ref[offsets, 0, j] = y[j]

        t0 = times_ref[0]
        tf = times_ref[n_save - 1]
        t = jnp.where(valid, t0, tf)
        dt = jnp.full((block_size,), dt0_ref[()], dtype=jnp.float64)
        save_idx = jnp.full((block_size,), 1, dtype=jnp.int32)
        n_loop = jnp.int32(0)
        accepted_steps = jnp.zeros((block_size,), dtype=jnp.int32)
        rejected_steps = jnp.zeros((block_size,), dtype=jnp.int32)

        def rhs(y_state, t_state):
            return ode_fn(_PallasMatrix(y_state), t_state, _PallasMatrix(p))

        def cond_fn(state):
            t, _dt, save_idx, n_loop, *_ = state
            unfinished_t = jnp.where(valid & (save_idx < n_save), t, tf)
            return (jnp.min(unfinished_t) < tf) & (n_loop < max_steps)

        def body_fn(state):
            (
                t,
                dt,
                save_idx,
                n_loop,
                y,
                accepted_steps,
                rejected_steps,
            ) = state
            active = valid & (save_idx < n_save) & (t < tf)
            safe_save_idx = jnp.minimum(save_idx, n_save - 1)
            next_target = times_ref[safe_save_idx]
            dt_use = jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30)

            # Jacobian via JVP — column j: tangent is e_j across n_vars_work
            jac_cols = []
            for j in range(n_vars):
                tangent_j = tuple(
                    jnp.ones((block_size,), dtype=jnp.float64) if k == j
                    else jnp.zeros((block_size,), dtype=jnp.float64)
                    for k in range(n_vars_work)
                )
                _, col = jax.jvp(lambda y_: rhs(y_, t), (y,), (tangent_j,))
                jac_cols.append(col)
            # jac_cols[j][i] = ∂f_i/∂y_j, shape (block_size,)

            dtgamma_inv = 1.0 / (dt_use * ck.GAMMA)
            mat = [
                [
                    (dtgamma_inv if i == j else jnp.zeros((block_size,), dtype=jnp.float64))
                    - jac_cols[j][i]
                    for j in range(n_vars)
                ]
                for i in range(n_vars)
            ]
            lu = _lu_factor(mat, n_vars)

            inv_dt = 1.0 / dt_use
            f0 = rhs(y, t)

            def lu_solve_rhs(rhs_vec):
                result = _lu_solve(lu, list(rhs_vec[:n_vars]), n_vars)
                if n_vars_work > n_vars:
                    result = result + tuple(
                        jnp.zeros((block_size,), dtype=jnp.float64)
                        for _ in range(n_vars_work - n_vars)
                    )
                return result

            k1 = lu_solve_rhs(f0)

            u = _tuple_axpy(y, 1.0, (ck.A21, k1))
            f1 = rhs(u[:n_vars_work] if len(u) == n_vars_work else u, t + ck.C2 * dt_use)
            k2_rhs = tuple(f1[i] + (ck.C21 * k1[i]) * inv_dt for i in range(n_vars))
            k2 = lu_solve_rhs(k2_rhs)

            u = _tuple_axpy(y, 1.0, (ck.A31, k1), (ck.A32, k2))
            f2 = rhs(u, t + ck.C3 * dt_use)
            k3_rhs = tuple(
                f2[i] + (ck.C31 * k1[i] + ck.C32 * k2[i]) * inv_dt for i in range(n_vars)
            )
            k3 = lu_solve_rhs(k3_rhs)

            u = _tuple_axpy(y, 1.0, (ck.A41, k1), (ck.A42, k2), (ck.A43, k3))
            f3 = rhs(u, t + ck.C4 * dt_use)
            k4_rhs = tuple(
                f3[i] + (ck.C41 * k1[i] + ck.C42 * k2[i] + ck.C43 * k3[i]) * inv_dt
                for i in range(n_vars)
            )
            k4 = lu_solve_rhs(k4_rhs)

            u = _tuple_axpy(y, 1.0, (ck.A51, k1), (ck.A52, k2), (ck.A53, k3), (ck.A54, k4))
            f4 = rhs(u, t + ck.C5 * dt_use)
            k5_rhs = tuple(
                f4[i]
                + (ck.C51 * k1[i] + ck.C52 * k2[i] + ck.C53 * k3[i] + ck.C54 * k4[i])
                * inv_dt
                for i in range(n_vars)
            )
            k5 = lu_solve_rhs(k5_rhs)

            t_end = t + dt_use
            u = _tuple_axpy(
                y, 1.0, (ck.A61, k1), (ck.A62, k2), (ck.A63, k3), (ck.A64, k4), (ck.A65, k5)
            )
            f5 = rhs(u, t_end)
            k6_rhs = tuple(
                f5[i]
                + (
                    ck.C61 * k1[i]
                    + ck.C62 * k2[i]
                    + ck.C63 * k3[i]
                    + ck.C64 * k4[i]
                    + ck.C65 * k5[i]
                )
                * inv_dt
                for i in range(n_vars)
            )
            k6 = lu_solve_rhs(k6_rhs)

            # u7 = u6 + k6
            u = tuple(u[i] + k6[i] for i in range(n_vars))
            # pad back to n_vars_work if needed
            if n_vars_work > n_vars:
                u = u + tuple(
                    jnp.zeros((block_size,), dtype=jnp.float64)
                    for _ in range(n_vars_work - n_vars)
                )
            f6 = rhs(u, t_end)
            k7_rhs = tuple(
                f6[i]
                + (
                    ck.C71 * k1[i]
                    + ck.C72 * k2[i]
                    + ck.C73 * k3[i]
                    + ck.C74 * k4[i]
                    + ck.C75 * k5[i]
                    + ck.C76 * k6[i]
                )
                * inv_dt
                for i in range(n_vars)
            )
            k7 = lu_solve_rhs(k7_rhs)

            # u8 = u7 + k7
            u = tuple(u[i] + k7[i] for i in range(n_vars))
            if n_vars_work > n_vars:
                u = u + tuple(
                    jnp.zeros((block_size,), dtype=jnp.float64)
                    for _ in range(n_vars_work - n_vars)
                )
            f7 = rhs(u, t_end)
            k8_rhs = tuple(
                f7[i]
                + (
                    ck.C81 * k1[i]
                    + ck.C82 * k2[i]
                    + ck.C83 * k3[i]
                    + ck.C84 * k4[i]
                    + ck.C85 * k5[i]
                    + ck.C86 * k6[i]
                    + ck.C87 * k7[i]
                )
                * inv_dt
                for i in range(n_vars)
            )
            k8 = lu_solve_rhs(k8_rhs)

            # y_new = u8 + k8; error estimate = k8
            y_new_list = [u[i] + k8[i] for i in range(n_vars)]
            if n_vars_work > n_vars:
                y_new_list += [
                    jnp.zeros((block_size,), dtype=jnp.float64)
                    for _ in range(n_vars_work - n_vars)
                ]
            y_new = tuple(y_new_list)

            rtol_value = rtol_ref[()]
            atol_value = atol_ref[()]
            err_sum = jnp.zeros((block_size,), dtype=jnp.float64)
            for j in range(n_vars):
                scale = atol_value + rtol_value * jnp.maximum(jnp.abs(y[j]), jnp.abs(y_new[j]))
                ratio = k8[j] / scale
                err_sum = err_sum + ratio * ratio
            err_norm = jnp.sqrt(err_sum / n_vars)

            accept = active & (err_norm <= 1.0) & ~jnp.isnan(err_norm)
            rejected = active & ~accept

            t_new = jnp.where(accept, t + dt_use, t)
            y_out = tuple(jnp.where(accept, y_new[j], y[j]) for j in range(n_vars_work))

            reached = accept & (
                jnp.abs(t_new - next_target)
                <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
            )
            for j in range(n_vars_work):
                old_hist = hist_ref[offsets, safe_save_idx, j]
                hist_ref[offsets, safe_save_idx, j] = jnp.where(reached, y_out[j], old_hist)

            safe_err = jnp.where(
                jnp.isnan(err_norm) | (err_norm > 1e18),
                1e18,
                jnp.where(err_norm == 0.0, 1e-18, err_norm),
            )
            factor = jnp.clip(
                ck.SAFETY * safe_err ** (-1.0 / 6.0), ck.FACTOR_MIN, ck.FACTOR_MAX
            )
            return (
                t_new,
                jnp.where(active, dt_use * factor, dt),
                save_idx + reached.astype(jnp.int32),
                n_loop + 1,
                y_out,
                accepted_steps + accept.astype(jnp.int32),
                rejected_steps + rejected.astype(jnp.int32),
            )

        final = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                t,
                dt,
                save_idx,
                n_loop,
                y,
                accepted_steps,
                rejected_steps,
            ),
        )
        acc_ref[offsets] = final[-2]
        rej_ref[offsets] = final[-1]
        loop_ref[offsets] = jnp.full((block_size,), final[3], dtype=jnp.int32)

    out_shape = (
        jax.ShapeDtypeStruct((n_padded, n_save, n_vars_work), jnp.float64),
        jax.ShapeDtypeStruct((n_padded,), jnp.int32),
        jax.ShapeDtypeStruct((n_padded,), jnp.int32),
        jax.ShapeDtypeStruct((n_padded,), jnp.int32),
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(n_blocks,),
        compiler_params=pl_triton.CompilerParams(num_warps=1),
        name="rodas5ckp_generic_components",
    )(y0, times, params, n_actual, dt0, rtol, atol)


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
):
    del batch_size
    y0_arr, times, params_arr, dt0 = ck.normalize_inputs(y0, t_span, params, first_step)
    n_actual = y0_arr.shape[0]
    n_vars = y0_arr.shape[1]
    n_params = params_arr.shape[1]
    n_vars_work = 1 << (n_vars - 1).bit_length()
    pad_count = (-n_actual) % _BLOCK_SIZE
    if pad_count:
        y0_arr = np.pad(y0_arr, ((0, pad_count), (0, 0)))
        params_arr = np.pad(params_arr, ((0, pad_count), (0, 0)))
    if n_vars_work != n_vars:
        y0_arr = np.pad(y0_arr, ((0, 0), (0, n_vars_work - n_vars)))

    solution, accepted_steps, rejected_steps, loop_steps = _solve_kernel(
        ode_fn,
        jnp.asarray(y0_arr, dtype=jnp.float64),
        jnp.asarray(times, dtype=jnp.float64),
        jnp.asarray(params_arr, dtype=jnp.float64),
        jnp.asarray(n_actual, dtype=jnp.int32),
        jnp.asarray(dt0, dtype=jnp.float64),
        jnp.asarray(rtol, dtype=jnp.float64),
        jnp.asarray(atol, dtype=jnp.float64),
        n_save=times.shape[0],
        n_vars=n_vars,
        n_vars_work=n_vars_work,
        n_params=n_params,
        max_steps=int(max_steps),
        block_size=_BLOCK_SIZE,
    )
    solution = solution[:n_actual, :, :n_vars]
    accepted_steps = accepted_steps[:n_actual]
    rejected_steps = rejected_steps[:n_actual]
    loop_steps = loop_steps[:n_actual]
    if not return_stats:
        return solution
    stats = {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": jnp.ones_like(loop_steps),
    }
    return solution, stats
