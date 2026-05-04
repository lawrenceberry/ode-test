"""Generic Rodas5 custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math

import numpy as np
from numba import cuda
from nvmath.device import LUPivotSolver

from solvers import _rodas5ck_common as ck


def _as_launch_block_dim(block_dim):
    if isinstance(block_dim, int):
        return block_dim
    if isinstance(block_dim, (tuple, list)):
        return block_dim
    x = getattr(block_dim, "x", None)
    y = getattr(block_dim, "y", 1)
    z = getattr(block_dim, "z", 1)
    if x is not None:
        return (int(x), int(y), int(z))
    return 128


@functools.cache
def _make_kernel(ode_fn, jac_fn, n_vars: int):
    lu_solver = LUPivotSolver(
        size=(n_vars, n_vars, 1),
        precision=np.float32,
        execution="Block",
        arrangement=("row_major", "row_major"),
        batches_per_block="suggested",
        block_dim="suggested",
    )
    batches_per_block = int(lu_solver.batches_per_block)
    a_size = int(lu_solver.a_size())
    b_size = int(lu_solver.b_size())
    ipiv_size = int(lu_solver.ipiv_size)
    print(
        f"Rodas5ckn kernel: n_vars={n_vars}, batches_per_block={batches_per_block}, block_dim={lu_solver.block_dim}, a_size={a_size}, b_size={b_size}, ipiv_size={ipiv_size}"
    )

    @cuda.jit
    def kernel(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
        y,
        u,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        jac,
    ):
        block_start = cuda.blockIdx.x * batches_per_block
        tx = cuda.threadIdx.x

        n_save = times.shape[0]
        tf = times[n_save - 1]

        smem_lu = cuda.shared.array(shape=a_size, dtype=np.float32)
        smem_rhs = cuda.shared.array(shape=b_size, dtype=np.float32)
        smem_ipiv = cuda.shared.array(shape=ipiv_size, dtype=np.int32)
        smem_info = cuda.shared.array(shape=batches_per_block, dtype=np.int32)

        smem_t = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt_use = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_inv_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_t_end = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_next_target = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_save_idx = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_n_steps = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_accepted = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_rejected = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_continue = cuda.shared.array(shape=1, dtype=np.int32)

        if tx < batches_per_block:
            i = block_start + tx
            if i < y0.shape[0]:
                for j in range(n_vars):
                    y[i, j] = y0[i, j]
                    hist[i, 0, j] = y0[i, j]
                smem_t[tx] = times[0]
                smem_dt[tx] = dt0
                smem_save_idx[tx] = 1
            else:
                smem_t[tx] = tf
                smem_dt[tx] = dt0
                smem_save_idx[tx] = n_save
            smem_n_steps[tx] = 0
            smem_accepted[tx] = 0
            smem_rejected[tx] = 0

        if tx == 0:
            smem_continue[0] = 1
        cuda.syncthreads()

        while smem_continue[0] != 0:
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                a_offset = batch * n_vars * n_vars
                b_offset = batch * n_vars
                if active:
                    next_target = times[smem_save_idx[batch]]
                    dt_use = smem_dt[batch]
                    if dt_use > next_target - smem_t[batch]:
                        dt_use = next_target - smem_t[batch]
                    if dt_use < 1e-30:
                        dt_use = 1e-30
                    smem_next_target[batch] = next_target
                    smem_dt_use[batch] = dt_use
                    smem_inv_dt[batch] = 1.0 / dt_use
                    smem_t_end[batch] = smem_t[batch] + dt_use

                    jac_fn(y, smem_t[batch], params, jac, i)

                    # Build M = (1/(gamma*dt)) * I - J in row-major shared memory.
                    dtgamma_inv = 1.0 / (dt_use * ck.GAMMA)
                    for row in range(n_vars):
                        for col in range(n_vars):
                            if row == col:
                                smem_lu[a_offset + row * n_vars + col] = np.float32(
                                    dtgamma_inv - jac[i, row, col]
                                )
                            else:
                                smem_lu[a_offset + row * n_vars + col] = np.float32(
                                    -jac[i, row, col]
                                )
                else:
                    for row in range(n_vars):
                        for col in range(n_vars):
                            if row == col:
                                smem_lu[a_offset + row * n_vars + col] = 1.0
                            else:
                                smem_lu[a_offset + row * n_vars + col] = 0.0
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = 0.0
            cuda.syncthreads()
            lu_solver.factorize(smem_lu, smem_ipiv, smem_info)
            cuda.syncthreads()

            # ---- Stage 1 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    ode_fn(y, smem_t[batch], params, k1, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(k1[i, j])
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k1[i, j] = np.float64(smem_rhs[b_offset + j])

            # ---- Stage 2 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    for j in range(n_vars):
                        u[i, j] = y[i, j] + ck.A21 * k1[i, j]
                    ode_fn(u, smem_t[batch] + ck.C2 * smem_dt_use[batch], params, k2, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(
                            k2[i, j] + ck.C21 * k1[i, j] * smem_inv_dt[batch]
                        )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k2[i, j] = np.float64(smem_rhs[b_offset + j])

            # ---- Stage 3 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    for j in range(n_vars):
                        u[i, j] = y[i, j] + (ck.A31 * k1[i, j] + ck.A32 * k2[i, j])
                    ode_fn(u, smem_t[batch] + ck.C3 * smem_dt_use[batch], params, k3, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(
                            k3[i, j]
                            + (ck.C31 * k1[i, j] + ck.C32 * k2[i, j])
                            * smem_inv_dt[batch]
                        )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k3[i, j] = np.float64(smem_rhs[b_offset + j])

            # ---- Stage 4 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    for j in range(n_vars):
                        u[i, j] = y[i, j] + (
                            ck.A41 * k1[i, j] + ck.A42 * k2[i, j] + ck.A43 * k3[i, j]
                        )
                    ode_fn(u, smem_t[batch] + ck.C4 * smem_dt_use[batch], params, k4, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(
                            k4[i, j]
                            + (
                                ck.C41 * k1[i, j]
                                + ck.C42 * k2[i, j]
                                + ck.C43 * k3[i, j]
                            )
                            * smem_inv_dt[batch]
                        )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k4[i, j] = np.float64(smem_rhs[b_offset + j])

            # ---- Stage 5 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    for j in range(n_vars):
                        u[i, j] = y[i, j] + (
                            ck.A51 * k1[i, j]
                            + ck.A52 * k2[i, j]
                            + ck.A53 * k3[i, j]
                            + ck.A54 * k4[i, j]
                        )
                    ode_fn(u, smem_t[batch] + ck.C5 * smem_dt_use[batch], params, k5, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(
                            k5[i, j]
                            + (
                                ck.C51 * k1[i, j]
                                + ck.C52 * k2[i, j]
                                + ck.C53 * k3[i, j]
                                + ck.C54 * k4[i, j]
                            )
                            * smem_inv_dt[batch]
                        )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k5[i, j] = np.float64(smem_rhs[b_offset + j])

            # ---- Stage 6 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    for j in range(n_vars):
                        u[i, j] = y[i, j] + (
                            ck.A61 * k1[i, j]
                            + ck.A62 * k2[i, j]
                            + ck.A63 * k3[i, j]
                            + ck.A64 * k4[i, j]
                            + ck.A65 * k5[i, j]
                        )
                    ode_fn(u, smem_t_end[batch], params, k6, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(
                            k6[i, j]
                            + (
                                ck.C61 * k1[i, j]
                                + ck.C62 * k2[i, j]
                                + ck.C63 * k3[i, j]
                                + ck.C64 * k4[i, j]
                                + ck.C65 * k5[i, j]
                            )
                            * smem_inv_dt[batch]
                        )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k6[i, j] = np.float64(smem_rhs[b_offset + j])

            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                if active:
                    for j in range(n_vars):
                        u[i, j] += k6[i, j]

            # ---- Stage 7 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    ode_fn(u, smem_t_end[batch], params, k7, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(
                            k7[i, j]
                            + (
                                ck.C71 * k1[i, j]
                                + ck.C72 * k2[i, j]
                                + ck.C73 * k3[i, j]
                                + ck.C74 * k4[i, j]
                                + ck.C75 * k5[i, j]
                                + ck.C76 * k6[i, j]
                            )
                            * smem_inv_dt[batch]
                        )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k7[i, j] = np.float64(smem_rhs[b_offset + j])

            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                if active:
                    for j in range(n_vars):
                        u[i, j] += k7[i, j]

            # ---- Stage 8 ----
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                b_offset = batch * n_vars
                if active:
                    ode_fn(u, smem_t_end[batch], params, k8, i)
                    for j in range(n_vars):
                        smem_rhs[b_offset + j] = np.float32(
                            k8[i, j]
                            + (
                                ck.C81 * k1[i, j]
                                + ck.C82 * k2[i, j]
                                + ck.C83 * k3[i, j]
                                + ck.C84 * k4[i, j]
                                + ck.C85 * k5[i, j]
                                + ck.C86 * k6[i, j]
                                + ck.C87 * k7[i, j]
                            )
                            * smem_inv_dt[batch]
                        )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                b_offset = batch * n_vars
                if i < y0.shape[0]:
                    for j in range(n_vars):
                        k8[i, j] = np.float64(smem_rhs[b_offset + j])

            if tx < batches_per_block:
                batch = tx
                i = block_start + batch
                active = (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                )
                if active:
                    err_sum = 0.0
                    for j in range(n_vars):
                        y_new_j = u[i, j] + k8[i, j]
                        scale = atol + rtol * max(
                            math.fabs(y[i, j]), math.fabs(y_new_j)
                        )
                        ratio = k8[i, j] / scale
                        err_sum += ratio * ratio
                    err_norm = math.sqrt(err_sum / n_vars)
                    accept = err_norm <= 1.0 and not math.isnan(err_norm)

                    t_new = smem_t[batch]
                    if accept:
                        t_new = smem_t[batch] + smem_dt_use[batch]
                        for j in range(n_vars):
                            y[i, j] = u[i, j] + k8[i, j]
                        smem_accepted[batch] += 1
                    else:
                        smem_rejected[batch] += 1

                    reached = accept and (
                        abs(t_new - smem_next_target[batch])
                        <= 1e-12 * max(1.0, abs(smem_next_target[batch]))
                    )
                    if reached:
                        for j in range(n_vars):
                            hist[i, smem_save_idx[batch], j] = y[i, j]
                        smem_save_idx[batch] += 1

                    if math.isnan(err_norm) or err_norm > 1e18:
                        safe_err = 1e18
                    elif err_norm == 0.0:
                        safe_err = 1e-18
                    else:
                        safe_err = err_norm
                    factor = ck.SAFETY * safe_err ** (-1.0 / 6.0)
                    if factor < ck.FACTOR_MIN:
                        factor = ck.FACTOR_MIN
                    elif factor > ck.FACTOR_MAX:
                        factor = ck.FACTOR_MAX
                    smem_dt[batch] = smem_dt_use[batch] * factor
                    smem_t[batch] = t_new
                    smem_n_steps[batch] += 1

            cuda.syncthreads()
            if tx == 0:
                keep_going = 0
                for batch in range(batches_per_block):
                    i = block_start + batch
                    if (
                        i < y0.shape[0]
                        and smem_save_idx[batch] < n_save
                        and smem_t[batch] < tf
                        and smem_n_steps[batch] < max_steps
                    ):
                        keep_going = 1
                smem_continue[0] = keep_going
            cuda.syncthreads()

        if tx < batches_per_block:
            i = block_start + tx
            if i < y0.shape[0]:
                accepted_out[i] = smem_accepted[tx]
                rejected_out[i] = smem_rejected[tx]
                loop_out[i] = smem_n_steps[tx]

    return kernel, lu_solver


def solve(
    ode_fn,
    jac_fn,
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
    n, n_vars = y0_arr.shape

    hist_dev = cuda.device_array((n, times.shape[0], n_vars), dtype=np.float64)
    accepted_dev = cuda.device_array(n, dtype=np.int32)
    rejected_dev = cuda.device_array(n, dtype=np.int32)
    loop_dev = cuda.device_array(n, dtype=np.int32)

    flat_work = [cuda.device_array((n, n_vars), dtype=np.float64) for _ in range(10)]
    jac_dev = cuda.device_array((n, n_vars, n_vars), dtype=np.float64)

    kernel, lu_solver = _make_kernel(ode_fn, jac_fn, n_vars)
    threads = _as_launch_block_dim(lu_solver.block_dim)
    blocks = (n + int(lu_solver.batches_per_block) - 1) // int(
        lu_solver.batches_per_block
    )
    kernel[blocks, threads](
        cuda.to_device(y0_arr),
        cuda.to_device(times),
        cuda.to_device(params_arr),
        np.float64(dt0),
        np.float64(rtol),
        np.float64(atol),
        np.int32(max_steps),
        hist_dev,
        accepted_dev,
        rejected_dev,
        loop_dev,
        *flat_work,  # y, u, k1..k8
        jac_dev,
    )
    cuda.synchronize()

    solution = hist_dev.copy_to_host()
    if not return_stats:
        return solution
    accepted_steps = accepted_dev.copy_to_host()
    rejected_steps = rejected_dev.copy_to_host()
    loop_steps = loop_dev.copy_to_host()
    return solution, ck.numpy_stats(accepted_steps, rejected_steps, loop_steps)
