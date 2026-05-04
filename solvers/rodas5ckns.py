"""Single-batch cooperative Rodas5 custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math

import numpy as np
from numba import cuda

from solvers import _rodas5ck_common as ck
from solvers._rodas5ckn_common import (
    PreparedSolve,
    as_launch_block_dim,
    block_threads_x,
    get_workspace,
    make_lu_solver,
)

_WORKSPACE_CACHE: dict[tuple[int, int, int, int], object] = {}


@functools.cache
def _make_kernel(ode_fn, jac_fn, n_vars: int):
    lu_solver = make_lu_solver(n_vars, batches_per_block=1)

    block_dim = as_launch_block_dim(lu_solver.block_dim)
    block_threads = block_threads_x(block_dim)
    a_size = int(lu_solver.a_size())
    b_size = int(lu_solver.b_size())
    ipiv_size = int(lu_solver.ipiv_size)

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
        y_global,
        u_global,
        work_global,
        jac,
    ):
        i = cuda.blockIdx.x
        if i >= y0.shape[0]:
            return

        tx = cuda.threadIdx.x
        n_save = times.shape[0]
        tf = times[n_save - 1]

        smem_lu = cuda.shared.array(shape=a_size, dtype=np.float32)
        smem_rhs = cuda.shared.array(shape=b_size, dtype=np.float32)
        smem_ipiv = cuda.shared.array(shape=ipiv_size, dtype=np.int32)
        smem_info = cuda.shared.array(shape=1, dtype=np.int32)

        smem_y = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_u = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k1 = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k2 = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k3 = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k4 = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k5 = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k6 = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k7 = cuda.shared.array(shape=n_vars, dtype=np.float64)
        smem_k8 = cuda.shared.array(shape=n_vars, dtype=np.float64)

        smem_err = cuda.shared.array(shape=block_threads, dtype=np.float64)
        smem_t = cuda.shared.array(shape=1, dtype=np.float64)
        smem_dt = cuda.shared.array(shape=1, dtype=np.float64)
        smem_dt_use = cuda.shared.array(shape=1, dtype=np.float64)
        smem_inv_dt = cuda.shared.array(shape=1, dtype=np.float64)
        smem_t_end = cuda.shared.array(shape=1, dtype=np.float64)
        smem_next_target = cuda.shared.array(shape=1, dtype=np.float64)
        smem_save_idx = cuda.shared.array(shape=1, dtype=np.int32)
        smem_n_steps = cuda.shared.array(shape=1, dtype=np.int32)
        smem_accepted = cuda.shared.array(shape=1, dtype=np.int32)
        smem_rejected = cuda.shared.array(shape=1, dtype=np.int32)
        smem_continue = cuda.shared.array(shape=1, dtype=np.int32)
        smem_accept = cuda.shared.array(shape=1, dtype=np.int32)
        smem_reached = cuda.shared.array(shape=1, dtype=np.int32)
        smem_hist_idx = cuda.shared.array(shape=1, dtype=np.int32)

        for j in range(tx, n_vars, block_threads):
            val = y0[i, j]
            smem_y[j] = val
            hist[i, 0, j] = val
        if tx == 0:
            smem_t[0] = times[0]
            smem_dt[0] = dt0
            smem_save_idx[0] = 1
            smem_n_steps[0] = 0
            smem_accepted[0] = 0
            smem_rejected[0] = 0
            smem_continue[0] = 1
        cuda.syncthreads()

        while True:
            if tx == 0:
                smem_continue[0] = int(
                    smem_save_idx[0] < n_save
                    and smem_t[0] < tf
                    and smem_n_steps[0] < max_steps
                )
                if smem_continue[0] != 0:
                    next_target = times[smem_save_idx[0]]
                    dt_use = smem_dt[0]
                    if dt_use > next_target - smem_t[0]:
                        dt_use = next_target - smem_t[0]
                    if dt_use < 1e-30:
                        dt_use = 1e-30
                    smem_next_target[0] = next_target
                    smem_dt_use[0] = dt_use
                    smem_inv_dt[0] = 1.0 / dt_use
                    smem_t_end[0] = smem_t[0] + dt_use
            cuda.syncthreads()
            if smem_continue[0] == 0:
                break

            for j in range(tx, n_vars, block_threads):
                y_global[i, j] = smem_y[j]
            cuda.syncthreads()
            if tx == 0:
                jac_fn(y_global, smem_t[0], params, jac, i)
            cuda.syncthreads()

            dtgamma_inv = 1.0 / (smem_dt_use[0] * ck.GAMMA)
            for idx in range(tx, n_vars * n_vars, block_threads):
                row = idx // n_vars
                col = idx - row * n_vars
                if row == col:
                    smem_lu[idx] = np.float32(dtgamma_inv - jac[i, row, col])
                else:
                    smem_lu[idx] = np.float32(-jac[i, row, col])
            cuda.syncthreads()
            lu_solver.factorize(smem_lu, smem_ipiv, smem_info)
            cuda.syncthreads()

            if tx == 0:
                ode_fn(y_global, smem_t[0], params, work_global, i)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(work_global[i, j])
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k1[j] = np.float64(smem_rhs[j])

            for j in range(tx, n_vars, block_threads):
                smem_u[j] = smem_y[j] + ck.A21 * smem_k1[j]
                u_global[i, j] = smem_u[j]
            cuda.syncthreads()
            if tx == 0:
                ode_fn(
                    u_global, smem_t[0] + ck.C2 * smem_dt_use[0], params, work_global, i
                )
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(
                    work_global[i, j] + ck.C21 * smem_k1[j] * smem_inv_dt[0]
                )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k2[j] = np.float64(smem_rhs[j])

            for j in range(tx, n_vars, block_threads):
                smem_u[j] = smem_y[j] + (ck.A31 * smem_k1[j] + ck.A32 * smem_k2[j])
                u_global[i, j] = smem_u[j]
            cuda.syncthreads()
            if tx == 0:
                ode_fn(
                    u_global, smem_t[0] + ck.C3 * smem_dt_use[0], params, work_global, i
                )
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(
                    work_global[i, j]
                    + (ck.C31 * smem_k1[j] + ck.C32 * smem_k2[j]) * smem_inv_dt[0]
                )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k3[j] = np.float64(smem_rhs[j])

            for j in range(tx, n_vars, block_threads):
                smem_u[j] = smem_y[j] + (
                    ck.A41 * smem_k1[j] + ck.A42 * smem_k2[j] + ck.A43 * smem_k3[j]
                )
                u_global[i, j] = smem_u[j]
            cuda.syncthreads()
            if tx == 0:
                ode_fn(
                    u_global, smem_t[0] + ck.C4 * smem_dt_use[0], params, work_global, i
                )
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(
                    work_global[i, j]
                    + (ck.C41 * smem_k1[j] + ck.C42 * smem_k2[j] + ck.C43 * smem_k3[j])
                    * smem_inv_dt[0]
                )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k4[j] = np.float64(smem_rhs[j])

            for j in range(tx, n_vars, block_threads):
                smem_u[j] = smem_y[j] + (
                    ck.A51 * smem_k1[j]
                    + ck.A52 * smem_k2[j]
                    + ck.A53 * smem_k3[j]
                    + ck.A54 * smem_k4[j]
                )
                u_global[i, j] = smem_u[j]
            cuda.syncthreads()
            if tx == 0:
                ode_fn(
                    u_global, smem_t[0] + ck.C5 * smem_dt_use[0], params, work_global, i
                )
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(
                    work_global[i, j]
                    + (
                        ck.C51 * smem_k1[j]
                        + ck.C52 * smem_k2[j]
                        + ck.C53 * smem_k3[j]
                        + ck.C54 * smem_k4[j]
                    )
                    * smem_inv_dt[0]
                )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k5[j] = np.float64(smem_rhs[j])

            for j in range(tx, n_vars, block_threads):
                smem_u[j] = smem_y[j] + (
                    ck.A61 * smem_k1[j]
                    + ck.A62 * smem_k2[j]
                    + ck.A63 * smem_k3[j]
                    + ck.A64 * smem_k4[j]
                    + ck.A65 * smem_k5[j]
                )
                u_global[i, j] = smem_u[j]
            cuda.syncthreads()
            if tx == 0:
                ode_fn(u_global, smem_t_end[0], params, work_global, i)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(
                    work_global[i, j]
                    + (
                        ck.C61 * smem_k1[j]
                        + ck.C62 * smem_k2[j]
                        + ck.C63 * smem_k3[j]
                        + ck.C64 * smem_k4[j]
                        + ck.C65 * smem_k5[j]
                    )
                    * smem_inv_dt[0]
                )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k6[j] = np.float64(smem_rhs[j])
                smem_u[j] += smem_k6[j]
                u_global[i, j] = smem_u[j]
            cuda.syncthreads()

            if tx == 0:
                ode_fn(u_global, smem_t_end[0], params, work_global, i)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(
                    work_global[i, j]
                    + (
                        ck.C71 * smem_k1[j]
                        + ck.C72 * smem_k2[j]
                        + ck.C73 * smem_k3[j]
                        + ck.C74 * smem_k4[j]
                        + ck.C75 * smem_k5[j]
                        + ck.C76 * smem_k6[j]
                    )
                    * smem_inv_dt[0]
                )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k7[j] = np.float64(smem_rhs[j])
                smem_u[j] += smem_k7[j]
                u_global[i, j] = smem_u[j]
            cuda.syncthreads()

            if tx == 0:
                ode_fn(u_global, smem_t_end[0], params, work_global, i)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_rhs[j] = np.float32(
                    work_global[i, j]
                    + (
                        ck.C81 * smem_k1[j]
                        + ck.C82 * smem_k2[j]
                        + ck.C83 * smem_k3[j]
                        + ck.C84 * smem_k4[j]
                        + ck.C85 * smem_k5[j]
                        + ck.C86 * smem_k6[j]
                        + ck.C87 * smem_k7[j]
                    )
                    * smem_inv_dt[0]
                )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            for j in range(tx, n_vars, block_threads):
                smem_k8[j] = np.float64(smem_rhs[j])

            err_local = 0.0
            for j in range(tx, n_vars, block_threads):
                y_new_j = smem_u[j] + smem_k8[j]
                scale = atol + rtol * max(math.fabs(smem_y[j]), math.fabs(y_new_j))
                ratio = smem_k8[j] / scale
                err_local += ratio * ratio
            smem_err[tx] = err_local
            cuda.syncthreads()

            stride = block_threads // 2
            while stride > 0:
                if tx < stride:
                    smem_err[tx] += smem_err[tx + stride]
                cuda.syncthreads()
                stride //= 2

            if tx == 0:
                err_norm = math.sqrt(smem_err[0] / n_vars)
                accept = err_norm <= 1.0 and not math.isnan(err_norm)
                smem_accept[0] = 1 if accept else 0
                smem_reached[0] = 0
                smem_hist_idx[0] = smem_save_idx[0]

                t_new = smem_t[0]
                if accept:
                    t_new = smem_t[0] + smem_dt_use[0]
                    smem_accepted[0] += 1
                    reached = abs(t_new - smem_next_target[0]) <= 1e-12 * max(
                        1.0, abs(smem_next_target[0])
                    )
                    smem_reached[0] = 1 if reached else 0
                    if reached:
                        smem_save_idx[0] += 1
                else:
                    smem_rejected[0] += 1

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
                smem_dt[0] = smem_dt_use[0] * factor
                smem_t[0] = t_new
                smem_n_steps[0] += 1
            cuda.syncthreads()

            if smem_accept[0] != 0:
                for j in range(tx, n_vars, block_threads):
                    smem_y[j] = smem_u[j] + smem_k8[j]
            cuda.syncthreads()

            if smem_reached[0] != 0:
                hist_idx = smem_hist_idx[0]
                for j in range(tx, n_vars, block_threads):
                    hist[i, hist_idx, j] = smem_y[j]
            cuda.syncthreads()

        if tx == 0:
            accepted_out[i] = smem_accepted[0]
            rejected_out[i] = smem_rejected[0]
            loop_out[i] = smem_n_steps[0]

    return kernel, lu_solver


def prepare_solve(
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
):
    del batch_size
    y0_arr, times, params_arr, dt0 = ck.normalize_inputs(y0, t_span, params, first_step)
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]

    workspace = get_workspace(_WORKSPACE_CACHE, n, n_vars, n_save, n_params)
    workspace.y0_dev.copy_to_device(y0_arr)
    workspace.times_dev.copy_to_device(times)
    workspace.params_dev.copy_to_device(params_arr)

    kernel, lu_solver = _make_kernel(ode_fn, jac_fn, n_vars)
    threads = as_launch_block_dim(lu_solver.block_dim)
    blocks = n

    return PreparedSolve(
        kernel=kernel,
        lu_solver=lu_solver,
        workspace=workspace,
        dt0=np.float64(dt0),
        rtol=np.float64(rtol),
        atol=np.float64(atol),
        max_steps=np.int32(max_steps),
        blocks=blocks,
        threads=threads,
    )


def run_prepared(prepared: PreparedSolve, *, return_stats=False, copy_solution=True):
    workspace = prepared.workspace
    prepared.kernel[prepared.blocks, prepared.threads](
        workspace.y0_dev,
        workspace.times_dev,
        workspace.params_dev,
        prepared.dt0,
        prepared.rtol,
        prepared.atol,
        prepared.max_steps,
        workspace.hist_dev,
        workspace.accepted_dev,
        workspace.rejected_dev,
        workspace.loop_dev,
        workspace.work[0],
        workspace.work[1],
        workspace.work[2],
        workspace.jac_dev,
    )
    cuda.synchronize()

    solution = (
        workspace.hist_dev.copy_to_host() if copy_solution else workspace.hist_dev
    )
    if not return_stats:
        return solution

    accepted_steps = workspace.accepted_dev.copy_to_host()
    rejected_steps = workspace.rejected_dev.copy_to_host()
    loop_steps = workspace.loop_dev.copy_to_host()
    return solution, ck.numpy_stats(accepted_steps, rejected_steps, loop_steps)


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
    prepared = prepare_solve(
        ode_fn,
        jac_fn,
        y0,
        t_span,
        params,
        batch_size=batch_size,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )
    return run_prepared(prepared, return_stats=return_stats, copy_solution=True)
