"""KenCarp5 solver — linear IMEX ODE variant (split jac_fn path).

Accepts split Jacobians
``explicit_jac_fn(t, params) -> [n_vars, n_vars]`` and
``implicit_jac_fn(t, params) -> [n_vars, n_vars]``.

The explicit contribution is applied via matrix-vector products. The implicit
DIRK stage equation is solved via LU factorization of
``I - h * gamma * J_implicit`` at each stage.

Uses a single jax.lax.while_loop with the batch dimension inside the loop
body instead of vmap-over-while-loop.
"""

import functools
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# fmt: off
_GAMMA = 41.0 / 200.0

_B_SOL = np.array([
    -872700587467.0 / 9133579230613.0,
    0.0,
    0.0,
    22348218063261.0 / 9555858737531.0,
    -1143369518992.0 / 8141816002931.0,
    -39379526789629.0 / 19018526304540.0,
    32727382324388.0 / 42900044865799.0,
    _GAMMA,
], dtype=np.float64)
_B_EMBEDDED = np.array([
    -975461918565.0 / 9796059967033.0,
    0.0,
    0.0,
    78070527104295.0 / 32432590147079.0,
    -548382580838.0 / 3424219808633.0,
    -33438840321285.0 / 15594753105479.0,
    3629800801594.0 / 4656183773603.0,
    4035322873751.0 / 18575991585200.0,
], dtype=np.float64)
_B_ERROR = _B_SOL - _B_EMBEDDED
_C = np.array([
    0.0,
    41.0 / 100.0,
    2935347310677.0 / 11292855782101.0,
    1426016391358.0 / 7196633302097.0,
    92.0 / 100.0,
    24.0 / 100.0,
    3.0 / 5.0,
    1.0,
], dtype=np.float64)

_A_EXPLICIT = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [41.0 / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [367902744464.0 / 2072280473677.0, 677623207551.0 / 8224143866563.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1268023523408.0 / 10340822734521.0, 0.0, 1029933939417.0 / 13636558850479.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [14463281900351.0 / 6315353703477.0, 0.0, 66114435211212.0 / 5879490589093.0, -54053170152839.0 / 4284798021562.0, 0.0, 0.0, 0.0, 0.0],
    [14090043504691.0 / 34967701212078.0, 0.0, 15191511035443.0 / 11219624916014.0, -18461159152457.0 / 12425892160975.0, -281667163811.0 / 9011619295870.0, 0.0, 0.0, 0.0],
    [19230459214898.0 / 13134317526959.0, 0.0, 21275331358303.0 / 2942455364971.0, -38145345988419.0 / 4862620318723.0, -1.0 / 8.0, -1.0 / 8.0, 0.0, 0.0],
    [-19977161125411.0 / 11928030595625.0, 0.0, -40795976796054.0 / 6384907823539.0, 177454434618887.0 / 12078138498510.0, 782672205425.0 / 8267701900261.0, -69563011059811.0 / 9646580694205.0, 7356628210526.0 / 4942186776405.0, 0.0],
], dtype=np.float64)

_A_IMPLICIT = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, _GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [41.0 / 400.0, -567603406766.0 / 11931857230679.0, _GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0],
    [683785636431.0 / 9252920307686.0, 0.0, -110385047103.0 / 1367015193373.0, _GAMMA, 0.0, 0.0, 0.0, 0.0],
    [3016520224154.0 / 10081342136671.0, 0.0, 30586259806659.0 / 12414158314087.0, -22760509404356.0 / 11113319521817.0, _GAMMA, 0.0, 0.0, 0.0],
    [218866479029.0 / 1489978393911.0, 0.0, 638256894668.0 / 5436446318841.0, -1179710474555.0 / 5321154724896.0, -60928119172.0 / 8023461067671.0, _GAMMA, 0.0, 0.0],
    [1020004230633.0 / 5715676835656.0, 0.0, 25762820946817.0 / 25263940353407.0, -2161375909145.0 / 9755907335909.0, -211217309593.0 / 5846859502534.0, -4269925059573.0 / 7827059040719.0, _GAMMA, 0.0],
    [_B_SOL[0], _B_SOL[1], _B_SOL[2], _B_SOL[3], _B_SOL[4], _B_SOL[5], _B_SOL[6], _GAMMA],
], dtype=np.float64)
# fmt: on

_SAFETY = 0.9
_FACTOR_MIN = 0.2
_FACTOR_MAX = 10.0
_BATCHED_DOT_DIMENSION_NUMBERS = (((2,), (1,)), ((0,), (0,)))


def _batched_matvec(mat, vec, *, precision):
    out = lax.dot(
        mat,
        vec[..., None],
        dimension_numbers=_BATCHED_DOT_DIMENSION_NUMBERS,
        precision=precision,
    )
    return out[..., 0]


def make_solver(
    explicit_jac_fn,
    implicit_jac_fn,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    mv_precision: Literal["fp32", "fp64"] = "fp64",
    batch_size=None,
):
    """Create a reusable KenCarp5 ensemble solver for linear split ODEs."""
    lu_dtype = jnp.float32 if lu_precision == "fp32" else jnp.float64
    mv_dtype = jnp.float32 if mv_precision == "fp32" else jnp.float64
    dot_precision = (
        lax.DotAlgorithmPreset.TF32_TF32_F32
        if mv_precision == "fp32" and jax.default_backend() != "cpu"
        else lax.Precision.DEFAULT
    )

    lu_factor_batched = jax.vmap(jax.scipy.linalg.lu_factor)
    lu_solve_batched = jax.vmap(jax.scipy.linalg.lu_solve)
    _explicit_jac_batched = jax.vmap(explicit_jac_fn)
    _implicit_jac_batched = jax.vmap(implicit_jac_fn)

    @functools.partial(jax.jit, static_argnames=("n_save", "max_steps"))
    def _solve_impl(
        y0_arr, params_batches, times, *, n_save, max_steps, dt0, rtol, atol
    ):
        n_vars = y0_arr.shape[0]
        bs = params_batches.shape[1]
        eye = jnp.eye(n_vars, dtype=lu_dtype)[None, :, :]
        tf = times[-1]

        def _solve_batch(params_batch):
            y_init = jnp.broadcast_to(y0_arr, (bs, n_vars)).copy()
            hist_init = (
                jnp.zeros((bs, n_save, n_vars), dtype=jnp.float64)
                .at[:, 0, :]
                .set(y_init)
            )
            t_init = jnp.full((bs,), times[0], dtype=jnp.float64)
            dt_init = jnp.full((bs,), dt0, dtype=jnp.float64)
            save_idx_init = jnp.ones((bs,), dtype=jnp.int32)

            def _f_eval(jac, u):
                return _batched_matvec(
                    jac.astype(mv_dtype),
                    u.astype(mv_dtype),
                    precision=dot_precision,
                ).astype(jnp.float64)

            def _step_batch(y, t, dt):
                dt_col = dt[:, None]
                stage_y = []
                stage_fe = []
                stage_fi = []

                for i in range(8):
                    t_stage = t + _C[i] * dt
                    base = y
                    for j in range(i):
                        ae = _A_EXPLICIT[i, j]
                        ai = _A_IMPLICIT[i, j]
                        if ae != 0.0:
                            base = base + dt_col * ae * stage_fe[j]
                        if ai != 0.0:
                            base = base + dt_col * ai * stage_fi[j]

                    implicit_jac = _implicit_jac_batched(t_stage, params_batch)
                    if _A_IMPLICIT[i, i] != 0.0:
                        mat = eye - (
                            dt * _A_IMPLICIT[i, i]
                        ).astype(lu_dtype)[:, None, None] * implicit_jac.astype(
                            lu_dtype
                        )
                        lu = lu_factor_batched(mat)
                        y_stage = lu_solve_batched(lu, base.astype(lu_dtype)).astype(
                            jnp.float64
                        )
                    else:
                        y_stage = base

                    explicit_jac = _explicit_jac_batched(t_stage, params_batch)
                    fe_stage = _f_eval(explicit_jac, y_stage)
                    fi_stage = _f_eval(implicit_jac, y_stage)
                    stage_y.append(y_stage)
                    stage_fe.append(fe_stage)
                    stage_fi.append(fi_stage)

                y_new = stage_y[-1]
                err_est = jnp.zeros_like(y)
                for i in range(8):
                    total_stage = stage_fe[i] + stage_fi[i]
                    err_est = err_est + dt_col * _B_ERROR[i] * total_stage
                return y_new, err_est

            def cond_fn(state):
                t, _, _, _, save_idx, n_steps = state
                active = save_idx < n_save
                return (jnp.min(jnp.where(active, t, tf)) < tf) & (n_steps < max_steps)

            def body_fn(state):
                t, y, dt, hist, save_idx, n_steps = state
                active = save_idx < n_save
                next_target = times[save_idx]
                dt_use = jnp.where(
                    active,
                    jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30),
                    1e-30,
                )

                y_new, err_est = _step_batch(y, t, dt_use)

                scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
                err_norm = jnp.sqrt(jnp.mean((err_est / scale) ** 2, axis=1))

                accept = active & (err_norm <= 1.0) & ~jnp.isnan(err_norm)
                t_new = jnp.where(accept, t + dt_use, t)
                y_out = jnp.where(accept[:, None], y_new, y)

                reached = accept & (
                    jnp.abs(t_new - next_target)
                    <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
                )
                slot_mask = (
                    jax.nn.one_hot(save_idx, n_save, dtype=jnp.bool_) & reached[:, None]
                )
                hist_new = jnp.where(slot_mask[:, :, None], y_out[:, None, :], hist)
                save_idx_new = save_idx + reached.astype(jnp.int32)

                safe_err = jnp.where(
                    jnp.isnan(err_norm) | (err_norm > 1e18),
                    1e18,
                    jnp.where(err_norm == 0.0, 1e-18, err_norm),
                )
                factor = jnp.clip(
                    _SAFETY * safe_err ** (-1.0 / 5.0), _FACTOR_MIN, _FACTOR_MAX
                )
                dt_new = jnp.where(active, dt_use * factor, dt)
                return (t_new, y_out, dt_new, hist_new, save_idx_new, n_steps + 1)

            init = (t_init, y_init, dt_init, hist_init, save_idx_init, jnp.int32(0))
            _, _, _, hist_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
            return hist_final

        return jax.vmap(_solve_batch)(params_batches)

    def _solve(
        y0,
        t_span,
        params,
        *,
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
    ):
        y0_arr = jnp.asarray(y0, dtype=jnp.float64)
        params_arr = jnp.asarray(params)
        n_vars = int(y0_arr.shape[0])
        N = int(params_arr.shape[0])
        times = np.asarray(t_span, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError(
                f"t_span must be a 1D array of save times, got shape {times.shape}"
            )
        if times.size < 2:
            raise ValueError(
                f"t_span must contain at least 2 save times, got {times.size}"
            )
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("t_span must be strictly increasing")

        times_jnp = jnp.asarray(times, dtype=jnp.float64)
        dt0 = jnp.float64(
            first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
        )
        bs = N if batch_size is None else batch_size
        n_chunks = (N + bs - 1) // bs
        n_padded = n_chunks * bs

        if n_padded > N:
            pad_rows = jnp.broadcast_to(
                params_arr[-1:],
                (n_padded - N,) + params_arr.shape[1:],
            )
            params_padded = jnp.concatenate([params_arr, pad_rows], axis=0)
        else:
            params_padded = params_arr

        params_batches = params_padded.reshape((n_chunks, bs) + params_arr.shape[1:])
        results = _solve_impl(
            y0_arr,
            params_batches,
            times_jnp,
            n_save=int(times.size),
            max_steps=int(max_steps),
            dt0=float(dt0),
            rtol=float(rtol),
            atol=float(atol),
        )
        return results.reshape(n_padded, int(times.size), n_vars)[:N]

    return _solve
