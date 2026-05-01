"""Shared helpers for custom-kernel Rodas5 variants."""

from __future__ import annotations

import numpy as np

# fmt: off
GAMMA = 0.19

A21 = 2.0
A31 = 3.040894194418781;  A32 = 1.041747909077569
A41 = 2.576417536461461;  A42 = 1.622083060776640;  A43 = -0.9089668560264532
A51 = 2.760842080225597;  A52 = 1.446624659844071;  A53 = -0.3036980084553738;  A54 = 0.2877498600325443
A61 = -14.09640773051259; A62 = 6.925207756232704;  A63 = -41.47510893210728;   A64 = 2.343771018586405;  A65 = 24.13215229196062
A71 = A61;                A72 = A62;                A73 = A63;                   A74 = A64;               A75 = A65;               A76 = 1.0
A81 = A61;                A82 = A62;                A83 = A63;                   A84 = A64;               A85 = A65;               A86 = 1.0;  A87 = 1.0

C21 = -10.31323885133993
C31 = -21.04823117650003; C32 = -7.234992135176716
C41 = 32.22751541853323;  C42 = -4.943732386540191;  C43 = 19.44922031041879
C51 = -20.69865579590063; C52 = -8.816374604402768;  C53 = 1.260436877740897;   C54 = -0.7495647613787146
C61 = -46.22004352711257; C62 = -17.49534862857472;  C63 = -289.6389582892057;  C64 = 93.60855400400906;  C65 = 318.3822534212147
C71 = 34.20013733472935;  C72 = -14.15535402717690;  C73 = 57.82335640988400;   C74 = 25.83362985412365;  C75 = 1.408950972071624;  C76 = -6.551835421242162
C81 = 42.57076742291101;  C82 = -13.80770672017997;  C83 = 93.98938432427124;   C84 = 18.77919633714503;  C85 = -31.58359187223370;  C86 = -6.685968952921985;  C87 = -5.810979938412932

C2 = 0.38
C3 = 0.3878509998321533
C4 = 0.4839718937873840
C5 = 0.4570477008819580
# C6 = C7 = C8 = 1.0
# fmt: on

SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 6.0


def normalize_inputs(y0, t_span, params, first_step):
    y0_in = np.asarray(y0, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)
    times = np.asarray(t_span, dtype=np.float64)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
        params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
    else:
        n = y0_in.shape[0]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )

    if y0_arr.ndim != 2:
        raise ValueError("custom-kernel Rodas5 expects y0 shape (N, n_vars)")
    if params_arr.ndim != 2:
        raise ValueError("custom-kernel Rodas5 expects params shape (N, n_params)")
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("t_span must be a 1-D array with at least two save times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_span must be strictly increasing")

    dt0 = (
        np.float64(first_step)
        if first_step is not None
        else np.float64((times[-1] - times[0]) * 1e-6)
    )
    return y0_arr, times, params_arr, dt0


def numpy_stats(accepted_steps, rejected_steps, loop_steps):
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": np.ones_like(loop_steps, dtype=np.int32),
    }
