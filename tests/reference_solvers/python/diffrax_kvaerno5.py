"""Kvaerno5 solver via diffrax — ESDIRK method of order 5 for stiff ODEs."""

import diffrax
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402


def make_solver(ode_fn):
    """Create a reusable Kvaerno5 ensemble solver.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``.
    """

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
        """Solve an ensemble of ODEs.

        Parameters
        ----------
        y0 : array, shape [n_vars]
            Shared initial state.
        t_span : array-like, shape [n_save]
            Strictly increasing array of save times (len >= 2).
        params : array, shape [N, ...]
            Per-trajectory parameters.

        Returns
        -------
        array, shape [N, n_save, n_vars]
        """
        y0_arr = jnp.asarray(y0, dtype=jnp.float64)
        params_arr = jnp.asarray(params)
        save_times = jnp.asarray(np.asarray(t_span, dtype=np.float64), dtype=jnp.float64)
        t0 = float(save_times[0])
        tf = float(save_times[-1])
        dt0 = first_step if first_step is not None else (tf - t0) * 1e-6

        def _solve_one(p):
            term = diffrax.ODETerm(lambda t, y, args: ode_fn(y, t, p))
            solver = diffrax.Kvaerno5()
            controller = diffrax.PIDController(rtol=rtol, atol=atol)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=tf,
                dt0=dt0,
                y0=y0_arr,
                stepsize_controller=controller,
                max_steps=max_steps,
                saveat=diffrax.SaveAt(ts=save_times),
            )
            return sol.ys  # shape [n_save, n_vars]

        return jax.jit(jax.vmap(_solve_one))(params_arr)  # shape [N, n_save, n_vars]

    return _solve
