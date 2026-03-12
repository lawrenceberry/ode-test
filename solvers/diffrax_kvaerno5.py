"""Kvaerno5 solver via diffrax — ESDIRK method of order 5 for stiff ODEs."""

import diffrax
import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402


def solve(f, y0, t_span, *, rtol=1e-8, atol=1e-10, first_step=None, max_steps=100000):
    """Solve a stiff autonomous ODE system using diffrax Kvaerno5.

    Args:
        f: JAX function mapping state vector y -> dy/dt.
        y0: Initial conditions (list or array).
        t_span: Tuple (t0, tf) for integration bounds.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        first_step: Initial step size (optional).
        max_steps: Maximum number of steps.

    Returns:
        Final state array of shape (n_components,).
    """
    y0_arr = jnp.asarray(y0, dtype=jnp.float64)
    t0, tf = t_span
    dt0 = first_step if first_step is not None else (tf - t0) * 1e-6

    term = diffrax.ODETerm(lambda t, y, args: f(y))
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
        saveat=diffrax.SaveAt(t1=True),
    )
    return sol.ys[0]


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
    """Solve an ensemble of ODEs with different parameters using vmap.

    Args:
        f: JAX function (y, params) -> dy/dt.
        y0: Initial conditions, shared across ensemble.
        t_span: Tuple (t0, tf), shared across ensemble.
        params_batch: Array of shape (n_ensemble, ...) with parameters.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        first_step: Initial step size (optional).
        max_steps: Maximum number of steps.

    Returns:
        Array of shape (n_ensemble, n_components) with final states.
    """

    def _solve_one(params):
        def f_fn(y):
            return f(y, params)

        return solve(
            f_fn, y0, t_span, rtol=rtol, atol=atol, first_step=first_step, max_steps=max_steps
        )

    return jax.jit(jax.vmap(_solve_one))(params_batch)
