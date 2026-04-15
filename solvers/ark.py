"""Shared helpers for additive Runge-Kutta solvers.

This module contains the dynamic Gershgorin splitting utilities used by the
``kencarpgersh5`` solvers.  The split is row-based: for a Jacobian ``J`` and
trial step size ``dt``, each row is classified as stiff when

``sum(abs(J[i, :])) > gershgorin_scale / max(dt, 1e-30)``.

Rows classified as non-stiff are zeroed out of the implicit operator.  After
reordering the variables so the non-stiff rows come first, the stage system

``(I - coeff * J_stiff_rows) x = rhs``

becomes block lower triangular,

``[[I, 0], [-coeff * J_sn, I - coeff * J_ss]]``.

That lets us avoid a full ``n x n`` factorization: the non-stiff part is copied
directly from ``rhs`` and only the stiff-stiff block is LU-factorized.  JAX
requires static slice sizes under ``jit``, so the reduced solve is implemented
with ``lax.switch`` over all possible stiff block sizes.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax


def gershgorin_stiff_mask(jac, dt, gershgorin_scale):
    """Classify stiff rows using a Gershgorin row-sum bound."""
    row_bounds = jnp.sum(jnp.abs(jac), axis=-1)
    tau = gershgorin_scale / jnp.maximum(dt[..., None], 1e-30)
    return row_bounds > tau


def row_partition(mask):
    """Return a stable non-stiff-first permutation and its inverse."""
    perm = jnp.argsort(mask.astype(jnp.int32), axis=-1, stable=True)
    inv_perm = jnp.argsort(perm, axis=-1)
    n_stiff = jnp.sum(mask, axis=-1, dtype=jnp.int32)
    return perm, inv_perm, n_stiff


def permute_vector(vec, perm):
    """Apply a row permutation to the last axis of a batched vector."""
    return jnp.take_along_axis(vec, perm, axis=-1)


def unpermute_vector(vec_perm, inv_perm):
    """Undo a row permutation on the last axis of a batched vector."""
    return jnp.take_along_axis(vec_perm, inv_perm, axis=-1)


def permute_matrix(mat, perm):
    """Apply the same permutation to the row and column axes of a matrix."""
    mat_rows = jnp.take_along_axis(mat, perm[..., :, None], axis=-2)
    return jnp.take_along_axis(mat_rows, perm[..., None, :], axis=-1)


def split_rows(values, mask):
    """Split a vector into explicit and implicit row contributions."""
    implicit = jnp.where(mask, values, 0.0)
    explicit = values - implicit
    return explicit, implicit


def make_reduced_row_implicit_solver(n_vars, lu_dtype):
    """Create a batched reduced-order solve for row-masked implicit systems."""

    @functools.partial(jax.vmap, in_axes=(0, 0, 0, 0))
    def _solve_single(jac_perm, rhs_perm, n_stiff, coeff):
        branches = []

        for stiff_size in range(n_vars + 1):
            n_nonstiff = n_vars - stiff_size

            def _branch(args, *, _stiff_size=stiff_size, _n_nonstiff=n_nonstiff):
                jac_perm, rhs_perm, coeff = args
                if _stiff_size == 0:
                    return rhs_perm

                x_nonstiff = rhs_perm[:_n_nonstiff]
                jac_sn = jac_perm[_n_nonstiff:, :_n_nonstiff]
                jac_ss = jac_perm[_n_nonstiff:, _n_nonstiff:]
                mat_ss = (
                    jnp.eye(_stiff_size, dtype=lu_dtype)
                    - coeff.astype(lu_dtype) * jac_ss.astype(lu_dtype)
                )
                rhs_stiff = rhs_perm[_n_nonstiff:] + coeff * (
                    jac_sn.astype(jnp.float64) @ x_nonstiff.astype(jnp.float64)
                )
                lu_and_piv = jax.scipy.linalg.lu_factor(mat_ss)
                x_stiff = jax.scipy.linalg.lu_solve(
                    lu_and_piv, rhs_stiff.astype(lu_dtype)
                ).astype(jnp.float64)
                return jnp.concatenate((x_nonstiff, x_stiff), axis=0)

            branches.append(_branch)

        return lax.switch(n_stiff, branches, (jac_perm, rhs_perm, coeff))

    def _solve_batched(jac, rhs, mask, coeff):
        perm, inv_perm, n_stiff = row_partition(mask)
        jac_perm = permute_matrix(jac, perm)
        rhs_perm = permute_vector(rhs, perm)
        x_perm = _solve_single(jac_perm, rhs_perm, n_stiff, coeff)
        return unpermute_vector(x_perm, inv_perm)

    return _solve_batched
