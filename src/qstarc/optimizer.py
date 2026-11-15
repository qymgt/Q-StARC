"""Q-StARC: projected ambient Adam followed by Riemannian PR+ conjugate gradient.

The optimization kernels and legacy defaults retain the frozen 2026-11-02
implementation. Adam moments are ambient and are not transported; Adam returns
its last iterate. RCG uses projection transport, PR+, Armijo backtracking, and
phase-corrected thin QR. No Fisher ridge or probability clipping is added.

Paper notation (Secs. II and III): n is the Hilbert-space dimension, d the
number of parameters, s the number of outcomes, and Z the Stiefel matrix.
The historical argument/result name Q denotes Z, with shape (s,n) and
Q.conj().T @ Q = I_n; it is not the paper's kernel projection Q=I-P.
The argument mout is s. Outcome x has effect M_x=|u_x><u_x|, where u_x is
column x of Z.conj().T (returned as U), or outer(Q[x].conj(), Q[x]) in code.
The row count s includes all stored outcomes before pruning or merging.
The metric is Re(vdot(Y,T)); derivatives use real and imaginary parts of Z.

Legacy minimizeQ_stiefel minimizes f_I(Z)=Tr(F^-1) in the supplied coordinates.
optimize_povm optionally whitens a positive definite weight and records both
coordinate systems (tilde D and tilde F in Sec. III G). Its risk is the
numerical candidate c_num=Tr(W F_original^-1). Experiment-specific reporting
scales, such as N in Eq. (7.3), are applied separately by the example driver.
Returned risks are never certified upper bounds or claims of global optimality.

Input validation uses absolute tolerances (rtol=0): 1e-12 for Hermiticity
and 1e-10 for traces. This deliberately tightens the frozen validation's
relative-tolerance checks while leaving valid-input optimization kernels
and defaults unchanged. A density eigenvalue below -1e-12 is rejected.
"""

# %% setup
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# %% geometry
def pack_complex(Q):
    """Paper real-coordinate map P: complex Z of shape (s,n) -> (2,s,n)."""
    return jnp.stack([Q.real, Q.imag])


def unpack_complex(z):
    return z[0] + 1j * z[1]


def inner(X, Y):
    """Real Frobenius inner product; stays on the JAX device."""
    return jnp.real(jnp.vdot(X, Y))


def stiefel_project(Q, G):
    """Paper projection Pi_Z(Y)=Y-Z sym(Z^*Y); historical Q=Z and G=Y."""
    A = Q.conj().T @ G
    return G - Q @ ((A + A.conj().T) / 2)


def stiefel_retract(Y):
    """Paper qf(Y): thin QR with a nonnegative real diagonal in R.

    For a full-column-rank input the diagonal is positive, as in Appendix D 1.
    Applying this to Z+T gives the retraction R_Z(T).
    """
    Q, R = jnp.linalg.qr(Y, mode="reduced")
    diagonal = jnp.diag(R)
    magnitude = jnp.abs(diagonal)
    phase = diagonal / jnp.where(magnitude > 0, magnitude, 1.0)
    phase = jnp.where(magnitude > 0, phase, 1.0 + 0j)
    return Q * phase[None, :]


def random_stiefel(key, mout, dim):
    key_re, key_im = jax.random.split(key)
    Z = (jax.random.normal(key_re, (mout, dim), dtype=jnp.float64)
         + 1j * jax.random.normal(key_im, (mout, dim), dtype=jnp.float64))
    return stiefel_retract(Z)


# %% objective
def _probabilities(Q, rho, d_rho):
    # No (mout, D, D) array of POVM matrices is constructed.
    U = Q.conj().T
    p = jnp.einsum("as,ab,bs->s", U.conj(), rho, U).real
    dp = jnp.einsum("as,kab,bs->ks", U.conj(), d_rho, U).real
    return p, dp


@jax.jit
def classical_fisher_stiefel(Q, rho, d_rho):
    """Paper F(Z)_ij=sum_x (partial_i p_x)(partial_j p_x)/p_x; shape (d,d).

    The historical Q is Z. The derivatives d_rho are D_i in the supplied
    coordinates, or tilde D_i after weight whitening; F is then tilde F.

    Tiny positive probabilities are used without clipping. A zero vector
    contributes zero. A nonzero outcome with p <= 0 is outside this regular
    objective's domain and returns NaN (handled by the objective below).
    """
    p, dp = _probabilities(Q, rho, d_rho)
    zero_row = jnp.all(Q == 0, axis=1)
    valid = jnp.all((p > 0) | (zero_row & (p == 0)))
    denominator = jnp.where(p > 0, p, 1.0)
    F = (dp / denominator[None, :]) @ dp.T
    return jnp.where(valid, (F + F.T) / 2, jnp.full_like(F, jnp.nan))


def _objective_real(z, rho, d_rho):
    F = classical_fisher_stiefel(unpack_complex(z), rho, d_rho)
    # The risk is defined only for positive definite F. No ridge is added.
    positive_definite = jnp.all(jnp.isfinite(jnp.linalg.cholesky(F)))
    return jax.lax.cond(
        positive_definite,
        lambda A: jnp.trace(jnp.linalg.solve(A, jnp.eye(A.shape[0]))),
        lambda A: jnp.array(jnp.inf, dtype=jnp.float64),
        F,
    )


_value_and_grad_real = jax.value_and_grad(_objective_real)


@jax.jit
def value_Q(Q, rho, d_rho):
    return _objective_real(pack_complex(Q), rho, d_rho)


@jax.jit
def value_and_rgrad_Q(Q, rho, d_rho):
    value, grad_z = _value_and_grad_real(pack_complex(Q), rho, d_rho)
    return value, stiefel_project(Q, unpack_complex(grad_z))


def make_stiefel_objective(rho, d_rho):
    """Compatibility wrapper. JIT kernels above are shared across calls."""
    return (lambda Q: value_Q(Q, rho, d_rho),
            lambda Q: value_and_rgrad_Q(Q, rho, d_rho))


# %% validation
def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _validate_model(rho, d_rho, mout):
    rho, d_rho = np.asarray(rho), np.asarray(d_rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must have shape (D, D).")
    dim = rho.shape[0]
    if d_rho.ndim != 3 or d_rho.shape[1:] != (dim, dim) or len(d_rho) == 0:
        raise ValueError("d_rho must have shape (nparam, D, D), nparam >= 1.")
    _integer("mout", mout, dim)
    if mout <= len(d_rho):
        raise ValueError("An invertible Fisher matrix requires mout >= nparam + 1.")
    if not np.isfinite(rho).all() or not np.isfinite(d_rho).all():
        raise ValueError("Model matrices must be finite.")
    if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0):
        raise ValueError("rho must be Hermitian.")
    if not np.allclose(d_rho, d_rho.conj().swapaxes(-1, -2), atol=1e-12, rtol=0):
        raise ValueError("Every derivative must be Hermitian.")
    if not np.isclose(np.trace(rho), 1, atol=1e-10, rtol=0):
        raise ValueError("rho must have trace 1.")
    if not np.allclose(np.trace(d_rho, axis1=1, axis2=2), 0, atol=1e-10, rtol=0):
        raise ValueError("Density-matrix derivatives must have trace 0.")
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("rho must be positive semidefinite.")
    return jnp.asarray(rho, dtype=jnp.complex128), jnp.asarray(d_rho, dtype=jnp.complex128)


# %% adam
@jax.jit
def _adam_chunk(Q, first, second, count, nsteps, lr, rho, d_rho):
    """Original projected ambient Adam. Moments are NOT transported."""
    def step(i, state):
        def advance(state):
            Q, first, second, healthy = state
            value, G = value_and_rgrad_Q(Q, rho, d_rho)
            g = pack_complex(G)
            first_new = 0.9 * first + 0.1 * g
            second_new = 0.999 * second + 0.001 * g**2
            t = count + i + 1
            mhat = first_new / (1 - 0.9**t)
            vhat = second_new / (1 - 0.999**t)
            direction = unpack_complex(mhat / (jnp.sqrt(vhat) + 1e-8))
            direction = stiefel_project(Q, direction)
            candidate = stiefel_retract(Q - lr * direction)
            healthy = (jnp.isfinite(value) & jnp.all(jnp.isfinite(g))
                       & jnp.all(jnp.isfinite(candidate)))
            return (jnp.where(healthy, candidate, Q),
                    jnp.where(healthy, first_new, first),
                    jnp.where(healthy, second_new, second), healthy)
        return jax.lax.cond(state[3], advance, lambda s: s, state)

    return jax.lax.fori_loop(0, nsteps, step, (Q, first, second, jnp.array(True)))


def stiefel_adam(rho, d_rho, mout, seed=0, adam_steps=30000,
                 adam_lr=0.005, adam_chunk=5000, verbose=True):
    """Coarse search; returns the LAST iterate, as in the original code."""
    rho, d_rho = _validate_model(rho, d_rho, mout)
    _integer("seed", seed)
    _integer("adam_steps", adam_steps)
    _integer("adam_chunk", adam_chunk, 1)
    _positive("adam_lr", adam_lr)
    Q = random_stiefel(jax.random.PRNGKey(seed), mout, rho.shape[0])
    first = jnp.zeros((2,) + Q.shape, dtype=jnp.float64)
    second = jnp.zeros_like(first)
    start = time.perf_counter()
    for count in range(0, adam_steps, adam_chunk):
        nsteps = min(adam_chunk, adam_steps - count)
        Q, first, second, healthy = _adam_chunk(
            Q, first, second, count, nsteps, adam_lr, rho, d_rho)
        value = float(value_Q(Q, rho, d_rho))  # synchronize once per chunk
        if not bool(healthy) or not np.isfinite(value):
            raise FloatingPointError("Adam encountered an invalid objective or gradient; try another seed or smaller adam_lr.")
        if verbose:
            print(f"Adam {count + nsteps:6d}: f={value:.12f}")
    Q.block_until_ready()
    if verbose:
        print(f"Adam elapsed: {time.perf_counter() - start:.2f} s (includes compilation when needed)")
    return Q


# %% rcg
class _CGState(NamedTuple):
    Q: object
    value: object
    grad: object
    direction: object
    iteration: object
    status: object  # 0 running, 1 converged, 2 maxiter, 3 line search failed, 4 nonfinite
    line_search_evaluations: object


def _armijo(Q, value, direction, slope, rho, d_rho, alpha0, c1, shrink, limit):
    """Return accepted point, or the original point after total failure."""
    # State: trial count, next alpha, accepted flag, selected Q.
    def condition(state):
        count, _, accepted, _ = state
        return (count < limit) & ~accepted

    def trial(state):
        count, alpha, _, _ = state
        candidate = stiefel_retract(Q + alpha * direction)
        trial_value = value_Q(candidate, rho, d_rho)
        accepted = jnp.isfinite(trial_value) & (trial_value <= value + c1 * alpha * slope)
        return count + 1, alpha * shrink, accepted, jnp.where(accepted, candidate, Q)

    count, _, accepted, selected = jax.lax.while_loop(
        condition, trial, (jnp.array(0), jnp.asarray(alpha0), jnp.array(False), Q))
    return selected, accepted, count


@jax.jit
def _rcg_chunk(state, stop_at, rho, d_rho, grad_tol, alpha0, c1, shrink, limit):
    """PR+ updates and backtracking stay on the device between reports."""
    def condition(s):
        return (s.iteration < stop_at) & (s.status == 0)

    def step(s):
        slope = inner(s.grad, s.direction)
        direction = jnp.where(slope < 0, s.direction, -s.grad)
        slope = inner(s.grad, direction)
        Qnew, accepted, evaluations = _armijo(
            s.Q, s.value, direction, slope, rho, d_rho, alpha0, c1, shrink, limit)

        def update(_):
            value, grad = value_and_rgrad_Q(Qnew, rho, d_rho)
            finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(grad))
            # Projection transport compares tangent vectors at the SAME point.
            old_grad_here = stiefel_project(Qnew, s.grad)
            old_direction_here = stiefel_project(Qnew, direction)
            y = grad - old_grad_here
            beta = jnp.maximum(0.0, inner(grad, y) / jnp.maximum(inner(s.grad, s.grad), 1e-30))
            new_direction = -grad + beta * old_direction_here
            new_direction = jnp.where(inner(grad, new_direction) < 0, new_direction, -grad)
            status = jnp.where(jnp.sqrt(inner(grad, grad)) < grad_tol, 1, 0)
            return _CGState(
                jnp.where(finite, Qnew, s.Q), jnp.where(finite, value, s.value),
                jnp.where(finite, grad, s.grad), jnp.where(finite, new_direction, s.direction),
                s.iteration + finite.astype(s.iteration.dtype),
                jnp.where(finite, status, 4), s.line_search_evaluations + evaluations)

        return jax.lax.cond(
            accepted, update,
            lambda _: s._replace(status=jnp.array(3),
                                 line_search_evaluations=s.line_search_evaluations + evaluations),
            operand=None)

    return jax.lax.while_loop(condition, step, state)


def stiefel_rcg(rho, d_rho, Q0, maxiter=2000, grad_tol=1e-9,
                alpha0=0.2, armijo_c1=1e-4, backtrack=0.5,
                print_every=200, verbose=True, *, max_linesearch=40):
    """Paper Sec. III D / Appendix D 3: RCG with PR+ and stopping diagnostics.

    The historical input Q0 denotes Z_0; output Q is Z and output U is Z^*.
    """
    Q0 = np.asarray(Q0)
    if Q0.ndim != 2:
        raise ValueError("Q0 must be a matrix of shape (mout, D).")
    rho, d_rho = _validate_model(rho, d_rho, Q0.shape[0])
    if Q0.shape[1] != rho.shape[0] or not np.isfinite(Q0).all():
        raise ValueError("Q0 has an invalid shape or nonfinite entries.")
    if np.linalg.norm(Q0.conj().T @ Q0 - np.eye(Q0.shape[1])) > 1e-8:
        raise ValueError("Q0 must satisfy Q0.conj().T @ Q0 = I.")
    for name, value, minimum in [("maxiter", maxiter, 0), ("print_every", print_every, 1),
                                 ("max_linesearch", max_linesearch, 1)]:
        _integer(name, value, minimum)
    _positive("grad_tol", grad_tol)
    _positive("alpha0", alpha0)
    if not 0 < armijo_c1 < 1 or not 0 < backtrack < 1:
        raise ValueError("armijo_c1 and backtrack must lie strictly between 0 and 1.")
    Q = jnp.asarray(Q0, dtype=jnp.complex128)
    value, grad = value_and_rgrad_Q(Q, rho, d_rho)
    finite = bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(grad)))
    gradnorm = float(jnp.sqrt(inner(grad, grad)))
    initial_status = 4 if not finite else (1 if gradnorm < grad_tol else 0)
    state = _CGState(Q, value, grad, -grad, jnp.array(0), jnp.array(initial_status), jnp.array(0))
    chunk = print_every if verbose else max(1, maxiter)
    for stop_at in range(chunk, maxiter + chunk, chunk):
        if int(state.status) != 0:
            break
        state = _rcg_chunk(state, min(stop_at, maxiter), rho, d_rho,
                           grad_tol, alpha0, armijo_c1, backtrack, max_linesearch)
        if verbose:
            print(f"RCG {int(state.iteration):6d}: f={float(state.value):.12f}, "
                  f"|grad|={float(jnp.sqrt(inner(state.grad, state.grad))):.3e}")
    status_code = int(state.status) or 2
    status = {1: "converged", 2: "maxiter", 3: "line_search_failed", 4: "nonfinite"}[status_code]
    Q = np.asarray(state.Q)
    F = np.asarray(classical_fisher_stiefel(state.Q, rho, d_rho))
    eigenvalues = np.linalg.eigvalsh(F) if np.isfinite(F).all() else np.full(len(d_rho), np.nan)
    result = {
        "minimum": float(state.value), "Q": Q, "U": Q.conj().T,
        "gradient_norm": float(jnp.sqrt(inner(state.grad, state.grad))),
        "iterations": int(state.iteration), "status": status,
        "converged": status == "converged", "fisher": F,
        "completeness_error": float(np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1]))),
        "fisher_min_eigenvalue": float(eigenvalues.min()),
        "line_search_evaluations": int(state.line_search_evaluations),
    }
    if verbose:
        print(f"RCG status: {status}")
    return result


# %% public_api
def minimizeQ_stiefel(rho, d_rho, mout, seed=0, adam_steps=30000,
                      adam_lr=0.005, adam_chunk=5000, rcg_steps=2000,
                      grad_tol=1e-9, verbose=True):
    """Original call signature, with mout=s and returned Q=Z in paper notation.

    Inputs rho and d_rho have shapes (n,n) and (d,n,n). Returned fisher is
    F(Z) in those coordinates, and U=Z^*. The field minimum is the final
    local unit-weight objective f_I(Z), not a certified global minimum.
    With transformed inputs this is Tr(tilde F^-1); any experiment-specific
    reporting factor is applied separately. Result keys are retained for
    compatibility. Timings include compilation when it is needed.
    """
    # Validate both phases before starting a potentially long computation.
    _integer("rcg_steps", rcg_steps)
    _positive("grad_tol", grad_tol)
    start = time.perf_counter()
    Q = stiefel_adam(rho, d_rho, mout, seed, adam_steps, adam_lr, adam_chunk, verbose)
    adam_done = time.perf_counter()
    result = stiefel_rcg(rho, d_rho, Q, rcg_steps, grad_tol, verbose=verbose)
    result.update(seed=seed, adam_steps=adam_steps,
                  adam_seconds=adam_done - start,
                  rcg_seconds=time.perf_counter() - adam_done)
    result["elapsed_seconds"] = result["adam_seconds"] + result["rcg_seconds"]
    return result


def optimize_povm(rho, d_rho, mout, seed=0, *, weight=None, **optimizer_options):
    """Thin weighted wrapper using the coordinate reduction of paper Sec. III G.

    Paper n,d,s correspond to rho.shape[0], len(d_rho), and mout. Returned
    Q is the paper's Stiefel matrix Z; its columns in U=Z^* are u_x.

    The legacy fields ``minimum`` and ``fisher`` retain their meaning in the
    supplied-to-optimizer coordinates: these are the whitened coordinates
    when ``weight`` is supplied, so ``fisher`` is tilde F. ``risk`` equals
    the same numerical value c_num=Tr(weight F_original^-1).
    ``fisher_original`` restores F in the original coordinates, and
    ``whitening_transform`` stores T=W^-1/2, with theta-theta_0=T eta.

    The returned ``rho``, ``D`` (paper tilde D_i), and ``B`` (the explicitly
    symmetrized numerical inverse of tilde F) are the identity-weight
    fixed-B dual handoff. ``D_original``
    and ``weight`` preserve the requested model for input-error certification.
    They are numerical data, not an exact interpretation of the ideal model.

    ``risk_is_certified_upper`` is always False. A finite endpoint can have
    ``converged=False``; the original stopping status is preserved. No dual
    solve or lower-bound certificate is generated by this function.
    """
    from .models import whiten_derivatives, fisher_to_original
    original = np.asarray(d_rho, dtype=np.complex128)
    if weight is None:
        transformed = original
        p = len(original) if original.ndim else 0
        T = np.eye(p)
        W = np.eye(p)
    else:
        transformed, T = whiten_derivatives(original, weight)
        raw_weight = np.asarray(weight).real.astype(np.float64)
        W = (raw_weight + raw_weight.T) / 2
    result = minimizeQ_stiefel(rho, transformed, mout, seed=seed, **optimizer_options)
    fisher = result["fisher"]
    finite = bool(np.isfinite(result["minimum"]) and np.isfinite(fisher).all())
    inverse = np.linalg.inv(fisher) if finite else None
    B = (inverse + inverse.T) / 2 if finite else None
    result.update(
        risk=result["minimum"], risk_kind="numerical_local_candidate",
        risk_is_certified_upper=False, measurement_outcomes=int(result["Q"].shape[0]),
        rho=np.asarray(rho, dtype=np.complex128), D=transformed, D_original=original,
        weight=W, whitening_transform=T, fisher_coordinates="weight_whitened",
        fisher_whitened=fisher,
        fisher_original=fisher_to_original(fisher, T) if finite else None,
        B=B, B_convention="symmetric part of the numerical inverse whitened Fisher matrix",
        dual_handoff_coordinates="rho,D,B use the whitened identity-weight objective",
    )
    return result


def optimize_multistart(rho, d_rho, mout, seeds=range(10), *, weight=None, **optimizer_options):
    """Sequential starts, selecting the smallest finite *final* risk.

    Returns ``best``, ``runs`` (all full results), ``best_index``, and ``seeds``.
    No outcome pruning, averaging, continuation, or within-run best-iterate
    selection is performed. Equal risks select the first supplied seed.
    Finite nonconverged endpoints remain eligible and retain their flags.
    Numerical Adam failures are recorded with an error and infinite risk;
    invalid arguments propagate normally. If every start fails to produce a
    finite objective, a FloatingPointError is raised.
    """
    if "seed" in optimizer_options:
        raise ValueError("Use seeds=(...) instead of seed=... in optimize_multistart.")
    try:
        seeds = tuple(seeds)
    except TypeError as error:
        raise ValueError("seeds must be a nonempty iterable of nonnegative integers.") from error
    if not seeds:
        raise ValueError("seeds must be nonempty.")
    for seed in seeds:
        _integer("seed", seed)
    runs = []
    for seed in seeds:
        try:
            result = optimize_povm(rho, d_rho, mout, seed=seed, weight=weight, **optimizer_options)
        except FloatingPointError as error:
            result = dict(seed=int(seed), minimum=float("inf"), risk=float("inf"),
                          status="adam_nonfinite", converged=False, error=str(error),
                          risk_kind="numerical_failed_start", risk_is_certified_upper=False)
        runs.append(result)
    finite = [i for i, result in enumerate(runs) if np.isfinite(result["risk"])]
    if not finite:
        raise FloatingPointError("No start produced a finite risk; inspect the model, seeds, or optimizer settings.")
    best_index = min(finite, key=lambda i: runs[i]["risk"])
    return dict(best=runs[best_index], runs=runs, best_index=best_index,
                seeds=tuple(int(seed) for seed in seeds),
                selection="smallest finite final risk; first supplied seed breaks ties",
                risk_is_certified_upper=False)


def sweep_r(r_values, ncopy, mout, seeds=(0,), **optimizer_options):
    """Legacy sequential sweep in single-copy-SLD-normalized coordinates.

    The legacy Tr(F^-1) column means Tr(tilde F^-1) in paper Eq. (7.3),
    and N*Tr(F^-1) is c_num. Neither is an exact upper certificate; no dual
    solve is performed. ``ncopy`` is N and ``mout`` is the outcome count s.
    """
    from .models import qbit_model_n
    rows = []
    options = {"verbose": False, **optimizer_options}
    if "seed" in options:
        raise ValueError("Use seeds=(...) instead of seed=... in sweep_r.")
    for r in r_values:
        rho, d_rho = qbit_model_n(float(r), ncopy)
        for seed in seeds:
            result = minimizeQ_stiefel(rho, d_rho, mout, seed=seed, **options)
            rows.append({"r": float(r), "ncopy": ncopy, "mout": mout, "seed": seed,
                         "Tr(F^-1)": result["minimum"],
                         "N*Tr(F^-1)": ncopy * result["minimum"],
                         **{key: result[key] for key in ["status", "gradient_norm", "iterations",
                                                        "completeness_error", "elapsed_seconds"]}})
    return rows


__all__ = ["minimizeQ_stiefel", "optimize_povm", "optimize_multistart", "sweep_r",
           "classical_fisher_stiefel", "stiefel_adam", "stiefel_rcg",
           "make_stiefel_objective", "value_Q", "value_and_rgrad_Q",
           "stiefel_project", "stiefel_retract", "random_stiefel",
           "pack_complex", "unpack_complex", "inner"]


