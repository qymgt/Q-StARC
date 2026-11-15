"""Coordinate and weight conventions for local quantum estimation models.

Paper notation: n=dim(H), d is the parameter dimension, and N is the copy
number. The local state is rho=rho_{theta_0}, and D_i=partial_i rho_{theta_0}.
``d_rho[i]`` is this derivative in the i-th *supplied local coordinate*,
with zero-based array indexing. Bloch and SLD-normalized coordinates differ.
The legacy ``qbit_model_n`` uses single-copy SLD-normalized tangents, for which
the N-copy SLD information is N I, so its SLD-weighted risk is N Tr(F^-1).
``qubit_model_bloch`` instead uses the original three Bloch derivatives.

Weight whitening is the parameter-coordinate transformation of Sec. III G,
theta-theta_0=T eta with T=W^-1/2, not state whitening. The returned Dwhite
is paper tilde D_i=sum_j D_j T[j,i], and Fwhite is paper tilde F=T.T F T.
Thus Tr(tilde F^-1)=Tr(W F^-1), the numerical candidate cost c_num when
evaluated at the optimized measurement. Additional reporting factors in
the paper examples are applied separately.
All such identities here are evaluated numerically; no returned matrix or
risk is an interval or exact-arithmetic certificate.

NumPy weight helpers work without JAX. Legacy qubit construction imports JAX
only when called and retains its original complex128 product-rule operations.
"""
import numpy as np


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _jax_numpy():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    return jnp


def _paulis(jnp):
    return (jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128),
            jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128),
            jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128))


def qbit_rho(r):
    """Legacy state at the Bloch point (0,0,r): (I+r sigma_z)/2.

    This low-level legacy formula does not validate r. The regular model
    constructors below require a real finite r with |r|<1.
    """
    jnp = _jax_numpy()
    return (jnp.eye(2, dtype=jnp.complex128) + r * _paulis(jnp)[2]) / 2


def dqbit_rho(r, i):
    """Legacy one-copy SLD-normalized LOCAL tangent at (0,0,r).

    These are (sigma_x, sigma_y, sqrt(1-r^2) sigma_z)/2, not the
    three original Bloch-coordinate derivatives.
    """
    if i not in (0, 1, 2):
        raise ValueError("i must be 0, 1, or 2.")
    jnp = _jax_numpy()
    sx, sy, sz = _paulis(jnp)
    return (sx / 2, sy / 2, jnp.sqrt(1 - r**2) * sz / 2)[i]


def kron_all(matrices):
    """Legacy tensor-product helper, returning a complex128 JAX array."""
    jnp = _jax_numpy()
    product = jnp.array([[1]], dtype=jnp.complex128)
    for matrix in matrices:
        product = jnp.kron(product, matrix)
    return product


def qbit_model_n(r, ncopy):
    """Legacy iid qubits in the SLD-normalized coordinates of paper Eq. (7.3).

    Here ncopy=N, n=2**N, and d=3. Returns (rho, d_rho), with shapes (n,n)
    and (d,n,n). The derivatives are tilde D for T=diag(1,1,sqrt(1-r^2)),
    with single-copy SLD information I and N-copy SLD information N I.
    The optimizer minimizes Tr(tilde F^-1); c_num=N Tr(tilde F^-1) is the
    SLD-Fisher-weighted risk. No N factor is silently included in the state,
    derivatives, or optimizer.
    """
    _integer("ncopy", ncopy, 1)
    if not np.isfinite(r) or not abs(r) < 1:
        raise ValueError("These regular 3-parameter coordinates require |r| < 1.")
    jnp = _jax_numpy()
    rho1 = qbit_rho(r)
    derivatives1 = jnp.stack([dqbit_rho(r, i) for i in range(3)])
    rho = jnp.array([[1]], dtype=jnp.complex128)
    derivatives = jnp.zeros((3, 1, 1), dtype=jnp.complex128)
    for _ in range(ncopy):
        derivatives = jnp.stack([
            jnp.kron(derivatives[i], rho1) + jnp.kron(rho, derivatives1[i])
            for i in range(3)])
        rho = jnp.kron(rho, rho1)
    return rho, derivatives


def _regular_radius(r):
    value = np.asarray(r)
    if value.ndim != 0 or np.iscomplexobj(value) or not np.isfinite(value) or not abs(value) < 1:
        raise ValueError("r must be real and finite with |r| < 1.")
    return float(value)


def qubit_model_sld(r, ncopy=1):
    """Explicitly named single-copy-SLD-normalized iid qubit coordinates.

    This is the validated-name wrapper for ``qbit_model_n``; the N-copy SLD
    information is N I. Its d_rho contains tilde D from paper Eq. (7.3),
    not the original Bloch-coordinate derivatives D_i.
    """
    return qbit_model_n(_regular_radius(r), ncopy)


def qubit_model_bloch(r, ncopy=1):
    """Iid qubits in the original three Bloch coordinates at (0,0,r).

    The one-copy derivatives D_i are sigma_i/2, as in paper Eq. (7.2).
    Here ncopy=N, n=2**N, and d=3; J_{theta_0}^{(N)} is
    N diag(1,1,1/(1-r^2)). An identity weight here differs from an identity
    weight in ``qubit_model_sld`` coordinates.
    """
    r = _regular_radius(r)
    _integer("ncopy", ncopy, 1)
    jnp = _jax_numpy()
    rho1 = qbit_rho(r)
    derivatives1 = jnp.stack([matrix / 2 for matrix in _paulis(jnp)])
    rho = jnp.array([[1]], dtype=jnp.complex128)
    derivatives = jnp.zeros((3, 1, 1), dtype=jnp.complex128)
    for _ in range(ncopy):
        derivatives = jnp.stack([
            jnp.kron(derivatives[i], rho1) + jnp.kron(rho, derivatives1[i])
            for i in range(3)])
        rho = jnp.kron(rho, rho1)
    return rho, derivatives


def whiten_derivatives(d_rho, weight):
    """Return ``(Dwhite, T)``: paper tilde D and T=W^-1/2 (Sec. III G).

    ``weight`` is W and must be a finite real positive definite d-by-d matrix.
    Symmetry is checked with absolute tolerance 1e-12 and no relative
    tolerance; accepted roundoff asymmetry is interpreted through its
    symmetric part (its antisymmetric part contributes zero to the risk).
    No eigenvalue clipping, regularization, or ridge is used.

    T is the numerical principal inverse square root of that symmetric
    weight, and Dwhite_i=sum_j d_rho_j T[j,i]. The optimizer's Fwhite is
    tilde F, with Tr(tilde F^-1)=Tr(W F^-1) in the original coordinates.
    Use (rho,Dwhite,Fwhite^-1) for an identity-weight fixed-B dual handoff.
    Preserve weight, T, and the original tangents if a later certificate must
    also account for numerical coordinate-transform errors.
    """
    derivatives = np.asarray(d_rho, dtype=np.complex128)
    if derivatives.ndim != 3 or derivatives.shape[1] != derivatives.shape[2] or len(derivatives) == 0:
        raise ValueError("d_rho must have shape (nparam, dimension, dimension).")
    if not np.isfinite(derivatives).all():
        raise ValueError("d_rho must be finite.")
    if not np.allclose(derivatives, derivatives.conj().swapaxes(-1, -2), atol=1e-12, rtol=0):
        raise ValueError("Every derivative must be Hermitian.")
    raw = np.asarray(weight)
    if np.iscomplexobj(raw) and np.any(raw.imag != 0):
        raise ValueError("weight must be real symmetric, not complex Hermitian.")
    W = np.asarray(raw.real, dtype=np.float64)
    p = len(derivatives)
    if W.shape != (p, p) or not np.isfinite(W).all():
        raise ValueError("weight must be a finite nparam-by-nparam matrix.")
    if not np.allclose(W, W.T, atol=1e-12, rtol=0):
        raise ValueError("weight must be symmetric.")
    W = (W + W.T) / 2
    eigenvalues, vectors = np.linalg.eigh(W)
    if np.any(eigenvalues <= 0):
        raise ValueError("weight must be positive definite; no regularization is applied.")
    T = (vectors * (1 / np.sqrt(eigenvalues))[None, :]) @ vectors.T
    T = (T + T.T) / 2
    transformed = np.einsum("ji,jab->iab", T, derivatives)
    return transformed, T


def fisher_to_original(fisher_whitened, transform):
    """Recover paper F from tilde F=T.T F T using linear solves, without a ridge."""
    Fwhite = np.asarray(fisher_whitened, dtype=np.float64)
    T = np.asarray(transform, dtype=np.float64)
    if Fwhite.ndim != 2 or Fwhite.shape[0] != Fwhite.shape[1] or T.shape != Fwhite.shape:
        raise ValueError("fisher_whitened and transform must be square matrices of equal shape.")
    if not np.isfinite(Fwhite).all() or not np.isfinite(T).all():
        raise ValueError("fisher_whitened and transform must be finite.")
    left = np.linalg.solve(T.T, Fwhite)
    return np.linalg.solve(T.T, left.T).T


__all__ = ["qbit_rho", "dqbit_rho", "kron_all", "qbit_model_n",
           "qubit_model_sld", "qubit_model_bloch", "whiten_derivatives",
           "fisher_to_original"]
