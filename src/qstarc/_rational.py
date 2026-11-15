"""Internal exact arithmetic and input checks; no solver dependencies.

The implementation's p, n, rank, and level correspond to the paper's
parameter dimension d, Hilbert-space dimension n, support rank r, and SOS
level m. The local alias F below denotes Fraction, not Fisher information.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
from math import inf, isqrt, nextafter

import numpy as np


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
    return value


def validate_model(rho, derivatives, B):
    """Return binary64 copies; do not silently symmetrize or normalize inputs."""
    rho = np.array(rho, dtype=np.complex128, copy=True)
    D = np.array(derivatives, dtype=np.complex128, copy=True)
    raw = np.asarray(B)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not len(rho):
        raise ValueError("rho must be a nonempty square matrix")
    n = len(rho)
    if D.ndim != 3 or D.shape[1:] != (n, n) or not len(D):
        raise ValueError("derivatives must have shape (p,n,n), with p positive")
    p = len(D)
    if raw.shape != (p, p) or np.any(np.imag(raw)):
        raise ValueError("B must be a real p by p matrix")
    B = np.array(np.real(raw), dtype=np.float64, copy=True)
    if not all(np.all(np.isfinite(x)) for x in (rho, D, B)):
        raise ValueError("rho, derivatives, and B must be finite")
    if np.any(rho.imag) or np.any(rho != np.diag(np.diag(rho))):
        raise ValueError("rho must be exactly real and diagonal")
    diagonal = rho.diagonal().real
    if np.any(diagonal < 0) or not np.any(diagonal > 0):
        raise ValueError("rho must be nonzero and positive semidefinite")
    rank = int(np.count_nonzero(diagonal))
    if np.any(diagonal[:rank] <= 0) or np.any(diagonal[rank:]):
        raise ValueError("Place the positive support of rho before its zero kernel")
    if np.any(D != D.conj().transpose(0, 2, 1)):
        raise ValueError("derivatives must be exactly Hermitian")
    if np.any(D[:, rank:, rank:]):
        raise ValueError("The derivative kernel-kernel blocks must be zero")
    return rho, D, B, rank


@lru_cache(None)
def monomials(p, degree):
    def exact(total, length):
        if length == 1:
            yield (total,)
        else:
            for first in range(total, -1, -1):
                for rest in exact(total-first, length-1):
                    yield (first,) + rest
    return tuple(a for total in range(degree+1) for a in exact(total, p))


def full_basis(p, n, rank, level):
    """Return the columns of V_m, using the saved component-major ordering.

    A tuple (alpha_1, ..., alpha_d, a) represents xi**alpha times e_a, with
    zero-based component a. Support components have degree at most m+1;
    kernel components have degree at most m. Eq. (5.5) uses the same basis
    with monomials before components within each support/kernel block.
    The two column orderings differ only by a permutation, also applied
    to Gram rows and columns. Keep the saved order for certificate replay.
    """
    return tuple(a + (i,) for i in range(n)
                 for a in monomials(p, level + int(i < rank)))


def weight_coefficients(p, level):
    out = {(0,) * p: 1}
    for _ in range(level):
        new = {}
        for a, c in out.items():
            new[a] = new.get(a, 0) + c
            for i in range(p):
                b = list(a)
                b[i] += 2
                b = tuple(b)
                new[b] = new.get(b, 0) + c
        out = new
    return out


def arrays(n):
    return np.zeros((n, n), dtype=object), np.zeros((n, n), dtype=object)


def fraction_matrix(x):
    re, im = arrays(len(x))
    for i, j in np.ndindex(x.shape):
        re[i, j] = F(float(x[i, j].real))
        im[i, j] = F(float(x[i, j].imag))
    return re, im


def exact_model(rho, D, B):
    """Interpret rho,D,B as dyadics and form A_i = sum_j B_ij D_j exactly.

    This is the B orientation in Eq. (4.6); no transpose is taken.
    """
    n, p = len(rho), len(D)
    bf = [[F(float(B[i, j])) for j in range(p)] for i in range(p)]
    df = [fraction_matrix(d) for d in D]
    af = []
    for i in range(p):
        ar, ai = arrays(n)
        for j in range(p):
            ar += bf[i][j] * df[j][0]
            ai += bf[i][j] * df[j][1]
        af.append((ar, ai))
    return bf, df, af


def entry(poly, key, n):
    if key not in poly:
        poly[key] = arrays(n)
    return poly[key]


def add_matrix(poly, key, matrix, n, factor=1):
    re, im = entry(poly, key, n)
    re += factor * matrix[0]
    im += factor * matrix[1]


def homogenize(poly, p, level, rank, n):
    """Mixed degrees are 2m+2, 2m+1, 2m on support/cross/kernel entries."""
    out = {}
    for alpha, (re, im) in poly.items():
        for i, j in np.ndindex((n, n)):
            if not (re[i, j] or im[i, j]):
                continue
            power = 2*level + int(i < rank) + int(j < rank) - sum(alpha)
            if power < 0:
                raise ValueError("Polynomial exceeds the specified column degrees")
            rr, ii = entry(out, (power,) + alpha, n)
            rr[i, j] += re[i, j]
            ii[i, j] += im[i, j]
    return out


@lru_cache(None)
def monomial_bound(alpha, bits=40):
    """Outward rational bound on |z**alpha| for real ||z||=1."""
    degree = sum(alpha)
    if not degree:
        return F(1)
    square = F(1)
    for exponent in alpha:
        if exponent:
            square *= F(exponent, degree) ** exponent
    scaled = square * (1 << (2*bits))
    numerator = isqrt(scaled.numerator // scaled.denominator)
    if F(numerator*numerator) < scaled:
        numerator += 1
    return F(numerator, 1 << bits)


def upward(x):
    try:
        value = float(x)
    except OverflowError:
        return inf if x > 0 else -np.finfo(float).max
    return nextafter(value, inf) if F(value) < x else value


def downward(x):
    try:
        value = float(x)
    except OverflowError:
        return np.finfo(float).max if x > 0 else -inf
    return nextafter(value, -inf) if F(value) > x else value


def validate_basis_blocks(data, p, n, rank, level):
    gram_ids = sorted(int(k[2:]) for k in data if k.startswith('G_') and k[2:].isdigit())
    if not gram_ids or gram_ids != list(range(len(gram_ids))):
        raise ValueError("Gram blocks must be consecutively numbered from zero")
    if {k for k in data if k.startswith('G_')} != {f'G_{i}' for i in gram_ids}:
        raise ValueError("Unexpected Gram block name")
    if {k for k in data if k.startswith('basis_')} != {f'basis_{i}' for i in gram_ids}:
        raise ValueError("Each Gram block needs exactly one basis")
    bases, seen = [], set()
    expected = set(full_basis(p, n, rank, level))
    for idx in gram_ids:
        raw = np.asarray(data[f'basis_{idx}'])
        if raw.ndim != 2 or raw.shape[1] != p+1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("A basis must be a nonempty integer array with p+1 columns")
        basis = [tuple(map(int, row)) for row in raw]
        if any(row not in expected or row in seen for row in basis) or len(set(basis)) != len(basis):
            raise ValueError("Invalid or duplicate monomial-column basis row")
        seen.update(basis)
        gram = np.asarray(data[f'G_{idx}'])
        if gram.shape != (len(basis), len(basis)) or not np.all(np.isfinite(gram)):
            raise ValueError("Gram block shape or finite-value check failed")
        bases.append(np.asarray(basis, dtype=np.int64))
    if seen != expected:
        raise ValueError("Gram blocks must partition the full denominator basis")
    return bases
