"""Fixed-coefficient denominator-SOS hierarchy, with exact sign symmetry.

This module implements the fixed-B calculation in Sec. V of the paper.
In its notation, n = dim(H), d is the parameter dimension,
rho = rho_{theta_0}, D_j = partial_j rho_{theta_0}, and m is the denominator
level. Internal ``p`` means d, and the API keyword ``level`` means m.
For A_i = sum_j B_ij D_j and xi in R^d, the constraint is

    w(xi)^m Q_{B,X}(xi) = V_m(xi) G V_m(xi)^dagger,
    w(xi) = 1 + ||xi||^2,
    Q_{B,X}(xi) = X + ||xi||^2 rho - 2 sum_i xi_i A_i.

Here B is fixed, X is Hermitian, and G is a positive-semidefinite Gram
matrix. The objective approximates d_m(B) = inf tr(X). This API uses
identity weight, giving c_m(I; B) = 2 Tr(B) - d_m(B); it does not implement
the joint optimization c_m(W) of Sec. VI. No transpose of B is required.

This module solves a numerical SDP; its output is *not* a certified bound.
Call :func:`qstarc.certification.certify_risk_bound` on the saved solution for a
rational residual repair. The cost is Tr(F^{-1}) in the supplied coordinates;
whitening a weight is a separate operation and is not certified here.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import warnings

import numpy as np

from ._rational import (exact_model, full_basis, integer, validate_model,
                        weight_coefficients)

SOLUTION_FORMAT = 'qstarc-sos-proposal-v1'


def _characters(rho, D, B, enabled, states, scores):
    if not enabled and states is None and scores is None:
        return None, None
    n, p = len(rho), len(D)
    if states is None:
        states = [0] + [1 << j for j in range(n-1)]
    if len(states) != n:
        raise ValueError('state_characters must have length n')
    states = tuple(integer(x, 'state character') for x in states)
    _, _, A = exact_model(rho, D, B)
    inferred = []
    for re, im in A:
        labels = {states[i] ^ states[j] for i in range(n) for j in range(n)
                  if re[i, j] or im[i, j]}
        if len(labels) > 1:
            raise ValueError('Fixed B breaks the requested sign symmetry; do not split this input')
        inferred.append(next(iter(labels), 0))
    if scores is None:
        scores = inferred
    if len(scores) != p:
        raise ValueError('score_characters must have length p')
    scores = tuple(integer(x, 'score character') for x in scores)
    for score, (re, im) in zip(scores, A):
        for i, j in np.ndindex((n, n)):
            if (re[i, j] or im[i, j]) and (states[i] ^ states[j]) != score:
                raise ValueError('Exact binary64 A=B D does not obey the requested sign characters')
    return states, scores


def _basis_blocks(p, n, rank, level, states, scores):
    """Partition stored V_m columns by sign character, preserving block order.

    These Gram symmetry blocks are distinct from sufficient-algebra blocks
    of the physical model. ``p`` is the paper's parameter dimension d.
    """
    basis = full_basis(p, n, rank, level)
    if states is None:
        return [basis]
    blocks = {}
    for row in basis:
        character = states[row[-1]]
        for exponent, score in zip(row[:-1], scores):
            if exponent % 2:
                character ^= score
        blocks.setdefault(character, []).append(row)
    return [tuple(blocks[k]) for k in sorted(blocks)]


def _hermitian_basis(n, states):
    out = [np.diag(np.eye(n)[i]).astype(complex) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if states is not None and states[i] != states[j]:
                continue
            re = np.zeros((n, n), complex)
            re[i, j] = re[j, i] = 1
            im = np.zeros((n, n), complex)
            im[i, j] = 1j
            im[j, i] = -1j
            out.extend((re, im))
    return np.asarray(out)


def _polynomial(rho, A, X, level, constant=True):
    p = len(A)
    base = {(0,)*p: X}
    if constant:
        for i in range(p):
            key = [0]*p
            key[i] = 1
            base[tuple(key)] = -2*A[i]
            key[i] = 2
            base[tuple(key)] = rho
    out = {}
    for alpha, coefficient in weight_coefficients(p, level).items():
        for beta, matrix in base.items():
            key = tuple(a+b for a, b in zip(alpha, beta))
            out[key] = out.get(key, 0) + coefficient*matrix
    return out


def _build(rho, A, rank, level, weights, states, scores):
    import cvxpy as cp
    import scipy.sparse as sp

    start = time.perf_counter()
    n, p = len(rho), len(A)
    hb = _hermitian_basis(n, states)
    zero = np.zeros((n, n), complex)
    constant = _polynomial(rho, A, zero, level)
    x_polys = [_polynomial(rho, A, h, level, False) for h in hb]
    bases = _basis_blocks(p, n, rank, level, states, scores)
    maps, keys = [], set()
    for basis in bases:
        entries = []
        size = len(basis)
        for a, ea in enumerate(basis):
            for b, eb in enumerate(basis):
                i, j = ea[-1], eb[-1]
                if i > j:
                    continue
                key = (tuple(x+y for x, y in zip(ea[:-1], eb[:-1])), i, j)
                keys.add(key)
                entries.append((key, a+b*size))
        maps.append(entries)
    for poly in [constant] + x_polys:
        for exponent, matrix in poly.items():
            for i in range(n):
                for j in range(i, n):
                    if matrix[i, j] != 0:
                        keys.add((exponent, i, j))
    keys = sorted(keys)
    position = {key: i for i, key in enumerate(keys)}
    c = np.zeros(len(keys), complex)
    rows, columns, values = [], [], []
    for k, (exponent, i, j) in enumerate(keys):
        c[k] = constant.get(exponent, zero)[i, j]
        for a, poly in enumerate(x_polys):
            value = poly.get(exponent, zero)[i, j]
            if value != 0:
                rows.append(k)
                columns.append(a)
                values.append(value)
    target = sp.csc_matrix((values, (rows, columns)), shape=(len(keys), len(hb)))
    x = cp.Variable(len(hb), name='X_coordinates')
    grams = [cp.Variable((len(b), len(b)), hermitian=True, name=f'G_{i}')
             for i, b in enumerate(bases)]
    expression = 0
    for basis, gram, entries in zip(bases, grams, maps):
        matrix = sp.csc_matrix((np.ones(len(entries)),
            ([position[key] for key, _ in entries], [col for _, col in entries])),
            shape=(len(keys), len(basis)**2))
        expression = expression + matrix @ cp.vec(gram, order='F')
    residual = expression - (target @ x + c)
    problem = cp.Problem(cp.Minimize(weights @ x[:n]),
                         [g >> 0 for g in grams] + [residual == 0])
    return dict(problem=problem, x=x, hb=hb, grams=grams, bases=bases,
                residual=residual, coefficient_rows=len(keys),
                build_seconds=time.perf_counter()-start)


def save_sos_proposal(result, path):
    """Save a numerical fixed-B SOS proposal to an NPZ path and return Path.

    Inputs and Gram/basis arrays are included for certification. The file
    contains no absolute output path. This function does not certify the
    numerical proposal and may overwrite an existing file at ``path``.
    """
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('The solution path must end in .npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    array_keys = {'rho', 'D', 'B', 'X', 'grams', 'bases', 'solution_path'}
    metadata = {k: v for k, v in result.items() if k not in array_keys}
    arrays = dict(format_version=SOLUTION_FORMAT, method='denominator',
                  rho=result['rho'], D=result['D'], B=result['B'], X=result['X'],
                  r=result['rank'], level=result['level'],
                  metadata_json=json.dumps(metadata))
    arrays.update({f'G_{i}': g for i, g in enumerate(result['grams'])})
    arrays.update({f'basis_{i}': np.asarray(b, dtype=np.int64)
                   for i, b in enumerate(result['bases'])})
    np.savez_compressed(path, **arrays)
    return path


def solve_fixed_b_sos(rho, derivatives, B, *, level=1, output_path=None,
               tolerance=1e-8, max_iterations=250, time_limit=90.0,
               sign_symmetry=False, state_characters=None, score_characters=None,
               precondition=True, verbose=False):
    """Solve the fixed-B, unit-weight relaxation d_m(B) with CLARABEL.

    Parameters
    ----------
    rho : (n,n) array
        Exactly real diagonal binary64 matrix with nonnegative diagonal.
        Positive support entries must precede any zero entries.
    derivatives : (d,n,n) array
        The matrices D_j in the supplied coordinates, exactly Hermitian.
        Kernel-kernel blocks must vanish if rho
        is singular. Inputs are copied; no normalization or repair is hidden.
    B : (d,d) real array
        Fixed support coefficient: A_i = sum_j B_ij D_j. Usually B is the
        candidate inverse Fisher matrix in these unit-weight coordinates.
        It need not be positive definite or symmetric. For general weights,
        first transform the derivatives as in Sec. III G; passing the
        original-coordinate coefficient W^(1/2) F^(-1) alone does not make
        this API a weighted-risk calculation.
    level : int
        The nonnegative denominator exponent m; default 1. The polynomial is
        (1+||xi||^2)^m [X+||xi||^2 rho-2 sum_i xi_i (B D)_i].
        xi is an auxiliary real d-vector, distinct from the outcome count s.
    sign_symmetry : bool
        Infer the canonical diagonal-sign characters if True. For more
        general Z2^k actions, supply integer bit-mask state_characters and
        score_characters. Fixed B is never projected or modified. Requested
        covariance is checked with exact rational contractions of B and D.
    output_path : path-like or None
        Optional .npz destination, overwritten if it exists. All model and
        proof-proposal arrays are included for subsequent certification.
    tolerance : float
        Positive CLARABEL primal, dual, and gap tolerances.
    max_iterations : int
        Positive CLARABEL iteration limit.
    time_limit : float or None
        Positive solver time limit in seconds. Individual solver operations
        can finish after this limit; this is not an external process timeout.
    precondition : bool
        Congruence by the inverse support square root before the SDP. Saved
        arrays are mapped back to the supplied coordinates.
    verbose : bool
        Whether CLARABEL prints its progress.

    Returns a dictionary including arrays, numerical diagnostics, and a
    solution_path when saved. ``raw_support_upper`` is the numerical tr(X),
    and ``raw_risk_lower`` is 2 Tr(B) minus that value; neither is certified.
    ``rank`` is r = rank(rho). Saved basis tuples list the d exponents then
    the zero-based Hilbert-space component. Before optional sign partitioning,
    their component-major order is a column permutation of the paper's V_m
    in Eq. (5.5); each sign block retains the induced order.
    Apply
    certify_risk_bound to the saved file. Even successful solver termination alone
    does not establish feasibility. Rank-deficient support handling does not
    establish that the derivatives extend to a physical statistical model.
    """
    rho, D, B, rank = validate_model(rho, derivatives, B)
    level = integer(level, 'level')
    max_iterations = integer(max_iterations, 'max_iterations', 1)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('time_limit must be positive and finite, or None')
    states, scores = _characters(rho, D, B, sign_symmetry, state_characters, score_characters)
    n, p = len(rho), len(D)
    weights = np.ones(n)
    if precondition:
        weights[:rank] = rho.diagonal()[:rank].real
    root = np.sqrt(weights)
    invroot = 1/root
    A = np.einsum('ij,jab->iab', B, D)
    if states is not None:
        # _characters proved these entries zero in exact rational arithmetic.
        # Remove only cancellation roundoff in the numerical contraction.
        for j, score in enumerate(scores):
            for a, b in np.ndindex((n, n)):
                if states[a] ^ states[b] != score:
                    A[j, a, b] = 0
    work_rho = rho*invroot[:, None]*invroot[None, :]
    work_A = A*invroot[None, :, None]*invroot[None, None, :]
    if not np.all(np.isfinite(work_A)) or not np.all(np.isfinite(work_rho)):
        raise ValueError('Preconditioned coefficients overflow; rescale inputs or disable preconditioning')
    built = _build(work_rho, work_A, rank, level, weights, states, scores)
    import cvxpy as cp
    problem = built['problem']
    start = time.perf_counter()
    problem.get_problem_data(cp.CLARABEL)
    canonical = time.perf_counter()-start
    options = dict(tol_gap_abs=tolerance, tol_gap_rel=tolerance,
                   tol_feas=tolerance, max_iter=max_iterations, max_threads=1)
    if time_limit is not None:
        options['time_limit'] = float(time_limit)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        problem.solve(solver=cp.CLARABEL, warm_start=False, verbose=verbose, **options)
    elapsed = time.perf_counter()-start
    if built['x'].value is None or any(g.value is None for g in built['grams']):
        raise RuntimeError(f'CLARABEL returned no numerical proposal: {problem.status}')
    X = np.einsum('a,aij->ij', built['x'].value, built['hb'])*root[:, None]*root[None, :]
    grams = []
    for gram, basis in zip(built['grams'], built['bases']):
        columns = np.asarray(basis)[:, -1]
        grams.append(np.asarray(gram.value)*root[columns, None]*root[columns[None, :]])
    if not np.all(np.isfinite(X)) or any(not np.all(np.isfinite(g)) for g in grams):
        raise RuntimeError('Solver proposal contains nonfinite values')
    result = dict(rho=rho, D=D, B=B, X=X, grams=grams,
        bases=[np.asarray(b, dtype=np.int64) for b in built['bases']],
        rank=rank, dimension=n, parameters=p, level=level, method='denominator',
        status=problem.status, solver='CLARABEL', solver_threads=1,
        raw_support_upper=float(problem.value),
        raw_risk_lower=float(2*np.trace(B)-problem.value), certified=False,
        gram_sizes=[len(b) for b in built['bases']],
        build_seconds=built['build_seconds'], canonicalize_seconds=canonical,
        solve_seconds=elapsed, coefficient_rows=built['coefficient_rows'],
        coefficient_residual_max=float(np.max(np.abs(built['residual'].value))),
        iterations=problem.solver_stats.num_iters,
        sign_symmetry=states is not None, state_characters=states,
        score_characters=scores, preconditioned=bool(precondition),
        warnings=[str(w.message) for w in caught], solution_path=None,
        scope=('Numerical fixed-B proposal for the saved binary64 Fisher functional; '
               'no certification or ideal-model conversion yet.'))
    if output_path is not None:
        result['solution_path'] = str(Path(output_path))
        save_sos_proposal(result, output_path)
    return result
