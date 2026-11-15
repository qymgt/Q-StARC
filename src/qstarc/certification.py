"""Exact rational residual repair and solver-free certificate replay.

The guaranteed quantity is a lower bound on inf_M Tr(F(M)^(-1)), where F is
formed from the *saved binary64* rho and derivatives. No ideal-state, ideal
derivative, or ideal-weight conversion is included. If the arrays are not an
exact normalized density/tangent model, the assertion is about that algebraic
Fisher functional only. Singular F has infinite inverse cost.

The repaired support bound is valid for all POVMs. It is not claimed to be a
feasible point of the same finite SOS cone, or the exact optimum of that SDP.
In the notation of Sec. VI C, ``support_upper_exact`` is h_bar >= h_B and
``risk_lower_exact`` is c_cert_lower = 2 Tr(B) - h_bar, with W = I in the saved
coordinates. Their float counterparts are rounded upward and downward,
respectively. At m = 1 this is the type of quantity denoted c_1_lower in
Sec. VII A, but here the guarantee is only for the saved binary64 input;
the paper's additional ideal-model corrections are not included.
Eigenvalue computations choose dyadic factors only during certificate creation;
replay uses integer/rational arithmetic, hashes, and no solver or eigensystem.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from ._rational import (add_matrix, arrays, downward, entry, exact_model,
    fraction_matrix, homogenize, integer, monomial_bound, upward,
    validate_basis_blocks, validate_model, weight_coefficients)

SOLUTION_FORMAT = 'qstarc-sos-proposal-v1'
CERTIFICATE_FORMAT = 'qstarc-risk-certificate-v1'
SCOPE = (
    'Exact dyadic interpretation of saved binary64 rho,D,B. All-POVM lower '
    'bound on the associated Fisher inverse cost. No ideal-state, '
    'ideal-derivative or ideal-weight transfer is included. Exact normalization '
    'flags do not prove local physical extendibility of a rank-deficient model.'
)
PROOF = (
    'Exact PSD factor products; sphere-homogenized coefficient residual; '
    'outward monomial bounds and weighted Gershgorin; convex anchor repair. '
    'Not a certified feasible point or exact optimum of the original finite SDP.'
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar(data, name):
    if name not in data or np.asarray(data[name]).shape != ():
        raise ValueError(f'Missing or nonscalar {name}')
    return np.asarray(data[name]).item()


def _load(path):
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    if _scalar(data, 'format_version') != SOLUTION_FORMAT:
        raise ValueError('Unsupported proposal format; use the certifier version matching '
                         'this file. Do not rewrite archived artifacts.')
    if _scalar(data, 'method') != 'denominator':
        raise ValueError('Only denominator-SOS proposals are supported')
    for key in ('rho', 'D', 'B', 'X'):
        if key not in data:
            raise ValueError(f'Missing solution array {key}')
    rho, D, B, rank = validate_model(data['rho'], data['D'], data['B'])
    if integer(_scalar(data, 'r'), 'r', 1) != rank:
        raise ValueError('Saved support rank does not match rho')
    level = integer(_scalar(data, 'level'), 'level')
    X = np.asarray(data['X'], dtype=complex)
    if X.shape != rho.shape or not np.all(np.isfinite(X)):
        raise ValueError('X must be finite and have the same shape as rho')
    bases = validate_basis_blocks(data, len(D), len(rho), rank, level)
    return data, rho, D, B, X, rank, level, bases


def _integer_factor(raw, size):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] != size or not raw.shape[1]:
        raise ValueError('Proof factor has invalid dimensions')
    if raw.dtype.kind not in 'USiu':
        raise ValueError('Proof factors must contain integer strings or integers')
    out = np.empty(raw.shape, dtype=object)
    for index in np.ndindex(raw.shape):
        text = str(raw[index])
        if not re.fullmatch(r'[+-]?[0-9]+', text):
            raise ValueError('Noninteger proof factor entry')
        out[index] = int(text)
    return out


def _factor_proposals(data, bases, bits):
    proof = {'factor_bits': np.array(bits, dtype=np.int64)}
    minima = []
    for idx, basis in enumerate(bases):
        G = np.asarray(data[f'G_{idx}'], dtype=complex)
        ev, vectors = np.linalg.eigh((G+G.conj().T)/2)
        minima.append(float(ev.min()))
        L = vectors*np.sqrt(np.maximum(ev, 0))[None, :]
        with np.errstate(over='ignore', invalid='ignore'):
            rr = np.rint(np.ldexp(L.real, bits))
            ii = np.rint(np.ldexp(L.imag, bits))
        if not np.all(np.isfinite(rr)) or not np.all(np.isfinite(ii)):
            raise ValueError('Dyadic factor scaling overflowed')
        proof[f'factor_real_{idx}'] = np.vectorize(lambda x: str(int(x)), otypes=[str])(rr)
        proof[f'factor_imag_{idx}'] = np.vectorize(lambda x: str(int(x)), otypes=[str])(ii)
        proof[f'basis_{idx}'] = basis
    return proof, min(minima)


def _repaired_bound(loaded, proof, factor_bits, monomial_bits):
    data, rho, D, B, X, rank, level, bases = loaded
    n, p = len(rho), len(D)
    if integer(_scalar(proof, 'factor_bits'), 'factor_bits', 0, 256) != factor_bits:
        raise ValueError('Factor precision does not match the certificate')
    expected = {'factor_bits'} | {f'{kind}_{i}' for i in range(len(bases))
        for kind in ('factor_real', 'factor_imag', 'basis')}
    if set(proof) != expected:
        raise ValueError('Unexpected or missing proof arrays')
    bf, df, af = exact_model(rho, D, B)
    xr, xi = fraction_matrix(X)
    xf = ((xr+xr.T)/F(2), (xi-xi.T)/F(2))
    rf = fraction_matrix(rho)
    qpoly = {(0,)*p: xf}
    for j in range(p):
        key = [0]*p
        key[j] = 1
        add_matrix(qpoly, tuple(key), af[j], n, -2)
        key[j] = 2
        add_matrix(qpoly, tuple(key), rf, n)
    target = {}
    for alpha, c in weight_coefficients(p, level).items():
        for beta, matrix in qpoly.items():
            add_matrix(target, tuple(a+b for a, b in zip(alpha, beta)), matrix, n, c)
    sos = {}
    denominator = 1 << (2*factor_bits)
    for idx, basis in enumerate(bases):
        if not np.array_equal(proof[f'basis_{idx}'], basis):
            raise ValueError('Proof basis differs from the saved proposal')
        lr = _integer_factor(proof[f'factor_real_{idx}'], len(basis))
        li = _integer_factor(proof[f'factor_imag_{idx}'], len(basis))
        if lr.shape != li.shape:
            raise ValueError('Real and imaginary factor shapes differ')
        # These integer products define a PSD Gram exactly, regardless of
        # whether the floating-point solver Gram was actually PSD.
        gr = lr@lr.T + li@li.T
        gi = li@lr.T - lr@li.T
        for a, ea in enumerate(basis):
            ia = int(ea[-1])
            for b, eb in enumerate(basis):
                ib = int(eb[-1])
                key = tuple(int(x+y) for x, y in zip(ea[:-1], eb[:-1]))
                re_part, im_part = entry(sos, key, n)
                re_part[ia, ib] += int(gr[a, b])
                im_part[ia, ib] += int(gi[a, b])
    for rr, ii in sos.values():
        for i, j in np.ndindex((n, n)):
            rr[i, j] = F(int(rr[i, j]), denominator)
            ii[i, j] = F(int(ii[i, j]), denominator)
    target = homogenize(target, p, level, rank, n)
    sos = homogenize(sos, p, level, rank, n)
    row_errors = [F(0) for _ in range(n)]
    empty = arrays(n)
    for key in set(target) | set(sos):
        tr, ti = target.get(key, empty)
        sr, si = sos.get(key, empty)
        dr, di = tr-sr, ti-si
        if np.any(dr != dr.T) or np.any(di != -di.T):
            raise ArithmeticError('Residual polynomial is not exactly Hermitian')
        maximum = monomial_bound(key, monomial_bits)
        for i in range(n):
            row_errors[i] += maximum*sum(abs(dr[i, j])+abs(di[i, j]) for j in range(n))
    diagonal = [F(float(rho[i, i].real)) for i in range(rank)]
    anchor_metric = diagonal + [F(1)]*(n-rank)
    epsilon = max(error/anchor for error, anchor in zip(row_errors, anchor_metric))
    trace_x = sum(xf[0][i, i] for i in range(n))
    # anchor_metric=diag(rho_support,I_kernel). A block-diagonal X_anchor
    # satisfies H(X_anchor)>=anchor_metric/2 on the sphere. Its trace is
    # computed exactly below. This metric is not the paper's SLD J_theta_0,
    # and X_anchor is not the support block X^0 from Eq. (5.4).
    # Support block: rho/2 + 4 sum A_SS rho^-1 A_SS.
    # Kernel block: I/2 + 16 sum A_SK^* rho^-1 A_SK.
    trace_anchor = sum(diagonal)/2 + F(n-rank, 2)
    for ar, ai in af:
        for i in range(rank):
            for j in range(rank):
                trace_anchor += 4*(ar[i, j]**2+ai[i, j]**2)/diagonal[j]
            for j in range(rank, n):
                trace_anchor += 16*(ar[i, j]**2+ai[i, j]**2)/diagonal[i]
    # H(X)>=-epsilon anchor_metric; repair with X_anchor makes H nonnegative.
    support_upper = (trace_x+2*epsilon*trace_anchor)/(1+2*epsilon)
    trace_b = sum(bf[i][i] for i in range(p))
    lower = 2*trace_b-support_upper
    trace_rho = sum(diagonal)
    traces_D = [sum(re_part[i, i] for i in range(n)) for re_part, _ in df]
    exact_density = trace_rho == 1 and all(x == 0 for x in traces_D)
    return dict(risk_lower_exact=str(lower), risk_lower=downward(lower),
        support_upper_exact=str(support_upper), support_upper=upward(support_upper),
        epsilon_exact=str(epsilon), epsilon_upper=upward(epsilon),
        trace_B_exact=str(trace_b), trace_X_exact=str(trace_x),
        trace_anchor_exact=str(trace_anchor),
        repair_increase_upper=upward(support_upper-trace_x),
        trace_rho_exact=str(trace_rho), derivative_traces_exact=list(map(str, traces_D)),
        exact_density_normalization=exact_density,
        dimension=n, parameters=p, rank=rank, level=level,
        gram_sizes=[len(b) for b in bases])


def certify_risk_bound(solution_path, *, factor_bits=40, monomial_bits=40, output_dir=None):
    """Create an exact all-POVM bound for the saved binary64 input model.

    PSD factors are proposed by floating-point eigendecomposition, rounded to
    dyadic integers, then interpreted exactly. All residual coefficients and
    the repair are computed with Fraction arithmetic. For ideal parameters,
    rounded whitening matrices, or ideal weights this is not by itself a
    certificate: a separate input-error transfer is required.

    Files default to ``certificates/<stem>_certificate.json`` and
    ``certificates/<stem>_dyadic_factors.npz`` next to the solution. File
    references in the certificate are relative, permitting directory moves.

    ``solution_path`` must be an NPZ produced by solve_fixed_b_sos or
    save_sos_proposal. ``factor_bits`` and ``monomial_bits`` are integer
    dyadic precisions between 0 and 256. ``output_dir`` optionally changes the
    destination. Existing certificate/factor files at that location are
    overwritten. The returned dict contains exact Fraction strings, outward
    float bounds, normalization flags, and the two artifact paths.

    ``support_upper_exact`` stores h_bar >= h_B, and ``risk_lower_exact``
    stores c_cert_lower = 2 Tr(B) - h_bar in the paper's unit-weight case.
    ``support_upper`` rounds h_bar upward; ``risk_lower`` rounds the lower
    bound downward. These are corrected bounds, not exact values of d_m(B)
    or c_m(I; B). Display-scale conversions are separate.
    """
    start = time.perf_counter()
    path = Path(solution_path).resolve()
    input_hash = _hash(path)
    bits = integer(factor_bits, 'factor_bits', 0, 256)
    mbits = integer(monomial_bits, 'monomial_bits', 0, 256)
    loaded = _load(path)
    proof, minimum = _factor_proposals(loaded[0], loaded[-1], bits)
    values = _repaired_bound(loaded, proof, bits, mbits)
    if _hash(path) != input_hash:
        raise ValueError('Solution file changed during certification')
    destination = Path(output_dir).resolve() if output_dir is not None else path.parent/'certificates'
    destination.mkdir(parents=True, exist_ok=True)
    proof_path = destination/(path.stem+'_dyadic_factors.npz')
    certificate_path = destination/(path.stem+'_certificate.json')
    np.savez_compressed(proof_path, **proof)
    result = dict(schema=CERTIFICATE_FORMAT, **values, factor_bits=bits, monomial_bits=mbits,
        input_file=os.path.relpath(path, destination), input_sha256=input_hash,
        proof_file=proof_path.name, proof_sha256=_hash(proof_path),
        certified=True, ideal_model_transfer_included=False,
        same_finite_SOS_cone_feasibility_claimed=False,
        minimum_proposal_Gram_eigenvalue=minimum,
        certificate_seconds=time.perf_counter()-start,
        scope=SCOPE, proof=PROOF)
    certificate_path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return dict(result, certificate_path=str(certificate_path), proof_path=str(proof_path))


def verify_risk_certificate(certificate_path):
    """Recompute and compare every mathematical claim without a solver/eigensystem.

    This checks the stored factors against the saved input, not against an
    independently supplied ideal physical model. Altered hashes, factors,
    basis partitions, exact values, or normalization flags cause an error.

    ``certificate_path`` is a JSON certificate produced by certify_risk_bound.
    Return a dictionary with ``verified=True``, exact/rational bound fields,
    normalization flags, and replay time. All model/proof files are read-only.
    """
    start = time.perf_counter()
    path = Path(certificate_path).resolve()
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema') != CERTIFICATE_FORMAT:
        raise ValueError('Unsupported certificate format; use the verifier version matching '
                         'this certificate. Do not rewrite archived artifacts.')
    for key in ('input_file', 'proof_file', 'input_sha256', 'proof_sha256'):
        if not isinstance(record.get(key), str):
            raise ValueError(f'Missing certificate field {key}')
    source, proof_path = path.parent/record['input_file'], path.parent/record['proof_file']
    if _hash(source) != record['input_sha256'] or _hash(proof_path) != record['proof_sha256']:
        raise ValueError('Certificate artifact hash mismatch')
    bits = integer(record.get('factor_bits'), 'factor_bits', 0, 256)
    mbits = integer(record.get('monomial_bits'), 'monomial_bits', 0, 256)
    with np.load(proof_path, allow_pickle=False) as stored:
        proof = {k: stored[k] for k in stored.files}
    values = _repaired_bound(_load(source), proof, bits, mbits)
    for key, value in values.items():
        if key not in record or record[key] != value:
            raise ValueError(f'Certificate recomputation mismatch: {key}')
    if record.get('certified') is not True or record.get('ideal_model_transfer_included') is not False:
        raise ValueError('Certificate overstates or changes its scope')
    if record.get('same_finite_SOS_cone_feasibility_claimed') is not False:
        raise ValueError('The repair does not certify original finite-cone feasibility')
    if record.get('scope') != SCOPE or record.get('proof') != PROOF:
        raise ValueError('Certificate scope/proof text differs from this format')
    return dict(verified=True, solver_free=True, eigendecomposition_free=True,
                **values, seconds=time.perf_counter()-start,
                certificate_sha256=_hash(path), scope=record['scope'])
