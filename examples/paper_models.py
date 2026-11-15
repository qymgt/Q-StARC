"""Numerical inputs for the selected paper examples; no search archive needed.

``build_model`` preserves the *historical optimizer coordinates*. Its ``D``
is passed directly to ``minimizeQ_stiefel`` without a second weight whitening.
In the paper's notation, these are transformed-coordinate derivatives
tilde D_i; the optimizer's Fisher matrix is tilde F. Hilbert dimension is n,
parameter count is d, outcome count is s, and the frame Z is saved as Q.
``D0,W,T`` describe the historical transform D_i=sum_j D0_j T[j,i].
This T is an additional historical transform: in experiments 1 and 2,
D0 is already in SLD coordinates and T=I. In experiment 3 diagonal, D0
is in SLD coordinates while paper_D0 is in Bloch coordinates.
``paper_D0,paper_W`` describe the reported coordinates/weight; multiply the
optimizer objective by ``risk_scale`` only after optimization to obtain the
floating-point c_num (saved as paper_risk). In particular, experiment 3
diagonal uses the original SLD-whitened inputs and a 153/223
reporting factor, rather than running with freshly Bloch-whitened tangents.

The returned arrays are binary64 approximations, not interval enclosures of
the rational parameters or sqrt(2)/sqrt(6) basis. Ideal-input correction is
outside these examples. NumPy is sufficient to import this module; building
legacy SLD qubits also uses the package's JAX model constructor.
"""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np


def load_settings():
    """Read the adjacent, portable paper-settings file as a fresh dictionary."""
    return json.loads(Path(__file__).with_name("paper_settings.json").read_text())


def _fraction(value):
    if isinstance(value, bool):
        raise ValueError("A radius must be a real number or rational string.")
    return Fraction(str(value))


def _diagonal(values):
    return np.diag([float(Fraction(v)) for v in values])


def _bloch_two_copy(r):
    # Original experiment-3 model() operation order, without a JAX conversion.
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                      [[1, 0], [0, -1]]], complex)
    rho1 = (np.eye(2) + r * pauli[2]) / 2
    rho = np.kron(rho1, rho1)
    D0 = np.array([np.kron(s / 2, rho1) + np.kron(rho1, s / 2) for s in pauli])
    return rho, D0


def _qutrit_basis():
    """Adapted GMM order: four 0-tail, three tail, and one radial tangent."""
    out = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        x = np.zeros((3, 3), complex)
        x[i, j] = x[j, i] = 1 / np.sqrt(2)
        out.append(x)
        y = np.zeros((3, 3), complex)
        y[i, j], y[j, i] = -1j / np.sqrt(2), 1j / np.sqrt(2)
        out.append(y)
    out.extend([np.diag([0, 1, -1]) / np.sqrt(2),
                np.diag([2, -1, -1]) / np.sqrt(6)])
    return np.array(out)


def _historical_whitening(D0, W, *, symmetrize):
    # Keep the original division operation and optional symmetrization: the
    # general public helper uses a mathematically equivalent operation order.
    eigenvalues, U = np.linalg.eigh(W)
    if np.any(eigenvalues <= 0):
        raise ValueError("The example weight must be positive definite.")
    T = (U / np.sqrt(eigenvalues)) @ U.T
    if symmetrize:
        T = (T + T.T) / 2
    D = np.einsum("ji,jab->iab", T, D0)
    if symmetrize:
        D = (D + D.conj().transpose(0, 2, 1)) / 2
    return D, T


def build_model(example_id, *, ncopy=None, r=None):
    """Build one configured point and return arrays plus reporting metadata.

    Experiment 1 requires ``ncopy`` and ``r`` from the saved grid. Experiment
    2 requires ``ncopy`` and fixes r=1/2. Other examples have fixed inputs.
    The return keys ``rho,D,B`` (B is produced later from the optimizer's F)
    form an identity-weight dual input, whose risk scale is recorded here.
    That generic binary64 input alone is not a paper ideal-model certificate.
    """
    settings = load_settings()
    if example_id not in settings["examples"]:
        raise ValueError(f"Unknown example {example_id!r}.")
    spec = settings["examples"][example_id]
    if "ncopy_values" in spec:
        if isinstance(ncopy, bool) or not isinstance(ncopy, (int, np.integer)) or ncopy not in spec["ncopy_values"]:
            raise ValueError("Choose --ncopy from 2, 3, 4, 5, 6.")
        ncopy = int(ncopy)
    else:
        fixed_n = spec.get("ncopy", 1)
        if ncopy is not None and (isinstance(ncopy, bool) or not isinstance(ncopy, (int, np.integer)) or ncopy != fixed_n):
            raise ValueError(f"This example fixes ncopy={fixed_n}.")
        ncopy = fixed_n
    if "r_values_exact" in spec:
        if r is None or _fraction(r) not in map(Fraction, spec["r_values_exact"]):
            raise ValueError("Experiment 1 requires --r in 0, 0.1, ..., 0.9.")
        radius = _fraction(r)
    elif "r_exact" in spec:
        radius = Fraction(spec["r_exact"])
        if r is not None and _fraction(r) != radius:
            raise ValueError(f"This example fixes r={radius}.")
    else:
        if r is not None:
            raise ValueError("The qutrit examples specify a spectrum, not a qubit radius.")
        radius = None

    kind = spec["model"]
    if kind in ("iid_qubit_sld", "two_qubit_sld_weighted"):
        from qstarc.models import qbit_model_n
        rho, D0 = map(np.asarray, qbit_model_n(float(radius), ncopy))
        if kind == "iid_qubit_sld":
            W, T = np.eye(3), np.eye(3)
            D = D0
            paper_D0, paper_W, scale = D0, ncopy * np.eye(3), Fraction(ncopy)
        else:
            W = np.diag([0.1, 1, 1])
            W = W * 3 / np.trace(W)  # original stored binary64 weight
            D, T = _historical_whitening(D0, W, symmetrize=False)
            _, paper_D0 = _bloch_two_copy(float(radius))
            paper_W = _diagonal(spec["paper_weight_diagonal_exact"])
            scale = Fraction(spec["paper_risk_scale_exact"])
    elif kind == "two_qubit_bloch_weighted":
        rho, D0 = _bloch_two_copy(float(radius))
        W = np.array([[float(Fraction(v)) for v in row] for row in spec["paper_weight_exact"]])
        D, T = _historical_whitening(D0, W, symmetrize=True)
        paper_D0, paper_W, scale = D0, W, Fraction(1)
    elif kind == "qutrit_full8_weighted":
        rho = np.diag([float(Fraction(v)) for v in spec["spectrum_exact"]]).astype(complex)
        D0 = _qutrit_basis()
        W = _diagonal(spec["paper_weight_diagonal_exact"])
        D, T = _historical_whitening(D0, W, symmetrize=True)
        paper_D0, paper_W, scale = D0, W, Fraction(1)
    else:
        raise ValueError(f"Unsupported model kind {kind!r}.")
    mout = spec["M_by_N"][str(ncopy)] if "M_by_N" in spec else spec["M"]
    return dict(rho=rho, D=D, D0=D0, W=W, T=T, paper_D0=paper_D0,
                paper_W=paper_W, risk_scale=float(scale), metadata=dict(
                    example_id=example_id, ncopy=ncopy,
                    r_exact=str(radius) if radius is not None else None,
                    spectrum_exact=spec.get("spectrum_exact"),
                    qstarc_support_size=mout, risk_scale_exact=str(scale),
                    optimizer_objective="Tr(F_optimizer^-1)",
                    paper_risk=spec["paper_risk"], paper_coordinates=spec["paper_coordinates"],
                    derivative_convention="D_i=sum_j D0_j T[j,i]; D is passed directly, with no additional whitening",
                    dual_input_scope="rho,D and inverse F use identity weight in optimizer coordinates; apply risk_scale afterward",
                    exact_input_certification=False, spec=spec))


__all__ = ["load_settings", "build_model"]
