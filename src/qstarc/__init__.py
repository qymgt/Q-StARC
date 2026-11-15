"""Q-StARC: local POVM optimization and SOS risk-bound certification.

Optional optimization and SDP dependencies are loaded only when their
functions are accessed. Certificate replay does not require JAX or CVXPY.
"""

from importlib import import_module

__version__ = "1.0.1"

_EXPORTS = {
    "optimize_povm": "optimizer",
    "optimize_multistart": "optimizer",
    "minimizeQ_stiefel": "optimizer",
    "stiefel_adam": "optimizer",
    "stiefel_rcg": "optimizer",
    "classical_fisher_stiefel": "optimizer",
    "qubit_model_bloch": "models",
    "qubit_model_sld": "models",
    "qbit_model_n": "models",
    "whiten_derivatives": "models",
    "solve_fixed_b_sos": "sos",
    "save_sos_proposal": "sos",
    "certify_risk_bound": "certification",
    "verify_risk_certificate": "certification",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
