"""Measurement search -> fixed-coefficient SOS (m=1) -> rational replay.

Run from an installed source tree. Each stage can use a separate Python
environment; the exchange files contain no pickled objects or absolute paths.
The fixed r=1/2, N=2 Bloch model is exactly dyadic and trace normalized.
Paper notation: n=4, d=3, default s=8, Z is saved as Q, and W=I_3.
c_num is numerical; c_cert_lower denotes the certified lower bound
underline c_cert for this saved-input model. The fixed-coefficient SOS
calculation uses denominator exponent m=level.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from qstarc import __version__


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def versions(names: tuple[str, ...]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def require_new(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; choose a new output directory or use --overwrite")


def optimize(args: argparse.Namespace) -> None:
    from qstarc import optimize_multistart, qubit_model_bloch

    target = args.output / "candidate.npz"
    summary = args.output / "optimization.json"
    require_new(target, args.overwrite)
    require_new(summary, args.overwrite)
    rho, derivatives = qubit_model_bloch(r=0.5, ncopy=2)
    search = optimize_multistart(
        rho, derivatives, mout=args.outcomes, seeds=range(args.starts),
        adam_steps=args.adam_steps, rcg_steps=args.rcg_steps,
        grad_tol=1e-9, verbose=False,
    )
    best = search["best"]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, rho=best["rho"], D=best["D"], B=best["B"],
        Q=best["Q"], F=best["fisher"], weight=best["weight"],
        risk=best["risk"], seed=best["seed"], status=best["status"],
        format_version="qstarc.candidate.v1",
    )
    record = {
        "method": "Q-StARC", "package_version": __version__,
        "model": "two-copy qubit at (0, 0, 1/2), standard Bloch coordinates",
        "weight": "identity", "risk_normalization": "per two-copy block",
        "outcome_slots": int(best["Q"].shape[0]),
        "seeds": list(range(args.starts)), "best_seed": int(best["seed"]),
        "adam_steps": args.adam_steps, "rcg_steps": args.rcg_steps,
        "risk_numerical": float(best["risk"]),
        "risk_is_certified_upper": False,
        "status": best["status"], "converged": bool(best["converged"]),
        "gradient_norm": float(best["gradient_norm"]),
        "completeness_error": float(best["completeness_error"]),
        "candidate_file": target.name, "python": sys.version.split()[0],
        "dependencies": versions(("numpy", "jax", "jaxlib")),
    }
    summary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"c_num (numerical candidate): {best['risk']:.12f} ({best['status']})")


def dual(args: argparse.Namespace) -> None:
    from qstarc import certify_risk_bound, solve_fixed_b_sos

    target = args.output / "solution.npz"
    require_new(target, args.overwrite)
    with np.load(args.output / "candidate.npz", allow_pickle=False) as data:
        result = solve_fixed_b_sos(
            data["rho"], data["D"], data["B"], level=1,
            tolerance=1e-9, output_path=target,
        )
    certificate = certify_risk_bound(target)
    print(f"Fixed-B SOS, m=1 (uncertified): {result['raw_risk_lower']:.12f} ({result['status']})")
    print(f"c_cert_lower (saved-input model): {certificate['risk_lower']:.12f}")


def verify(args: argparse.Namespace) -> None:
    from qstarc import verify_risk_certificate

    path = args.output / "certificates" / "solution_certificate.json"
    result = verify_risk_certificate(path)
    if not result["verified"]:
        raise RuntimeError("Certificate replay did not verify")
    print(f"PASS: exact rational replay; lower bound {result['risk_lower']:.12f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "optimize", "dual", "verify"), default="all")
    parser.add_argument("--output", type=Path, default=Path("demo_results"))
    parser.add_argument("--starts", type=positive_integer, default=2)
    parser.add_argument("--adam-steps", type=positive_integer, default=1000)
    parser.add_argument("--rcg-steps", type=positive_integer, default=200)
    parser.add_argument("--outcomes", type=positive_integer, default=8, help="outcome count s (rows of Z)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing generated results")
    args = parser.parse_args()
    for name, action in (("optimize", optimize), ("dual", dual), ("verify", verify)):
        if args.stage in ("all", name):
            action(args)


if __name__ == "__main__":
    main()
