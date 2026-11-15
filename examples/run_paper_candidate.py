"""Run one selected paper model with explicit, auditable candidate settings.

Examples (after installing the package with its optimization dependencies)::

    python examples/run_paper_candidate.py experiment2 --ncopy 2 --dry-run
    python examples/run_paper_candidate.py experiment2 --ncopy 2 --output results/run_n2
    python examples/run_paper_candidate.py experiment4_A --dry-run

Unknown historical learning rates/RCG caps remain null in a dry run. Actual
execution requires --adam-lr and --rcg-steps for those examples; this creates a
new, explicitly chosen protocol, not a claim to recover missing history.
Use --adam-steps 5 --rcg-steps 5 --seeds 0 for a small smoke check. No NH,
dual-SDP, ideal-input certification, or historical auxiliary warm start is run.

Paper notation: Z is saved as Q, the outcome count s is the legacy M/mout,
and the reported floating-point cost c_num is saved as paper_risk.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

# Historical candidate drivers used CPU/x64 and one-thread linear algebra.
# Set these before importing NumPy/JAX; no package implementation is modified.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from paper_models import build_model, load_settings


def _save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _evaluate(Q, model):
    u = Q.conj().T
    p = np.einsum("as,ab,bs->s", u.conj(), model["rho"], u).real
    if not np.all(p > 0):
        raise FloatingPointError("Nonpositive measurement probability; no clipping is applied.")
    dp = np.einsum("as,kab,bs->ks", u.conj(), model["D"], u).real
    F = (dp / p) @ dp.T
    F = (F + F.T) / 2
    inverse = np.linalg.inv(F)
    cost = float(np.trace(inverse))
    dp0 = np.einsum("as,kab,bs->ks", u.conj(), model["paper_D0"], u).real
    F0 = (dp0 / p) @ dp0.T
    direct = float(np.trace(model["paper_W"] @ np.linalg.inv(F0)))
    completeness = float(np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1])))
    if not math.isfinite(cost) or np.linalg.eigvalsh(F).min() <= 0 or completeness > 1e-8:
        raise FloatingPointError("Endpoint failed the finite positive-Fisher/completeness check.")
    return F, (inverse + inverse.T) / 2, dict(
        optimizer_risk=cost, paper_risk=model["risk_scale"] * cost,
        direct_paper_weight_risk=direct,
        coordinate_risk_residual=abs(direct-model["risk_scale"]*cost),
        completeness_residual=completeness, probability_min=float(p.min()),
        qstarc_support_size=int(len(Q)), min_effect_trace=float(np.sum(abs(Q)**2, axis=1).min()))


def main(argv=None):
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=settings["examples"])
    parser.add_argument("--ncopy", type=int)
    parser.add_argument("--r", help="Decimal or rational string, e.g. 0.5 or 1/2")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--dry-run", action="store_true", help="Show inputs and unresolved settings without optimizing or writing")
    parser.add_argument("--mout", type=int, help="Explicit override of the recorded outcome count s (legacy M)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit override of the recorded seeds 0,...,9")
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--adam-lr", type=float)
    parser.add_argument("--adam-chunk", type=int)
    parser.add_argument("--rcg-steps", type=int)
    parser.add_argument("--grad-tol", type=float)
    args = parser.parse_args(argv)
    try:
        model = build_model(args.example, ncopy=args.ncopy, r=args.r)
    except ValueError as error:
        parser.error(str(error))
    spec = model["metadata"]["spec"]
    options = dict(spec["optimizer"])
    overrides = {}
    for key in options:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
            overrides[key] = value
    seeds = args.seeds if args.seeds is not None else settings["seed_list"]
    mout = args.mout if args.mout is not None else model["metadata"]["qstarc_support_size"]
    if args.seeds is not None:
        overrides["seeds"] = seeds
    if args.mout is not None:
        overrides["mout"] = mout
    for key in ("adam_steps", "rcg_steps", "adam_chunk"):
        if options[key] is not None and options[key] < (1 if key == "adam_chunk" else 0):
            parser.error(f"{key} is outside its allowed range.")
    for key in ("adam_lr", "grad_tol"):
        if options[key] is not None and (not math.isfinite(options[key]) or options[key] <= 0):
            parser.error(f"{key} must be finite and positive.")
    if min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error("Seeds must be distinct nonnegative integers.")
    if mout < len(model["rho"]):
        parser.error("mout must be at least the Hilbert dimension.")
    missing = [key for key, value in options.items() if value is None]
    same_known = all(options[k] == v for k, v in spec["optimizer"].items())
    matches = bool(spec["recorded_settings_complete"] and same_known and seeds == settings["seed_list"]
                   and mout == model["metadata"]["qstarc_support_size"])
    protocol = dict(model=model["metadata"], optimizer_options=options,
                    qstarc_support_size=mout, seeds=seeds, explicit_overrides=overrides,
                    unrecorded_historical_settings=spec.get("unrecorded", []),
                    unresolved_options=missing, matches_recorded_candidate_settings=matches,
                    scope="New numerical candidates only; no exact upper or lower certificate, and no guaranteed reproduction of historical endpoints/timings",
                    retry_policy="No automatic learning-rate fallback; the chosen value is recorded explicitly",
                    timing_scope="Per-call wall time includes compilation when encountered; no independent JIT warmup or benchmark",
                    command=sys.argv if argv is None else list(argv))
    if args.dry_run:
        print(json.dumps(protocol, indent=2, allow_nan=False))
        return 0
    if missing:
        parser.error("Historical settings are unrecorded. Specify " + ", ".join("--"+key.replace("_", "-") for key in missing) + ".")
    if args.output is None:
        parser.error("Execution requires --output pointing to a new directory.")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to preserve previous results.")
    from qstarc.optimizer import minimizeQ_stiefel
    import jax
    protocol["environment"] = dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__, platform=platform.platform())
    args.output.mkdir(parents=True, exist_ok=False)
    _save(args.output / "protocol.json", protocol)
    identity = dict(risk_scale_exact=model["metadata"]["risk_scale_exact"], example_id=args.example)
    np.savez_compressed(args.output / "model.npz", **identity,
                        **{k:model[k] for k in ("rho","D","D0","W","T","paper_D0","paper_W","risk_scale")})
    runs, best = [], None
    for seed in seeds:
        start = time.perf_counter()
        try:
            result = minimizeQ_stiefel(model["rho"], model["D"], mout=mout, seed=seed, verbose=False, **options)
            Q = result["Q"]
            F, B, evaluated = _evaluate(Q, model)
            rec = {k:v for k,v in result.items() if k not in ("Q","U","fisher")}
            rec.update(evaluated, risk_is_certified_upper=False)
            if best is None or rec["optimizer_risk"] < best["optimizer_risk"]:
                best = rec.copy()
                np.savez_compressed(args.output / "best.npz", Q=Q, F=F, B=B,
                                    rho=model["rho"], D=model["D"], risk_scale=model["risk_scale"], **identity)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            rec = dict(seed=seed, status="numerical_failure", converged=False,
                       optimizer_risk=None, paper_risk=None, error=str(error), risk_is_certified_upper=False)
        rec["runner_seconds"] = time.perf_counter() - start
        runs.append(rec)
        _save(args.output / "runs.json", runs)
        print(json.dumps({k:rec.get(k) for k in ("seed","status","paper_risk","qstarc_support_size")}, allow_nan=False), flush=True)
    summary = dict(complete=True, best=best, runs=runs, risk_scale_exact=model["metadata"]["risk_scale_exact"],
                   selection=settings["selection"], risk_is_certified_upper=False,
                   matches_recorded_candidate_settings=matches,
                   B_convention="Symmetric part of the numerical inverse Fisher in optimizer coordinates; no experiment4 dual symmetry projection")
    _save(args.output / "summary.json", summary)
    if best is None:
        raise FloatingPointError("No finite candidate was obtained; inspect runs.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
