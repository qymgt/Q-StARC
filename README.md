# Q-StARC

**Quantum Estimator Design via Stiefel Adam–RCG Optimization with Certified Risk Bounds**

Q-StARC searches for rank-one quantum measurements with small weighted
inverse-Fisher cost. It combines projected Adam, QR retraction, and
Riemannian conjugate-gradient refinement, then provides separate SOS
lower-bound calculation and rational certificate verification.

## Installation

Use Python 3.12 in a new virtual environment, from this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

JAX runs the optimizer; CVXPY and the open-source Clarabel solver run the
SDP. No commercial solver is required. The tested dependency versions are
pinned in `requirements.txt`. Certificate replay needs only NumPy and
Python's standard library.

## Quick start: optimize a measurement

Specify the state `rho`, its local derivatives `D`, and the number of
outcomes `mout`. Here $n=2$, $d=2$, and $s=3$: a two-parameter qubit at the
maximally mixed state, with identity weight.

```python
import numpy as np
from qstarc import optimize_multistart

rho = np.eye(2, dtype=complex) / 2
D = np.array([
    [[0, 0.5], [0.5, 0]],
    [[0, -0.5j], [0.5j, 0]],
])

search = optimize_multistart(
    rho, D, mout=3, seeds=[0, 1],
    adam_steps=1000, rcg_steps=200, grad_tol=1e-9,
)
best = search["best"]
Z = best["Q"]
F = best["fisher_original"]
c_num = best["risk"]
print("c_num =", c_num, "status =", best["status"])
```

The state has shape $(n,n)$ and the derivatives have shape $(d,n,n)$.
Each row of $Z$ defines an effect: in Python, construct `M[x]` as
`np.outer(Z[x].conj(), Z[x])`.
The example selects the smallest final finite cost over two initializations;
the short iteration settings are a functional example, not a paper run.

| Output | Meaning |
| --- | --- |
| `best["Q"]` | Measurement frame $Z$, shape $(s,n)$ |
| `best["fisher_original"]` | Fisher matrix in the original supplied coordinates |
| `best["risk"]` | Numerical weighted cost $c_{\mathrm{num}}$ |
| `best["status"]`, `best["converged"]` | Stopping diagnostics |
| `best["rho"]`, `best["D"]`, `best["B"]` | Inputs for the lower-bound calculation below |

To use a general positive definite weight, add, for example,
`weight=np.diag([1.0, 2.0])` to `optimize_multistart`.
The candidate cost is a floating-point evaluation, not a certified upper
bound or proof of global optimality.

### Main search options

| Argument | Meaning | Default |
| --- | --- | --- |
| `mout` | Outcome count $s$ | Required |
| `seeds` | Independent initialization seeds | `range(10)` |
| `weight` | Positive definite parameter-space weight $W$ | Identity |
| `adam_steps` | Number of Adam updates | `30000` |
| `adam_lr` | Adam learning rate | `0.005` |
| `rcg_steps` | Maximum RCG iterations | `2000` |
| `grad_tol` | RCG gradient stopping threshold | `1e-9` |

## Compute and verify a lower bound

Continue in the same Python session, using `best` from the quick start:

```python
from qstarc import solve_fixed_b_sos, certify_risk_bound, verify_risk_certificate

solution_path = "results/quickstart/solution.npz"
proposal = solve_fixed_b_sos(
    best["rho"], best["D"], best["B"],
    level=1, output_path=solution_path,
)
certificate = certify_risk_bound(solution_path)
verified = verify_risk_certificate(certificate["certificate_path"])
print("certified lower =", certificate["risk_lower"])
print("exact rational value =", certificate["risk_lower_exact"])
print("verified =", verified["verified"])
```

`level=1` means the denominator-SOS level $m=1$, with $B$ held fixed.
The raw SDP objective is uncertified; `certify_risk_bound` repairs it and
`verify_risk_certificate` checks the saved proof without a solver.
The lower bound concerns the saved state and derivatives; the quick-start
model has exactly representable dyadic entries. See the certification scope
below for general inputs. These API calls overwrite files at their chosen
destinations.

Proposal files use format `qstarc-sos-proposal-v1`; certificates use
`qstarc-risk-certificate-v1`. Each verifier accepts its declared format.
For other archived formats, use the matching archived verifier; do not
edit certificate contents or referenced files to change their format.

## Run the supplied end-to-end demo

```sh
python examples/qubit_workflow.py --output results/demo
```

This separate demo uses $N=2$ qubits at Bloch vector $(0,0,1/2)$,
standard Bloch derivatives, identity weight, $s=8$, seeds 0 and 1,
1,000 Adam updates, and at most 200 RCG iterations. Its state and
derivatives also have exact dyadic entries.

The output directory contains `candidate.npz` and `optimization.json`,
the SDP proposal `solution.npz`, and the certificate and factors under
`certificates/`. To run stages separately, use `--stage optimize`,
`--stage dual`, or `--stage verify` with the same output directory.
Existing outputs are protected unless `--overwrite` is supplied.

## Run a paper model

Inspect the settings before starting a paper-sized candidate search:

```sh
python examples/run_paper_candidate.py experiment2 --ncopy 2 --dry-run
python examples/run_paper_candidate.py experiment2 --ncopy 2 --output results/paper_n2
```

The second command uses ten seeds, 30,000 Adam updates, and up to 2,000
RCG iterations per seed. Use `--help` for other models and overrides.
The runner saves the selected numerical candidate and its run settings;
it does not perform certification or comparisons with other methods.

`examples/paper_settings.json` records the models, weights, outcome counts,
seeds, and stopping conditions. `examples/paper_models.py` constructs the
historical numerical inputs. Missing historical learning rates or iteration
limits remain `null` and require explicit overrides when running those
cases; an override does not recover an unknown historical setting.

## Notation and numerical conventions

We follow the accompanying manuscript, *Efficient numerical measurement design
and certified risk bounds for multiparameter quantum estimation*. Existing API
names and saved-array keys are retained for compatibility.

| Paper notation | Meaning | Code or saved key |
| --- | --- | --- |
| $n$ | Hilbert-space dimension | `rho.shape[0]`, `dimension`, `dim` |
| $d$ | Number of parameters | `len(D)`, `parameters`, internal `p` or `nparam` |
| $s$ | Number of outcomes | `mout`, `measurement_outcomes`, `M`, `M_by_N` |
| $Z\in\mathrm{St}_{\mathbb C}(s,n)$ | Measurement frame, $Z^\dagger Z=I_n$ | `Q`; `U=Q.conj().T` |
| $\rho=\rho_{\theta_0}$, $D_j=\partial_j\rho_{\theta_0}$ | State and supplied local derivatives | `rho`, `D`, `d_rho`, `derivatives` |
| $F_{\theta_0}[M]$, $W$ | Classical Fisher information and weight | `fisher` or `F`, `weight` or `W` |
| $\widetilde D_j$, $\widetilde F$ | Derivatives and Fisher matrix in transformed coordinates | `D`, `fisher` after whitening |
| $B$, $X$, $G$ | Fixed support coefficient, Hermitian majorant, PSD Gram matrix | `B`, `X`, `grams` / `G_i` |
| $\xi\in\mathbb R^d$, $m$ | SOS auxiliary variable and denominator exponent | monomial exponents; `level` |
| $c_{\mathrm{num}}$ | Floating-point candidate risk | weighted-wrapper `risk`; example-runner `paper_risk` |
| $\underline c_{\mathrm{cert}}$ | Certified all-POVM risk lower bound | `risk_lower_exact`; downward float `risk_lower` |
| $N$, $r$ in qubit examples | Copy number and Bloch radius | `ncopy`, `r`; Hilbert dimension is $n=2^N$ |

The paper's $M=(M_x)$ is a POVM, not an outcome count:
$u_x=Z^\dagger e_x$ and $M_x=u_xu_x^\dagger$. The legacy key `Q` denotes
$Z$, not the kernel projector $Q=I-P$ used in the theory. Likewise, the
certificate key `rank` denotes $\mathrm{rank}(\rho)$, not the Bloch radius.
We use $\mathrm{tr}$ for operator traces and $\mathrm{Tr}$ for
parameter-matrix traces, as in the paper; NumPy uses `trace` for both.

For general weights, Sec. III G uses $T=W^{-1/2}$,
$\widetilde D_i=\sum_j D_jT_{ji}$ and $\widetilde F=T^{\mathsf T}FT$, so
$\mathrm{Tr}(\widetilde F^{-1})=\mathrm{Tr}(WF^{-1})$.
The optimizer's `minimum` is the unit-weight objective in the supplied
coordinates; apply any example-specific `risk_scale` to obtain `paper_risk`.

In Experiments 1 and 2, Eq. (7.3) gives
$c_{\mathrm{num}}=N\mathrm{Tr}(\widetilde F^{-1})$. The search optimizes
$\mathrm{Tr}(\widetilde F^{-1})$ before applying the factor $N$.
The diagonal two-copy example retains its historical coordinates and the
final `153/223` reporting scale.

The outcome count $s$ includes all rows of $Z$ before removing tiny effects
or merging directions; it is not a minimum-outcome claim.
`maxiter` and `line_search_failed` are stopping statuses, not convergence.
Adam returns its last iterate, and multiple starts compare their final
finite costs. No probability floor or Fisher ridge is used.

## Certification scope and SOS formulation

The candidate risk $c_{\mathrm{num}}=\mathrm{Tr}(WF^{-1})$ is a floating-point evaluation of the
constructed measurement. It is not a rigorous upper certificate or proof
of global optimality. A raw SDP objective is also not a certificate.

`solve_fixed_b_sos` implements the fixed-coefficient denominator-SOS
formulation of Secs. IV and V. The denominator exponent is `level=m`.
It uses identity weight in the supplied derivative coordinates, with
$A_i=\sum_j B_{ij}D_j$ and
$Q_{B,X}(\xi)=X+\|\xi\|^2\rho-2\sum_i\xi_i A_i$.
It minimizes $\mathrm{tr}(X)$ subject to a Gram representation of
$(1+\|\xi\|^2)^m Q_{B,X}(\xi)$.
The raw value `2*trace(B)-raw_support_upper` is a numerical approximation
to the fixed-$B$ optimum $c_m(I_d;B)=2\mathrm{Tr}(B)-d_m(B)$; feasibility
and optimality are not certified by the solver. The joint optimization over $B$ defining
$c_m(W)$ in Sec. VI is not implemented here.

In transformed coordinates the usual proposal is
$B=\widehat{\widetilde F}^{-1}$. The general-weight formula
$B=W^{1/2}\widehat F^{-1}$ in Eq. (4.3) refers to the original coordinates;
the API does not accept an additional general weight.
Saved basis rows `(alpha_1, ..., alpha_d, i)` use a zero-based component index
and mean the column $\xi^\alpha e_{i+1}$ of the paper's reduced basis $V_m(\xi)$.
The code orders components before monomials, a permutation of Eq. (5.5);
sign-symmetry blocks group these columns by character. Certificate replay
uses the saved order.

Rational post-processing constructs a positive-semidefinite Gram factor,
bounds the polynomial coefficient residual, and repairs the lower bound.
Replay checks the saved factors and exact rational values without calling
an SDP solver or an eigendecomposition. The guarantee concerns the exact
dyadic interpretation of the stored binary64 state and derivatives, in
the stored derivative coordinates with identity weight. Any reporting-scale
conversion is separate. A floating-point whitening or basis transformation
does not itself certify an ideal input model.
The resulting $\underline c_{\mathrm{cert}}$ at $m=1$ has the role of
$\underline c_1$ in Sec. VII, but the paper's ideal-model tables also
require the input-error corrections described there.

The SDP interface uses an exactly diagonal state. Any positive support
must precede its zero kernel, and derivative kernel-kernel blocks must
vanish. Exact trace checks are reported separately; they do not establish
physical extendibility of arbitrary singular-model tangents. The repaired
bound need not be feasible in the original finite SOS cone.

This release does **not** include comparisons with other methods, figure
generation, large saved results, or the full paper experiment archive.
The experiment-specific ideal-input transfers and Schur-block workflow
used for the paper's certified tables are outside this minimal package.
It does not claim to reproduce all paper experiments or their ideal-model
certified bounds. The generic SOS calculation and certificate replay are
exercised by the small demo above.

## Verified behavior and public scope


```sh
python -m unittest discover -s tests -v
```

Tests cover optimizer regression against a small frozen reference, model
and weight conventions, SOS splitting, and exact certificate validation.
Optional-component tests skip if their dependencies are absent. All 23
tests passed in a clean Python 3.12.14 environment on macOS arm64, using
NumPy 2.5.4, JAX 0.11.2, CVXPY 1.9.3, and Clarabel 0.11.1.
The two Python quick-start blocks above were also executed together:
the candidate cost was approximately `4.0` (status `converged`), the
certified lower bound exceeded `3.999999720943`, and replay returned
`verified=True`.

The complete `examples/qubit_workflow.py` demo produced numerical risk
`2.741511036406` (RCG status `maxiter`) and
certified lower bound `2.741046056648`, with exact rational replay passing.
These values are rounded for display; the certificate stores exact rational
values. Small candidate runs passed for all six example configurations;
the full paper experiments were not rerun.

No software license has been selected; see `NOTICE`. Dependency source
trees, credentials, and historical development files are not bundled.
