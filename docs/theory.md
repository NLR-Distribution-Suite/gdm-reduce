# Phase-Aware Kron Reduction

Chevalier and Almassalkhi {cite:p}`chevalier2022` formulate balanced-network
Opti-KRON clustering as an optimization problem. Mokhtari, Chevalier, and
Almassalkhi {cite:p}`mokhtari2026` extend the formulation to unbalanced
three-phase feeders and describe exhaustive candidate search and radialization.
These papers motivate GDM-Reduce; the package does **not** implement the earlier
MILP solver, global optimality certification, or the later GPU acceleration.

## Existing Phase Terminals

Let $\mathcal{T}=\{(b,\phi):\phi\text{ exists at bus }b\}$ index the network:

```{math}
:label: nodal-equation
I=YV,\qquad Y\in\mathbb{C}^{|\mathcal{T}|\times|\mathcal{T}|}.
```

Voltages are volts, currents amperes, admittances siemens; positive current
injects into the passive network. Missing phases have no padded rows. Supported
terminals are A/B/C and grounded S1/S2, not explicit or floating neutrals.
Source phasors are fixed, not source currents. With sources $S$ and unknowns $L$:

```{math}
:label: source-conditioned-equation
Y_{LL}V_L=I_L-Y_{LS}V_S.
```

The full $Y$ can be singular while $Y_{LL}$ is nonsingular. GDM-Reduce uses
factorizations and condition checks, rejecting singular source-fixed blocks.
The three-phase paper's padded absent-phase formulation uses a pseudoinverse;
that differs from this package's existing-terminal indexing and rejection policy.

## Exact Elimination

Keep terminals $K$, remove $R$, and retain every source bus. Schur elimination is

```{math}
:label: exact-schur
Y_{\mathrm{red}}=Y_{KK}-Y_{KR}Y_{RR}^{-1}Y_{RK},
\qquad I_{\mathrm{red}}=I_K-Y_{KR}Y_{RR}^{-1}I_R.
```

After the retained solve, reconstruct eliminated voltages:

```{math}
:label: voltage-reconstruction
V_R=Y_{RR}^{-1}(I_R-Y_{RK}V_K).
```

Inverse notation expresses the identity; implementation uses factorized solves,
not explicit inverses. For a supplied passive matrix and prescribed currents,
this is exact up to numerical tolerance. Fill-in can make the equivalent dense.
It does not validate the original translation or a nonlinear reduced load model.

## Current Pooling

Let $r(b)$ assign each bus to a retained root. A terminal assignment operator
$A_{\mathcal{T}}$ pools same-phase currents at roots:

```{math}
:label: pooled-injection
I^{\mathrm{agg}}_{(k,\phi),s}=\sum_{b:r(b)=k}I^{\mathrm{ref}}_{(b,\phi),s},
\qquad I^{\mathrm{agg}}=A_{\mathcal{T}}I^{\mathrm{ref}}.
```

Non-root entries are zero. An original terminal is represented by its root:
$V^{\mathrm{rep}}_{(b,\phi),s}=V^{\mathrm{agg}}_{(r(b),\phi),s}$.
This is **not** electrical reconstruction in Equation {eq}`voltage-reconstruction`.
Training magnitude errors and the local SMICE selection score are

```{math}
:label: local-error-score
e_{t,s}=\frac{\left|\left|V^{\mathrm{rep}}_{t,s}\right|
-\left|V^{\mathrm{ref}}_{t,s}\right|\right|}{V^{\mathrm{base}}_t},
\qquad\mathrm{SMICE}=\sum_s\sum_C\max_{t\in C}e_{t,s}.
```

Feasibility requires $\max_{t,s}e_{t,s}\leq\epsilon$ and applicable differential
constraints. Adjacent whole-bus merges are ranked by SMICE. Search stops at the
target, iteration limit, or absence of feasible merges; targets are not guaranteed.

Assignments preserve connectivity, matching phase sets and nominal-voltage
bases, and split-phase transformer/primary service domains. Sources, transformer
terminals, switches, DER, capacitors, controllers and reactors protect boundaries.
Matching phase sets are more restrictive than the paper's subset-availability rule.

## Grounded Split Phase

With two legs, $V^{\mathrm{LL}}=V_{S1}-V_{S2}$ and the additional constraint is

```{math}
:label: differential-error
e^{\mathrm{LL}}_s=
\frac{\left|(V^{\mathrm{rep}}_{S1,s}-V^{\mathrm{rep}}_{S2,s})
-(V^{\mathrm{ref}}_{S1,s}-V^{\mathrm{ref}}_{S2,s})\right|}
{V^{\mathrm{base}}_{S1}+V^{\mathrm{base}}_{S2}}\leq\epsilon.
```

This detects polarity/angle errors hidden by leg magnitudes. Center-tap grounding
is assumed on a solver copy by default and recorded in provenance. Set
`assume_center_taps_grounded=False` to require original grounding flags. Neither
policy supplies explicit-neutral or floating-neutral physics.

## Radialization and Equipment

Kron elimination can create meshed fill-in. Radialization restores branching
junctions in the minimal subtree connecting roots, while contracting degree-two
paths. Restored junctions carry zero pooled injection and count toward the final
bus count. Selected-root complex voltages are checked against the numerical
equivalent.

Export is a separate gate. Supported plain matrix-line paths become passive,
reciprocal equipment; protected devices and load UUIDs remain. Relocated loads
retain P/Q, equipment and profiles. A restamped series-only Y-bus must match the
radial Schur equivalent. Charging on discarded lines requires explicit
`allow_series_only=True`, with omissions recorded in the report.

## Nonlinear Evaluation

For constant-power devices $I(V)=\overline{S/V}$, so relocated P/Q is not
prescribed-current pooling. Solve the export separately at frozen, distinct
held-out scales. The notebook uses training 0.5 and 1.0, evaluation 0.75, and a
chosen 0.01 pu limit. The resulting voltage bound does not certify losses,
thermal loading, OPF feasibility, ZIP/time-series responses, or OpenDSS fidelity.

## References

```{bibliography}
```