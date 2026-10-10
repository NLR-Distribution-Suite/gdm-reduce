# Numerical Contract

## Existing Terminals and Source Boundaries

Rows and columns correspond only to existing `(bus_name, phase)` terminals.
Current support is A/B/C/S1/S2 with a ground reference, not explicit N or floating
neutrals. Missing phases have no padded rows. A source mapping
prescribes complex terminal voltages; it does not prescribe source current.

For fixed source terminals S and unknown terminals L:

$$Y_{LL}V_L = I_L - Y_{LS}V_S.$$

The full matrix can be singular. The source-conditioned block must be
nonsingular. An estimated one-norm condition number above the configured limit
(default $10^{12}$) is rejected. This estimate is a diagnostic, not a rigorous
upper bound. The topology check is necessary but does not establish grounding
or transformer-model correctness. No pseudoinverse conceals a singular solve.

## Exact Elimination

Partition terminals into kept K and removed R; all source buses are kept.

$$Y_{red} = Y_{KK} - Y_{KR}Y_{RR}^{-1}Y_{RK}.$$

For prescribed current injections at eliminated terminals:

$$I_{red} = I_K - Y_{KR}Y_{RR}^{-1}I_R.$$

Solve the retained source-boundary problem, then reconstruct:

$$V_R = Y_{RR}^{-1}(I_R - Y_{RK}V_K).$$

These are factorized solves, not explicit inverses. Schur columns are computed
in configurable batches. The output matrix may become dense through fill-in.
Only full-source-fixed and eliminated-block conditioning estimates are checked
during reduction, followed by the retained source-fixed block.

Current arrays have original terminal indexing. Voltage arrays returned by
`result.solve()` have retained indexing. `result.reconstruct()` returns original
indexing. Single vectors and terminal-by-scenario matrices are supported.

## Scope of Exactness

The identities apply to the supplied passive matrix and prescribed currents.
They do not certify the matrix's agreement with an independently translated
GDM/OpenDSS feeder. Nonzero source-current entries are not boundary conditions;
the solver ignores them while holding source voltages fixed.

Exact elimination is distinct from binary cluster current pooling. Moving a
loaded bus's current to a retained bus introduces a separate approximation
evaluated by candidate search. Constant-power, ZIP, delta, and controller-dependent
currents are not recomputed by this API. Their nonlinear deployment accuracy
must be validated separately. Input networks are not modified by reduction.

## Current-Pooled Search

For each scenario, aggregate every cluster's original terminal currents at its
root, phase by phase, and zero the other terminals. Keep the original matrix and
fixed source voltages throughout candidate evaluation. Represent each original
terminal by its cluster root's matching phase voltage. Define

$$e_{i,s} = \frac{\left| |V_{r(i),s}| - |V^{ref}_{i,s}| \right|}{V^{base}_i}.$$

The feasibility condition is $\max_{i,s} e_{i,s} \leq \epsilon$. Algorithm 1's
selection objective is $\sum_s\sum_C\max_{i\in C} e_{i,s}$ (SMICE), rather than
Equation (15)'s global maximum objective. Errors include all original members
of previously merged clusters. Phases cannot be invented, source buses cannot
be removed, and currents cannot be pooled across nominal-voltage boundaries.
Split-phase pooling also requires the same transformer/primary service domain.
Where both legs exist, complex S1-minus-S2 voltage error divided by the sum of
the leg bases must not exceed the same tolerance. This detects polarity/angle
errors invisible to leg magnitudes. SMICE remains a per-leg magnitude objective.
Leg imbalance is the signed leg-magnitude difference divided by the average leg
base; reports compare reference and represented imbalance without inventing
differential observations for single-leg buses.

At completion, exact Schur elimination of zero-injection removed terminals
produces the selected numerical equivalent. Solve it with the selected pooled
currents. Electrical reconstruction with `equivalent.reconstruct()` differs
from `represented_voltage`: the former solves eliminated terminal voltages,
whereas the latter assigns the root voltage used for cluster error scoring.
Source-current entries remain bookkeeping values; actual network source
injections are obtained from the solved voltages and matrix, not those entries.

Training magnitude thresholds do not bound voltage angles, losses, source power,
thermal loading, held-out conditions, or nonlinear aggregated-device behavior.
This result itself is not a physical feeder export; arbitrary Schur fill-in is
not silently converted into GDM line/transformer equipment. A separate guarded
equipment exporter is described below.

## Radial Numerical Equivalent

For a radial original physical bus graph, take the minimal subtree connecting
the selected roots. Restore its non-root vertices of degree at least three;
degree-two paths can still be eliminated. Removed dangling branches do not
force restoration. Recompute the Schur equivalent from the original matrix
with all selected roots and restored junctions kept.

The pooled original-index current matrix is unchanged. Restored junctions have
zero injection and are not new assignment roots. Therefore eliminating those
junctions again recovers the selected numerical equivalent; selected-terminal
complex voltages must agree within configured tolerances (defaults: 1e-7 V
absolute, 1e-9 relative). Both the original matrix's inter-bus nonzero support
and the radial matrix's support must exactly match their respective physical
trees. No coupling is thresholded away to obtain a tree. Inputs with unsupported
nonlocal couplings or numerical artifacts inconsistent with that contract are
rejected rather than presented as physical radial feeders.

The reported radial reduction counts restored buses. Search cluster membership,
UUID mapping, and training scores remain separate and unchanged. Restored
junction voltages are electrical solutions, not the representative root values
assigned during training-error scoring. Equipment realizability, nonlinear
device aggregation, holdouts, and independent feeder validation remain separate
release/export gates.

## GDM Equipment Export

`build_reduced_gdm()` works on a deep copy and retains protected equipment.
Degree-two paths consisting of plain matrix-impedance lines are replaced by
series equipment derived from the radial matrix's inter-bus blocks. Reciprocal
passive impedance, original retained terminals and a restamped Y-bus match are
required. Defaults are 1e-6 S absolute and 1e-9 relative comparison tolerance.
Non-realizable self-shunts, parallel branches and unsupported contracted branch
types fail rather than being approximated without acknowledgment.

Loads retain their original device UUIDs, equipment, per-phase P/Q, ZIP fields
and time-series attachments but move to their selected assignment roots.
Restored junctions are not assignment roots. Surviving bus identities and
equipment stay intact; replacement lines receive deterministic UUIDs, original
path UUID provenance, total path length and minimum path ampacity. Metadata on
discarded buses or lines is rejected. The original system remains unchanged.

References and export verification currently omit line charging. Nonzero
charging on removed lines requires explicit `allow_series_only=True` or CLI
`--export-series-only`; affected branch UUIDs are recorded. Even then the
restamped series-only matrix must match. Charged Schur equivalents with residual
shunts cannot be represented by this exporter. Grounding assumptions normalize
only copied center-tap secondary windings and remain explicit in the report.

Preserving and relocating P/Q devices does not reproduce fixed-current pooling.
`validate_reduced_gdm()` therefore solves the exported equipment separately at
recorded load/solar scales, using the original source phasors and root mapping.
Leg magnitude and applicable S1-S2 complex limits determine its pass flag.
Convergence alone does not establish accuracy: the report separately records
solver convergence and passing voltage limits. This check covers gdm-flow's
static constant-P/Q behavior, not independent OpenDSS, ZIP, time-series, thermal
limits or OPF feasibility. Programmatic callers own scenario independence.

`save_reduced_gdm()` uses canonical GDM serialization and companion profile
storage. A separate `.export-report.json` preserves assumptions, mappings,
lineage and validation; numerical result bundles remain distinct artifacts.
Existing files require explicit overwrite permission. The CLI additionally
forbids input/model/report path collisions and returns failure when either its
numerical or separately solved exported-device reports fail.

## Frozen-Cluster Evaluation And Artifacts

Evaluation requires the original terminal order, matrix, voltage bases, fixed
sources and model signature to match the selection fingerprint. Pool new
reference currents with the existing assignment; do not select new clusters
using held-out voltages. Magnitude and complex-voltage errors use each original
terminal's nominal phase base and the assigned root's voltage. Angle differences
are wrapped to [-180, 180] degrees; a zero representative makes the scenario's
maximum angle error undefined (`None`).

Report the full-network complex-voltage reconstruction error separately from
the reduction errors, backend convergence mismatch and reference KCL residual.
Source P/Q and passive active-absorption differences use electrically reconstructed
voltages and actual `YV` currents, not assignment-root voltages or pooled source
bookkeeping. These differences are signed reduced minus reference. Passive
absorption includes all passive elements represented by Y and is not labeled
as line-only loss or independently validated operating behavior.

Leg magnitude and applicable complex S1-S2 limits determine the report pass flag. Other diagnostics
remain explicit and are not hidden in that flag. The CLI rejects overlapping
training/holdout load scales, records per-stage timing/environment metadata,
and emits reports even on validation failure. The programmatic caller owns
the reference split and must not treat repeated training data as independent
held-out evidence.

Versioned JSON artifacts contain CSR data in SI units, fixed sources, labels,
mapping/lineage, separate reference/current snapshots and diagnostics. Decoding
validates sparse structure and the numerical network; no LU caches or GDM
equipment models are serialized. Existing files are not overwritten without
explicit permission. A saved matrix can replay its retained-current solve,
but does not establish physical equipment realizability or nonlinear accuracy.

## Center-Tap Grounding Assumption

GDM reference generation assumes supported center taps are grounded by default.
Only the solver copy's secondary winding flags are normalized; original models
and files are unchanged. Service domains and fingerprints distinguish
`assumed-grounded` from strict `model-grounded` policy, even for identical numeric
matrices. Scenario provenance and v2 result bundles preserve the assumption.
Use `assume_center_taps_grounded=False` or `--require-grounding-metadata` for
strict original-metadata validation. Ingested results must already have been
solved using the declared policy; ingestion does not modify their matrix.

This assumption supplies an implicit zero-voltage neutral reference, not a new
explicit N terminal or a floating-neutral model. Unsupported winding layouts
and unsupported device physics are still rejected. Nonnegative no-load core
loss is a conductance on winding two, using primary rated VA and the squared
tap-adjusted winding-two voltage base; its primitive is checked against OpenDSS.
Complete one-based pair indices are normalized to zero-based indices without
inventing missing reactances. Split-phase line-to-line bus bases are divided by two.
For a two-leg DELTA load, gdm-flow uses opposite currents from the actual
S1-minus-S2 voltage and the sum of GDM phase-load powers. Reduction freezes those
reference currents; exported devices retain their individual P/Q and must be
solved separately. The exported-device check does not implement ZIP evaluation.