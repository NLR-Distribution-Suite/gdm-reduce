# Numerical Contract

## Existing Terminals and Source Boundaries

Rows and columns correspond only to existing `(bus_name, phase)` terminals.
Current support is A/B/C. Missing phases have no padded rows. A source mapping
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

At completion, exact Schur elimination of zero-injection removed terminals
produces the selected numerical equivalent. Solve it with the selected pooled
currents. Electrical reconstruction with `equivalent.reconstruct()` differs
from `represented_voltage`: the former solves eliminated terminal voltages,
whereas the latter assigns the root voltage used for cluster error scoring.
Source-current entries remain bookkeeping values; actual network source
injections are obtained from the solved voltages and matrix, not those entries.

Training magnitude thresholds do not bound voltage angles, losses, source power,
thermal loading, held-out conditions, or nonlinear aggregated-device behavior.
This result is not a physical feeder export; arbitrary Schur fill-in is not
silently converted into GDM line/transformer equipment.

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

The magnitude limit alone determines the report pass flag. Other diagnostics
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