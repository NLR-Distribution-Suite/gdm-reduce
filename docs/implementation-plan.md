# Implementation Roadmap

## Ownership

grid-data-models owns equipment, units, identities, and time series. gdm-flow
owns matrix stamping and nonlinear feeder solves. gdm-reduce owns electrical
reduction, clustering, mapping, radialization, and accuracy reports. OpenDSS
provides independent reference validation. Do not duplicate the stamping code.

## Milestones

1. **Foundation and exact reduction (implemented):** package scaffold, CI,
   BSD-3-Clause licensing, source-conditioned Schur elimination, exact loaded
   current transformation, batched solves, and voltage reconstruction.
2. **Scenario ingestion:** public gdm-flow adapter; original component UUID
   lineage; nominal voltage bases; configuration fingerprints; scenario shape,
   source, KCL residual, and reference-convergence validation.
3. **Candidate search:** whole-bus phase-compatible adjacent cluster merges,
   protected source/transformer/control/DER buses, accumulated cluster currents,
   original-scenario error checks, deterministic ties, and bounded batches.
4. **Radialization:** minimal required junction restoration, unchanged injection
   assignments, radiality tests, and retained-voltage equivalence checks.
5. **Validated alpha release:** CLI, versioned result bundles, synthetic and
   IEEE examples, held-out scenario reports, and reproducible CPU benchmarks.
6. **GDM feeder export:** supported equipment reconstruction, device/time-series
   lineage, and independent nonlinear electrical validation. Reject unsupported
   equivalents instead of claiming all Schur matrices are ordinary feeders.

## Search Contract

Use Algorithm 1's sum of maximum intra-cluster magnitude errors (SMICE), with
a separate maximum per-phase error feasibility threshold. Document that
Equation (15) in the paper instead states a maximum-error selection objective.
Enumeration is exhaustive over available one-merge candidates, not globally
optimal over final partitions. Evaluate all original buses against original
training voltages after every merge, composing complete cluster memberships.

The source-conditioned admittance and electrical configuration remain fixed
within a library. Reuse factorization and batch candidate computations. Avoid
a dense inverse and unbounded candidate-voltage tensors. Report whether search
stopped because of a target, infeasibility, iteration limit, or resource limit.

## Validation Gates

- Exact identities on well-conditioned synthetic feeders: approximately
  $10^{-9}$ relative voltage error or better, with independent closed-form and
  nodal-residual checks. Retain mutual coupling and missing phases.
- Track reference-model, prescribed-current reduction, and nonlinear exported
  model errors separately. Baseline errors must be materially smaller than the
  requested reduction tolerance; do not hide them in aggregate metrics.
- IEEE-13 first; IEEE-123 after regulator and transformer baseline discrepancies
  are resolved. Match controller settings, taps, capacitors, loads, and sources.
- Holdouts include independently varying phase demand, reverse flow, and
  stress conditions. Training thresholds are not holdout guarantees.
- Validate source P/Q, losses, voltage angles, and reconstructed thermal loading
  where supported before making operational claims.

## Deferred Scope and Upstream Dependencies

Explicit neutrals, split-phase, meshed-input reduction, GPU acceleration, and
OPF disaggregation are deferred. An optional flow extra declares the planned
dependency but does not yet expose a GDM adapter. True sparse assembly is an
upstream gdm-flow requirement before utility-scale benchmarks: its current
`sparse=True` path first allocates a dense matrix. Keep private utility data and
copyrighted paper PDFs outside this repository.