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
2. **Scenario ingestion (implemented):** public gdm-flow adapter; original component UUID
   lineage; nominal voltage bases; configuration fingerprints; scenario shape,
  source, KCL residual, and reference-convergence validation. Generates static
  load-scale snapshots and ingests caller-provided converged results; temporal
  profile interpretation and independent reference validation remain future work.
3. **Candidate search (implemented):** whole-bus phase-compatible adjacent cluster
  merges on radial input graphs; matching nominal voltage bases; protected
  source/transformer/regulator/switch/control/DER/capacitor buses; accumulated
  cluster currents; original-terminal scenario error checks; deterministic ties;
  bounded batches; complete name/UUID mappings; and numerical equivalents.
  IEEE-13 ingestion and search are exercised together. Output is not yet a
  reduced GDM feeder or a validated nonlinear deployment model.
4. **Radialization (implemented):** prune non-root leaves, restore non-root branch
  vertices of the subtree connecting selected roots, and suppress degree-two
  paths. Preserve cluster/UUID mappings and pooled injections, verify original
  and reduced matrix bus connectivity, and check selected complex-voltage
  equivalence. Synthetic branching/coupled/missing-phase cases and IEEE-13 are
  exercised. Final bus counts include restored zero-injection junctions; this
  is structural restoration, not equipment reconstruction or nonlinear validation.
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
OPF disaggregation are deferred. The optional flow extra enables reference
generation through the GDM adapter. True sparse assembly is an
upstream gdm-flow requirement before utility-scale benchmarks: its current
`sparse=True` path first allocates a dense matrix. Keep private utility data and
copyrighted paper PDFs outside this repository.