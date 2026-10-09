# Scope and Validation

## Ownership

grid-data-models owns equipment, units, identities, and time series. gdm-flow
owns matrix stamping and nonlinear feeder solves. gdm-reduce owns electrical
reduction, clustering, mapping, radialization, and accuracy reports. OpenDSS
provides independent reference validation. Do not duplicate the stamping code.

## Implemented Capabilities

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
  IEEE-13 ingestion and search are exercised together. Search returns a numerical
  equivalent; physical equipment export is a separate step.
4. **Radialization (implemented):** prune non-root leaves, restore non-root branch
  vertices of the subtree connecting selected roots, and suppress degree-two
  paths. Preserve cluster/UUID mappings and pooled injections, verify original
  and reduced matrix bus connectivity, and check selected complex-voltage
  equivalence. Synthetic branching/coupled/missing-phase cases and IEEE-13 are
  exercised. Final bus counts include restored zero-injection junctions; this
  is structural restoration, not equipment reconstruction or nonlinear validation.
5. **Validated alpha release (tooling implemented; certification pending):** CLI
  with independent train/holdout load-scale splits and failure exit codes;
  versioned, replayable numerical JSON bundles; frozen-cluster magnitude,
  complex/angle and electrical power diagnostics; synthetic stress, reverse-flow
  and phase-specific tests; IEEE-13 holdout runs; per-stage CPU timings and
  environment metadata. Independent baseline fidelity, representative benchmark
  campaigns and nonlinear deployment certification remain open release gates.
6. **GDM feeder export (implemented for supported radial equivalents):** passive
  matrix-impedance path contraction, retained protected equipment and bus UUIDs,
  device/time-series lineage, canonical GDM serialization, a restamped Y-bus
  check, and separately solved constant-P/Q training/holdout validation. Charging
  omission requires explicit series-only authorization. Unsupported equivalents
  are rejected; independent whole-feeder OpenDSS validation remains separate.

## Search Contract
Ground-referenced S1/S2 support is implemented with per-leg bases, fingerprinted
service identities, protected transformer boundaries, complex differential-voltage
constraints, imbalance diagnostics, and v2 replay metadata. Grounded center taps
are an explicit default assumption on solver copies, with strict metadata mode
available. Independent OpenDSS service fixtures exercise unequal/leg-to-leg
loads, mutual coupling, stress, and reverse power. Center-tap no-load core losses
and complete one-based winding-pair indices are supported and independently
checked against OpenDSS. Explicit/floating neutrals remain unsupported. Exported
constant-P/Q devices are validated separately; feeder reference convergence and
independent whole-feeder validation remain separate gates.

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

Explicit/floating neutrals, arbitrary split-phase device physics, ZIP evaluation,
profile-driven time-series validation, meshed-input reduction, GPU acceleration,
batch percentage CLI export, and OPF disaggregation are deferred. The optional
flow extra enables reference
generation through the GDM adapter. True sparse assembly is an
upstream gdm-flow requirement before utility-scale benchmarks: its current
`sparse=True` path first allocates a dense matrix. Keep private utility data and
copyrighted paper PDFs outside this repository.