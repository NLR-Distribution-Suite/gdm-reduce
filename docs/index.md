# GDM-Reduce

GDM-Reduce constructs phase-aware equivalents of distribution networks and,
where equipment is realizable, exports a smaller grid-data-models (GDM) system.
This book separates exact elimination of prescribed currents, approximate
current-pooling decisions, and accuracy of a separately solved equipment model.

Start with [installation](installation.md), then run the
[original-versus-reduced comparison](comparison.ipynb). At the demonstrated
settings, reduction targets advance in 5% steps against the geographic SFO
primary baseline. The notebook reloads each saved GDM system and measures a
held-out operating point. One slider controls native GDM maps, voltage curves,
and error plots, with achieved bus counts and error reported at every step.

## Workflow

1. Acquire a GDM model or convert a tested OpenDSS circuit with DiTTo.
2. Solve fixed-configuration training scenarios on the original system.
3. Select connected, phase-compatible whole-bus clusters.
4. Restore necessary junctions to obtain a radial numerical equivalent.
5. Export supported equipment and verify its restamped admittance matrix.
6. Reload the export and solve its devices at distinct evaluation scenarios.

The [model catalog](models.md) describes copied circuit fixtures and gdmloader
acquisition. [Theory](theory.md) develops the equations and connects them to
the two Opti-KRON papers. The [numerical contract](numerical-contract.md) states
the detailed guarantees and limitations. [API and CLI](api.md) provides examples.

## What Is Not Certified

A current-pooling voltage threshold does not guarantee nonlinear accuracy,
losses, thermal loading, OPF feasibility, time-series behavior, or independent
OpenDSS fidelity. GDM references and export checks currently use gdm-flow's
static constant-P/Q semantics. Unsupported or non-realizable exports fail
rather than being silently approximated.