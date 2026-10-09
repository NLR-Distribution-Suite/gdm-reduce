# gdm-reduce

Phase-aware Kron reduction for distribution systems in the
[NLR Distribution Suite](https://github.com/NLR-Distribution-Suite).

This alpha implements exact source-conditioned electrical reduction and
phase-aware Opti-KRON candidate search with radial numerical equivalents. It uses
existing A/B/C bus-phase terminals, retains whole buses, preserves phase
coupling, and reconstructs eliminated voltages using sparse factorized solves.

**This is a prescribed-current numerical equivalent, not an aggregated
nonlinear power-flow model or an OPF feasibility certificate.** Reduced GDM
feeder export is not yet implemented. GDM metadata and fixed-configuration reference-scenario
ingestion are available through the optional gdm-flow integration.

## Install

Python 3.11 or newer is required.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pre-commit install
```

## Exact Reduction

```python
import numpy as np
from gdm_reduce import ElectricalNetwork, kron_reduce

network = ElectricalNetwork(
    admittance=np.array([[10, -10, 0], [-10, 20, -10], [0, -10, 10]]),
    labels=(("source", "A"), ("middle", "A"), ("load", "A")),
    source_voltages={("source", "A"): 120},
)
currents = np.array([0, -1, -2], dtype=complex)
result = kron_reduce(network, retained_buses={"source", "load"})
retained_voltage = result.solve(currents)
full_voltage = result.reconstruct(currents, retained_voltage)
```

Admittance is in siemens, voltages in volts, and currents in amperes, positive
into the passive network. Negative currents represent consumption. Currents
always use the original terminal indexing, even when passed to a reduced result.
Source current entries are ignored by voltage solves: source injections are
determined by the fixed voltages and can be obtained from `network.admittance @
full_voltage`. Multiple scenarios use a `(terminals, scenarios)` current array.

Every fixed-source bus must be retained. Disconnected terminals, singular or
excessively ill-conditioned blocks, nonfinite inputs, and neutral/split-phase
labels are rejected. Inputs are copied and reduction does not mutate them.
Kron equivalents can have significant fill-in even when the input is sparse.

## GDM Scenario Ingestion

```sh
python -m pip install -e '.[dev,flow]'
python examples/ingest_gdm.py ../gdm-flow/tests/data/ieee-13/gdm/ieee13_system.json
```

```python
from gdm.distribution import DistributionSystem
from gdm_reduce import build_gdm_scenario_library

system = DistributionSystem.from_json("model.json")
library = build_gdm_scenario_library(system, load_scales=[0.5, 1.0])
```

The library stores original terminal indexing, nominal phase-voltage bases,
bus/component UUID lineage, configuration fingerprints, and solver provenance.
Each scenario must converge, share the same source state and passive matrix,
and satisfy the voltage/current KCL consistency tolerance. The generator copies
the model and rejects backend hardware-state mutations. It uses the backend's
constant-P/Q semantics; ZIP/delta fidelity and OpenDSS baseline validation are
not established by this check. Time-series profiles are not interpreted yet.

`from_gdm_flow_results(system, results)` accepts a mapping of names to solved
results; callers must supply their original model. Matrices are taken from the
results, not independently restamped. Fingerprints include the full static GDM
component serialization, so models differing in UUIDs or static load settings
are conservatively rejected when mixing references. No utility data is copied.

External IEEE-13 integration tests discover the sibling gdm-flow data directory.
Set `GDM_REDUCE_TEST_DATA` to override it. They skip if the data or optional
backend is unavailable; ordinary consistency tests run without gdm-flow.

The supplied IEEE-13 model is exercised end to end. IEEE-123 must include its
referenced `ieee_123_node_time_series` companion storage before GDM can load it.
The supplied P4U model includes S1/S2 terminals and is explicitly outside the
current A/B/C adapter scope; no phases or storage metadata are silently dropped.

## Candidate Search

```sh
python examples/search_gdm.py ../gdm-flow/tests/data/ieee-13/gdm/ieee13_system.json
```

```python
from gdm_reduce import get_gdm_search_constraints, optimize_kron_reduction

constraints = get_gdm_search_constraints(system, library)
selected = optimize_kron_reduction(
    library,
    constraints.bus_edges,
    protected_buses=constraints.protected_buses,
    voltage_error_limit_pu=0.001,
    target_reduction_fraction=0.5,
)
retained_voltage = selected.equivalent.solve(selected.aggregated_currents)
```

Search enumerates adjacent whole-bus merges on a connected radial input graph.
The removed bus's phases must exist at the receiver, with matching nominal
phase-voltage bases. GDM topology comes from active physical equipment, not
off-diagonal admittance entries. Sources, transformer/regulator/switch boundaries,
DER, capacitors, and controller measurement buses are conservatively protected.
Caller-specified protected buses can be added to this set. Configuration checks
reject a GDM model changed since reference generation.

Candidates pool the accumulated cluster currents at the receiver. Algorithm 1's
SMICE objective sums cluster-wise maximum magnitude errors over scenarios;
a separate maximum original-terminal error enforces feasibility. Every step
checks all original terminals against the original references, not just surviving
buses. Candidate solves reuse one sparse LU in bounded batches. Ties within
1e-12 pu use receiver/removed bus-name order. This is exhaustive one-merge search,
not a globally optimal partition or a held-out/nonlinear accuracy guarantee.

`selected` contains clusters, complete name and UUID mappings, original-index
pooled currents, represented training voltages, step diagnostics, a stop reason,
and a `KronReductionResult`. **It is not a GDM `DistributionSystem`.** The eventual
export still requires supported equipment reconstruction and independent
nonlinear validation. Use `selected.aggregated_currents`, rather than the original
library currents, to solve the selected current-pooled equivalent.

## Radialization

```python
from gdm_reduce import radialize_reduction

radial = radialize_reduction(selected, constraints.bus_edges)
retained_voltage = radial.equivalent.solve(selected.aggregated_currents)
```

Eliminating a branching junction can turn a tree into a meshed Schur equivalent.
Radialization prunes non-root leaves from the original physical tree, restores
non-root branching junctions in the subtree connecting selected roots, and
suppresses degree-two paths. This is minimal structural restoration for that
tree, not a claim of globally minimal equipment reconstruction.

Restored junctions have zero pooled injection; they do not become new cluster
roots or receive reassigned devices. Original cluster/UUID mappings, currents,
and training scores remain in `radial.selection`. `radial.retained_buses` includes
both selected roots and restored junctions, so `radial.reduction_fraction` can
be smaller than the search reduction. The example reports both percentages.

Both original and radial matrix bus couplings must match their physical trees.
No matrix entries are discarded to conceal mesh edges. Selected-terminal complex
voltages must agree within 1e-7 V absolute and 1e-9 relative tolerance by default;
the result reports their maximum difference. Coupled and missing A/B/C phases
are preserved. This remains a numerical equivalent, not GDM feeder equipment
or nonlinear validation.

## Held-Out Validation

```python
from gdm_reduce import evaluate_reduction

holdout = build_gdm_scenario_library(system, load_scales=[0.75, 1.25])
report = evaluate_reduction(radial, holdout, voltage_error_limit_pu=0.001)
print(report.passed, report.max_voltage_magnitude_error_pu)
```

Evaluation freezes the selected clusters, applies them to new reference currents,
and checks the configuration fingerprint before solving. It does not re-optimize
on holdouts. Per-scenario reports separate magnitude, complex-voltage and wrapped
angle errors, full-network reference consistency, backend convergence mismatch,
and KCL residual. An angle error is `None` if a representative voltage is zero.
Power diagnostics use electrical reconstruction and actual `YV` currents, not
the ignored pooled source entries. Signed power differences are reduced minus
reference; passive active absorption includes every passive element in the matrix,
not just line losses. These are diagnostics, not thermal or nonlinear certificates.

The evaluator accepts training or held-out references; callers own the split.
Tests exercise unseen heavier demand, reverse flow and phase-specific changes,
including a stress case that passes training and fails the held-out threshold.

## CLI And Result Bundles

```sh
python -m pip install -e '.[dev,flow]'
gdm-reduce ../gdm-flow/tests/data/ieee-13/gdm/ieee13_system.json \
  --train-load-scales 0.5 1.0 --holdout-load-scales 0.75 1.25 \
  --voltage-error-limit-pu 0.001 --output results/ieee13.json
```

The CLI rejects overlapping training/holdout scales. It saves a failed validation
report rather than silently changing the clusters. Exit codes: `0` when both
reports pass the magnitude limit, `1` for a validation failure, `2` for invalid
arguments, incompatible configurations, solver failures or file errors. Without
`--output`, the JSON bundle is printed to stdout. Existing files require explicit
`--overwrite`. `python -m gdm_reduce.cli` is also available.

The `gdm-reduce-result-v1` bundle stores the reduced CSR matrix, fixed source
phasors, original/retained indexing, clusters and UUID mappings, restored junctions,
step history, reference and pooled currents, separate reports, and provenance.
It is explicitly labeled `radial-prescribed-current-equivalent`, with
`gdm_model_exported: false`. It is **not** accepted as a GDM feeder model.

```python
import json
from pathlib import Path
import numpy as np
from gdm_reduce import network_from_result_bundle, solve_voltages

bundle = json.loads(Path("results/ieee13.json").read_text())
network = network_from_result_bundle(bundle)
values = bundle["holdout"]["pooled_retained_currents"]
currents = np.array(values["real"]) + 1j * np.array(values["imag"])
retained_voltage = solve_voltages(network, currents)
```

`build_result_bundle()` and `save_result_bundle()` provide the same artifact API
for programmatic workflows. The decoder validates the numerical equivalent;
it does not certify manually edited reports or reconstruct original equipment.

Per-stage wall-clock timings, package/Python/platform versions, CPU count and
BLAS/OpenMP thread environment settings are recorded for reproducible CPU runs.
Compare repeated runs on matching hardware and thread settings; these are
single-run measurements, not hardware-independent performance guarantees.
Current dense-first gdm-flow assembly still prevents utility-scale claims.

IEEE-13 is exercised with 0.5/1.0 training and 0.75/1.25 holdout snapshots. This
does not establish independent OpenDSS baseline fidelity or nonlinear reduced
GDM device behavior; those release/export gates remain open.

## Development

```sh
pytest --cov=gdm_reduce --cov-report=term-missing
ruff check .
ruff format --check .
python -m build
python examples/exact_reduction.py
```

See [the numerical contract](docs/numerical-contract.md) for source-boundary
and current-transformation semantics, and [the implementation roadmap](docs/implementation-plan.md)
for the next milestones. GDM and gdm-flow compatibility does not yet imply
support for all devices or successful reference-solver validation.

## Research Basis

- Chevalier and Almassalkhi (2022), *Towards Optimal Kron-based Reduction Of
  Networks (Opti-KRON) for the Electric Power Grid*,
  [DOI: 10.1109/CDC51059.2022.9992730](https://doi.org/10.1109/CDC51059.2022.9992730).
- Mokhtari, Chevalier, and Almassalkhi (2026), *Optimal Kron-based Reduction of
  Networks (Opti-KRON) for Three-phase Distribution Feeders*,
  [arXiv:2510.19608v3](https://arxiv.org/abs/2510.19608v3).

The implementation is independently written from the mathematical formulation.
The papers and private utility datasets are not distributed with this package.

## License

BSD-3-Clause. See [LICENSE](LICENSE).