# gdm-reduce

Phase-aware Kron reduction for distribution systems in the
[NLR Distribution Suite](https://github.com/NLR-Distribution-Suite).

This alpha implements exact source-conditioned electrical reduction and
phase-aware Opti-KRON candidate search with radial numerical equivalents. It uses
existing A/B/C/S1/S2 bus-phase terminals, retains whole buses, preserves phase
coupling, and reconstructs eliminated voltages using sparse factorized solves.

**This is a prescribed-current numerical equivalent, not an aggregated
nonlinear power-flow model or an OPF feasibility certificate.** Supported radial
equivalents can now be exported as separate GDM equipment models, with independent
nonlinear validation. GDM metadata and fixed-configuration reference-scenario
ingestion are available through the optional gdm-flow integration.

## Install

Python 3.11+ and grid-data-models 2.4.0+ are required. Imports use the canonical
`gdm.systems.distribution` namespace. This is an alpha source release. From a
repository checkout, install the core package or the optional GDM power-flow
integration:

```sh
python -m pip install .
python -m pip install '.[flow]'
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
excessively ill-conditioned blocks, nonfinite inputs, and explicit neutral
labels are rejected. Inputs are copied and reduction does not mutate them.
Kron equivalents can have significant fill-in even when the input is sparse.

## GDM Scenario Ingestion

```sh
python examples/ingest_gdm.py /path/to/model.json
```

```python
from gdm.systems.distribution import DistributionSystem
from gdm_reduce import build_gdm_scenario_library

system = DistributionSystem.from_json("model.json")
library = build_gdm_scenario_library(system, load_scales=[0.5, 1.0])
```

The library stores original terminal indexing, nominal phase-voltage bases,
bus/component UUID lineage, configuration fingerprints, and solver provenance.
Each scenario must converge, share the same source state and passive matrix,
and satisfy the voltage/current KCL consistency tolerance. The generator copies
the model and rejects backend hardware-state mutations. It uses the backend's
constant-P/Q semantics; ZIP evaluation and independent whole-feeder OpenDSS
baseline validation are not established by this check. Supported grounded
split-phase line-to-line loads are described below. Time-series profiles are
not interpreted by the scenario generator.

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
Center taps are treated as grounded under the assumption below. The updated
stamper includes nonnegative no-load core losses and accepts complete zero- or
one-based winding-pair indices. These features do not by themselves establish
reference convergence or independent baseline fidelity for every feeder.

## GDM 2.4 Station Equipment

With the updated gdm-flow backend, series reactors and supported two-winding
station transformers participate in AC references. Reactor, transformer, and
switch terminals are protected during candidate selection. Export retains these
devices, their concrete equipment types, UUIDs, and nameplate metadata; only
plain matrix-impedance line paths are synthesized or contracted. Station breaker,
disconnector, and earthing equipment retain their static switch states. Protection
trips, thermal behavior, and unsupported transformer layouts are not simulated.

## Ground-Referenced Split Phase

S1/S2 references require the updated companion gdm-flow center-tap stamper and
connection-aware power-flow solver. Install the updated sibling during development
with `python -m pip install -e ../gdm-flow`; its version number alone does not
guarantee the required split-phase capabilities.

By default, `build_gdm_scenario_library()` assumes all supported center taps are
grounded, including when GDM winding flags say otherwise. Only the solver's deep
copy is normalized; input models and files remain unchanged. The assumption is
recorded as `assumed-grounded` in scenario provenance, service-domain fingerprints,
and result bundles. It is a modeling assumption, not verified grounding evidence.
Pass `assume_center_taps_grounded=False`, or CLI `--require-grounding-metadata`,
to require the original secondary winding grounding flags instead.

Supported services have one A/B/C primary and two single-phase STAR secondary
windings sharing a center-tap bus. Explicit N, floating neutrals, ambiguous service
connections and unsupported device connections remain rejected. No-load core
loss uses a tap-adjusted winding-two conductance, checked against OpenDSS.
Secondary winding order and S2/S1 bus order are supported;
missing legs are not padded. Split line-to-line bus ratings are divided by two,
not square root of three. Transformer/primary UUIDs and primary phase identify
each service domain; current pooling cannot cross domains or nominal bases.

Grounded STAR loads use per-leg constant P/Q. A two-leg DELTA load uses the sum
of its GDM phase-load powers as one S1-to-S2 device, with opposite currents computed
from the actual differential voltage. ZIP coefficients are not evaluated.
Independent OpenDSS service tests cover winding primitives, mutual line coupling,
unequal legs, leg-to-leg loads, heavy loading, and reverse power. They do not
certify an arbitrary feeder or reduced nonlinear device model.

Selection and holdouts enforce complex S1-minus-S2 voltage error as well as leg
magnitude limits. Differential voltage uses the sum of the leg bases. Reports
include differential magnitude/complex error and signed leg-imbalance differences.
`audit_split_phase_grounding(system)` inspects original flags without modifying
them, even when the generator uses the grounded assumption.

## Candidate Search

```sh
python examples/search_gdm.py /path/to/model.json
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
phase-voltage bases and split-phase service domains. GDM topology comes from active physical equipment, not
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
and a `KronReductionResult`. **It is not a GDM `DistributionSystem`.** Use the
separate equipment exporter after radialization and independently validate its
devices. Use `selected.aggregated_currents`, rather than the original
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
the result reports their maximum difference. Coupled and missing A/B/C/S1/S2 phases
are preserved. This remains a numerical equivalent, not GDM feeder equipment
or nonlinear validation.

## Reduced GDM Export

```python
from gdm_reduce import build_reduced_gdm, save_reduced_gdm, validate_reduced_gdm

exported = build_reduced_gdm(system, radial, library)
validation = validate_reduced_gdm(exported, holdout, split="holdout")
save_reduced_gdm(exported, "results/reduced.json")
```

The exporter deep-copies the original system, retains existing bus UUIDs and
protected transformers/regulators/switches/DER, prunes removed line stubs, and
replaces degree-two matrix-impedance paths with reciprocal passive series-line
equipment. New lines have deterministic UUIDs, summed path length, conservative
minimum ampacity and original-line UUID provenance. It independently restamps
the exported model and rejects a Y-bus mismatch with the radial equivalent
(default absolute tolerance 1e-6 S, relative tolerance 1e-9).

Loads move to their assigned roots without merging devices: UUIDs, phases,
individual P/Q, ZIP coefficients and attached time-series profiles survive the
GDM serialization round trip. Original models are not modified. Center-tap
grounding assumptions are applied only to the exported copy and recorded.
Profiles or supplemental metadata attached to removed buses/lines are rejected
rather than silently discarded. Contracted geometry/sequence lines, parallel
paths, nonreciprocal/nonpassive equivalents and non-realizable Schur shunts are
not supported.

Nonzero charging on contracted/pruned lines is not silently dropped. An
explicit `allow_series_only=True` acknowledges its omission, consistent with
the series-only reference workflow. The exported Y-bus must still match; this
option cannot turn a charged Schur equivalent into a pure series model. This
limitation matters for feeders with nonzero capacitance. Uncontracted equipment
remains unchanged.

`validate_reduced_gdm()` solves the exported devices at recorded static load and
solar scales and compares original-terminal representative voltages separately
from current-pooled errors. It records convergence, magnitude/complex errors and
S1-S2 complex error in the export report. Missing scale provenance is rejected.
Its backend is gdm-flow's constant-P/Q solver; this is not independent OpenDSS,
ZIP, profile-driven time-series, thermal or OPF validation. A failed solve or
voltage limit fails validation. Programmatic callers own the training/holdout
split; use explicit split labels when recording multiple evaluations.

The model is canonical GDM JSON, including its companion time-series storage.
Assumptions, mappings, path provenance and validation are saved separately as
`reduced.export-report.json`; neither file is overwritten by default.

```sh
gdm-reduce model.json --target-reduction 0.5 \
  --output results/numerical.json --export-gdm results/reduced.json
```

For an explicitly authorized series-only export of charged contracted lines,
add `--export-series-only`. The CLI independently validates exported-device
training and holdout snapshots, saves failing reports and exits `1` if either
numerical or exported-device validation fails. Input/model/report paths must
be distinct, even with `--overwrite`.

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
gdm-reduce /path/to/model.json \
  --train-load-scales 0.5 1.0 --holdout-load-scales 0.75 1.25 \
  --voltage-error-limit-pu 0.001 --target-reduction 0.5 \
  --output results/numerical.json --export-gdm results/reduced.json
```

`--target-reduction` accepts one fraction between 0 and 1, not a percentage list.
Search is accuracy-limited; structural protections and restored junctions can
make actual radial bus removal smaller than the requested target. Batch
percentage export and automatic primary-only preprocessing are not CLI features.

The CLI rejects overlapping training/holdout scales. It saves a failed validation
report rather than silently changing the clusters. Exit codes: `0` when both
numerical reports and any requested exported-device validation pass the magnitude
and applicable S1-S2 complex limits, `1` for a validation failure, `2` for invalid
arguments, incompatible configurations, solver failures or file errors. Without
`--output`, the JSON bundle is printed to stdout. Existing files require explicit
`--overwrite`. `python -m gdm_reduce.cli` is also available.

The `gdm-reduce-result-v1` bundle stores the reduced CSR matrix, fixed source
phasors, original/retained indexing, clusters and UUID mappings, restored junctions,
step history, reference and pooled currents, separate reports, and provenance.
It is explicitly labeled `radial-prescribed-current-equivalent`, with
`gdm_model_exported: false` unless a separate CLI GDM export was saved. It is
**not** accepted as a GDM feeder model, even when it references that export.
Split-phase bundles use `gdm-reduce-result-v2`, adding service domains, the
ground-reference contract, and grounding-policy provenance. Existing ABC v1
bundles remain readable; split-phase data without v2 metadata is rejected.

```python
import json
from pathlib import Path
import numpy as np
from gdm_reduce import network_from_result_bundle, solve_voltages

bundle = json.loads(Path("results/numerical.json").read_text())
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

IEEE-13 is exercised with 0.5/1.0 training and 0.75/1.25 holdout snapshots.
Exported constant-P/Q devices are separately solved and checked; independent
whole-feeder OpenDSS validation and operational certification remain outside
these checks.

## Development

```sh
python -m pip install -e '.[dev,flow,opendss]'
pre-commit install
pytest --cov=gdm_reduce --cov-report=term-missing
ruff check .
ruff format --check .
python -m build
python examples/exact_reduction.py
```

See [the numerical contract](docs/numerical-contract.md) for source-boundary
and current-transformation semantics, and [scope and validation](docs/implementation-plan.md)
for implemented features and remaining limits. GDM and gdm-flow compatibility does not imply
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