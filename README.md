# gdm-reduce

Phase-aware Kron reduction for distribution systems in the
[NLR Distribution Suite](https://github.com/NLR-Distribution-Suite).

This alpha implements exact source-conditioned electrical reduction. It uses
existing A/B/C bus-phase terminals, retains whole buses, preserves phase
coupling, and reconstructs eliminated voltages using sparse factorized solves.

**This is a prescribed-current numerical equivalent, not an aggregated
nonlinear power-flow model or an OPF feasibility certificate.** Opti-KRON
candidate selection, radialization, and equipment export are planned but not
yet implemented. GDM metadata and fixed-configuration reference-scenario
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