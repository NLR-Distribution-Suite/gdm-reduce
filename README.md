# gdm-reduce

Phase-aware Kron reduction for distribution systems in the
[NLR Distribution Suite](https://github.com/NLR-Distribution-Suite).

This alpha implements exact source-conditioned electrical reduction. It uses
existing A/B/C bus-phase terminals, retains whole buses, preserves phase
coupling, and reconstructs eliminated voltages using sparse factorized solves.

**This is a prescribed-current numerical equivalent, not an aggregated
nonlinear power-flow model or an OPF feasibility certificate.** Opti-KRON
candidate selection, GDM adapters, radialization, and equipment export are
planned but not yet implemented.

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