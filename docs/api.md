# API and Command Line

## Exact Numerical Reduction

```python
import numpy as np
from gdm_reduce import ElectricalNetwork, kron_reduce

network = ElectricalNetwork(
    admittance=np.array([[10, -10, 0], [-10, 20, -10], [0, -10, 10]]),
    labels=(("source", "A"), ("middle", "A"), ("load", "A")),
    source_voltages={("source", "A"): 120},
)
currents = np.array([0, -1, -2], dtype=complex)
equivalent = kron_reduce(network, retained_buses={"source", "load"})
retained_voltage = equivalent.solve(currents)
original_voltage = equivalent.reconstruct(currents, retained_voltage)
```

Current inputs always have original indexing. `solve()` returns retained
indexing; `reconstruct()` returns original indexing. Multiple snapshots use
`(terminals, scenarios)` arrays. Source current entries are not voltage boundaries.

## GDM Selection and Equipment Export

Run from the repository root:

```python
from pathlib import Path
from gdm.systems.distribution import DistributionSystem
from gdm_reduce import (
    build_gdm_scenario_library, get_gdm_search_constraints,
    optimize_kron_reduction, radialize_reduction,
)
from gdm_reduce.gdm_export import (
    build_reduced_gdm, save_reduced_gdm, validate_reduced_gdm,
)

system = DistributionSystem.from_json(Path("tests/data/circuits/ieee13/model.json"))
training = build_gdm_scenario_library(system, load_scales=(0.5, 1.0))
constraints = get_gdm_search_constraints(system, training)
selection = optimize_kron_reduction(
    training, constraints.bus_edges,
    protected_buses=constraints.protected_buses,
    voltage_error_limit_pu=0.01, target_reduction_fraction=0.5,
)
radial = radialize_reduction(selection, constraints.bus_edges)
export = build_reduced_gdm(system, radial, training, allow_series_only=True)
heldout = build_gdm_scenario_library(system, load_scales=(0.75,))
validation = validate_reduced_gdm(export, heldout, voltage_error_limit_pu=0.01)
assert validation["passed"]
save_reduced_gdm(export, Path("results/reduced.json"))
```

| API | Contract |
| --- | --- |
| `build_gdm_scenario_library` | Copied-model static P/Q references with fixed hardware, source state, bases and provenance |
| `from_gdm_flow_results` | Ingest solved Flow snapshots with original model and matching declared assumptions |
| `get_gdm_search_constraints` | Physical edges and protected equipment boundaries |
| `optimize_kron_reduction` | Adjacent whole-bus current pooling, original-terminal errors, SMICE ranking |
| `radialize_reduction` | Restore junctions and verify selected-root complex voltage agreement |
| `build_reduced_gdm` | Deep-copy passive matrix-line export with restamped Y-bus verification |
| `validate_reduced_gdm` | Separate relocated-device solves; convergence and voltage pass flags |
| `save_reduced_gdm` | Canonical JSON, profile storage and separate export report |

## CLI and Examples

```sh
gdm-reduce --help
python examples/search_gdm.py tests/data/circuits/ieee13/model.json \
  --load-scales 0.5 1.0 --voltage-error-limit-pu 0.01 --target-reduction 0.5
python examples/compare_gdm.py tests/data/circuits/ieee13/model.json \
  --output results/comparison
```

The comparison example is shared by the notebook and its regression test.
It reloads saved equipment before validation and plotting. Its 0.01 pu limit
and 50% target are demonstration settings, not defaults or guaranteed outcomes.

## Failure and Deployment Gates

Singular source-fixed blocks, explicit neutrals, incompatible phase/base/domain
assignments, nonradial graphs, unsupported contracted equipment, discarded
metadata and non-realizable shunts are rejected. Export does not erase hardware
to force success. `assume_center_taps_grounded=False` requires original flags.

Inspect termination reason, achieved radial reduction, restoration count,
charging omissions, Y-bus verification, nonlinear mismatch and voltage errors.
Convergence is not accuracy; validation must pass. Callers own train/holdout
independence. See the numerical contract for detailed guarantees.