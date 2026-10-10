# Models, Conversion and Provenance

## Copied DiTTo Circuit Fixtures

The source is `ditto/tests/data/opendss_circuit_models`, the collection used by
DiTTo's reader tests. Five top-level primary circuits are converted:

| Circuit | GDM buses | Documentation verification |
| --- | ---: | --- |
| ieee13 | 16 | Round trip, reduction/export, reloaded held-out AC solve |
| P4U | 181 | Canonical GDM round trip |
| ckt7 | 1255 | Canonical GDM round trip |
| ckt24 | 6058 | Canonical GDM round trip |
| SFO | 3517 | Canonical GDM round trip |

Larger circuits are portability fixtures, not claims of converged Flow solutions
or supported reduction exports. Nested substation/feeder subexports are not duplicated.

`tests/data/circuits/<name>/model.json` and companion `model_time_series/`
directories are canonical GDM artifacts. Keep both together. Tests and notebook
load local files without downloads or runtime DiTTo conversion.

The fixture manifest records source DSS/CSV checksums, conversion versions,
bus counts and removed converter annotation classes. DiTTo-only supplemental
annotations are removed through GDM public APIs before serialization, keeping
database associations consistent. Electrical components, UUIDs, equipment,
settings and profiles are not normalized. These copies support GDM analysis,
not guaranteed lossless OpenDSS writer round trips after annotation removal.

## Refresh Copies

```sh
python examples/prepare_circuit_models.py \
  /path/to/ditto/tests/data/opendss_circuit_models --output tests/data/circuits
```

Use `--cases ieee13 P4U` for an explicitly selected smaller refresh. Preparation
calls `Reader(Master.dss).get_system()` directly, without editing source circuits
or bypassing validation. Failures stop preparation, not skipped tests. Review
refreshed UUIDs, profiles and provenance together.

## Acquire Published Models with gdmloader

Acquisition is separate from offline validation:

```python
from gdm.systems.distribution import DistributionSystem
from gdmloader.constants import GCS_CASE_SOURCE
from gdmloader.source import SystemLoader

loader = SystemLoader()
loader.add_source(GCS_CASE_SOURCE)
system = loader.load_dataset(
    system_type=DistributionSystem,
    source_name=GCS_CASE_SOURCE.name,
    dataset_name="p5r",
)
```

Save a portable local copy with the provided command:

```sh
python examples/acquire_gdm_model.py p5r --output results/acquired/p5r.json
```

Network access is required. Published supplemental annotations can require a
compatible DiTTo installation or local source on `PYTHONPATH`. The script records
loader/GDM versions and the saved JSON checksum. Inspect metadata before making
a stable fixture. The book comparison does not silently substitute a remotely
changed feeder for the copied IEEE-13 case.

## Comparison Artifacts

```sh
python examples/compare_gdm.py tests/data/circuits/ieee13/model.json \
  --output results/ieee13
```

Outputs include actual reduced GDM JSON and profile storage, export report,
per-terminal comparison JSON, native GDM interactive maps, and voltage PNG. Eliminated terminals are
represented by the matching root phase. Reduced voltages come from reloading
and separately solving relocated P/Q devices, not electrical reconstruction
mislabeled as equipment accuracy.

For a synchronized slider at 5% target increments:

```sh
python examples/compare_gdm.py tests/data/circuits/SFO/model.json \
  --primary-first --reference-tolerance 1e-7 --steps \
  --minimum-percent 5 --maximum-percent 60 --resume --output results/SFO-steps
```

`slider.html` contains the baseline/reduced native `DistributionSystem.plot()`
maps, terminal voltages, and voltage errors. `steps.json` records requested and
achieved reduction, bus counts, validation, and stopping reasons. Each target
has a separate `step-NN` model export. Targets run from 5% through 60% in 5%
increments. Saturated targets reuse the last achievable model and retain its
stopping reason; failed held-out validation stops the sweep with an error.
Use `--maximum-percent 25` to request a shorter sweep. Large feeders can require
substantial search time. Primary-first errors are relative to the saved primary
baseline, not the original secondary network. `--resume` reuses completed steps
in the same output directory; use it only with the same input model. Solve
settings are checked before reuse. The slider refreshes after each completed step.

### Measured SFO Results

The copied SFO model has 3,517 buses; GDM primary reduction produces the
832-bus baseline used for these comparisons. Training scales are 0.5 and 1.0;
the held-out scale is 0.75. All displayed exports pass the 0.01 pu magnitude-error
gate after reload and a separate constant-P/Q solve.

| Requested reduction | Exported buses | Maximum held-out magnitude error (pu) |
| --- | ---: | ---: |
| 5% | 790 | 0.00000225213 |
| 10% | 749 | 0.0000112545 |
| 15% | 708 | 0.0000264040 |
| 20% | 666 | 0.000174176 |
| 25% through 60% | 643 | 0.00113052 |

The search exhausts feasible merges at the 25% target. The achievable physical
reduction is approximately 22.7%, not 60%. Later slider positions intentionally
show the same validated model with `no_feasible_merge`. These errors are relative
to the primary baseline and do not certify the removed secondary network.