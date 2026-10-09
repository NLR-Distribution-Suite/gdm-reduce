"""Build a fixed-configuration reference library from an external GDM model."""

import argparse
from pathlib import Path

import numpy as np
from gdm.distribution import DistributionSystem

from gdm_reduce import build_gdm_scenario_library, kron_reduce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--load-scales", nargs="+", type=float, default=[0.5, 1.0])
    arguments = parser.parse_args()
    system = DistributionSystem.from_json(str(arguments.model))
    library = build_gdm_scenario_library(
        system,
        load_scales=arguments.load_scales,
        provenance={"model_path": str(arguments.model.resolve())},
    )
    result = kron_reduce(library.network, {system.get_source_bus().name})
    reconstructed = result.reconstruct(library.currents, result.solve(library.currents))
    error = float(np.max(np.abs(reconstructed - library.voltage)))
    np.testing.assert_allclose(reconstructed, library.voltage, rtol=1e-9, atol=1e-7)
    print(f"Original buses: {len(library.bus_uuids)}")
    print(f"Terminals/scenarios: {library.voltage.shape}")
    print(f"Configuration fingerprint: {library.configuration_fingerprint}")
    print(
        f"Maximum KCL consistency residual (A): {max(library.max_kcl_residual_a):.3g}"
    )
    print(f"Maximum exact reconstruction error (V): {error:.3g}")
    print(
        "Reference snapshots only; not independent OpenDSS or reduced-model validation."
    )


if __name__ == "__main__":
    main()
