"""Select phase-aware clusters from static GDM reference snapshots."""

import argparse
from pathlib import Path

from gdm.distribution import DistributionSystem

from gdm_reduce import (
    build_gdm_scenario_library,
    get_gdm_search_constraints,
    optimize_kron_reduction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--load-scales", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--voltage-error-limit-pu", type=float, default=0.001)
    parser.add_argument("--target-reduction", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int)
    arguments = parser.parse_args()
    system = DistributionSystem.from_json(str(arguments.model))
    library = build_gdm_scenario_library(
        system,
        load_scales=arguments.load_scales,
        provenance={"model_path": str(arguments.model.resolve())},
    )
    constraints = get_gdm_search_constraints(system, library)
    selected = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=arguments.voltage_error_limit_pu,
        target_reduction_fraction=arguments.target_reduction,
        max_iterations=arguments.max_iterations,
    )
    retained_voltage = selected.equivalent.solve(selected.aggregated_currents)
    print(f"Original/retained buses: {len(library.bus_uuids)}/{len(selected.clusters)}")
    print(f"Protected buses: {len(selected.protected_buses)}")
    print(f"Reduction: {selected.reduction_fraction:.1%}")
    print(f"Maximum training magnitude error (pu): {selected.max_voltage_error_pu:.6g}")
    print(f"SMICE (pu): {selected.smice:.6g}")
    print(f"Stop reason: {selected.termination_reason}")
    print(f"Retained terminals/scenarios: {retained_voltage.shape}")
    print(f"UUID mappings: {len(selected.bus_uuid_to_retained_uuid)}")
    for step in selected.history:
        print(
            f"  {step.removed_bus} -> {step.receiver_bus}: "
            f"max error {step.max_voltage_error_pu:.6g} pu; "
            f"{step.feasible_candidates}/{step.evaluated_candidates} feasible"
        )
    print("Numerical equivalent only; reduced GDM feeder export is not implemented.")


if __name__ == "__main__":
    main()
