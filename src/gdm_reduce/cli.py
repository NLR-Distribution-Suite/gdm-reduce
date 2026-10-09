"""Generate a radial numerical equivalent and held-out validation artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .adapters import build_gdm_scenario_library, get_gdm_search_constraints
from .artifacts import build_result_bundle, save_result_bundle
from .radialization import radialize_reduction
from .search import optimize_kron_reduction


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--train-load-scales", nargs="+", type=float, default=[0.5, 1.0]
    )
    parser.add_argument(
        "--holdout-load-scales", nargs="+", type=float, default=[0.75, 1.25]
    )
    parser.add_argument("--voltage-error-limit-pu", type=float, default=0.001)
    parser.add_argument("--target-reduction", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--protect-bus", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if set(arguments.train_load_scales) & set(arguments.holdout_load_scales):
        parser.error("Training and holdout load scales must be disjoint.")
    if (
        arguments.output is not None
        and arguments.output.exists()
        and not arguments.overwrite
    ):
        parser.error("Output exists; choose another path or pass --overwrite.")
    timings = {}
    try:
        from gdm.distribution import DistributionSystem

        started = perf_counter()
        system = DistributionSystem.from_json(str(arguments.model))
        timings["model_load"] = perf_counter() - started
        provenance = {"model_path": str(arguments.model.resolve())}
        started = perf_counter()
        training = build_gdm_scenario_library(
            system,
            load_scales=arguments.train_load_scales,
            provenance={**provenance, "split": "training"},
        )
        timings["training_references"] = perf_counter() - started
        started = perf_counter()
        holdout = build_gdm_scenario_library(
            system,
            load_scales=arguments.holdout_load_scales,
            provenance={**provenance, "split": "holdout"},
        )
        timings["holdout_references"] = perf_counter() - started
        constraints = get_gdm_search_constraints(system, training)
        started = perf_counter()
        selected = optimize_kron_reduction(
            training,
            constraints.bus_edges,
            protected_buses=constraints.protected_buses
            | frozenset(arguments.protect_bus),
            voltage_error_limit_pu=arguments.voltage_error_limit_pu,
            target_reduction_fraction=arguments.target_reduction,
            max_iterations=arguments.max_iterations,
            batch_size=arguments.batch_size,
        )
        timings["search"] = perf_counter() - started
        started = perf_counter()
        radial = radialize_reduction(selected, constraints.bus_edges)
        timings["radialization"] = perf_counter() - started
        started = perf_counter()
        bundle = build_result_bundle(
            radial,
            training,
            holdout,
            voltage_error_limit_pu=arguments.voltage_error_limit_pu,
            timings_s=timings,
        )
        bundle["timings_s"]["evaluation_and_bundle"] = perf_counter() - started
        bundle["request"] = {
            "model_path": provenance["model_path"],
            "train_load_scales": arguments.train_load_scales,
            "holdout_load_scales": arguments.holdout_load_scales,
            "voltage_error_limit_pu": arguments.voltage_error_limit_pu,
            "target_reduction": arguments.target_reduction,
            "max_iterations": arguments.max_iterations,
            "batch_size": arguments.batch_size,
            "protected_buses": arguments.protect_bus,
        }
        if arguments.output is None:
            print(json.dumps(bundle, indent=2, allow_nan=False))
        else:
            save_result_bundle(arguments.output, bundle, overwrite=arguments.overwrite)
            print(f"Saved numerical equivalent: {arguments.output}")
            print(f"Radial buses: {len(radial.retained_buses)}")
            report = bundle["holdout"]["report"]
            print(
                "Holdout maximum magnitude error (pu): "
                f"{report['max_voltage_magnitude_error_pu']:.6g}"
            )
            print(f"Holdout passed: {report['passed']}")
        return (
            0
            if bundle["training"]["report"]["passed"]
            and bundle["holdout"]["report"]["passed"]
            else 1
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
