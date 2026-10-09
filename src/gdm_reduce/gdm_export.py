"""Export realizable radial equivalents as copied GDM equipment models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import networkx as nx
import numpy as np
from gdm.distribution import DistributionSystem
from gdm.distribution.components import (
    DistributionBus,
    DistributionLoad,
    DistributionTransformer,
    MatrixImpedanceBranch,
)
from gdm.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)
from gdm.distribution.components.base.distribution_switch_base import (
    DistributionSwitchBase,
)
from gdm.distribution.equipment import MatrixImpedanceBranchEquipment

from .adapters import get_gdm_search_constraints
from .radialization import RadializationResult
from .scenarios import ScenarioLibrary, split_phase_voltage_errors
from .validation import pool_scenario_currents


@dataclass(frozen=True)
class GDMModelExport:
    system: DistributionSystem
    report: dict


def build_reduced_gdm(
    system: DistributionSystem,
    result: RadializationResult,
    library: ScenarioLibrary,
    *,
    allow_series_only: bool = False,
    ybus_atol_s: float = 1e-6,
    ybus_rtol: float = 1e-9,
) -> GDMModelExport:
    """Build a GDM feeder; load relocation is not current pooling.

    Original load UUIDs, equipment, P/Q, ZIP coefficients, and time series are
    preserved. Exported-model nonlinear accuracy must be evaluated separately.
    Charged contracted lines require explicit series-only authorization.
    """
    from gdm_flow.ybus import calculate_ybus

    if any(not np.isfinite(value) or value < 0 for value in (ybus_atol_s, ybus_rtol)):
        raise ValueError("Y-bus tolerances must be finite and nonnegative.")
    constraints = get_gdm_search_constraints(system, library)
    pool_scenario_currents(result.selection, library)
    retained = set(result.retained_buses)
    if not constraints.protected_buses <= retained:
        raise ValueError("Export must retain protected equipment boundaries.")
    tree = nx.Graph(constraints.bus_edges)
    working = system.deepcopy()
    buses = {bus.name: bus for bus in working.get_components(DistributionBus)}
    mapping = dict(result.selection.original_to_retained)
    if set(mapping) != set(buses) or not set(mapping.values()) <= retained:
        raise ValueError("Export bus mapping must cover the original GDM buses.")
    for bus in retained:
        if bus not in buses:
            raise ValueError("Retained bus is missing from the original GDM model.")
    edge_components = {}
    modified_lines = []
    for component in working.get_components(DistributionBranchBase):
        edge = tuple(sorted(bus.name for bus in component.buses))
        if component.in_service:
            if edge in edge_components:
                raise ValueError(
                    "Parallel branches are not supported by radial export."
                )
            edge_components[edge] = component
        if not all(bus.name in retained for bus in component.buses):
            if isinstance(component, DistributionSwitchBase):
                raise ValueError("Export cannot remove a switching-device boundary.")
            if not isinstance(component, MatrixImpedanceBranch):
                raise ValueError(
                    "Contracted lines must use matrix-impedance equipment."
                )
            if any(
                working.has_time_series(owner)
                or working.has_supplemental_attribute(owner)
                for owner in (component, component.equipment)
            ):
                raise ValueError(
                    "Cannot discard contracted-line time series or attributes."
                )
            modified_lines.append(component)
    charged = [
        str(line.uuid)
        for line in modified_lines
        if np.any(line.equipment.c_matrix.magnitude)
    ]
    if charged and not allow_series_only:
        raise ValueError(
            "Contracted lines have charging; authorize with allow_series_only=True."
        )
    reassigned = {}
    for component in working.get_components(DistributionLoad):
        destination = buses[mapping[component.bus.name]]
        if destination.uuid != component.bus.uuid:
            if not set(component.phases) <= set(destination.phases):
                raise ValueError(
                    "Relocated load phases are unavailable at its destination."
                )
            reassigned[str(component.uuid)] = {
                "original_bus_uuid": str(component.bus.uuid),
                "retained_bus_uuid": str(destination.uuid),
            }
            component.bus = destination
    for component in working.get_components():
        if isinstance(
            component, DistributionBus | DistributionLoad | DistributionBranchBase
        ):
            continue
        attachments = (
            [component.bus]
            if isinstance(getattr(component, "bus", None), DistributionBus)
            else []
        )
        attachments += list(getattr(component, "buses", []))
        if any(
            isinstance(bus, DistributionBus) and bus.name not in retained
            for bus in attachments
        ):
            raise ValueError(
                f"Cannot relocate unsupported component {component.name!r}."
            )
    grounding = any(
        scenario.provenance.get("center_tap_grounding") == "assumed-grounded"
        for scenario in library.scenarios
    )
    if grounding:
        for transformer in working.get_components(DistributionTransformer):
            if transformer.in_service and any(
                phase.value in {"S1", "S2"}
                for group in transformer.winding_phases
                for phase in group
            ):
                for winding in transformer.equipment.windings[1:]:
                    winding.is_grounded = True
    labels = result.equivalent.retained_labels
    indices = {label: index for index, label in enumerate(labels)}
    matrix = result.equivalent.reduced_network.admittance
    lineages = {}
    for first, second in result.bus_edges:
        path = nx.shortest_path(tree, first, second)
        if len(path) == 2:
            continue
        originals = [
            edge_components[tuple(sorted((start, end)))]
            for start, end in zip(path, path[1:])
        ]
        if any(
            not isinstance(line, MatrixImpedanceBranch)
            or isinstance(line, DistributionSwitchBase)
            for line in originals
        ):
            raise ValueError("Only plain matrix-impedance lines can be combined.")
        template = originals[0]
        phases = [
            phase
            for phase in buses[first].phases
            if phase in buses[second].phases
            and (first, phase.value) in indices
            and abs(
                matrix[indices[(first, phase.value)], indices[(second, phase.value)]]
            )
            > 0
        ]
        if not phases:
            raise ValueError("Equivalent line has no matching terminal phases.")
        first_indices = [indices[(first, phase.value)] for phase in phases]
        second_indices = [indices[(second, phase.value)] for phase in phases]
        series = -matrix[np.ix_(first_indices, second_indices)].toarray()
        try:
            impedance = np.linalg.inv(series)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Equivalent line has singular series admittance.") from exc
        if not np.allclose(impedance, impedance.T, rtol=1e-9, atol=1e-9):
            raise ValueError("Equivalent line is not reciprocal.")
        if np.min(np.linalg.eigvalsh((impedance.real + impedance.real.T) / 2)) < -1e-9:
            raise ValueError("Equivalent line is not passive.")
        length = sum(float(line.length.to("m").magnitude) for line in originals)
        if not np.isfinite(length) or length <= 0:
            raise ValueError(
                "Equivalent path must have a positive finite physical length."
            )
        identity = ":".join(str(line.uuid) for line in originals)
        equipment = MatrixImpedanceBranchEquipment(
            name=f"reduced-equipment-{first}-{second}",
            uuid=uuid5(NAMESPACE_URL, f"gdm-reduce:equipment:{identity}"),
            construction=template.equipment.construction,
            r_matrix=type(template.equipment.r_matrix)(
                impedance.real / length, "ohm/m"
            ),
            x_matrix=type(template.equipment.x_matrix)(
                impedance.imag / length, "ohm/m"
            ),
            c_matrix=type(template.equipment.c_matrix)(
                np.zeros_like(impedance.real), "farad/m"
            ),
            ampacity=type(template.equipment.ampacity)(
                min(
                    float(line.equipment.ampacity.to("A").magnitude)
                    for line in originals
                ),
                "A",
            ),
        )
        equivalent = MatrixImpedanceBranch(
            name=f"reduced-line-{first}-{second}",
            uuid=uuid5(NAMESPACE_URL, f"gdm-reduce:line:{identity}"),
            buses=[buses[first], buses[second]],
            phases=phases,
            length=type(template.length)(length, "m"),
            equipment=equipment,
            feeder=template.feeder,
            substation=template.substation,
        )
        working.add_component(equivalent)
        lineages[str(equivalent.uuid)] = [str(line.uuid) for line in originals]
    working.rebuild_component_associations()
    for line in modified_lines:
        working.remove_component(line, cascade_down=False)
    working.rebuild_component_associations()
    for name, bus in buses.items():
        if name not in retained:
            if working.has_time_series(bus) or working.has_supplemental_attribute(bus):
                raise ValueError(
                    "Cannot discard removed-bus time series or supplemental attributes."
                )
            working.remove_component(bus, cascade_down=False)
    working.rebuild_component_associations()
    stamped = calculate_ybus(working, include_shunt=False)
    order = [stamped.label_to_index[label] for label in labels]
    if set(stamped.index_to_label) != set(labels):
        raise ValueError(
            "Exported GDM terminal set differs from the radial equivalent."
        )
    observed = stamped.ybus[np.ix_(order, order)]
    expected = matrix.toarray()
    error = float(np.max(np.abs(observed - expected)))
    if not np.allclose(observed, expected, rtol=ybus_rtol, atol=ybus_atol_s):
        raise ValueError(
            f"Equivalent is not realizable as GDM series lines (error {error:g} S)."
        )
    return GDMModelExport(
        working,
        {
            "kind": "gdm-radial-equipment-model",
            "gdm_model_exported": True,
            "configuration_fingerprint": library.configuration_fingerprint,
            "original_buses": len(buses),
            "retained_buses": len(retained),
            "bus_uuid_to_retained_uuid": dict(
                result.selection.bus_uuid_to_retained_uuid
            ),
            "equivalent_branch_lineage": lineages,
            "relocated_loads": reassigned,
            "charging_policy": "series-only" if charged else "unchanged",
            "omitted_charging_branch_uuids": charged,
            "center_tap_grounding": "assumed-grounded"
            if grounding
            else "model-metadata",
            "load_policy": "preserve individual device P/Q, ZIP, UUIDs and time series",
            "prescribed_current_certificate_applies": False,
            "nonlinear_accuracy_verified": False,
            "ybus_verification": {
                "include_shunt": False,
                "max_absolute_error_s": error,
                "absolute_tolerance_s": ybus_atol_s,
                "relative_tolerance": ybus_rtol,
            },
        },
    )


def validate_reduced_gdm(
    export: GDMModelExport,
    library: ScenarioLibrary,
    *,
    voltage_error_limit_pu: float = 0.001,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
    split: str = "evaluation",
) -> dict:
    """Solve exported devices at recorded static load/solar scales.

    This evaluates gdm-flow's constant-P/Q model, not independent OpenDSS
    fidelity or time-series/ZIP behavior. Failed solves fail the report.
    """
    from gdm_flow.ac_pf import solve_ac_power_flow_from_components

    if library.configuration_fingerprint != export.report["configuration_fingerprint"]:
        raise ValueError(
            "Nonlinear references must match the original export configuration."
        )
    if not np.isfinite(voltage_error_limit_pu) or voltage_error_limit_pu < 0:
        raise ValueError("Voltage error limit must be finite and nonnegative.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Solver tolerance must be finite and positive.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("Solver iteration count must be a positive integer.")
    if not isinstance(split, str) or not split:
        raise ValueError("Validation split must be a nonempty string.")
    records = []
    uuid_names = {uuid: name for name, uuid in library.bus_uuids.items()}
    representatives = {
        uuid_names[original]: uuid_names[retained]
        for original, retained in export.report["bus_uuid_to_retained_uuid"].items()
    }
    source_bus = export.system.get_source_bus().name
    for index, scenario in enumerate(library.scenarios):
        provenance = scenario.provenance
        if "load_scale" not in provenance or "solar_scale" not in provenance:
            raise ValueError(
                "Nonlinear validation requires load/solar scale provenance."
            )
        load_scale, solar_scale = (
            float(provenance["load_scale"]),
            float(provenance["solar_scale"]),
        )
        if any(
            not np.isfinite(value) or value < 0 for value in (load_scale, solar_scale)
        ):
            raise ValueError(
                "Nonlinear scenario scales must be finite and nonnegative."
            )
        if provenance.get("include_shunt", "False") != "False":
            raise ValueError(
                "Nonlinear validation currently supports series-only references."
            )
        retained = {bus.name for bus in export.system.get_components(DistributionBus)}
        warm = {
            label: complex(library.voltage[position, index])
            for position, label in enumerate(library.network.labels)
            if label[0] in retained and label[0] != source_bus
        }
        solved = solve_ac_power_flow_from_components(
            export.system.deepcopy(),
            load_scale=load_scale,
            solar_scale=solar_scale,
            include_shunt=False,
            include_capacitor=provenance.get("include_capacitor", "True") == "True",
            v0_complex=warm,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        record = {
            "name": scenario.name,
            "load_scale": load_scale,
            "solar_scale": solar_scale,
            "converged": bool(solved.success),
            "message": solved.message,
            "passed": False,
        }
        if solved.success:
            for label, expected in library.network.source_voltages.items():
                observed = solved.voltage[solved.ybus_result.label_to_index[label]]
                if not np.isclose(observed, expected, rtol=1e-10, atol=1e-8):
                    raise ValueError(
                        "Exported-model solve changed the reference source phasors."
                    )
            represented = np.array(
                [
                    solved.voltage[
                        solved.ybus_result.label_to_index[(representatives[bus], phase)]
                    ]
                    for bus, phase in library.network.labels
                ]
            )
            reference = library.voltage[:, index]
            magnitude = float(
                np.max(
                    np.abs(np.abs(represented) - np.abs(reference))
                    / library.voltage_bases
                )
            )
            complex_error = float(
                np.max(np.abs(represented - reference) / library.voltage_bases)
            )
            differential = float(
                split_phase_voltage_errors(
                    library, represented[:, None], reference[:, None]
                )[1][0]
            )
            record.update(
                {
                    "max_voltage_magnitude_error_pu": magnitude,
                    "max_complex_voltage_error_pu": complex_error,
                    "max_line_to_line_complex_error_pu": differential,
                    "max_mismatch_pu": float(solved.max_mismatch_pu),
                    "passed": magnitude <= voltage_error_limit_pu
                    and differential <= voltage_error_limit_pu,
                }
            )
        records.append(record)
    report = {
        "backend": "gdm-flow",
        "device_model": "constant-P/Q",
        "voltage_error_limit_pu": voltage_error_limit_pu,
        "passed": all(record["passed"] for record in records),
        "scenarios": records,
    }
    validations = export.report.setdefault("nonlinear_validation", {})
    validations[split] = report
    export.report["nonlinear_solves_converged"] = all(
        record["converged"]
        for item in validations.values()
        for record in item["scenarios"]
    )
    export.report["nonlinear_accuracy_verified"] = all(
        item["passed"] for item in validations.values()
    )
    return report


def save_reduced_gdm(
    export: GDMModelExport, path: str | Path, *, overwrite: bool = False
) -> None:
    """Use GDM serialization, including its companion time-series store."""
    destination = Path(path)
    report_path = destination.with_name(f"{destination.stem}.export-report.json")
    if not overwrite and (destination.exists() or report_path.exists()):
        raise FileExistsError("Reduced GDM model or export report already exists.")
    report_content = json.dumps(export.report, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    export.system.to_json(destination, overwrite=overwrite, indent=2)
    with report_path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(report_content)
