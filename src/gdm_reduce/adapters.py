"""Public GDM metadata and gdm-flow result ingestion without matrix restamping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import numpy as np

from .models import BusPhaseLabel, ElectricalNetwork
from .scenarios import (
    ReferenceScenario,
    ScenarioLibrary,
    configuration_fingerprint,
)

if TYPE_CHECKING:
    from gdm.distribution import DistributionSystem
    from gdm_flow.ac_pf import ACPowerFlowResult


def _model_signature(system: DistributionSystem) -> str:
    components = sorted(system.iter_all_components(), key=lambda item: str(item.uuid))
    records = [
        {"type": type(item).__name__, "data": item.model_dump(mode="json")}
        for item in components
    ]
    payload = json.dumps(records, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _metadata(
    system: DistributionSystem, labels: Sequence[BusPhaseLabel]
) -> tuple[np.ndarray, dict[str, str], dict[str, tuple[str, ...]]]:
    from gdm.distribution.components import DistributionBus
    from gdm.distribution.enums import VoltageTypes

    buses = {bus.name: bus for bus in system.get_components(DistributionBus)}
    expected_labels = {
        (bus.name, phase.value) for bus in buses.values() for phase in bus.phases
    }
    if any(phase not in {"A", "B", "C"} for _, phase in expected_labels):
        raise ValueError("GDM adapter currently supports A/B/C terminals only.")
    if set(labels) != expected_labels:
        raise ValueError("Reference indexing does not cover the GDM bus phases.")
    bases = []
    for bus_name, _ in labels:
        bus = buses[bus_name]
        nominal = float(bus.rated_voltage.to("volt").magnitude)
        if bus.voltage_type == VoltageTypes.LINE_TO_LINE:
            nominal /= np.sqrt(3)
        bases.append(nominal)
    bus_uuids = {name: str(bus.uuid) for name, bus in buses.items()}
    lineage = {}
    for component in system.iter_all_components():
        associated = []
        if isinstance(component, DistributionBus):
            associated.append(component.name)
        if getattr(component, "bus", None) is not None:
            associated.append(component.bus.name)
        associated.extend(bus.name for bus in getattr(component, "buses", ()))
        lineage[str(component.uuid)] = tuple(sorted(set(associated)))
    return np.array(bases), bus_uuids, lineage


def _flow_version() -> str:
    try:
        return version("gdm-flow")
    except PackageNotFoundError:
        return "unknown"


def from_gdm_flow_results(
    system: DistributionSystem,
    results: Mapping[str, ACPowerFlowResult],
    *,
    scenario_provenance: Mapping[str, Mapping[str, str]] | None = None,
    max_reference_mismatch_pu: float = 1e-5,
) -> ScenarioLibrary:
    """Ingest converged results belonging to one unchanged GDM system.

    Result matrices are authoritative, not independently restamped. Currents
    come from reported powers and voltages and are checked against YV. Callers
    must pair imported results with their original model/configuration.
    """
    if not results:
        raise ValueError("At least one gdm-flow reference result is required.")
    signature = _model_signature(system)
    source_bus = system.get_source_bus().name
    scenarios = []
    provenance = scenario_provenance or {}
    network = None
    bases = None
    bus_uuids = None
    lineage = None
    for name, result in results.items():
        if result.success is not True:
            raise ValueError(f"Reference solver did not converge for {name!r}.")
        labels = tuple(result.ybus_result.index_to_label)
        voltage = np.asarray(result.voltage, dtype=complex)
        power = np.asarray(result.power_injection, dtype=complex)
        if (
            voltage.shape != (len(labels),)
            or power.shape != voltage.shape
            or not np.all(np.isfinite(voltage))
            or not np.all(np.isfinite(power))
            or np.any(voltage == 0)
        ):
            raise ValueError("Reference voltage/power arrays are invalid.")
        sources = {
            label: complex(voltage[index])
            for index, label in enumerate(labels)
            if label[0] == source_bus
        }
        candidate_bases, candidate_uuids, candidate_lineage = _metadata(system, labels)
        candidate = ElectricalNetwork(result.ybus_result.ybus, labels, sources)
        fingerprint = configuration_fingerprint(
            candidate, candidate_bases, model_signature=signature
        )
        scenario_metadata = {
            **provenance.get(name, {}),
            "backend": "gdm-flow",
            "backend_version": _flow_version(),
            "grid_data_models_version": version("grid-data-models"),
            "reference_message": result.message,
        }
        scenarios.append(
            ReferenceScenario(
                name,
                labels,
                voltage,
                np.conj(power / voltage),
                fingerprint,
                result.success,
                result.max_mismatch_pu,
                scenario_metadata,
            )
        )
        if network is None:
            network, bases = candidate, candidate_bases
            bus_uuids, lineage = candidate_uuids, candidate_lineage
    assert network is not None and bases is not None
    assert bus_uuids is not None and lineage is not None
    return ScenarioLibrary(
        network,
        bases,
        scenarios,
        model_signature=signature,
        bus_uuids=bus_uuids,
        component_buses=lineage,
        max_reference_mismatch_pu=max_reference_mismatch_pu,
    )


def build_gdm_scenario_library(
    system: DistributionSystem,
    *,
    load_scales: Sequence[float] = (0.5, 1.0),
    include_shunt: bool = False,
    include_capacitor: bool = True,
    solar_scale: float = 1.0,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    provenance: Mapping[str, str] | None = None,
) -> ScenarioLibrary:
    """Generate references using constant-P/Q semantics on a copied GDM model.

    Hardware state must stay fixed. This is not OpenDSS validation, control
    optimization, or time-series interpretation. Each scale defines a snapshot.
    """
    scales = np.asarray(load_scales, dtype=float)
    if scales.ndim != 1 or not len(scales) or not np.all(np.isfinite(scales)):
        raise ValueError("load_scales must be a nonempty finite sequence.")
    if np.any(scales < 0) or not np.isfinite(solar_scale) or solar_scale < 0:
        raise ValueError("Load and solar scales must be nonnegative.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Reference tolerance must be finite and positive.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    try:
        from gdm_flow.ac_pf import solve_ac_power_flow_from_components
    except ModuleNotFoundError as exc:
        if exc.name not in {"gdm_flow", "gdm_flow.ac_pf"}:
            raise
        raise RuntimeError("Install with pip install 'gdm-reduce[flow]'.") from exc
    working_system = system.deepcopy()
    from gdm.distribution.components import DistributionBus

    labels = tuple(
        (bus.name, phase.value)
        for bus in working_system.get_components(DistributionBus)
        for phase in bus.phases
    )
    _metadata(working_system, labels)
    signature = _model_signature(working_system)
    results = {}
    metadata = {}
    for index, scale in enumerate(scales):
        name = f"load-{index:03d}"
        results[name] = solve_ac_power_flow_from_components(
            working_system,
            load_scale=float(scale),
            solar_scale=float(solar_scale),
            include_shunt=include_shunt,
            include_capacitor=include_capacitor,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if _model_signature(working_system) != signature:
            raise ValueError("Reference solver changed the fixed GDM configuration.")
        metadata[name] = {
            **(provenance or {}),
            "load_scale": str(float(scale)),
            "solar_scale": str(float(solar_scale)),
            "include_shunt": str(include_shunt),
            "include_capacitor": str(include_capacitor),
            "requested_tolerance_pu": str(tolerance),
        }
    return from_gdm_flow_results(
        working_system,
        results,
        scenario_provenance=metadata,
        max_reference_mismatch_pu=tolerance,
    )
