"""Public GDM metadata and gdm-flow result ingestion without matrix restamping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import numpy as np

from .models import SUPPORTED_PHASES, BusPhaseLabel, ElectricalNetwork
from .scenarios import (
    ReferenceScenario,
    ScenarioLibrary,
    configuration_fingerprint,
)

if TYPE_CHECKING:
    from gdm.distribution import DistributionSystem
    from gdm_flow.ac_pf import ACPowerFlowResult


@dataclass(frozen=True)
class GDMSearchConstraints:
    """Original physical bus edges and conservative protected equipment buses."""

    bus_edges: tuple[tuple[str, str], ...]
    protected_buses: frozenset[str]


def get_gdm_search_constraints(
    system: DistributionSystem, library: ScenarioLibrary
) -> GDMSearchConstraints:
    """Extract topology without inferring branch edges from mutual Y entries.

    Sources, transformer/regulator/switch boundaries, DER, capacitors, and
    controller measurement buses are retained. Open switches contribute no
    edge. The search kernel separately verifies connected radial topology.
    """
    from gdm.distribution.components import (
        DistributionBattery,
        DistributionBranchBase,
        DistributionCapacitor,
        DistributionRegulator,
        DistributionSolar,
        DistributionTransformer,
    )
    from gdm.distribution.components.base.distribution_switch_base import (
        DistributionSwitchBase,
    )

    if library.model_signature != _model_signature(system):
        raise ValueError("GDM topology must match the reference library configuration.")
    _metadata(system, library.network.labels)
    edges = set()
    protected = {bus for bus, _ in library.network.source_voltages}
    for component in system.iter_all_components():
        if not getattr(component, "in_service", True):
            continue
        terminals = getattr(component, "buses", ())
        is_switch = isinstance(component, DistributionSwitchBase)
        is_transformer = isinstance(
            component, DistributionTransformer | DistributionRegulator
        )
        if is_switch or is_transformer:
            protected.update(bus.name for bus in terminals)
        if isinstance(component, DistributionBranchBase) or is_transformer:
            if not is_switch or any(component.is_closed):
                edges.update(
                    tuple(sorted((terminals[0].name, terminal.name)))
                    for terminal in terminals[1:]
                )
        if isinstance(
            component, DistributionSolar | DistributionBattery | DistributionCapacitor
        ):
            protected.add(component.bus.name)
        for controller in getattr(component, "controllers", ()) or ():
            controlled_bus = getattr(controller, "controlled_bus", None)
            if controlled_bus is not None:
                protected.add(controlled_bus.name)
    return GDMSearchConstraints(tuple(sorted(edges)), frozenset(protected))


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
    if any(
        not {phase.value for phase in bus.phases} <= SUPPORTED_PHASES
        or (
            {phase.value for phase in bus.phases} & {"S1", "S2"}
            and {phase.value for phase in bus.phases} & {"A", "B", "C"}
        )
        for bus in buses.values()
    ):
        raise ValueError(
            "A/B/C terminals only on primary buses; S1/S2 services cannot "
            "mix ABC or explicit N terminals."
        )
    if set(labels) != expected_labels:
        raise ValueError("Reference indexing does not cover the GDM bus phases.")
    bases = []
    for bus_name, _ in labels:
        bus = buses[bus_name]
        nominal = float(bus.rated_voltage.to("volt").magnitude)
        if bus.voltage_type == VoltageTypes.LINE_TO_LINE:
            nominal /= (
                2
                if any(phase.value in {"S1", "S2"} for phase in bus.phases)
                else np.sqrt(3)
            )
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


def audit_split_phase_grounding(
    system: DistributionSystem,
) -> tuple[dict[str, str], ...]:
    """List unsupported center-tap grounding without changing model metadata."""
    from gdm.distribution.components import DistributionTransformer

    issues = []
    for transformer in system.get_components(DistributionTransformer):
        if not transformer.in_service or not any(
            phase.value in {"S1", "S2"}
            for group in transformer.winding_phases
            for phase in group
        ):
            continue
        windings = transformer.equipment.windings
        if len(windings) != 3 or any(
            not winding.is_grounded for winding in windings[1:]
        ):
            issues.append(
                {
                    "transformer": transformer.name,
                    "uuid": str(transformer.uuid),
                    "reason": (
                        "Both center-tap secondary windings must explicitly be "
                        "grounded; verify importer/source provenance, "
                        "do not override flags silently."
                    ),
                }
            )
    return tuple(issues)


def _split_phase_domains(
    system: DistributionSystem, *, assume_center_taps_grounded: bool = True
) -> dict[str, str]:
    import networkx as nx
    from gdm.distribution.components import (
        DistributionBranchBase,
        DistributionBus,
        DistributionTransformer,
    )

    split_buses = {
        bus.name
        for bus in system.get_components(DistributionBus)
        if any(phase.value in {"S1", "S2"} for phase in bus.phases)
    }
    if not split_buses:
        return {}
    issues = audit_split_phase_grounding(system)
    if issues and not assume_center_taps_grounded:
        raise ValueError(f"Unsupported split-phase grounding: {issues}")
    graph = nx.Graph()
    graph.add_nodes_from(split_buses)
    for branch in system.get_components(DistributionBranchBase):
        if not branch.in_service or (
            hasattr(branch, "is_closed") and not any(branch.is_closed)
        ):
            continue
        names = [bus.name for bus in branch.buses]
        if all(name in split_buses for name in names):
            graph.add_edge(*names)
    services = {}
    for transformer in system.get_components(DistributionTransformer):
        if not transformer.in_service or not any(
            bus.name in split_buses for bus in transformer.buses
        ):
            continue
        groups = [
            {phase.value for phase in group} for group in transformer.winding_phases
        ]
        if (
            len(groups) != 3
            or len(transformer.buses) != 3
            or groups[0] not in ({"A"}, {"B"}, {"C"})
            or {frozenset(group) for group in groups[1:]}
            != {frozenset({"N", "S1"}), frozenset({"N", "S2"})}
            or transformer.buses[1].name != transformer.buses[2].name
        ):
            raise ValueError(
                "Split-phase services require one ABC primary and a shared "
                "grounded center-tap secondary bus."
            )
        secondary = transformer.buses[1].name
        policy = "assumed-grounded" if assume_center_taps_grounded else "model-grounded"
        domain = (
            f"grounded-center-tap-v1:{policy}:{transformer.uuid}:"
            f"{transformer.buses[0].uuid}:{next(iter(groups[0]))}"
        )
        if secondary in services:
            raise ValueError("Ambiguous split-phase service has multiple transformers.")
        services[secondary] = domain
    domains = {}
    for members in nx.connected_components(graph):
        identities = {services[bus] for bus in members if bus in services}
        if len(identities) != 1:
            raise ValueError(
                "Every split-phase service must connect to exactly one "
                "grounded center-tap transformer."
            )
        domains.update({bus: next(iter(identities)) for bus in members})
    from gdm.distribution.components import DistributionLoad

    for component in system.iter_all_components():
        if (
            not getattr(component, "in_service", True)
            or getattr(getattr(component, "bus", None), "name", None) not in split_buses
        ):
            continue
        connection = getattr(
            getattr(component, "equipment", None), "connection_type", None
        )
        connection = getattr(connection, "value", connection)
        if connection not in {None, "STAR"}:
            if not (
                isinstance(component, DistributionLoad)
                and connection == "DELTA"
                and {phase.value for phase in component.phases} == {"S1", "S2"}
            ):
                raise ValueError(
                    "Split-phase non-STAR devices require explicitly supported "
                    "connection-aware physics."
                )
    return domains


def _require_split_phase_backend(domains: Mapping[str, str]) -> None:
    if domains:
        from gdm_flow import ac_pf, ybus

        if (
            getattr(ybus, "SPLIT_PHASE_CONTRACT", None) != "grounded-center-tap-v1"
            or getattr(ac_pf, "SPLIT_PHASE_POWER_FLOW_CONTRACT", None)
            != "grounded-center-tap-pq-v1"
        ):
            raise RuntimeError(
                "Split-phase references require grounded-center-tap-v1 stamping "
                "and grounded-center-tap-pq-v1 power flow; upgrade the backend."
            )


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
    assume_center_taps_grounded: bool = True,
) -> ScenarioLibrary:
    """Ingest converged results belonging to one unchanged GDM system.

    Result matrices are authoritative, not independently restamped. Currents
    come from reported powers and voltages and are checked against YV. Callers
    must pair imported results with their original model/configuration.
    Results must already use the declared center-tap grounding policy; no
    matrix or voltage is recomputed during ingestion.
    """
    if not results:
        raise ValueError("At least one gdm-flow reference result is required.")
    signature = _model_signature(system)
    domains = _split_phase_domains(
        system, assume_center_taps_grounded=assume_center_taps_grounded
    )
    _require_split_phase_backend(domains)
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
            candidate,
            candidate_bases,
            model_signature=signature,
            service_domains=domains,
        )
        scenario_metadata = {
            **provenance.get(name, {}),
            "backend": "gdm-flow",
            "backend_version": _flow_version(),
            "grid_data_models_version": version("grid-data-models"),
            "reference_message": result.message,
            **(
                {
                    "center_tap_grounding": "assumed-grounded"
                    if assume_center_taps_grounded
                    else "model-grounded"
                }
                if domains
                else {}
            ),
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
        service_domains=domains,
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
    assume_center_taps_grounded: bool = True,
) -> ScenarioLibrary:
    """Generate references using constant-P/Q semantics on a copied GDM model.

    Center taps are assumed grounded by default; only the working copy is
    normalized. Set assume_center_taps_grounded=False to require model flags.
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
    from gdm.distribution.components import DistributionBus, DistributionTransformer

    labels = tuple(
        (bus.name, phase.value)
        for bus in working_system.get_components(DistributionBus)
        for phase in bus.phases
    )
    _metadata(working_system, labels)
    domains = _split_phase_domains(
        working_system, assume_center_taps_grounded=assume_center_taps_grounded
    )
    _require_split_phase_backend(domains)
    if assume_center_taps_grounded:
        for transformer in working_system.get_components(DistributionTransformer):
            if transformer.in_service and any(
                phase.value in {"S1", "S2"}
                for group in transformer.winding_phases
                for phase in group
            ):
                for winding in transformer.equipment.windings[1:]:
                    winding.is_grounded = True
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
        system,
        results,
        scenario_provenance=metadata,
        max_reference_mismatch_pu=tolerance,
        assume_center_taps_grounded=assume_center_taps_grounded,
    )
