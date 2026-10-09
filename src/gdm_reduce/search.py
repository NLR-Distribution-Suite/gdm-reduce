"""Bounded-batch, one-bus Opti-KRON candidate enumeration on radial bus graphs."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import networkx as nx
import numpy as np

from .kron import KronReductionResult, _boundary, _condition_limit, _solve, kron_reduce
from .scenarios import ScenarioLibrary


@dataclass(frozen=True)
class ReductionStep:
    removed_bus: str
    receiver_bus: str
    retained_bus_count: int
    evaluated_candidates: int
    feasible_candidates: int
    max_voltage_error_pu: float
    smice: float


@dataclass(frozen=True)
class CandidateSearchResult:
    """Cluster selection and a numerical equivalent, not a GDM feeder export.

    Aggregated currents retain original terminal indexing. Source entries are
    bookkeeping sums, not prescribed source injections. Represented voltages
    assign each original bus its selected root's matching phase voltage.
    """

    equivalent: KronReductionResult
    original_to_retained: Mapping[str, str]
    clusters: Mapping[str, tuple[str, ...]]
    protected_buses: frozenset[str]
    aggregated_currents: np.ndarray
    represented_voltage: np.ndarray
    history: tuple[ReductionStep, ...]
    termination_reason: Literal[
        "target_reached", "no_feasible_merge", "iteration_limit"
    ]
    max_voltage_error_pu: float
    smice: float
    configuration_fingerprint: str
    bus_uuid_to_retained_uuid: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_to_retained",
            MappingProxyType(dict(self.original_to_retained)),
        )
        object.__setattr__(self, "clusters", MappingProxyType(dict(self.clusters)))
        object.__setattr__(
            self,
            "bus_uuid_to_retained_uuid",
            MappingProxyType(dict(self.bus_uuid_to_retained_uuid)),
        )
        for name in ("aggregated_currents", "represented_voltage"):
            values = np.array(getattr(self, name), dtype=complex, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def retained_buses(self) -> tuple[str, ...]:
        return tuple(sorted(self.clusters))

    @property
    def reduction_fraction(self) -> float:
        return 1 - len(self.clusters) / len(self.original_to_retained)


def _represent(
    library: ScenarioLibrary,
    mapping: Mapping[str, str],
    voltage: np.ndarray,
    indices: Mapping[tuple[str, str], int],
    reference_voltage: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    representatives = np.array(
        [indices[(mapping[bus], phase)] for bus, phase in library.network.labels]
    )
    represented = voltage[representatives]
    errors = np.abs(np.abs(represented) - np.abs(reference_voltage))
    errors /= library.voltage_bases[:, None]
    roots = np.array([mapping[bus] for bus, _ in library.network.labels])
    smice = sum(
        float(np.max(errors[roots == root], axis=0).sum())
        for root in sorted(set(mapping.values()))
    )
    return represented, float(np.max(errors)), smice


def _bus_tree(
    buses: Collection[str], bus_edges: Collection[tuple[str, str]]
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(buses)
    for edge in bus_edges:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or any(bus not in graph for bus in edge)
            or edge[0] == edge[1]
        ):
            raise ValueError("Bus edges must connect two distinct known bus names.")
        graph.add_edge(*edge)
    if not nx.is_tree(graph):
        raise ValueError("Candidate search requires a connected radial bus graph.")
    return graph


def optimize_kron_reduction(
    library: ScenarioLibrary,
    bus_edges: Collection[tuple[str, str]],
    *,
    protected_buses: Collection[str] = (),
    voltage_error_limit_pu: float = 0.001,
    target_reduction_fraction: float = 1.0,
    max_iterations: int | None = None,
    batch_size: int = 32,
    max_condition_number: float = 1e12,
) -> CandidateSearchResult:
    """Select connected, phase-compatible bus clusters by Algorithm 1 SMICE.

    Enumerates all currently admissible one-bus merges, not all final partitions.
    One source-conditioned LU is reused for bounded batches of scenarios.
    Sources are automatically protected; callers must protect other critical
    equipment boundaries. Pooling across nominal voltage bases is prohibited.
    Scores within 1e-12 pu are tied lexicographically by receiver/removed bus.
    """
    if not np.isfinite(voltage_error_limit_pu) or voltage_error_limit_pu < 0:
        raise ValueError("voltage_error_limit_pu must be finite and nonnegative.")
    if (
        not np.isfinite(target_reduction_fraction)
        or not 0 <= target_reduction_fraction <= 1
    ):
        raise ValueError("target_reduction_fraction must be between zero and one.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if max_iterations is not None and (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 0
    ):
        raise ValueError("max_iterations must be a nonnegative integer or None.")
    network = library.network
    buses = sorted({bus for bus, _ in network.labels})
    if isinstance(protected_buses, str) or not set(protected_buses) <= set(buses):
        raise ValueError("Protected buses must be a collection of known bus names.")
    protected = frozenset(protected_buses) | frozenset(
        bus for bus, _ in network.source_voltages
    )
    graph = _bus_tree(buses, bus_edges)
    indices = {label: index for index, label in enumerate(network.labels)}
    phases = {
        bus: {phase for name, phase in network.labels if name == bus} for bus in buses
    }
    boundary = _boundary(network, _condition_limit(max_condition_number))
    sources, unknown, source_voltage, factor, _ = boundary
    mapping = {bus: bus for bus in buses}
    clusters = {bus: {bus} for bus in buses}
    currents = library.currents.copy()
    reference_voltage = library.voltage
    voltage = _solve(network, currents, sources, unknown, source_voltage, factor)
    represented, error, smice = _represent(
        library, mapping, voltage, indices, reference_voltage
    )
    if error > voltage_error_limit_pu:
        raise ValueError("Full-network reference consistency exceeds the error limit.")
    history = []
    while True:
        if 1 - len(clusters) / len(buses) >= target_reduction_fraction:
            reason = "target_reached"
            break
        if max_iterations is not None and len(history) >= max_iterations:
            reason = "iteration_limit"
            break
        candidates = sorted(
            (receiver, removed)
            for removed in graph
            if removed not in protected
            for receiver in graph.neighbors(removed)
            if phases[removed] <= phases[receiver]
            and all(
                np.isclose(
                    library.voltage_bases[indices[(removed, phase)]],
                    library.voltage_bases[indices[(receiver, phase)]],
                    rtol=1e-10,
                    atol=1e-6,
                )
                for phase in phases[removed]
            )
        )
        best = None
        feasible = 0
        scenario_count = len(library.scenarios)
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            trial_currents = np.tile(currents, (1, len(batch)))
            for offset, (receiver, removed) in enumerate(batch):
                columns = slice(offset * scenario_count, (offset + 1) * scenario_count)
                removed_indices = [
                    indices[(removed, phase)] for phase in sorted(phases[removed])
                ]
                receiver_indices = [
                    indices[(receiver, phase)] for phase in sorted(phases[removed])
                ]
                trial_currents[receiver_indices, columns] += trial_currents[
                    removed_indices, columns
                ]
                trial_currents[removed_indices, columns] = 0
            trial_voltage = _solve(
                network, trial_currents, sources, unknown, source_voltage, factor
            )
            for offset, (receiver, removed) in enumerate(batch):
                columns = slice(offset * scenario_count, (offset + 1) * scenario_count)
                candidate_mapping = mapping.copy()
                for member in clusters[removed]:
                    candidate_mapping[member] = receiver
                candidate_voltage, candidate_error, candidate_smice = _represent(
                    library,
                    candidate_mapping,
                    trial_voltage[:, columns],
                    indices,
                    reference_voltage,
                )
                if candidate_error > voltage_error_limit_pu:
                    continue
                feasible += 1
                if best is None or candidate_smice < best[0] - 1e-12:
                    best = (
                        candidate_smice,
                        receiver,
                        removed,
                        candidate_mapping,
                        candidate_voltage.copy(),
                        candidate_error,
                        trial_currents[:, columns].copy(),
                    )
        if best is None:
            reason = "no_feasible_merge"
            break
        smice, receiver, removed, mapping, represented, error, currents = best
        clusters[receiver].update(clusters.pop(removed))
        for neighbor in list(graph.neighbors(removed)):
            if neighbor != receiver:
                graph.add_edge(receiver, neighbor)
        graph.remove_node(removed)
        history.append(
            ReductionStep(
                removed,
                receiver,
                len(clusters),
                len(candidates),
                feasible,
                error,
                smice,
            )
        )
    equivalent = kron_reduce(
        network, set(clusters), max_condition_number=max_condition_number
    )
    return CandidateSearchResult(
        equivalent,
        mapping,
        {root: tuple(sorted(members)) for root, members in clusters.items()},
        protected,
        currents,
        represented,
        tuple(history),
        reason,
        error,
        smice,
        library.configuration_fingerprint,
        {
            library.bus_uuids[original]: library.bus_uuids[root]
            for original, root in mapping.items()
            if library.bus_uuids
        },
    )
