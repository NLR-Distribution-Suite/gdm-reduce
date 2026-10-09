"""Restore tree junctions while preserving selected current assignments."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .kron import KronReductionResult, kron_reduce
from .models import ElectricalNetwork
from .search import CandidateSearchResult, _bus_tree


@dataclass(frozen=True)
class RadializationResult:
    """A radial numerical equivalent, not reconstructed GDM equipment.

    Restored junctions are passive terminals, not new cluster roots. Original
    cluster mappings and current assignments remain in ``selection``.
    """

    selection: CandidateSearchResult
    equivalent: KronReductionResult
    restored_buses: tuple[str, ...]
    bus_edges: tuple[tuple[str, str], ...]
    max_selected_voltage_difference_v: float

    @property
    def retained_buses(self) -> tuple[str, ...]:
        return tuple(sorted({bus for bus, _ in self.equivalent.retained_labels}))

    @property
    def reduction_fraction(self) -> float:
        return 1 - len(self.retained_buses) / len(self.selection.original_to_retained)


def _matrix_edges(network: ElectricalNetwork) -> set[tuple[str, str]]:
    matrix = network.admittance.tocoo()
    return {
        tuple(sorted((network.labels[row][0], network.labels[column][0])))
        for row, column, value in zip(matrix.row, matrix.col, matrix.data, strict=True)
        if value != 0 and network.labels[row][0] != network.labels[column][0]
    }


def radialize_reduction(
    selection: CandidateSearchResult,
    bus_edges: Collection[tuple[str, str]],
    *,
    max_condition_number: float = 1e12,
    voltage_atol_v: float = 1e-7,
    voltage_rtol: float = 1e-9,
) -> RadializationResult:
    """Restore non-root branch vertices in the tree connecting selected roots.

    Prune non-root leaves, restore non-root degree-three-or-higher junctions,
    and suppress degree-two paths. This is minimal structural restoration for
    the supplied tree, not an equipment-realizability certificate. Both the
    original and final matrix bus couplings must match their physical trees.
    No admittance entries are dropped to force radiality.
    """
    for name, tolerance in (
        ("voltage_atol_v", voltage_atol_v),
        ("voltage_rtol", voltage_rtol),
    ):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    network = selection.equivalent.network
    buses = {bus for bus, _ in network.labels}
    tree = _bus_tree(buses, bus_edges)
    edges = {tuple(sorted(edge)) for edge in tree.edges}
    if _matrix_edges(network) != edges:
        raise ValueError("Original admittance couplings must match the physical tree.")
    roots = set(selection.retained_buses)
    removed_indices = [
        index for index, (bus, _) in enumerate(network.labels) if bus not in roots
    ]
    if np.any(selection.aggregated_currents[removed_indices] != 0):
        raise ValueError("Removed buses must have zero pooled current.")
    leaves = deque(
        sorted(bus for bus in tree if tree.degree(bus) <= 1 and bus not in roots)
    )
    while leaves:
        bus = leaves.popleft()
        if bus not in tree or bus in roots or tree.degree(bus) > 1:
            continue
        neighbors = list(tree.neighbors(bus))
        tree.remove_node(bus)
        leaves.extend(neighbors)
    restored = {bus for bus in tree if bus not in roots and tree.degree(bus) >= 3}
    kept = roots | restored
    for bus in sorted(set(tree) - kept):
        first, second = sorted(tree.neighbors(bus))
        tree.add_edge(first, second)
        tree.remove_node(bus)
    radial_edges = tuple(sorted(tuple(sorted(edge)) for edge in tree.edges))
    equivalent = kron_reduce(network, kept, max_condition_number=max_condition_number)
    if not nx.is_tree(tree) or _matrix_edges(equivalent.reduced_network) != set(
        radial_edges
    ):
        raise ValueError("Reduced admittance couplings do not form the restored tree.")
    previous_voltage = selection.equivalent.solve(selection.aggregated_currents)
    radial_voltage = equivalent.solve(selection.aggregated_currents)
    indices = {label: index for index, label in enumerate(equivalent.retained_labels)}
    selected_voltage = radial_voltage[
        [indices[label] for label in selection.equivalent.retained_labels]
    ]
    difference = float(np.max(np.abs(selected_voltage - previous_voltage)))
    if not np.allclose(
        selected_voltage, previous_voltage, rtol=voltage_rtol, atol=voltage_atol_v
    ):
        raise ValueError("Radialization changed selected-terminal voltages.")
    return RadializationResult(
        selection, equivalent, tuple(sorted(restored)), radial_edges, difference
    )
