from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from gdm_reduce import (
    ElectricalNetwork,
    ReferenceScenario,
    ScenarioLibrary,
    configuration_fingerprint,
    optimize_kron_reduction,
    radialization,
    radialize_reduction,
    solve_voltages,
)


def star_selection():
    network = ElectricalNetwork(
        [[10, -10, 0, 0], [-10, 30, -10, -10], [0, -10, 10, 0], [0, -10, 0, 10]],
        (("source", "A"), ("junction", "A"), ("alpha", "A"), ("beta", "A")),
        {("source", "A"): 120},
    )
    currents = np.array([[3, 6], [0, 0], [-1, -2], [-2, -4]])
    voltage = solve_voltages(network, currents)
    bases = np.full(4, 120.0)
    fingerprint = configuration_fingerprint(network, bases)
    scenarios = [
        ReferenceScenario(
            f"test-{index}",
            network.labels,
            voltage[:, index],
            currents[:, index],
            fingerprint,
            True,
            0,
        )
        for index in range(2)
    ]
    library = ScenarioLibrary(network, bases, scenarios)
    edges = (("source", "junction"), ("junction", "alpha"), ("junction", "beta"))
    selected = optimize_kron_reduction(
        library,
        edges,
        protected_buses={"alpha", "beta"},
        voltage_error_limit_pu=1,
    )
    return selected, edges


def test_star_junction_restoration_removes_mesh_and_preserves_voltages():
    selected, edges = star_selection()
    assert selected.retained_buses == ("alpha", "beta", "source")
    assert selected.equivalent.admittance.nnz == 9
    radial = radialize_reduction(selected, edges)
    assert radial.restored_buses == ("junction",)
    assert radial.retained_buses == ("alpha", "beta", "junction", "source")
    assert nx.is_tree(nx.Graph(radial.bus_edges))
    assert radial.max_selected_voltage_difference_v < 1e-9
    assert radial.selection is selected
    assert selected.original_to_retained["junction"] == "alpha"
    np.testing.assert_array_equal(radial.selection.aggregated_currents[1], [0, 0])
    np.testing.assert_allclose(
        radial.equivalent.solve(selected.aggregated_currents),
        solve_voltages(selected.equivalent.network, selected.aggregated_currents),
    )
    assert radial.reduction_fraction == 0


def select_network(network, edges, protected_buses, **options):
    currents = np.zeros(len(network.labels), dtype=complex)
    for index, (bus, phase) in enumerate(network.labels):
        if bus in protected_buses:
            angle = {"A": 0, "B": -2 * np.pi / 3, "C": 2 * np.pi / 3}[phase]
            currents[index] = -np.exp(1j * angle)
    voltage = solve_voltages(network, np.column_stack((currents, 2 * currents)))
    currents = network.admittance @ voltage
    bases = np.full(len(network.labels), 120.0)
    fingerprint = configuration_fingerprint(network, bases)
    references = [
        ReferenceScenario(
            f"test-{index}",
            network.labels,
            voltage[:, index],
            currents[:, index],
            fingerprint,
            True,
            0,
        )
        for index in range(2)
    ]
    library = ScenarioLibrary(network, bases, references)
    return optimize_kron_reduction(
        library,
        edges,
        protected_buses=protected_buses,
        voltage_error_limit_pu=1,
        **options,
    )


def select_tree(edges, protected_buses, **options):
    graph = nx.Graph(edges)
    buses = sorted(graph)
    network = ElectricalNetwork(
        10 * nx.laplacian_matrix(graph, nodelist=buses).toarray(),
        tuple((bus, "A") for bus in buses),
        {("source", "A"): 120},
    )
    return select_network(network, edges, protected_buses, **options)


@pytest.mark.parametrize(
    "edges, protected, restored",
    [
        ((("source", "middle"), ("middle", "alpha")), {"alpha"}, ()),
        (
            (("source", "middle"), ("middle", "alpha"), ("middle", "dead")),
            {"alpha"},
            (),
        ),
        (
            (
                ("source", "junction1"),
                ("alpha", "junction1"),
                ("junction1", "middle"),
                ("middle", "junction2"),
                ("junction2", "beta"),
                ("junction2", "gamma"),
            ),
            {"alpha", "beta", "gamma"},
            ("junction1", "junction2"),
        ),
    ],
)
def test_only_branch_vertices_of_selected_root_subtree_are_restored(
    edges, protected, restored
):
    selected = select_tree(edges, protected)
    assert set(selected.retained_buses) == protected | {"source"}
    radial = radialize_reduction(selected, edges)
    assert radial.restored_buses == restored
    assert nx.is_tree(nx.Graph(radial.bus_edges))
    assert radial.max_selected_voltage_difference_v < 1e-9
    assert set(radial.retained_buses) == protected | {"source"} | set(restored)
    np.testing.assert_array_equal(
        radial.selection.aggregated_currents, selected.aggregated_currents
    )


def test_full_retention_and_source_only_cases():
    _, edges = star_selection()
    for options in ({"target_reduction_fraction": 0}, {}):
        selected = select_tree(edges, set(), **options)
        radial = radialize_reduction(selected, edges)
        assert radial.restored_buses == ()
        assert radial.retained_buses == selected.retained_buses
        graph = nx.Graph(radial.bus_edges)
        graph.add_nodes_from(radial.retained_buses)
        assert nx.is_tree(graph)
        assert radial.reduction_fraction == selected.reduction_fraction


def test_missing_phases_and_mutual_coupling_survive_junction_restoration():
    admittance = np.zeros((6, 6))
    coupling = np.array([[10, 1], [1, 8]])
    admittance[:2, :2] = admittance[2:4, 2:4] = coupling
    admittance[:2, 2:4] = admittance[2:4, :2] = -coupling
    for terminal, junction_terminal in ((4, 2), (5, 3)):
        admittance[terminal, terminal] += 10
        admittance[junction_terminal, junction_terminal] += 10
        admittance[terminal, junction_terminal] -= 10
        admittance[junction_terminal, terminal] -= 10
    network = ElectricalNetwork(
        admittance,
        (
            ("source", "A"),
            ("source", "B"),
            ("junction", "A"),
            ("junction", "B"),
            ("alpha", "A"),
            ("beta", "B"),
        ),
        {("source", "A"): 120, ("source", "B"): 120 * np.exp(-2j * np.pi / 3)},
    )
    _, edges = star_selection()
    selected = select_network(network, edges, {"alpha", "beta"})
    radial = radialize_reduction(selected, edges)
    assert radial.restored_buses == ("junction",)
    assert radial.equivalent.retained_labels == network.labels
    assert radial.max_selected_voltage_difference_v < 1e-9
    np.testing.assert_allclose(radial.equivalent.admittance.toarray(), admittance)


@pytest.mark.parametrize(
    "options",
    [
        {"voltage_atol_v": -1},
        {"voltage_atol_v": np.nan},
        {"voltage_rtol": -1},
        {"voltage_rtol": np.inf},
        {"max_condition_number": 0},
    ],
)
def test_invalid_radialization_options(options):
    selected, edges = star_selection()
    with pytest.raises(ValueError):
        radialize_reduction(selected, edges, **options)


def test_physical_tree_must_match_original_matrix():
    selected, _ = star_selection()
    with pytest.raises(ValueError, match="Original admittance couplings"):
        radialize_reduction(
            selected,
            (
                ("source", "alpha"),
                ("alpha", "junction"),
                ("junction", "beta"),
            ),
        )


def test_unpooled_eliminated_current_is_rejected():
    selected, edges = star_selection()
    currents = selected.aggregated_currents.copy()
    currents[1, 0] = -1
    invalid = replace(selected, aggregated_currents=currents)
    with pytest.raises(ValueError, match="zero pooled current"):
        radialize_reduction(invalid, edges)


def test_mesh_is_not_silently_dropped(monkeypatch):
    selected, edges = star_selection()
    monkeypatch.setattr(
        radialization, "kron_reduce", lambda *args, **kwargs: selected.equivalent
    )
    with pytest.raises(ValueError, match="do not form the restored tree"):
        radialize_reduction(selected, edges)


def test_selected_voltage_equivalence_is_a_required_gate(monkeypatch):
    selected, edges = star_selection()
    reduce = radialization.kron_reduce

    def corrupted(*args, **kwargs):
        equivalent = reduce(*args, **kwargs)
        return replace(equivalent, _source_voltage=equivalent._source_voltage + 0.01)

    monkeypatch.setattr(radialization, "kron_reduce", corrupted)
    with pytest.raises(ValueError, match="changed selected-terminal voltages"):
        radialize_reduction(selected, edges)
