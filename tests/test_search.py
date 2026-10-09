import numpy as np
import pytest

from gdm_reduce import ElectricalNetwork, solve_voltages
from gdm_reduce.scenarios import (
    ReferenceScenario,
    ScenarioLibrary,
    configuration_fingerprint,
)
from gdm_reduce.search import optimize_kron_reduction


@pytest.fixture
def chain_library():
    network = ElectricalNetwork(
        np.array([[10, -10, 0], [-10, 20, -10], [0, -10, 10]]),
        (("source", "A"), ("middle", "A"), ("load", "A")),
        {("source", "A"): 120},
    )
    bases = np.full(3, 120.0)
    references = []
    for name, scale in (("light", 0.5), ("heavy", 1.0)):
        currents = np.array([3, -1, -2]) * scale
        voltage = solve_voltages(network, currents)
        references.append(
            ReferenceScenario(
                name,
                network.labels,
                voltage,
                currents,
                configuration_fingerprint(network, bases),
                True,
                0,
            )
        )
    return ScenarioLibrary(network, bases, references)


EDGES = (("source", "middle"), ("middle", "load"))


def test_closed_form_merge_and_original_bus_error(chain_library):
    result = optimize_kron_reduction(chain_library, EDGES, voltage_error_limit_pu=0.002)
    assert result.termination_reason == "no_feasible_merge"
    assert len(result.history) == 1
    assert result.history[0].removed_bus == "load"
    assert result.history[0].receiver_bus == "middle"
    assert result.clusters["middle"] == ("load", "middle")
    assert result.original_to_retained["load"] == "middle"
    assert result.protected_buses == frozenset({"source"})
    np.testing.assert_allclose(result.aggregated_currents[:, 1], [3, -3, 0])
    assert result.max_voltage_error_pu == pytest.approx(0.2 / 120)
    assert result.smice == pytest.approx(0.3 / 120)
    np.testing.assert_allclose(result.represented_voltage[:, 1], [120, 119.7, 119.7])
    retained = result.equivalent.solve(result.aggregated_currents)
    np.testing.assert_allclose(retained[:, 1], [120, 119.7])


def test_full_cluster_memberships_survive_successive_merges(chain_library):
    result = optimize_kron_reduction(chain_library, EDGES, voltage_error_limit_pu=0.01)
    assert len(result.history) == 2
    assert result.retained_buses == ("source",)
    assert set(result.clusters["source"]) == {"source", "middle", "load"}
    assert result.max_voltage_error_pu == pytest.approx(0.5 / 120)
    assert result.reduction_fraction == pytest.approx(2 / 3)
    assert set(result.original_to_retained.values()) == {"source"}


def test_batch_sizes_and_stop_conditions(chain_library):
    results = [
        optimize_kron_reduction(
            chain_library,
            EDGES,
            voltage_error_limit_pu=0.01,
            target_reduction_fraction=0.3,
            batch_size=size,
        )
        for size in (1, 4)
    ]
    assert results[0].history == results[1].history
    assert results[0].termination_reason == "target_reached"
    stopped = optimize_kron_reduction(chain_library, EDGES, max_iterations=0)
    assert stopped.termination_reason == "iteration_limit"
    unchanged = optimize_kron_reduction(
        chain_library, EDGES, target_reduction_fraction=0
    )
    assert unchanged.termination_reason == "target_reached"
    assert unchanged.reduction_fraction == 0


def test_protected_buses_are_not_eliminated(chain_library):
    result = optimize_kron_reduction(
        chain_library,
        EDGES,
        protected_buses={"middle", "load"},
    )
    assert result.history == ()
    assert result.termination_reason == "no_feasible_merge"


def make_library(network, bases, currents):
    voltage = solve_voltages(network, np.asarray(currents))
    reference = ReferenceScenario(
        "test",
        network.labels,
        voltage,
        network.admittance @ voltage,
        configuration_fingerprint(network, bases),
        True,
        0,
    )
    return ScenarioLibrary(network, bases, [reference])


def test_missing_phase_receiver_is_excluded_with_mutual_coupling():
    coupling = np.array([[10, 1], [1, 8]])
    admittance = np.zeros((5, 5))
    admittance[:2, :2] = admittance[2:4, 2:4] = coupling
    admittance[:2, 2:4] = admittance[2:4, :2] = -coupling
    admittance[2, 2] += 10
    admittance[4, 4] = 10
    admittance[2, 4] = admittance[4, 2] = -10
    network = ElectricalNetwork(
        admittance,
        (
            ("source", "A"),
            ("source", "B"),
            ("middle", "A"),
            ("middle", "B"),
            ("load", "A"),
        ),
        {("source", "A"): 120, ("source", "B"): 120 * np.exp(-2j * np.pi / 3)},
    )
    library = make_library(network, np.full(5, 120.0), [0, 0, -1, -2, -3])
    selected = optimize_kron_reduction(
        library,
        EDGES,
        protected_buses={"load"},
        voltage_error_limit_pu=1,
    )
    assert len(selected.history) == 1
    assert selected.history[0].evaluated_candidates == 1
    assert selected.history[0].receiver_bus == "source"
    assert selected.original_to_retained["middle"] == "source"
    retained = selected.equivalent.solve(selected.aggregated_currents)
    indices = {
        label: index for index, label in enumerate(selected.equivalent.retained_labels)
    }
    for original_index, (bus, phase) in enumerate(network.labels):
        root = selected.original_to_retained[bus]
        np.testing.assert_allclose(
            selected.represented_voltage[original_index],
            retained[indices[(root, phase)]],
        )


def test_incompatible_voltage_bases_prevent_pooling(chain_library):
    library = make_library(
        chain_library.network,
        np.array([120.0, 240.0, 360.0]),
        [3, -1, -2],
    )
    selected = optimize_kron_reduction(library, EDGES, voltage_error_limit_pu=1)
    assert selected.history == ()
    assert selected.termination_reason == "no_feasible_merge"


def test_equal_scores_use_lexicographic_tie_break():
    network = ElectricalNetwork(
        np.array([[20, -10, -10], [-10, 10, 0], [-10, 0, 10]]),
        (("source", "A"), ("alpha", "A"), ("beta", "A")),
        {("source", "A"): 120},
    )
    library = make_library(network, np.full(3, 120.0), [4, -2, -2])
    selected = optimize_kron_reduction(
        library,
        (("source", "beta"), ("source", "alpha")),
        voltage_error_limit_pu=1,
        max_iterations=1,
    )
    assert selected.history[0].removed_bus == "alpha"
    assert selected.history[0].feasible_candidates == 2


@pytest.mark.parametrize(
    "options",
    [
        {"voltage_error_limit_pu": -1},
        {"voltage_error_limit_pu": np.nan},
        {"target_reduction_fraction": -0.1},
        {"target_reduction_fraction": 1.1},
        {"target_reduction_fraction": np.inf},
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"max_iterations": -1},
        {"max_iterations": True},
        {"max_iterations": 1.5},
        {"protected_buses": "source"},
        {"protected_buses": {"unknown"}},
        {"max_condition_number": 0},
    ],
)
def test_invalid_search_options(chain_library, options):
    with pytest.raises(ValueError):
        optimize_kron_reduction(chain_library, EDGES, **options)


@pytest.mark.parametrize(
    "edges",
    [
        (),
        (("source", "middle"),),
        (("source", "source"),),
        (("source", "unknown"),),
        (("source",),),
        (["source", "middle"],),
        (*EDGES, ("source", "load")),
    ],
)
def test_invalid_or_nonradial_graphs(chain_library, edges):
    with pytest.raises(ValueError):
        optimize_kron_reduction(chain_library, edges)


def test_reference_consistency_must_fit_requested_error_limit(chain_library):
    references = []
    for scenario in chain_library.scenarios:
        voltage = scenario.voltage.copy()
        voltage[1] += 1e-9
        references.append(
            ReferenceScenario(
                scenario.name,
                scenario.labels,
                voltage,
                scenario.currents,
                scenario.configuration_fingerprint,
                True,
                0,
            )
        )
    library = ScenarioLibrary(
        chain_library.network, chain_library.voltage_bases, references
    )
    with pytest.raises(ValueError, match="reference consistency"):
        optimize_kron_reduction(library, EDGES, voltage_error_limit_pu=0)


def test_inputs_and_result_mapping_are_not_mutable(chain_library):
    currents = chain_library.currents.copy()
    voltage = chain_library.voltage.copy()
    admittance = chain_library.network.admittance.toarray().copy()
    selected = optimize_kron_reduction(
        chain_library, EDGES, voltage_error_limit_pu=0.01
    )
    np.testing.assert_array_equal(chain_library.currents, currents)
    np.testing.assert_array_equal(chain_library.voltage, voltage)
    np.testing.assert_array_equal(
        chain_library.network.admittance.toarray(), admittance
    )
    with pytest.raises(TypeError):
        selected.original_to_retained["load"] = "load"
    with pytest.raises(ValueError):
        selected.aggregated_currents[0, 0] = 0


def test_single_bus_source_only_network():
    network = ElectricalNetwork([[1]], (("source", "A"),), {("source", "A"): 120})
    library = make_library(network, np.array([120.0]), [120])
    selected = optimize_kron_reduction(library, ())
    assert selected.retained_buses == ("source",)
    assert selected.termination_reason == "no_feasible_merge"
