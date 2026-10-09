import json
from copy import deepcopy
from importlib.metadata import PackageNotFoundError

import numpy as np
import pytest

from gdm_reduce import ElectricalNetwork, artifacts, solve_voltages
from gdm_reduce.artifacts import (
    build_result_bundle,
    network_from_result_bundle,
    save_result_bundle,
)
from gdm_reduce.radialization import radialize_reduction
from gdm_reduce.scenarios import (
    ReferenceScenario,
    ScenarioLibrary,
    configuration_fingerprint,
)
from gdm_reduce.search import optimize_kron_reduction
from gdm_reduce.validation import evaluate_reduction, pool_scenario_currents


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
    holdout = make_library(network, library.voltage_bases, [0, 0, 0, -20, 0])
    report = evaluate_reduction(selected, holdout, voltage_error_limit_pu=1)
    assert report.scenarios[0].max_voltage_angle_error_deg > 0
    radial = radialize_reduction(selected, EDGES)
    assert evaluate_reduction(radial, holdout, voltage_error_limit_pu=1) == report


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


def test_frozen_clusters_can_fail_unseen_stress_conditions(chain_library):
    selected = optimize_kron_reduction(
        chain_library, EDGES, voltage_error_limit_pu=0.002
    )
    trained = evaluate_reduction(selected, chain_library, voltage_error_limit_pu=0.002)
    assert trained.passed
    assert trained.max_voltage_magnitude_error_pu == pytest.approx(
        selected.max_voltage_error_pu
    )
    holdout = make_library(
        chain_library.network, chain_library.voltage_bases, [12, -4, -8]
    )
    report = evaluate_reduction(selected, holdout, voltage_error_limit_pu=0.002)
    assert not report.passed
    assert report.max_voltage_magnitude_error_pu == pytest.approx(0.8 / 120)
    assert report.scenarios[0].full_network_max_complex_error_pu < 1e-12
    np.testing.assert_allclose(
        pool_scenario_currents(selected, holdout)[:, 0], [12, -12, 0]
    )
    assert selected.original_to_retained["load"] == "middle"
    np.testing.assert_allclose(selected.aggregated_currents[:, 1], [3, -3, 0])


def test_result_bundle_round_trip_replays_holdout_and_preserves_failure(
    chain_library, tmp_path
):
    selected = optimize_kron_reduction(
        chain_library, EDGES, voltage_error_limit_pu=0.002
    )
    radial = radialize_reduction(selected, EDGES)
    holdout = make_library(
        chain_library.network, chain_library.voltage_bases, [12, -4, -8]
    )
    bundle = build_result_bundle(
        radial,
        chain_library,
        holdout,
        voltage_error_limit_pu=0.002,
        timings_s={"search": 0.1},
    )
    path = tmp_path / "results" / "reduction.json"
    save_result_bundle(path, bundle)
    loaded = json.loads(path.read_text())
    network = network_from_result_bundle(loaded)
    values = loaded["holdout"]["pooled_retained_currents"]
    currents = np.array(values["real"]) + 1j * np.array(values["imag"])
    np.testing.assert_allclose(
        solve_voltages(network, currents),
        radial.equivalent.solve(pool_scenario_currents(selected, holdout)),
    )
    assert loaded["training"]["report"]["passed"] is True
    assert loaded["holdout"]["report"]["passed"] is False
    assert loaded["gdm_model_exported"] is False
    assert loaded["timings_s"] == {"search": 0.1}
    with pytest.raises(FileExistsError):
        save_result_bundle(path, bundle)
    save_result_bundle(path, bundle, overwrite=True)


def test_reverse_flow_uses_frozen_clusters(chain_library):
    selected = optimize_kron_reduction(
        chain_library, EDGES, voltage_error_limit_pu=0.002
    )
    holdout = make_library(
        chain_library.network, chain_library.voltage_bases, [-6, 2, 4]
    )
    report = evaluate_reduction(selected, holdout, voltage_error_limit_pu=0.004)
    assert report.passed
    assert report.max_voltage_magnitude_error_pu == pytest.approx(0.4 / 120)
    np.testing.assert_allclose(
        pool_scenario_currents(selected, holdout)[:, 0], [-6, 6, 0]
    )


@pytest.mark.parametrize("kind", ["matrix", "source", "bases", "indexing"])
def test_incompatible_holdouts_are_rejected(chain_library, kind):
    original = chain_library.network
    matrix = original.admittance * (2 if kind == "matrix" else 1)
    labels = original.labels[::-1] if kind == "indexing" else original.labels
    sources = {("source", "A"): 121 if kind == "source" else 120}
    network = ElectricalNetwork(matrix, labels, sources)
    bases = chain_library.voltage_bases * (2 if kind == "bases" else 1)
    holdout = make_library(network, bases, [3, -1, -2])
    selected = optimize_kron_reduction(
        chain_library, EDGES, voltage_error_limit_pu=0.002
    )
    with pytest.raises(ValueError, match="selected configuration"):
        evaluate_reduction(selected, holdout)


@pytest.mark.parametrize("limit", [-1, np.nan, np.inf])
def test_invalid_evaluation_limits(chain_library, limit):
    selected = optimize_kron_reduction(chain_library, EDGES)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_reduction(selected, chain_library, voltage_error_limit_pu=limit)


def test_zero_representative_has_undefined_angle(chain_library):
    bases = np.array([120.0, 240.0, 240.0])
    training = make_library(chain_library.network, bases, [3, -1, -2])
    selected = optimize_kron_reduction(
        training,
        EDGES,
        protected_buses={"load"},
        voltage_error_limit_pu=1,
    )
    holdout = make_library(chain_library.network, bases, [600, -200, -400])
    report = evaluate_reduction(selected, holdout)
    assert report.scenarios[0].max_voltage_angle_error_deg is None
    assert not report.passed


def test_source_power_uses_electrical_currents_not_bookkeeping(chain_library):
    matrix = chain_library.network.admittance.toarray()
    matrix[2, 2] += 1
    network = ElectricalNetwork(
        matrix, chain_library.network.labels, chain_library.network.source_voltages
    )
    library = make_library(network, chain_library.voltage_bases, [3, -1, -2])
    selected = optimize_kron_reduction(library, EDGES, voltage_error_limit_pu=1)
    pooled = pool_scenario_currents(selected, library)
    voltage = solve_voltages(network, pooled)
    actual = network.admittance @ voltage
    reference_power = library.voltage[0] * np.conj(library.currents[0])
    expected = (voltage[0] * np.conj(actual[0]) - reference_power).real
    report = evaluate_reduction(selected, library, voltage_error_limit_pu=1)
    assert report.scenarios[0].source_active_power_error_w == pytest.approx(expected[0])
    assert abs(actual[0, 0] - pooled[0, 0]) > 0.1


def test_nonfinite_electrical_voltage_is_rejected(chain_library, monkeypatch):
    selected = optimize_kron_reduction(chain_library, EDGES)
    monkeypatch.setattr(
        type(selected.equivalent),
        "reconstruct",
        lambda *args: np.full(chain_library.voltage.shape, np.nan),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_reduction(selected, chain_library)


@pytest.mark.parametrize("timings", [{"search": -1}, {"search": np.nan}, {1: 0}])
def test_invalid_bundle_timings(chain_library, timings):
    selected = optimize_kron_reduction(chain_library, EDGES)
    radial = radialize_reduction(selected, EDGES)
    with pytest.raises(ValueError, match="Timings"):
        build_result_bundle(radial, chain_library, chain_library, timings_s=timings)


def test_bundle_without_optional_backend_version(chain_library, monkeypatch):
    original = artifacts.version

    def package_version(name):
        if name == "gdm-flow":
            raise PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(artifacts, "version", package_version)
    selected = optimize_kron_reduction(chain_library, EDGES)
    radial = radialize_reduction(selected, EDGES)
    bundle = build_result_bundle(radial, chain_library, chain_library)
    assert bundle["environment"]["versions"]["gdm-flow"] is None


@pytest.mark.parametrize(
    "kind", ["schema", "kind", "missing", "csr", "offsets", "complex", "sources"]
)
def test_malformed_bundle_is_rejected(chain_library, kind):
    selected = optimize_kron_reduction(chain_library, EDGES)
    radial = radialize_reduction(selected, EDGES)
    bundle = deepcopy(build_result_bundle(radial, chain_library, chain_library))
    if kind in {"schema", "kind"}:
        bundle[kind] = "unsupported"
    elif kind == "missing":
        del bundle["equivalent"]
    elif kind == "csr":
        bundle["equivalent"]["admittance"]["indices"][0] = 100
    elif kind == "offsets":
        bundle["equivalent"]["admittance"]["indptr"][0] = 0.5
    elif kind == "complex":
        bundle["equivalent"]["admittance"]["imag"] = [0]
    else:
        bundle["equivalent"]["source_voltages"] *= 2
    with pytest.raises(ValueError):
        network_from_result_bundle(bundle)
