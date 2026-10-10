import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gdm_reduce import ElectricalNetwork, kron_reduce, solve_voltages


@pytest.fixture
def coupled_feeder():
    labels = tuple(
        (bus, phase) for bus in ("source", "middle", "load") for phase in "ABC"
    )
    impedance = np.array(
        [
            [0.4 + 0.8j, 0.03 + 0.1j, 0.02 + 0.08j],
            [0.03 + 0.1j, 0.5 + 0.9j, 0.04 + 0.09j],
            [0.02 + 0.08j, 0.04 + 0.09j, 0.6 + 1.0j],
        ]
    )
    series = np.linalg.solve(impedance, np.eye(3))
    admittance = np.zeros((9, 9), dtype=complex)
    for upstream, downstream, scale in ((0, 3, 1.0), (3, 6, 0.7)):
        upstream_indices = np.arange(upstream, upstream + 3)
        downstream_indices = np.arange(downstream, downstream + 3)
        admittance[np.ix_(upstream_indices, upstream_indices)] += scale * series
        admittance[np.ix_(downstream_indices, downstream_indices)] += scale * series
        admittance[np.ix_(upstream_indices, downstream_indices)] -= scale * series
        admittance[np.ix_(downstream_indices, upstream_indices)] -= scale * series
    sources = {
        ("source", "A"): 7200 + 0j,
        ("source", "B"): 7200 * np.exp(-2j * np.pi / 3),
        ("source", "C"): 7200 * np.exp(2j * np.pi / 3),
    }
    return ElectricalNetwork(csr_matrix(admittance), labels, sources)


@pytest.mark.parametrize("scenarios", [None, 4])
def test_exact_loaded_bus_reduction_and_reconstruction(coupled_feeder, scenarios):
    rng = np.random.default_rng(42)
    shape = (9,) if scenarios is None else (9, scenarios)
    currents = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    original_admittance = coupled_feeder.admittance.copy()
    original_currents = currents.copy()
    full_voltage = solve_voltages(coupled_feeder, currents)
    result = kron_reduce(coupled_feeder, {"source", "load"}, block_size=2)
    retained_voltage = result.solve(currents)
    reconstructed = result.reconstruct(currents, retained_voltage)
    np.testing.assert_allclose(
        retained_voltage, full_voltage[result.retained_indices], rtol=1e-10
    )
    np.testing.assert_allclose(reconstructed, full_voltage, rtol=1e-10)
    np.testing.assert_allclose(
        (coupled_feeder.admittance @ reconstructed)[3:], currents[3:], atol=1e-10
    )
    np.testing.assert_array_equal(currents, original_currents)
    np.testing.assert_array_equal(
        coupled_feeder.admittance.toarray(), original_admittance.toarray()
    )
    assert result.retained_labels == tuple(
        label for label in coupled_feeder.labels if label[0] != "middle"
    )
    assert result.admittance.shape == (6, 6)


def test_source_currents_are_outputs_not_prescribed(coupled_feeder):
    currents = np.zeros(9, dtype=complex)
    expected = solve_voltages(coupled_feeder, currents)
    currents[:3] = 1000 + 20j
    np.testing.assert_allclose(solve_voltages(coupled_feeder, currents), expected)


def test_source_elimination_is_rejected(coupled_feeder):
    with pytest.raises(ValueError, match="source bus"):
        kron_reduce(coupled_feeder, {"load"})


def test_disconnected_terminals_are_rejected():
    with pytest.raises(ValueError, match="connected"):
        ElectricalNetwork(
            np.eye(2), (("source", "A"), ("island", "A")), {("source", "A"): 120}
        )


def test_closed_form_scalar_feeder():
    network = ElectricalNetwork(
        np.array([[10, -10, 0], [-10, 20, -10], [0, -10, 10]]),
        (("source", "A"), ("middle", "A"), ("load", "A")),
        {("source", "A"): 120},
    )
    currents = np.array([0, -1, -2])
    result = kron_reduce(network, {"source", "load"})
    np.testing.assert_allclose(result.admittance.toarray(), [[5, -5], [-5, 5]])
    np.testing.assert_allclose(result.transform_currents(currents), [-0.5, -2.5])
    np.testing.assert_allclose(result.solve(currents), [120, 119.5])
    np.testing.assert_allclose(
        result.reconstruct(currents, result.solve(currents)), [120, 119.7, 119.5]
    )


def test_missing_phases_and_whole_bus_elimination():
    labels = (
        ("source", "A"),
        ("source", "B"),
        ("source", "C"),
        ("middle", "A"),
        ("middle", "B"),
        ("load", "A"),
    )
    admittance = np.zeros((6, 6), dtype=complex)
    series = np.linalg.solve(
        np.array([[0.4 + 0.8j, 0.03 + 0.1j], [0.03 + 0.1j, 0.5 + 0.9j]]),
        np.eye(2),
    )
    admittance[np.ix_([0, 1], [0, 1])] += series
    admittance[np.ix_([3, 4], [3, 4])] += series
    admittance[np.ix_([0, 1], [3, 4])] -= series
    admittance[np.ix_([3, 4], [0, 1])] -= series
    downstream = 1 / (0.4 + 0.7j)
    admittance[3, 3] += downstream
    admittance[5, 5] += downstream
    admittance[3, 5] -= downstream
    admittance[5, 3] -= downstream
    sources = {
        label: 120 * np.exp(-2j * np.pi * index / 3)
        for index, label in enumerate(labels[:3])
    }
    network = ElectricalNetwork(admittance, labels, sources)
    currents = np.array([0, 0, 0, -1 + 0.2j, -2 + 0.5j, -3 + 0.4j])
    result = kron_reduce(network, {"source", "load"})
    assert result.retained_labels == labels[:3] + (labels[-1],)
    np.testing.assert_allclose(
        result.reconstruct(currents, result.solve(currents)),
        solve_voltages(network, currents),
        rtol=1e-10,
    )


def test_no_elimination_and_source_only_reduction(coupled_feeder):
    currents = np.arange(9) - 5j
    result = kron_reduce(coupled_feeder, {"source", "middle", "load"})
    assert len(result.eliminated_indices) == 0
    np.testing.assert_array_equal(result.transform_currents(currents), currents)
    np.testing.assert_allclose(
        result.admittance.toarray(), coupled_feeder.admittance.toarray()
    )
    full_voltage = solve_voltages(coupled_feeder, currents)
    np.testing.assert_allclose(
        result.reconstruct(currents, result.solve(currents)), full_voltage
    )
    source_only = kron_reduce(coupled_feeder, {"source"})
    np.testing.assert_allclose(
        source_only.reconstruct(currents, source_only.solve(currents)), full_voltage
    )


def test_all_terminals_are_sources():
    network = ElectricalNetwork(
        np.zeros((1, 1)), (("source", "A"),), {("source", "A"): 120}
    )
    currents = np.array([[2, 3]])
    np.testing.assert_allclose(solve_voltages(network, currents), [[120, 120]])
    result = kron_reduce(network, {"source"})
    np.testing.assert_allclose(result.solve(currents), [[120, 120]])


@pytest.mark.parametrize(
    "currents", [np.zeros(8), np.zeros((9, 0)), np.full(9, np.nan), np.array(1)]
)
def test_invalid_current_shapes_and_values(coupled_feeder, currents):
    with pytest.raises(ValueError, match="currents"):
        solve_voltages(coupled_feeder, currents)


def test_reconstruction_requires_matching_scenarios(coupled_feeder):
    result = kron_reduce(coupled_feeder, {"source", "load"})
    with pytest.raises(ValueError, match="matching scenario"):
        result.reconstruct(np.zeros((9, 2)), np.zeros(6))


@pytest.mark.parametrize("retained", ["source", set(), {"unknown"}])
def test_invalid_retained_buses(coupled_feeder, retained):
    with pytest.raises(ValueError, match="[Rr]etained"):
        kron_reduce(coupled_feeder, retained)


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_invalid_block_size(coupled_feeder, block_size):
    with pytest.raises(ValueError, match="block_size"):
        kron_reduce(coupled_feeder, {"source", "load"}, block_size=block_size)


@pytest.mark.parametrize("condition_limit", [0, np.inf, np.nan])
def test_invalid_condition_limit(coupled_feeder, condition_limit):
    with pytest.raises(ValueError, match="max_condition_number"):
        solve_voltages(
            coupled_feeder, np.zeros(9), max_condition_number=condition_limit
        )


def test_singular_and_ill_conditioned_grounded_blocks():
    labels = (("source", "A"), ("middle", "A"), ("load", "A"))
    sources = {("source", "A"): 120}
    singular = ElectricalNetwork(np.ones((3, 3)), labels, sources)
    with pytest.raises(ValueError, match="singular"):
        solve_voltages(singular, np.zeros(3))
    ill_conditioned = ElectricalNetwork(
        np.array([[1 + 1e-12, -1e-12, -1], [-1e-12, 1e-12, 0], [-1, 0, 1]]),
        labels,
        sources,
    )
    with pytest.raises(ValueError, match="ill-conditioned"):
        kron_reduce(ill_conditioned, {"source", "load"}, max_condition_number=1e6)


@pytest.mark.parametrize(
    "labels",
    [
        (),
        (("source", "N"),),
        (("source", "L1"),),
        (("", "A"),),
        (("source", "A"), ("source", "A")),
    ],
)
def test_invalid_terminal_labels(labels):
    with pytest.raises(ValueError, match="[Ll]abels"):
        ElectricalNetwork(np.eye(len(labels)), labels, {("source", "A"): 120})


@pytest.mark.parametrize(
    "admittance", [np.eye(2), np.array([[np.inf]]), np.array([[np.nan]])]
)
def test_invalid_admittance(admittance):
    with pytest.raises(ValueError, match="Admittance"):
        ElectricalNetwork(admittance, (("source", "A"),), {("source", "A"): 120})


@pytest.mark.parametrize(
    "sources",
    [{}, {("unknown", "A"): 120}, {("source", "A"): 0}, {("source", "A"): np.inf}],
)
def test_invalid_sources(sources):
    with pytest.raises(ValueError, match="[Ss]ource"):
        ElectricalNetwork(np.eye(1), (("source", "A"),), sources)


def test_network_copies_caller_owned_inputs():
    matrix = np.eye(1, dtype=complex)
    sources = {("source", "A"): 120}
    labels = [("source", "A")]
    network = ElectricalNetwork(matrix, labels, sources)
    matrix[0, 0] = 100
    sources[("source", "A")] = 240
    labels.append(("other", "A"))
    assert network.admittance[0, 0] == 1
    assert network.source_voltages[("source", "A")] == 120
    assert network.labels == (("source", "A"),)
