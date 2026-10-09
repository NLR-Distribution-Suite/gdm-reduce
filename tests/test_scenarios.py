from dataclasses import replace

import numpy as np
import pytest

from gdm_reduce import ElectricalNetwork, solve_voltages
from gdm_reduce.scenarios import (
    ReferenceScenario,
    ScenarioLibrary,
    configuration_fingerprint,
)


@pytest.fixture
def reference():
    network = ElectricalNetwork(
        np.array([[10, -10], [-10, 10]]),
        (("source", "A"), ("load", "A")),
        {("source", "A"): 120},
    )
    bases = np.array([120.0, 120.0])
    voltage = solve_voltages(network, np.array([0, -2]))
    scenario = ReferenceScenario(
        "nominal",
        network.labels,
        voltage,
        np.array([2, -2]),
        configuration_fingerprint(network, bases),
        True,
        1e-9,
        {"backend": "closed-form"},
    )
    return network, bases, scenario


def test_reference_library_preserves_original_data(reference):
    network, bases, scenario = reference
    library = ScenarioLibrary(network, bases, [scenario])
    np.testing.assert_allclose(library.max_kcl_residual_a, [0], atol=1e-12)
    assert library.voltage.shape == library.currents.shape == (2, 1)
    bases[0] = 1
    network.admittance.data[:] = 0
    assert library.voltage_bases[0] == 120
    assert library.network.admittance.nnz == 4
    with pytest.raises(ValueError):
        scenario.voltage[0] = 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"labels": (("load", "A"), ("source", "A"))}, "indexing"),
        ({"configuration_fingerprint": "other"}, "configuration"),
        ({"reference_mismatch_pu": 0.1}, "convergence mismatch"),
        ({"voltage": np.array([121, 119.8])}, "source"),
        ({"currents": np.array([2, -3])}, "KCL"),
    ],
)
def test_incompatible_references_are_rejected(reference, change, message):
    network, bases, scenario = reference
    with pytest.raises(ValueError, match=message):
        ScenarioLibrary(network, bases, [replace(scenario, **change)])


def test_missing_or_duplicate_scenarios_are_rejected(reference):
    network, bases, scenario = reference
    for scenarios in ([], [scenario, scenario]):
        with pytest.raises(ValueError, match="uniquely named"):
            ScenarioLibrary(network, bases, scenarios)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " "},
        {"reference_success": False},
        {"reference_mismatch_pu": np.nan},
        {"reference_mismatch_pu": -1},
        {"voltage": np.zeros(2)},
        {"currents": np.zeros(3)},
        {"currents": np.array([np.nan, 0])},
        {"labels": []},
        {"configuration_fingerprint": ""},
        {"provenance": {"scale": 0.5}},
    ],
)
def test_invalid_scenario_fields(reference, changes):
    with pytest.raises(ValueError):
        replace(reference[2], **changes)


@pytest.mark.parametrize("bases", [np.zeros(2), np.ones(3), [120, np.inf]])
def test_invalid_voltage_bases(reference, bases):
    with pytest.raises(ValueError, match="Voltage bases"):
        ScenarioLibrary(reference[0], bases, [reference[2]])


def test_fingerprint_tracks_electrical_changes(reference):
    network, bases, scenario = reference
    assert (
        configuration_fingerprint(network, bases) == scenario.configuration_fingerprint
    )
    assert (
        configuration_fingerprint(network, bases * 2)
        != scenario.configuration_fingerprint
    )
    assert configuration_fingerprint(network, bases, model_signature="changed") != (
        scenario.configuration_fingerprint
    )


def test_lineage_and_validation_policy(reference):
    network, bases, scenario = reference
    library = ScenarioLibrary(
        network,
        bases,
        [scenario],
        bus_uuids={"source": "uuid-1", "load": "uuid-2"},
        component_buses={"branch-uuid": ("source", "load")},
    )
    assert library.component_buses["branch-uuid"] == ("source", "load")
    for options in (
        {"bus_uuids": {"source": "uuid-1"}},
        {"component_buses": {"branch": ("unknown",)}},
        {"kcl_atol_a": -1},
    ):
        with pytest.raises(ValueError):
            ScenarioLibrary(network, bases, [scenario], **options)
