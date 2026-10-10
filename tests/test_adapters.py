import builtins
import os
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionBus,
    DistributionCapacitor,
    DistributionRegulator,
    DistributionSolar,
    DistributionTransformer,
    DistributionVoltageSource,
    MatrixImpedanceBranch,
    MatrixImpedanceSwitch,
)
from gdm.systems.distribution.enums import Phase, VoltageTypes
from scipy.sparse import csr_matrix

from gdm_reduce import (
    adapters,
    build_gdm_scenario_library,
    from_gdm_flow_results,
    get_gdm_search_constraints,
    kron_reduce,
    optimize_kron_reduction,
    radialize_reduction,
)


@pytest.fixture
def reference_model():
    branch = MatrixImpedanceBranch.example()
    source = DistributionVoltageSource.example()
    source.bus = branch.buses[0]
    system = DistributionSystem(auto_add_composed_components=True, name="adapter-test")
    system.add_component(branch)
    system.add_component(source)
    labels = tuple(
        (bus.name, phase.value) for bus in branch.buses for phase in bus.phases
    )
    nominal = float(source.bus.rated_voltage.to("volt").magnitude)
    if source.bus.voltage_type == VoltageTypes.LINE_TO_LINE:
        nominal /= np.sqrt(3)
    phases = nominal * np.exp(1j * np.array([0, -2 * np.pi / 3, 2 * np.pi / 3]))
    admittance = csr_matrix(
        np.block([[np.eye(3), -np.eye(3)], [-np.eye(3), np.eye(3)]])
    )
    result = SimpleNamespace(
        success=True,
        message="Converged",
        max_mismatch_pu=0,
        ybus_result=SimpleNamespace(ybus=admittance, index_to_label=labels),
        voltage=np.tile(phases, 2),
        power_injection=np.zeros(6, dtype=complex),
    )
    return system, result, branch


def test_ingest_voltage_bases_and_uuid_lineage(reference_model):
    system, result, branch = reference_model
    library = from_gdm_flow_results(system, {"light": result, "heavy": result})
    assert library.voltage.shape == library.currents.shape == (6, 2)
    assert library.bus_uuids[branch.buses[0].name] == str(branch.buses[0].uuid)
    assert library.component_buses[str(branch.uuid)] == tuple(
        sorted(bus.name for bus in branch.buses)
    )
    np.testing.assert_allclose(library.voltage_bases, abs(result.voltage))
    assert library.scenarios[0].provenance["backend"] == "gdm-flow"
    branch.length = branch.length * 2
    other = from_gdm_flow_results(system, {"light": result})
    assert other.configuration_fingerprint != library.configuration_fingerprint


def test_failed_and_malformed_backend_results(reference_model):
    system, result, _ = reference_model
    with pytest.raises(ValueError, match="At least one"):
        from_gdm_flow_results(system, {})
    result.success = False
    with pytest.raises(ValueError, match="did not converge"):
        from_gdm_flow_results(system, {"failed": result})
    result.success = True
    result.power_injection = np.zeros(5)
    with pytest.raises(ValueError, match="arrays are invalid"):
        from_gdm_flow_results(system, {"bad-shape": result})


def test_changed_matrix_and_phase_coverage(reference_model):
    system, result, _ = reference_model
    changed = SimpleNamespace(**vars(result))
    changed.ybus_result = SimpleNamespace(
        ybus=result.ybus_result.ybus * 2,
        index_to_label=result.ybus_result.index_to_label,
    )
    with pytest.raises(ValueError, match="configuration"):
        from_gdm_flow_results(system, {"one": result, "two": changed})
    result.ybus_result.index_to_label = tuple(
        ("unknown", phase) if bus.endswith("2") else (bus, phase)
        for bus, phase in result.ybus_result.index_to_label
    )
    with pytest.raises(ValueError, match="GDM bus phases"):
        from_gdm_flow_results(system, {"unknown-bus": result})


def test_line_to_line_bases_and_unsupported_neutral(reference_model):
    system, result, branch = reference_model
    for bus in branch.buses:
        bus.voltage_type = VoltageTypes.LINE_TO_LINE
    library = from_gdm_flow_results(system, {"nominal": result})
    np.testing.assert_allclose(library.voltage_bases, abs(result.voltage) / np.sqrt(3))
    branch.buses[1].phases = [Phase.A, Phase.B, Phase.C, Phase.N]
    with pytest.raises(ValueError, match="A/B/C terminals only"):
        from_gdm_flow_results(system, {"neutral": result})


def install_fake_solver(monkeypatch, solver):
    package = ModuleType("gdm_flow")
    module = ModuleType("gdm_flow.ac_pf")
    module.solve_ac_power_flow_from_components = solver
    monkeypatch.setitem(sys.modules, "gdm_flow", package)
    monkeypatch.setitem(sys.modules, "gdm_flow.ac_pf", module)


def test_generated_scenarios_use_copy_and_record_scales(reference_model, monkeypatch):
    system, result, _ = reference_model
    calls = []

    def solve(working, **options):
        assert working is not system
        calls.append(options)
        return result

    install_fake_solver(monkeypatch, solve)
    library = build_gdm_scenario_library(system, load_scales=[0.5, 1.0])
    assert [call["load_scale"] for call in calls] == [0.5, 1.0]
    assert [item.name for item in library.scenarios] == ["load-000", "load-001"]
    assert library.scenarios[0].provenance["load_scale"] == "0.5"


def test_backend_configuration_mutation_is_rejected(reference_model, monkeypatch):
    system, result, _ = reference_model

    def solve(working, **options):
        working.get_source_bus().in_service = False
        return result

    install_fake_solver(monkeypatch, solve)
    with pytest.raises(ValueError, match="changed the fixed"):
        build_gdm_scenario_library(system)
    assert system.get_source_bus().in_service is True


def test_unsupported_phases_are_rejected_before_solving(reference_model, monkeypatch):
    system, _, branch = reference_model
    branch.buses[1].phases = [Phase.A, Phase.B, Phase.C, Phase.S1]

    def solve(*args, **kwargs):
        pytest.fail("Unsupported terminals must be rejected before the backend solve.")

    install_fake_solver(monkeypatch, solve)
    with pytest.raises(ValueError, match="A/B/C terminals only"):
        build_gdm_scenario_library(system)


@pytest.mark.parametrize(
    "options",
    [
        {"load_scales": []},
        {"load_scales": [np.nan]},
        {"load_scales": [-1]},
        {"solar_scale": -1},
        {"tolerance": 0},
        {"max_iterations": 0},
    ],
)
def test_invalid_generation_options(reference_model, options):
    with pytest.raises(ValueError):
        build_gdm_scenario_library(reference_model[0], **options)


@pytest.mark.parametrize("missing", ["gdm_flow", "scipy.missing"])
def test_missing_backend_and_unrelated_import_errors(
    reference_model, monkeypatch, missing
):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "gdm_flow.ac_pf":
            raise ModuleNotFoundError(missing, name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    expected = RuntimeError if missing == "gdm_flow" else ModuleNotFoundError
    with pytest.raises(expected):
        build_gdm_scenario_library(reference_model[0])


def test_imported_results_without_installed_backend_version(
    reference_model, monkeypatch
):
    original = adapters.version

    def package_version(name):
        if name == "gdm-flow":
            raise PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(adapters, "version", package_version)
    library = from_gdm_flow_results(reference_model[0], {"one": reference_model[1]})
    assert library.scenarios[0].provenance["backend_version"] == "unknown"


def test_gdm_physical_topology_and_stale_configuration(reference_model):
    system, result, branch = reference_model
    library = from_gdm_flow_results(system, {"nominal": result})
    constraints = get_gdm_search_constraints(system, library)
    assert constraints.bus_edges == (tuple(sorted(bus.name for bus in branch.buses)),)
    assert constraints.protected_buses == frozenset({system.get_source_bus().name})
    branch.in_service = False
    with pytest.raises(ValueError, match="reference library configuration"):
        get_gdm_search_constraints(system, library)


def test_transformer_and_switch_boundaries_are_protected(reference_model):
    system, result, branch = reference_model
    transformer = DistributionTransformer.example()
    for winding, bus in zip(transformer.equipment.windings, branch.buses, strict=True):
        winding.rated_voltage = bus.rated_voltage * np.sqrt(3)
    transformer.buses = list(branch.buses)
    system.add_component(transformer)
    switch = MatrixImpedanceSwitch.example()
    switch.buses = list(branch.buses)
    switch.is_closed = [False, False, False]
    system.add_component(switch)
    library = from_gdm_flow_results(system, {"nominal": result})
    constraints = get_gdm_search_constraints(system, library)
    assert constraints.protected_buses == frozenset(bus.name for bus in branch.buses)
    assert len(constraints.bus_edges) == 1


def test_open_switch_is_not_a_physical_edge(reference_model):
    system, result, branch = reference_model
    branch.in_service = False
    switch = MatrixImpedanceSwitch.example()
    switch.buses = list(branch.buses)
    switch.is_closed = [False, False, False]
    system.add_component(switch)
    library = from_gdm_flow_results(system, {"nominal": result})
    assert get_gdm_search_constraints(system, library).bus_edges == ()


@pytest.mark.parametrize(
    "component_type",
    [
        DistributionSolar,
        DistributionBattery,
        DistributionCapacitor,
    ],
)
def test_der_and_capacitor_buses_are_protected(reference_model, component_type):
    system, result, branch = reference_model
    component = component_type.example()
    component.bus = branch.buses[1]
    system.add_component(component)
    library = from_gdm_flow_results(system, {"nominal": result})
    assert (
        branch.buses[1].name
        in get_gdm_search_constraints(system, library).protected_buses
    )


def test_remote_controller_measurement_bus_is_protected(reference_model):
    system, result, branch = reference_model
    regulator = DistributionRegulator.example()
    for winding, bus in zip(regulator.equipment.windings, branch.buses, strict=True):
        winding.rated_voltage = bus.rated_voltage * np.sqrt(3)
    regulator.buses = list(branch.buses)
    remote = DistributionBus.example()
    for controller in regulator.controllers:
        controller.controlled_bus = remote
    system.add_component(regulator)
    extended = SimpleNamespace(**vars(result))
    extended.voltage = np.tile(result.voltage[:3], 3)
    extended.power_injection = np.zeros(9, dtype=complex)
    identity = np.eye(3)
    zero = np.zeros((3, 3))
    extended.ybus_result = SimpleNamespace(
        index_to_label=(
            *result.ybus_result.index_to_label,
            *((remote.name, phase.value) for phase in remote.phases),
        ),
        ybus=csr_matrix(
            np.block(
                [
                    [2 * identity, -identity, -identity],
                    [-identity, identity, zero],
                    [-identity, zero, identity],
                ]
            )
        ),
    )
    library = from_gdm_flow_results(system, {"nominal": extended})
    constraints = get_gdm_search_constraints(system, library)
    assert remote.name not in {bus.name for bus in regulator.buses}
    assert remote.name in constraints.protected_buses


def test_ieee13_library_and_exact_reconstruction():
    pytest.importorskip("gdm_flow")
    default = Path(__file__).resolve().parents[2] / "gdm-flow" / "tests" / "data"
    root = Path(os.environ.get("GDM_REDUCE_TEST_DATA", default))
    model = root / "ieee-13" / "gdm" / "ieee13_system.json"
    if not model.exists():
        pytest.skip("Set GDM_REDUCE_TEST_DATA to the external feeder-data directory.")
    system = DistributionSystem.from_json(str(model))
    library = build_gdm_scenario_library(system, load_scales=[0.5, 1.0])
    source = system.get_source_bus().name
    result = kron_reduce(library.network, {source})
    reconstructed = result.reconstruct(library.currents, result.solve(library.currents))
    np.testing.assert_allclose(reconstructed, library.voltage, rtol=1e-9, atol=1e-7)
    assert library.voltage.shape == (41, 2)
    constraints = get_gdm_search_constraints(system, library)
    selected = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=0.001,
    )
    assert selected.max_voltage_error_pu <= 0.001
    assert constraints.protected_buses <= set(selected.retained_buses)
    assert len(selected.bus_uuid_to_retained_uuid) == len(library.bus_uuids)
    radial = radialize_reduction(selected, constraints.bus_edges)
    assert radial.max_selected_voltage_difference_v < 1e-7
    assert set(selected.retained_buses) <= set(radial.retained_buses)
