from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest
from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBus,
    DistributionLoad,
    DistributionReactor,
    DistributionTransformer,
    DistributionVoltageSource,
    MatrixImpedanceBranch,
    MatrixImpedanceSwitch,
)
from gdm.systems.distribution.equipment import (
    CircuitBreakerEquipment,
    DisconnectorEquipment,
    EarthingSwitchEquipment,
    PowerTransformerEquipment,
)
from infrasys import SingleTimeSeries

from gdm_reduce import (
    from_gdm_flow_results,
    get_gdm_search_constraints,
    optimize_kron_reduction,
    radialize_reduction,
)
from gdm_reduce.gdm_export import (
    build_reduced_gdm,
    save_reduced_gdm,
    validate_reduced_gdm,
)

solve_ac_power_flow_from_components = pytest.importorskip(
    "gdm_flow.ac_pf"
).solve_ac_power_flow_from_components


@pytest.fixture
def export_case():
    first = MatrixImpedanceBranch.example()
    for bus in first.buses:
        bus.rated_voltage = type(bus.rated_voltage)(7200, "volt")
    end = first.buses[1].model_copy(deep=True, update={"uuid": uuid4(), "name": "end"})
    equipment = first.equipment.model_copy(
        deep=True, update={"uuid": uuid4(), "name": "second-equipment"}
    )
    second = first.model_copy(
        update={
            "uuid": uuid4(),
            "name": "second",
            "buses": [first.buses[1], end],
            "equipment": equipment,
        }
    )
    for line in (first, second):
        line.equipment.c_matrix = type(line.equipment.c_matrix)(
            np.zeros((3, 3)), "farad/m"
        )
    source = DistributionVoltageSource.example()
    source.bus = first.buses[0]
    load = DistributionLoad.example()
    load.bus = first.buses[1]
    system = DistributionSystem(auto_add_composed_components=True, name="export-test")
    for component in (first, second, source, load):
        system.add_component(component)
    reference = solve_ac_power_flow_from_components(
        system, v0_complex={}, tolerance=1e-8
    )
    assert reference.success
    library = from_gdm_flow_results(
        system,
        {"nominal": reference},
        scenario_provenance={"nominal": {"load_scale": "1.0", "solar_scale": "1.0"}},
    )
    constraints = get_gdm_search_constraints(system, library)
    selection = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses={source.bus.name, "end"},
        voltage_error_limit_pu=1.0,
        target_reduction_fraction=1 / 3,
    )
    radial = radialize_reduction(selection, constraints.bus_edges)
    return system, library, radial, load


def test_export_contracts_series_path_and_preserves_original(export_case, tmp_path):
    system, library, radial, load = export_case
    export = build_reduced_gdm(system, radial, library)
    assert len(list(export.system.get_components(DistributionBus))) == 2
    assert len(list(export.system.get_components(MatrixImpedanceBranch))) == 1
    copied_load = export.system.get_component_by_uuid(load.uuid)
    assert copied_load.bus.name == radial.selection.original_to_retained[load.bus.name]
    assert copied_load.equipment.model_dump() == load.equipment.model_dump()
    assert len(list(system.get_components(DistributionBus))) == 3
    assert export.report["ybus_verification"]["max_absolute_error_s"] < 1e-6
    assert not export.report["nonlinear_accuracy_verified"]
    path = tmp_path / "reduced.json"
    save_reduced_gdm(export, path)
    reloaded = DistributionSystem.from_json(path)
    assert len(list(reloaded.get_components(DistributionBus))) == 2
    assert reloaded.get_component_by_uuid(load.uuid).bus.name == copied_load.bus.name
    reference = solve_ac_power_flow_from_components(
        reloaded, v0_complex={}, tolerance=1e-8
    )
    assert reference.success


@pytest.mark.parametrize(
    "device_type",
    [
        DistributionReactor,
        CircuitBreakerEquipment,
        DisconnectorEquipment,
        EarthingSwitchEquipment,
        PowerTransformerEquipment,
    ],
)
def test_export_preserves_gdm24_device_boundaries(export_case, device_type, tmp_path):
    system, _, _, _ = export_case
    end = system.get_component(DistributionBus, "end")
    terminal = end.model_copy(
        deep=True, update={"uuid": uuid4(), "name": "device-terminal"}
    )
    if device_type is DistributionReactor:
        device = DistributionReactor.example()
        device.buses = [end, terminal]
    elif device_type is PowerTransformerEquipment:
        device = DistributionTransformer.example()
        device.equipment = PowerTransformerEquipment.example()
        device.equipment.windings[1].is_grounded = True
        terminal.rated_voltage = type(terminal.rated_voltage)(400, "volt")
        terminal.voltage_type = "line-to-line"
        device.buses = [end, terminal]
    else:
        device = MatrixImpedanceSwitch.example()
        device.equipment = device_type.example()
        device.buses = [end, terminal]
    system.add_component(device)
    reference = solve_ac_power_flow_from_components(
        system, v0_complex={}, tolerance=1e-8
    )
    assert reference.success
    library = from_gdm_flow_results(system, {"nominal": reference})
    constraints = get_gdm_search_constraints(system, library)
    assert {end.name, terminal.name} <= constraints.protected_buses
    selection = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=1.0,
        target_reduction_fraction=0.25,
    )
    radial = radialize_reduction(selection, constraints.bus_edges)
    export = build_reduced_gdm(system, radial, library)
    path = tmp_path / "station-reduced.json"
    save_reduced_gdm(export, path)
    restored = DistributionSystem.from_json(path)
    copied_device = restored.get_component_by_uuid(device.uuid)
    assert type(copied_device) is type(device)
    assert type(copied_device.equipment) is type(device.equipment)
    assert copied_device.model_dump(mode="json") == device.model_dump(mode="json")
    assert len(list(restored.get_components(DistributionBus))) == 3
    assert export.report["ybus_verification"]["max_absolute_error_s"] < 1e-6


def test_export_requires_explicit_line_charging_policy(export_case):
    system, _, _, _ = export_case
    for line in system.get_components(MatrixImpedanceBranch):
        line.equipment.c_matrix = type(line.equipment.c_matrix)(
            np.eye(3), "nanofarad/m"
        )
    reference = solve_ac_power_flow_from_components(
        system, v0_complex={}, tolerance=1e-8
    )
    library = from_gdm_flow_results(system, {"nominal": reference})
    constraints = get_gdm_search_constraints(system, library)
    selection = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses={system.get_source_bus().name, "end"},
        voltage_error_limit_pu=1.0,
        target_reduction_fraction=1 / 3,
    )
    radial = radialize_reduction(selection, constraints.bus_edges)
    with pytest.raises(ValueError, match="charging"):
        build_reduced_gdm(system, radial, library)
    export = build_reduced_gdm(system, radial, library, allow_series_only=True)
    assert export.report["charging_policy"] == "series-only"
    assert export.report["omitted_charging_branch_uuids"]


def test_load_profiles_survive_relocation_and_serialization(export_case, tmp_path):
    system, library, radial, load = export_case
    profile = SingleTimeSeries.from_array(
        [1.0, 0.8, 1.2], "load_multiplier", datetime(2025, 1, 1), timedelta(hours=1)
    )
    system.add_time_series(profile, load)
    export = build_reduced_gdm(system, radial, library)
    path = tmp_path / "profiles.json"
    save_reduced_gdm(export, path)
    reloaded = DistributionSystem.from_json(path)
    owner = reloaded.get_component_by_uuid(load.uuid)
    np.testing.assert_array_equal(
        reloaded.get_time_series(owner, name="load_multiplier").data, profile.data
    )
    assert owner.bus.name == radial.selection.original_to_retained[load.bus.name]
    assert system.has_time_series(load)
    assert path.with_name("profiles.export-report.json").exists()
    with pytest.raises(FileExistsError):
        save_reduced_gdm(export, path)


def test_nonlinear_export_errors_are_measured_separately(export_case):
    system, library, radial, _ = export_case
    export = build_reduced_gdm(system, radial, library)
    report = validate_reduced_gdm(export, library, voltage_error_limit_pu=0.01)
    assert report["passed"]
    assert report["scenarios"][0]["converged"]
    assert report["scenarios"][0]["max_voltage_magnitude_error_pu"] >= 0
    assert not export.report["prescribed_current_certificate_applies"]
    assert export.report["nonlinear_validation"]["evaluation"] == report
    assert export.report["nonlinear_accuracy_verified"]


def test_convergence_does_not_certify_export_accuracy(export_case):
    system, library, radial, _ = export_case
    export = build_reduced_gdm(system, radial, library)
    report = validate_reduced_gdm(export, library, voltage_error_limit_pu=0.0)
    assert export.report["nonlinear_solves_converged"]
    assert not report["passed"]
    assert not export.report["nonlinear_accuracy_verified"]


def test_cli_exports_real_reduced_gdm(export_case, tmp_path):
    from gdm_reduce import cli

    system, _, _, _ = export_case
    original = tmp_path / "original.json"
    system.to_json(original)
    before = original.read_bytes()
    destination = tmp_path / "reduced.json"
    bundle = tmp_path / "numeric.json"
    assert (
        cli.main(
            [
                str(original),
                "--export-gdm",
                str(destination),
                "--output",
                str(bundle),
                "--target-reduction",
                str(1 / 3),
                "--protect-bus",
                "end",
            ]
        )
        == 0
    )
    reloaded = DistributionSystem.from_json(destination)
    assert len(list(reloaded.get_components(DistributionBus))) == 2
    assert original.read_bytes() == before
    import json

    report = json.loads(destination.with_name("reduced.export-report.json").read_text())
    assert report["nonlinear_accuracy_verified"]
    assert report["nonlinear_validation"]["holdout"]["passed"]
    with pytest.raises(SystemExit):
        cli.main([str(original), "--export-gdm", str(original), "--overwrite"])


@pytest.mark.parametrize("options", [{"ybus_atol_s": -1}, {"ybus_rtol": np.nan}])
def test_invalid_ybus_tolerances_are_rejected(export_case, options):
    system, library, radial, _ = export_case
    with pytest.raises(ValueError, match="tolerances"):
        build_reduced_gdm(system, radial, library, **options)


def test_charged_schur_equivalent_is_not_misrepresented_as_series(export_case):
    system, _, _, _ = export_case
    for line in system.get_components(MatrixImpedanceBranch):
        line.equipment.c_matrix = type(line.equipment.c_matrix)(
            np.eye(3) * 1e-6, "farad/m"
        )
    solved = solve_ac_power_flow_from_components(
        system, include_shunt=True, v0_complex={}, tolerance=1e-8
    )
    assert solved.success
    library = from_gdm_flow_results(system, {"nominal": solved})
    constraints = get_gdm_search_constraints(system, library)
    selected = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses={system.get_source_bus().name, "end"},
        voltage_error_limit_pu=1.0,
        target_reduction_fraction=1 / 3,
    )
    radial = radialize_reduction(selected, constraints.bus_edges)
    with pytest.raises(ValueError, match="not realizable"):
        build_reduced_gdm(system, radial, library, allow_series_only=True)


@pytest.mark.parametrize("owner_type", [DistributionBus, MatrixImpedanceBranch])
def test_metadata_on_removed_equipment_is_not_silently_discarded(
    export_case, owner_type
):
    system, library, radial, _ = export_case
    owner = next(
        component
        for component in system.get_components(owner_type)
        if (
            component.name not in radial.retained_buses
            if owner_type is DistributionBus
            else True
        )
    )
    profile = SingleTimeSeries.from_array(
        [1.0, 2.0], "equipment_profile", datetime(2025, 1, 1), timedelta(hours=1)
    )
    system.add_time_series(profile, owner)
    with pytest.raises(ValueError, match="time series"):
        build_reduced_gdm(system, radial, library)


def test_nonlinear_validation_rejects_unrecorded_operating_point(export_case):
    system, _, radial, _ = export_case
    solved = solve_ac_power_flow_from_components(system, v0_complex={}, tolerance=1e-8)
    library = from_gdm_flow_results(system, {"nominal": solved})
    export = build_reduced_gdm(system, radial, library)
    with pytest.raises(ValueError, match="provenance"):
        validate_reduced_gdm(export, library)


def test_contracted_line_equipment_profiles_are_not_lost(export_case):
    system, library, radial, _ = export_case
    equipment = next(system.get_components(MatrixImpedanceBranch)).equipment
    profile = SingleTimeSeries.from_array(
        [80.0, 90.0], "ampacity", datetime(2025, 1, 1), timedelta(hours=1)
    )
    system.add_time_series(profile, equipment)
    with pytest.raises(ValueError, match="time series"):
        build_reduced_gdm(system, radial, library)
