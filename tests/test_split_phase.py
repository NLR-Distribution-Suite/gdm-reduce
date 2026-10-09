from uuid import uuid4

import numpy as np
import pytest
from gdm.distribution import DistributionSystem
from gdm.distribution.components import (
    DistributionBus,
    DistributionLoad,
    DistributionTransformer,
    DistributionVoltageSource,
    MatrixImpedanceBranch,
)
from gdm.distribution.enums import Phase, VoltageTypes

from gdm_reduce import (
    ElectricalNetwork,
    ReferenceScenario,
    ScenarioLibrary,
    build_gdm_scenario_library,
    build_result_bundle,
    configuration_fingerprint,
    evaluate_reduction,
    get_gdm_search_constraints,
    kron_reduce,
    network_from_result_bundle,
    optimize_kron_reduction,
    radialize_reduction,
    solve_voltages,
)


@pytest.fixture
def grounded_system():
    pytest.importorskip("gdm_flow")

    def bus(name, phases, voltage):
        original = DistributionBus.example()
        return original.model_copy(
            update={
                "name": name,
                "uuid": uuid4(),
                "phases": phases,
                "rated_voltage": original.rated_voltage * (voltage / 400),
                "voltage_type": VoltageTypes.LINE_TO_GROUND,
            },
            deep=True,
        )

    primary = bus("primary", [Phase.A], 7200)
    secondary = bus("secondary", [Phase.S2, Phase.S1], 120)
    customer = bus("customer", [Phase.S2, Phase.S1], 120)
    sample = DistributionTransformer.example()
    winding_type = type(sample.equipment.windings[0])
    windings = [
        winding_type(
            name=f"winding-{index}",
            resistance=resistance,
            is_grounded=True,
            rated_voltage=sample.equipment.windings[0].rated_voltage
            * (voltage / 12470),
            voltage_type=VoltageTypes.LINE_TO_GROUND,
            rated_power=sample.equipment.windings[0].rated_power * (25000 / 56000),
            num_phases=1,
            connection_type=sample.equipment.windings[0].connection_type,
            tap_positions=[1.0],
        )
        for index, (voltage, resistance) in enumerate(
            ((7200, 0.2), (120, 0.4), (120, 0.4))
        )
    ]
    pair_type = type(sample.equipment.coupling_sequences[0])
    equipment = type(sample.equipment)(
        name="center-tap",
        windings=windings,
        pct_no_load_loss=0,
        pct_full_load_loss=0.6,
        winding_reactances=[2.0, 2.0, 1.0],
        coupling_sequences=[
            pair_type(from_index=first, to_index=second)
            for first, second in ((0, 1), (0, 2), (1, 2))
        ],
        is_center_tapped=True,
    )
    transformer = DistributionTransformer(
        name="center-tap",
        buses=[primary, secondary, secondary],
        equipment=equipment,
        winding_phases=[[Phase.A], [Phase.N, Phase.S1], [Phase.N, Phase.S2]],
    )
    branch_sample = MatrixImpedanceBranch.example()
    branch_equipment = type(branch_sample.equipment)(
        name="service-line",
        r_matrix=type(branch_sample.equipment.r_matrix)(
            [[0.002, 0.0003], [0.0003, 0.002]], "ohm/m"
        ),
        x_matrix=type(branch_sample.equipment.x_matrix)(
            [[0.001, 0.0002], [0.0002, 0.001]], "ohm/m"
        ),
        c_matrix=type(branch_sample.equipment.c_matrix)(
            np.zeros((2, 2)), "nanofarad/m"
        ),
        ampacity=branch_sample.equipment.ampacity,
    )
    branch = MatrixImpedanceBranch(
        name="service-line",
        buses=[secondary, customer],
        phases=[Phase.S2, Phase.S1],
        length=branch_sample.length
        * (10 / float(branch_sample.length.to("meter").magnitude)),
        equipment=branch_equipment,
    )
    source = DistributionVoltageSource.example()
    source.bus = primary
    source.phases = [Phase.A]
    system = DistributionSystem(
        name="grounded-service", auto_add_composed_components=True
    )
    system.add_components(source, transformer, branch)
    load_sample = DistributionLoad.example()
    for name, phase, active, reactive in (
        ("leg-one", Phase.S1, 1200, 400),
        ("leg-two", Phase.S2, 700, 200),
    ):
        phase_load = load_sample.equipment.phase_loads[0].model_copy(
            update={
                "name": name,
                "uuid": uuid4(),
                "real_power": type(load_sample.equipment.phase_loads[0].real_power)(
                    active, "watt"
                ),
                "reactive_power": type(
                    load_sample.equipment.phase_loads[0].reactive_power
                )(reactive, "var"),
                "z_real": 0.0,
                "z_imag": 0.0,
                "i_real": 0.0,
                "i_imag": 0.0,
                "p_real": 1.0,
                "p_imag": 1.0,
            },
            deep=True,
        )
        load_equipment = type(load_sample.equipment)(
            name=name,
            phase_loads=[phase_load],
            connection_type=load_sample.equipment.connection_type,
        )
        system.add_component(
            DistributionLoad(
                name=name, bus=customer, phases=[phase], equipment=load_equipment
            )
        )
    return system


def test_export_grounded_split_feeder_preserves_devices_and_assumption(
    grounded_system, tmp_path
):
    from gdm_reduce import (
        build_reduced_gdm,
        from_gdm_flow_results,
        save_reduced_gdm,
        validate_reduced_gdm,
    )

    solve = pytest.importorskip("gdm_flow.ac_pf").solve_ac_power_flow_from_components
    transformer = next(grounded_system.get_components(DistributionTransformer))
    for winding in transformer.equipment.windings[1:]:
        winding.is_grounded = False
    working = grounded_system.deepcopy()
    for winding in next(
        working.get_components(DistributionTransformer)
    ).equipment.windings[1:]:
        winding.is_grounded = True
    reference = solve(working, v0_complex={}, tolerance=1e-8)
    assert reference.success
    library = from_gdm_flow_results(
        grounded_system,
        {"nominal": reference},
        scenario_provenance={"nominal": {"load_scale": "1.0", "solar_scale": "1.0"}},
    )
    constraints = get_gdm_search_constraints(grounded_system, library)
    selected = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=0.1,
        target_reduction_fraction=1 / 3,
    )
    radial = radialize_reduction(selected, constraints.bus_edges)
    exported = build_reduced_gdm(grounded_system, radial, library)
    assert exported.report["center_tap_grounding"] == "assumed-grounded"
    assert all(
        not winding.is_grounded for winding in transformer.equipment.windings[1:]
    )
    assert len(list(exported.system.get_components(DistributionBus))) == 2
    for original in grounded_system.get_components(DistributionLoad):
        copied = exported.system.get_component_by_uuid(original.uuid)
        assert copied.bus.name == "secondary"
        assert copied.phases == original.phases
        assert copied.equipment.model_dump() == original.equipment.model_dump()
    assert validate_reduced_gdm(exported, library, voltage_error_limit_pu=0.01)[
        "passed"
    ]
    destination = tmp_path / "split.json"
    save_reduced_gdm(exported, destination)
    loaded = DistributionSystem.from_json(destination)
    assert solve(loaded, v0_complex={}, tolerance=1e-8).success


@pytest.mark.parametrize("no_load_loss", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("secondary_tap", [1.0, 1.02])
def test_real_grounded_center_tap_matches_opendss(
    grounded_system, no_load_loss, secondary_tap
):
    flow = pytest.importorskip("gdm_flow.ybus")
    dss = pytest.importorskip("opendssdirect")
    transformer = next(grounded_system.get_components(DistributionTransformer))
    transformer.equipment.pct_no_load_loss = no_load_loss
    transformer.tap_positions = [[1.0], [1.0, secondary_tap], [1.0, 1.0]]
    matrix = np.zeros((3, 3), dtype=complex)
    flow._stamp_transformer(
        matrix,
        {("primary", "A"): 0, ("secondary", "S1"): 1, ("secondary", "S2"): 2},
        transformer,
        False,
    )
    dss.Basic.ClearAll()
    dss.Text.Command("new circuit.split phases=1 bus1=primary.1 basekv=7.2 pu=1")
    dss.Text.Command(
        "new transformer.split phases=1 windings=3 "
        "wdg=1 bus=primary.1.0 conn=wye kv=7.2 kva=25 %r=0.2 "
        f"wdg=2 bus=secondary.1.0 conn=wye kv=0.12 kva=25 %r=0.4 tap={secondary_tap} "
        "wdg=3 bus=secondary.0.2 conn=wye kv=0.12 kva=25 %r=0.4 "
        f"xhl=2 xht=2 xlt=1 %noloadloss={no_load_loss} %imag=0 ppm_antifloat=0"
    )
    dss.Text.Command("solve")
    dss.Circuit.SetActiveElement("transformer.split")
    values = np.array(dss.CktElement.YPrim()).reshape(-1, 2)
    primitive = (values[:, 0] + 1j * values[:, 1]).reshape(6, 6)
    np.testing.assert_allclose(
        matrix, primitive[np.ix_([0, 2, 5], [0, 2, 5])], rtol=1e-11, atol=1e-10
    )


def split_library(domains=None):
    coupling = np.array([[10, 1], [1, 8]], dtype=complex)
    network = ElectricalNetwork(
        np.block([[coupling, -coupling], [-coupling, coupling]]),
        (("source", "S2"), ("source", "S1"), ("load", "S2"), ("load", "S1")),
        {("source", "S1"): 120, ("source", "S2"): -120},
    )
    domains = domains or {"source": "service", "load": "service"}
    currents = np.array([0, 0, 2 + 0.2j, -1 - 0.1j])
    voltage = solve_voltages(network, currents)
    bases = np.full(4, 120.0)
    reference = ReferenceScenario(
        "split",
        network.labels,
        voltage,
        network.admittance @ voltage,
        configuration_fingerprint(network, bases, service_domains=domains),
        True,
        0,
    )
    return ScenarioLibrary(network, bases, [reference], service_domains=domains)


def test_split_phase_exact_and_search_preserve_labels_and_domains():
    library = split_library()
    exact = kron_reduce(library.network, {"source"})
    np.testing.assert_allclose(
        exact.reconstruct(library.currents, exact.solve(library.currents)),
        library.voltage,
    )
    selected = optimize_kron_reduction(
        library, (("source", "load"),), voltage_error_limit_pu=1
    )
    assert selected.retained_buses == ("source",)
    radial = radialize_reduction(selected, (("source", "load"),))
    assert radial.retained_buses == ("source",)
    incompatible = split_library({"source": "first", "load": "second"})
    selected = optimize_kron_reduction(
        incompatible, (("source", "load"),), voltage_error_limit_pu=1
    )
    assert selected.history == ()


def test_grounded_gdm_ingestion_and_service_boundaries(grounded_system):
    library = build_gdm_scenario_library(grounded_system)
    assert set(library.service_domains) == {"secondary", "customer"}
    split_indices = [
        index
        for index, (_, phase) in enumerate(library.network.labels)
        if phase in {"S1", "S2"}
    ]
    np.testing.assert_allclose(library.voltage_bases[split_indices], 120)
    constraints = get_gdm_search_constraints(grounded_system, library)
    assert {"primary", "secondary"} <= constraints.protected_buses
    selected = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=0.1,
    )
    assert selected.original_to_retained["customer"] == "secondary"
    assert selected.original_to_retained["secondary"] == "secondary"


def test_ungrounded_service_rejected_without_mutation(grounded_system):
    transformer = next(grounded_system.get_components(DistributionTransformer))
    transformer.equipment.windings[1].is_grounded = False
    with pytest.raises(ValueError, match="verify importer/source provenance"):
        build_gdm_scenario_library(grounded_system, assume_center_taps_grounded=False)
    assert transformer.equipment.windings[1].is_grounded is False


def test_grounding_assumption_is_recorded_without_mutating_input(grounded_system):
    transformer = next(grounded_system.get_components(DistributionTransformer))
    for winding in transformer.equipment.windings[1:]:
        winding.is_grounded = False
    before = transformer.model_dump(mode="json")
    library = build_gdm_scenario_library(grounded_system)
    assert transformer.model_dump(mode="json") == before
    assert all(
        scenario.provenance["center_tap_grounding"] == "assumed-grounded"
        for scenario in library.scenarios
    )
    assert all(
        ":assumed-grounded:" in domain for domain in library.service_domains.values()
    )
    constraints = get_gdm_search_constraints(grounded_system, library)
    selected = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=0.1,
    )
    radial = radialize_reduction(selected, constraints.bus_edges)
    bundle = build_result_bundle(radial, library, library, voltage_error_limit_pu=0.1)
    assert bundle["center_tap_grounding"] == "assumed-grounded"


def test_grounding_policy_changes_fingerprint(grounded_system):
    assumed = build_gdm_scenario_library(grounded_system)
    explicit = build_gdm_scenario_library(
        grounded_system, assume_center_taps_grounded=False
    )
    assert assumed.configuration_fingerprint != explicit.configuration_fingerprint
    assert explicit.scenarios[0].provenance["center_tap_grounding"] == "model-grounded"


def test_differential_voltage_rejects_wrong_polarity():
    from gdm_reduce.search import _represent

    library = split_library()
    represented = library.voltage.copy()
    represented[0] *= -1
    represented[2] *= -1
    np.testing.assert_allclose(abs(represented), abs(library.voltage))
    _, error, _ = _represent(
        library,
        {"source": "source", "load": "load"},
        represented,
        {label: index for index, label in enumerate(library.network.labels)},
        library.voltage,
    )
    assert error > 0.99


def test_split_bundle_round_trip_and_holdouts(grounded_system):
    training = build_gdm_scenario_library(grounded_system)
    holdout = build_gdm_scenario_library(grounded_system, load_scales=[0.75, 1.25])
    constraints = get_gdm_search_constraints(grounded_system, training)
    selected = optimize_kron_reduction(
        training,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=0.1,
    )
    radial = radialize_reduction(selected, constraints.bus_edges)
    report = evaluate_reduction(radial, holdout, voltage_error_limit_pu=0.1)
    assert report.passed
    assert report.scenarios[0].max_line_to_line_complex_error_pu > 0
    assert report.scenarios[0].reference_max_leg_imbalance_pu > 0
    bundle = build_result_bundle(radial, training, holdout, voltage_error_limit_pu=0.1)
    assert bundle["schema"] == "gdm-reduce-result-v2"
    assert bundle["service_domains"] == dict(training.service_domains)
    restored = network_from_result_bundle(bundle)
    np.testing.assert_allclose(
        restored.admittance.toarray(),
        radial.equivalent.reduced_network.admittance.toarray(),
    )
    bundle["service_domains"]["customer"] = "wrong-service"
    with pytest.raises(ValueError, match="Invalid numerical equivalent"):
        network_from_result_bundle(bundle)


@pytest.mark.parametrize("line_to_line", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 4.0, -0.5])
def test_loaded_service_matches_opendss(grounded_system, line_to_line, scale):
    from gdm_flow.ac_pf import solve_ac_power_flow_from_components

    dss = pytest.importorskip("opendssdirect")
    if line_to_line:
        sample = next(grounded_system.get_components(DistributionLoad))
        from gdm.distribution.enums import ConnectionType

        phase_loads = [
            sample.equipment.phase_loads[0].model_copy(
                update={
                    "uuid": uuid4(),
                    "name": f"ll-{index}",
                    "real_power": type(sample.equipment.phase_loads[0].real_power)(
                        450, "watt"
                    ),
                    "reactive_power": type(
                        sample.equipment.phase_loads[0].reactive_power
                    )(150, "var"),
                },
                deep=True,
            )
            for index in range(2)
        ]
        equipment = type(sample.equipment)(
            name="line-to-line",
            phase_loads=phase_loads,
            connection_type=ConnectionType.DELTA,
        )
        grounded_system.add_component(
            DistributionLoad(
                name="line-to-line",
                bus=sample.bus,
                phases=[Phase.S2, Phase.S1],
                equipment=equipment,
            )
        )
    dss.Basic.ClearAll()
    dss.Text.Command("new circuit.split phases=1 bus1=primary.1 basekv=7.2 pu=1")
    dss.Text.Command(
        "new transformer.split phases=1 windings=3 "
        "wdg=1 bus=primary.1.0 conn=wye kv=7.2 kva=25 %r=0.2 "
        "wdg=2 bus=secondary.1.0 conn=wye kv=0.12 kva=25 %r=0.4 "
        "wdg=3 bus=secondary.0.2 conn=wye kv=0.12 kva=25 %r=0.4 "
        "xhl=2 xht=2 xlt=1 %noloadloss=0 %imag=0 ppm_antifloat=0"
    )
    dss.Text.Command(
        "new line.service phases=2 bus1=secondary.2.1 bus2=customer.2.1 "
        "length=10 units=m rmatrix=[0.002 | 0.0003 0.002] "
        "xmatrix=[0.001 | 0.0002 0.001] cmatrix=[0 | 0 0]"
    )
    dss.Text.Command(
        "new load.one phases=1 bus1=customer.1.0 conn=wye kv=0.12 "
        f"kw={1.2 * scale} kvar={0.4 * scale} model=1 vminpu=0.1 vmaxpu=2"
    )
    dss.Text.Command(
        "new load.two phases=1 bus1=customer.2.0 conn=wye kv=0.12 "
        f"kw={0.7 * scale} kvar={0.2 * scale} model=1 vminpu=0.1 vmaxpu=2"
    )
    if line_to_line:
        dss.Text.Command(
            "new load.ll phases=1 bus1=customer.1.2 conn=delta kv=0.24 "
            f"kw={0.9 * scale} kvar={0.3 * scale} model=1 vminpu=0.1 vmaxpu=2"
        )
    dss.Text.Command("set tolerance=1e-12 maxiterations=100")
    dss.Text.Command("solve")
    assert dss.Solution.Converged()
    voltages = {}
    for bus_name in ("primary", "secondary", "customer"):
        dss.Circuit.SetActiveBus(bus_name)
        values = np.array(dss.Bus.Voltages()).reshape(-1, 2)
        for node, (real, imag) in zip(dss.Bus.Nodes(), values):
            phase = "A" if bus_name == "primary" else f"S{node}"
            voltages[(bus_name, phase)] = complex(real, imag)
    result = solve_ac_power_flow_from_components(
        grounded_system,
        load_scale=scale,
        tolerance=1e-10,
        v0_complex={("primary", "A"): voltages[("primary", "A")]},
    )
    assert result.success, result.message
    reference = np.array(
        [voltages[label] for label in result.ybus_result.index_to_label]
    )
    np.testing.assert_allclose(result.voltage, reference, rtol=1e-9, atol=1e-7)
    if line_to_line and scale > 0:
        library = build_gdm_scenario_library(grounded_system, load_scales=[scale])
        assert library.service_domains


@pytest.mark.parametrize("phase", [Phase.B, Phase.C])
def test_primary_phase_identity_is_preserved(grounded_system, phase):
    transformer = next(grounded_system.get_components(DistributionTransformer))
    source = next(grounded_system.get_components(DistributionVoltageSource))
    transformer.buses[0].phases = [phase]
    transformer.winding_phases = [[phase], [Phase.N, Phase.S1], [Phase.N, Phase.S2]]
    source.phases = [phase]
    library = build_gdm_scenario_library(grounded_system)
    assert all(
        domain.endswith(f":{phase.value}")
        for domain in library.service_domains.values()
    )
    indices = {label: index for index, label in enumerate(library.network.labels)}
    primary_angle = np.angle(library.voltage[indices[("primary", phase.value)], 0])
    s1_angle = np.angle(library.voltage[indices[("secondary", "S1")], 0])
    assert abs(np.angle(np.exp(1j * (s1_angle - primary_angle)))) < 0.01


def test_split_line_to_line_bases(grounded_system):
    for bus in grounded_system.get_components(DistributionBus):
        if Phase.S1 in bus.phases:
            bus.voltage_type = VoltageTypes.LINE_TO_LINE
            bus.rated_voltage *= 2
    library = build_gdm_scenario_library(grounded_system)
    np.testing.assert_allclose(
        [
            base
            for base, (_, phase) in zip(library.voltage_bases, library.network.labels)
            if phase in {"S1", "S2"}
        ],
        120,
    )


def test_split_service_requires_exactly_one_transformer(grounded_system):
    transformer = next(grounded_system.get_components(DistributionTransformer))
    transformer.in_service = False
    with pytest.raises(ValueError, match="exactly one"):
        build_gdm_scenario_library(grounded_system)
    transformer.in_service = True
    duplicate = transformer.model_copy(
        update={"name": "duplicate", "uuid": uuid4()}, deep=True
    )
    grounded_system.add_component(duplicate)
    with pytest.raises(ValueError, match="multiple transformers"):
        build_gdm_scenario_library(grounded_system)


def test_explicit_neutral_and_missing_domain_rejected(grounded_system):
    library = split_library()
    with pytest.raises(ValueError, match="service domains"):
        ScenarioLibrary(library.network, library.voltage_bases, library.scenarios)
    from gdm_flow.ybus import calculate_ybus

    with pytest.raises(ValueError, match="Explicit-neutral"):
        calculate_ybus(grounded_system, include_neutral=True)


def test_legacy_backend_is_rejected(grounded_system, monkeypatch):
    from gdm_flow import ybus

    monkeypatch.delattr(ybus, "SPLIT_PHASE_CONTRACT")
    with pytest.raises(RuntimeError, match="upgrade the backend"):
        build_gdm_scenario_library(grounded_system)


def test_missing_secondary_leg_is_not_invented(grounded_system):
    sample = next(grounded_system.get_components(MatrixImpedanceBranch))
    customer = sample.buses[1]
    one_leg = customer.model_copy(
        update={"name": "one-leg", "uuid": uuid4(), "phases": [Phase.S1]}, deep=True
    )
    equipment = type(sample.equipment)(
        name="one-leg-line",
        r_matrix=type(sample.equipment.r_matrix)([[0.002]], "ohm/m"),
        x_matrix=type(sample.equipment.x_matrix)([[0.001]], "ohm/m"),
        c_matrix=type(sample.equipment.c_matrix)([[0.0]], "nanofarad/m"),
        ampacity=sample.equipment.ampacity,
    )
    grounded_system.add_component(
        MatrixImpedanceBranch(
            name="one-leg-line",
            buses=[customer, one_leg],
            phases=[Phase.S1],
            length=sample.length,
            equipment=equipment,
        )
    )
    library = build_gdm_scenario_library(grounded_system)
    assert ("one-leg", "S1") in library.network.labels
    assert ("one-leg", "S2") not in library.network.labels
    constraints = get_gdm_search_constraints(grounded_system, library)
    selected = optimize_kron_reduction(
        library,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=0.1,
    )
    radial = radialize_reduction(selected, constraints.bus_edges)
    assert evaluate_reduction(radial, library, voltage_error_limit_pu=0.1).passed


def test_secondary_winding_order_and_taps(grounded_system):
    from gdm_flow.ybus import _stamp_transformer

    transformer = next(grounded_system.get_components(DistributionTransformer))
    labels = {("primary", "A"): 0, ("secondary", "S1"): 1, ("secondary", "S2"): 2}
    transformer.tap_positions = [[1.0], [1.0, 1.02], [1.0, 0.98]]
    expected = np.zeros((3, 3), complex)
    _stamp_transformer(expected, labels, transformer, False)
    transformer.equipment.windings = [
        transformer.equipment.windings[index] for index in (0, 2, 1)
    ]
    transformer.winding_phases = [
        transformer.winding_phases[index] for index in (0, 2, 1)
    ]
    transformer.tap_positions = [[1.0], [1.0, 0.98], [1.0, 1.02]]
    actual = np.zeros((3, 3), complex)
    _stamp_transformer(actual, labels, transformer, False)
    np.testing.assert_allclose(actual, expected)


def test_real_split_phase_cli(grounded_system, tmp_path, capsys):
    import json

    from gdm_reduce.cli import main

    model = tmp_path / "service.json"
    output = tmp_path / "reduced.json"
    grounded_system.to_json(model)
    assert (
        main([str(model), "--output", str(output), "--voltage-error-limit-pu", "0.1"])
        == 0
    )
    bundle = json.loads(output.read_text())
    assert bundle["schema"] == "gdm-reduce-result-v2"
    assert bundle["holdout"]["report"]["passed"]
    assert not bundle["gdm_model_exported"]
    assert "Holdout passed: True" in capsys.readouterr().out
