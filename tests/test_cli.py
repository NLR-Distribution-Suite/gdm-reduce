import json
from types import SimpleNamespace

import numpy as np
import pytest
from gdm.systems.distribution import DistributionSystem

from gdm_reduce import (
    ElectricalNetwork,
    ReferenceScenario,
    ScenarioLibrary,
    cli,
    configuration_fingerprint,
    solve_voltages,
)


@pytest.fixture
def cli_backend(monkeypatch):
    network = ElectricalNetwork(
        [[10, -10, 0], [-10, 20, -10], [0, -10, 10]],
        (("source", "A"), ("middle", "A"), ("load", "A")),
        {("source", "A"): 120},
    )
    bases = np.full(3, 120.0)
    fingerprint = configuration_fingerprint(network, bases)

    def references(system, *, load_scales, provenance, assume_center_taps_grounded):
        scenarios = []
        for index, scale in enumerate(load_scales):
            currents = np.array([3, -1, -2]) * scale
            voltage = solve_voltages(network, currents)
            scenarios.append(
                ReferenceScenario(
                    f"load-{index}",
                    network.labels,
                    voltage,
                    currents,
                    fingerprint,
                    True,
                    0,
                    {
                        **provenance,
                        "load_scale": str(scale),
                        "center_tap_grounding": "assumed-grounded"
                        if assume_center_taps_grounded
                        else "model-grounded",
                    },
                )
            )
        return ScenarioLibrary(network, bases, scenarios)

    monkeypatch.setattr(DistributionSystem, "from_json", lambda path: object())
    monkeypatch.setattr(cli, "build_gdm_scenario_library", references)
    monkeypatch.setattr(
        cli,
        "get_gdm_search_constraints",
        lambda *args: SimpleNamespace(
            bus_edges=(("source", "middle"), ("middle", "load")),
            protected_buses=frozenset({"source"}),
        ),
    )


def test_cli_outputs_json_and_timings(cli_backend, capsys):
    assert (
        cli.main(
            [
                "model.json",
                "--voltage-error-limit-pu",
                "0.002",
                "--holdout-load-scales",
                "0.75",
            ]
        )
        == 0
    )
    bundle = json.loads(capsys.readouterr().out)
    assert bundle["holdout"]["report"]["passed"]
    assert bundle["gdm_model_exported"] is False
    assert all(value >= 0 for value in bundle["timings_s"].values())
    assert bundle["request"]["holdout_load_scales"] == [0.75]


def test_holdout_failure_is_saved_with_nonzero_exit(cli_backend, tmp_path, capsys):
    path = tmp_path / "failed.json"
    exit_code = cli.main(
        [
            "model.json",
            "--voltage-error-limit-pu",
            "0.002",
            "--holdout-load-scales",
            "4",
            "--output",
            str(path),
        ]
    )
    assert exit_code == 1
    assert json.loads(path.read_text())["holdout"]["report"]["passed"] is False
    assert "Holdout passed: False" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        cli.main(["model.json", "--output", str(path)])
    assert error.value.code == 2
    assert (
        cli.main(
            [
                "model.json",
                "--target-reduction",
                "0",
                "--output",
                str(path),
                "--overwrite",
            ]
        )
        == 0
    )


def test_overlapping_reference_scales_are_rejected(cli_backend):
    with pytest.raises(SystemExit) as error:
        cli.main(["model.json", "--holdout-load-scales", "1"])
    assert error.value.code == 2


def test_invalid_search_options_have_cli_error_exit(cli_backend):
    with pytest.raises(SystemExit) as error:
        cli.main(["model.json", "--batch-size", "0"])
    assert error.value.code == 2


def test_missing_model_has_cli_error_exit(monkeypatch):
    def load(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(DistributionSystem, "from_json", load)
    with pytest.raises(SystemExit) as error:
        cli.main(["missing.json"])
    assert error.value.code == 2
