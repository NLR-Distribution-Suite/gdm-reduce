import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import DistributionBus

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/data/circuits"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.mark.parametrize("record", MANIFEST, ids=lambda record: record["name"])
def test_copied_circuit_roundtrip(record, tmp_path):
    system = DistributionSystem.from_json(FIXTURES / record["model"])
    buses = {bus.uuid for bus in system.get_components(DistributionBus)}
    assert len(buses) == record["buses"]
    path = tmp_path / "roundtrip.json"
    system.to_json(path)
    restored = DistributionSystem.from_json(path)
    assert {bus.uuid for bus in restored.get_components(DistributionBus)} == buses
    assert record["electrical_normalization"] == "none"


def test_documented_ieee13_reduction_uses_reloaded_gdm(tmp_path):
    compare = runpy.run_path(str(ROOT / "examples/compare_gdm.py"))["compare_model"]
    report = compare(FIXTURES / "ieee13/model.json", tmp_path, make_plot=False)
    assert report["original_buses"] == 16
    assert report["reduced_buses"] == 11
    assert report["validation"]["passed"]
    assert report["heldout_load_scale"] not in report["training_load_scales"]
    errors = np.asarray(report["voltage_magnitude_error_pu"])
    assert 0 < errors.max() < 0.01
    assert len(errors) == len(report["labels"])
    assert (tmp_path / "reduced.export-report.json").exists()


def test_geographic_p5r_native_plot_comparison(tmp_path, monkeypatch):
    example = runpy.run_path(str(ROOT / "examples/compare_gdm.py"))
    compare = example["compare_model"]
    native_plot = DistributionSystem.plot
    plotted = []

    def record_plot(system, **kwargs):
        figure = native_plot(system, **kwargs)
        plotted.append((system, figure))
        return figure

    monkeypatch.setattr(DistributionSystem, "plot", record_plot)
    monkeypatch.setitem(compare.__globals__, "_plot", lambda summary, output: None)
    report = compare(FIXTURES / "p5r.json", tmp_path)
    assert report["original_buses"] == 10
    assert report["reduced_buses"] == 8
    assert report["validation"]["passed"]
    assert np.max(report["voltage_magnitude_error_pu"]) < 0.01
    assert len(plotted) == 2
    for axis, lower, upper in (("lat", -90, 90), ("lon", -180, 180)):
        values = [
            float(value)
            for _, figure in plotted
            for trace in figure.data
            for value in getattr(trace, axis)
            if value is not None and np.isfinite(value)
        ]
        assert values
        assert lower <= min(values) < max(values) <= upper
    reloaded = DistributionSystem.from_json(tmp_path / "reduced.json")
    assert {bus.uuid for bus in plotted[1][0].get_components(DistributionBus)} == {
        bus.uuid for bus in reloaded.get_components(DistributionBus)
    }
    assert (tmp_path / "systems.html").stat().st_size > 1000
    provenance = json.loads((FIXTURES / "p5r.provenance.json").read_text())
    assert provenance["dataset"] == "p5r"
    assert (
        hashlib.sha256((FIXTURES / "p5r.json").read_bytes()).hexdigest()
        == provenance["saved_json_sha256"]
    )


def test_five_percent_sweep_and_slider(tmp_path, monkeypatch):
    example = runpy.run_path(str(ROOT / "examples/compare_gdm.py"))
    summaries = example["compare_steps"](FIXTURES / "p5r.json", tmp_path)
    targets = [
        round(100 * summary["target_reduction_fraction"]) for summary in summaries
    ]
    assert targets == list(range(5, 61, 5))
    assert all(summary["validation"]["passed"] for summary in summaries)
    assert summaries[-1]["termination_reason"] == "no_feasible_merge"
    counts = [summary["reduced_buses"] for summary in summaries]
    assert counts == sorted(counts, reverse=True)
    figure = example["_plot_steps"](FIXTURES / "p5r.json", tmp_path, summaries)
    steps = figure.layout.sliders[0].steps
    assert [step.label for step in steps] == [f"{target}%" for target in targets]
    for step in steps:
        mask = step.args[0]["visible"]
        assert len(mask) == len(figure.data)
        visible = [trace for trace, enabled in zip(figure.data, mask) if enabled]
        assert {trace.geo for trace in visible if trace.type == "scattergeo"} == {
            "geo",
            "geo2",
        }
        assert {trace.name for trace in visible if trace.type == "scatter"} == {
            "Baseline",
            "Reduced",
            "Error",
        }
        assert "buses" in step.args[1]["title.text"]
        assert "error" in step.args[1]["title.text"]
    assert (tmp_path / "slider.html").stat().st_size > 1000
    rendered = figure._repr_mimebundle_()["text/html"]
    assert '<iframe title="GDM reduction comparison"' in rendered
    assert "srcdoc=" in rendered
    assert len(json.loads((tmp_path / "steps.json").read_text())) == len(summaries)

    def unexpected_recompute(*args, **kwargs):
        pytest.fail("Resume must reuse completed results")

    monkeypatch.setitem(
        example["compare_steps"].__globals__, "compare_model", unexpected_recompute
    )
    resumed = example["compare_steps"](FIXTURES / "p5r.json", tmp_path, resume=True)
    assert resumed == summaries
    with pytest.raises(ValueError, match="does not match"):
        example["compare_steps"](
            FIXTURES / "p5r.json", tmp_path, resume=True, reference_tolerance=1e-7
        )
