"""Compare original and reloaded, reduced GDM equipment at a held-out load scale."""

import argparse
import json
from pathlib import Path

import numpy as np
from gdm.systems.distribution import DistributionSystem, reduce_to_primary_system
from gdm.systems.distribution.components import DistributionBus
from gdm_flow.ac_pf import solve_ac_power_flow_from_components
from plotly.graph_objects import Figure

from gdm_reduce import (
    build_gdm_scenario_library,
    get_gdm_search_constraints,
    optimize_kron_reduction,
    radialize_reduction,
)
from gdm_reduce.gdm_export import (
    GDMModelExport,
    build_reduced_gdm,
    save_reduced_gdm,
    validate_reduced_gdm,
)


class NotebookFigure(Figure):
    def _ipython_display_(self):
        from IPython.display import display

        display(self._repr_mimebundle_(), raw=True)

    def _repr_mimebundle_(self, include=None, exclude=None):
        import html

        source = html.escape(
            self.to_html(full_html=True, include_plotlyjs=True), quote=True
        )
        return {
            "text/html": (
                '<iframe title="GDM reduction comparison" '
                'style="width:100%;height:920px;border:0" '
                f'srcdoc="{source}"></iframe>'
            )
        }


def compare_model(
    model: Path,
    output: Path,
    *,
    make_plot: bool = True,
    primary_first: bool = False,
    max_iterations: int | None = None,
    reference_tolerance: float = 1e-8,
    target_reduction_fraction: float = 0.5,
) -> dict:
    system = DistributionSystem.from_json(model)
    input_buses = len(list(system.get_components(DistributionBus)))
    output.mkdir(parents=True, exist_ok=True)
    if primary_first:
        system = reduce_to_primary_system(
            system, name=f"{system.name}_primary", agg_time_series=True
        )
        primary_path = output / "primary.json"
        system.to_json(primary_path, overwrite=True)
        system = DistributionSystem.from_json(primary_path)
    training = build_gdm_scenario_library(
        system, load_scales=(0.5, 1.0), tolerance=reference_tolerance
    )
    constraints = get_gdm_search_constraints(system, training)
    selection = optimize_kron_reduction(
        training,
        constraints.bus_edges,
        protected_buses=constraints.protected_buses,
        voltage_error_limit_pu=0.01,
        target_reduction_fraction=target_reduction_fraction,
        max_iterations=max_iterations,
    )
    radial = radialize_reduction(selection, constraints.bus_edges)
    export = build_reduced_gdm(system, radial, training, allow_series_only=True)
    evaluation = build_gdm_scenario_library(
        system, load_scales=(0.75,), tolerance=reference_tolerance
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reduced.json"
    save_reduced_gdm(export, path, overwrite=True)
    reloaded = GDMModelExport(DistributionSystem.from_json(path), export.report)
    validation = validate_reduced_gdm(reloaded, evaluation, voltage_error_limit_pu=0.01)
    retained = set(radial.retained_buses)
    warm = {
        label: complex(evaluation.voltage[index, 0])
        for index, label in enumerate(evaluation.network.labels)
        if label[0] in retained and label[0] != reloaded.system.get_source_bus().name
    }
    solved = solve_ac_power_flow_from_components(
        reloaded.system,
        load_scale=0.75,
        include_shunt=False,
        v0_complex=warm,
        tolerance=1e-6,
        max_iterations=200,
    )
    if not solved.success:
        raise RuntimeError(solved.message)
    represented = np.array(
        [
            solved.voltage[
                solved.ybus_result.label_to_index[
                    (selection.original_to_retained[bus], phase)
                ]
            ]
            for bus, phase in evaluation.network.labels
        ]
    )
    reference = evaluation.voltage[:, 0]
    summary = {
        "input_buses": input_buses,
        "primary_first": primary_first,
        "target_reduction_fraction": target_reduction_fraction,
        "max_iterations": max_iterations,
        "search_iterations": len(selection.history),
        "reference_tolerance_pu": reference_tolerance,
        "original_buses": len(list(system.get_components(DistributionBus))),
        "reduced_buses": len(list(reloaded.system.get_components(DistributionBus))),
        "search_reduction_fraction": selection.reduction_fraction,
        "radial_reduction_fraction": radial.reduction_fraction,
        "restored_junctions": list(radial.restored_buses),
        "termination_reason": selection.termination_reason,
        "training_load_scales": [0.5, 1.0],
        "heldout_load_scale": 0.75,
        "training_max_voltage_error_pu": selection.max_voltage_error_pu,
        "validation": validation,
        "labels": [list(label) for label in evaluation.network.labels],
        "original_voltage_magnitude_pu": (
            np.abs(reference) / evaluation.voltage_bases
        ).tolist(),
        "reduced_representative_voltage_magnitude_pu": (
            np.abs(represented) / evaluation.voltage_bases
        ).tolist(),
        "voltage_magnitude_error_pu": (
            np.abs(np.abs(represented) - np.abs(reference)) / evaluation.voltage_bases
        ).tolist(),
    }
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_reduced_gdm(reloaded, path, overwrite=True)
    if make_plot:
        _plot_systems(system, reloaded.system, output)
        _plot(summary, output)
    return summary


def _plot_systems(
    original: DistributionSystem, reduced: DistributionSystem, output: Path
) -> Figure:
    from plotly.subplots import make_subplots

    figures = [original.plot(show=False), reduced.plot(show=False)]
    comparison = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]],
        subplot_titles=("Original GDM", "Reloaded Reduced GDM"),
    )
    colors = ["#2864b4", "#ca5353", "#16856b", "#8650ab", "#b57913", "#1595aa"]
    categories = sorted({trace.name for figure in figures for trace in figure.data})
    palette = {
        name: colors[index % len(colors)] for index, name in enumerate(categories)
    }
    seen = set()
    for column, figure in enumerate(figures, start=1):
        for trace in figure.data:
            trace.legendgroup = trace.name
            trace.showlegend = trace.name not in seen
            trace.line.color = palette[trace.name]
            trace.marker.color = palette[trace.name]
            seen.add(trace.name)
            comparison.add_trace(trace, row=1, col=column)
        comparison.update_geos(
            fitbounds="locations",
            projection_type="mercator",
            visible=False,
            row=1,
            col=column,
        )
    comparison.update_layout(
        height=550,
        legend={"orientation": "h", "y": -0.1},
        margin={"t": 60, "b": 130},
    )
    comparison.write_html(output / "systems.html")
    return comparison


def compare_steps(
    model: Path,
    output: Path,
    *,
    primary_first: bool = False,
    reference_tolerance: float = 1e-8,
    minimum_percent: int = 5,
    maximum_percent: int = 60,
    resume: bool = False,
) -> list[dict]:
    if (
        minimum_percent not in range(0, 100, 5)
        or maximum_percent not in range(5, 100, 5)
        or minimum_percent > maximum_percent
    ):
        raise ValueError("Range must be ascending multiples of 5 between 0 and 95")
    summaries = []
    for percent in range(minimum_percent, maximum_percent + 1, 5):
        directory = output / f"step-{percent:02d}"
        saved = directory / "comparison.json"
        if resume and saved.exists():
            summary = json.loads(saved.read_text())
            if (
                summary["primary_first"] != primary_first
                or summary["reference_tolerance_pu"] != reference_tolerance
                or summary["target_reduction_fraction"] != percent / 100
                or not (directory / "reduced.json").exists()
            ):
                raise ValueError(f"Saved step {percent}% does not match this sweep")
        elif summaries and summaries[-1]["termination_reason"] == "no_feasible_merge":
            import shutil

            previous = round(100 * summaries[-1]["target_reduction_fraction"])
            shutil.copytree(
                output / f"step-{previous:02d}", directory, dirs_exist_ok=True
            )
            summary = dict(summaries[-1], target_reduction_fraction=percent / 100)
            saved.write_text(json.dumps(summary, indent=2) + "\n")
        else:
            summary = compare_model(
                model,
                directory,
                make_plot=False,
                primary_first=primary_first,
                reference_tolerance=reference_tolerance,
                target_reduction_fraction=percent / 100,
            )
        summaries.append(summary)
        print(
            f"Target {percent}%: {summary['original_buses']} -> "
            f"{summary['reduced_buses']} buses; "
            f"error {max(summary['voltage_magnitude_error_pu']):.6g} pu; "
            f"{summary['termination_reason']}",
            flush=True,
        )
        (output / "steps.json").write_text(json.dumps(summaries, indent=2) + "\n")
        _plot_steps(model, output, summaries)
        if not summary["validation"]["passed"]:
            raise RuntimeError(f"Held-out voltage validation failed at {percent}%")
    return summaries


def _plot_steps(model: Path, output: Path, summaries: list[dict]) -> Figure:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    model_name = model.parent.name if model.stem == "model" else model.stem
    first_percent = round(100 * summaries[0]["target_reduction_fraction"])
    baseline_path = (
        output / f"step-{first_percent:02d}/primary.json"
        if summaries[0]["primary_first"]
        else model
    )
    original = DistributionSystem.from_json(baseline_path).plot(show=False)
    colors = ["#2864b4", "#ca5353", "#16856b", "#8650ab", "#b57913", "#1595aa"]
    categories = sorted(trace.name for trace in original.data)
    palette = {
        name: colors[index % len(colors)] for index, name in enumerate(categories)
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}], [{}, {}]],
        subplot_titles=(
            "Original / primary baseline",
            "Reloaded reduced GDM",
            "Voltage magnitude",
            "Terminal magnitude error",
        ),
        vertical_spacing=0.16,
    )
    groups = []
    for index, summary in enumerate(summaries):
        percent = round(100 * summary["target_reduction_fraction"])
        reduced = DistributionSystem.from_json(
            output / f"step-{percent:02d}/reduced.json"
        ).plot(show=False)
        start = len(figure.data)
        for column, native in enumerate((original, reduced), start=1):
            for trace in native.data:
                trace = go.Scattergeo(trace.to_plotly_json())
                trace.visible = index == 0
                trace.showlegend = False
                trace.line.color = palette.get(trace.name, "#333333")
                trace.marker.color = palette.get(trace.name, "#333333")
                figure.add_trace(trace, row=1, col=column)
        terminals = [f"{bus}:{phase}" for bus, phase in summary["labels"]]
        for values, name, color, row, column in (
            (summary["original_voltage_magnitude_pu"], "Baseline", "#16856b", 2, 1),
            (
                summary["reduced_representative_voltage_magnitude_pu"],
                "Reduced",
                "#ca5353",
                2,
                1,
            ),
            (summary["voltage_magnitude_error_pu"], "Error", "#2864b4", 2, 2),
        ):
            figure.add_trace(
                go.Scatter(
                    x=terminals,
                    y=values,
                    name=name,
                    line={"color": color},
                    visible=index == 0,
                ),
                row=row,
                col=column,
            )
        groups.append((start, len(figure.data)))
    slider_steps = []
    for summary, (start, end) in zip(summaries, groups):
        target = 100 * summary["target_reduction_fraction"]
        achieved = 100 * (1 - summary["reduced_buses"] / summary["original_buses"])
        error = max(summary["voltage_magnitude_error_pu"])
        title = (
            f"{model_name}<br>"
            f"Target {target:.0f}% | {summary['original_buses']} -> "
            f"{summary['reduced_buses']} buses<br>"
            f"Achieved {achieved:.1f}% | max error {error:.3g} pu<br>"
            f"{summary['termination_reason']}"
        )
        slider_steps.append(
            {
                "label": f"{target:.0f}%",
                "method": "update",
                "args": [
                    {
                        "visible": [
                            start <= trace < end for trace in range(len(figure.data))
                        ]
                    },
                    {"title.text": title},
                ],
            }
        )
    figure.update_geos(fitbounds="locations", projection_type="mercator", visible=False)
    figure.update_yaxes(title_text="Voltage (pu)", row=2, col=1)
    figure.update_yaxes(title_text="Error (pu)", row=2, col=2)
    figure.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0.01,
        y1=0.01,
        xref="x2 domain",
        yref="y2",
        line={"dash": "dash", "color": "#333333"},
    )
    figure.update_xaxes(showticklabels=False, row=2)
    figure.update_layout(
        height=850,
        template="plotly_white",
        title={
            "text": slider_steps[0]["args"][1]["title.text"],
            "font": {"size": 14},
            "x": 0.5,
            "xanchor": "center",
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Reduction target: "},
                "steps": slider_steps,
            }
        ],
        margin={"t": 120, "b": 100, "l": 50, "r": 20},
    )
    figure.update_annotations(font={"size": 12})
    figure.write_html(output / "slider.html")
    return NotebookFigure(figure)


def _plot(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, panels = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    voltage_axis, error_axis = panels
    indices = np.arange(len(summary["labels"]))
    labels = [f"{bus}:{phase}" for bus, phase in summary["labels"]]
    voltage_axis.plot(
        indices,
        summary["original_voltage_magnitude_pu"],
        "o-",
        color="#007f73",
        label="Original",
    )
    voltage_axis.plot(
        indices,
        summary["reduced_representative_voltage_magnitude_pu"],
        "x--",
        color="#b54147",
        label="Reloaded reduced GDM",
    )
    voltage_axis.set_ylabel("Voltage magnitude (pu)")
    voltage_axis.legend()
    error_axis.bar(indices, summary["voltage_magnitude_error_pu"], color="#b54147")
    error_axis.axhline(
        0.01, color="#222222", linestyle="--", label="Chosen 0.01 pu limit"
    )
    error_axis.set_ylabel("Represented magnitude error (pu)")
    error_axis.legend()
    for axis in panels:
        axis.set_xticks(indices, labels, rotation=90, fontsize=7)
        axis.set_xlabel("Original bus:phase terminal")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Held-out 0.75 load scale | static constant-P/Q, series-only lines")
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    parser.add_argument("--primary-first", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--reference-tolerance", type=float, default=1e-8)
    parser.add_argument("--steps", action="store_true")
    parser.add_argument("--minimum-percent", type=int, default=5)
    parser.add_argument("--maximum-percent", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.steps:
        compare_steps(
            arguments.model,
            arguments.output,
            primary_first=arguments.primary_first,
            reference_tolerance=arguments.reference_tolerance,
            minimum_percent=arguments.minimum_percent,
            maximum_percent=arguments.maximum_percent,
            resume=arguments.resume,
        )
        return
    summary = compare_model(
        arguments.model,
        arguments.output,
        primary_first=arguments.primary_first,
        max_iterations=arguments.max_iterations,
        reference_tolerance=arguments.reference_tolerance,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("original_buses", "reduced_buses", "validation")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
