"""Versioned JSON artifacts for radial prescribed-current numerical equivalents."""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .models import ElectricalNetwork
from .radialization import RadializationResult
from .scenarios import ScenarioLibrary
from .validation import evaluate_reduction, pool_scenario_currents

SCHEMA = "gdm-reduce-result-v1"
SPLIT_PHASE_SCHEMA = "gdm-reduce-result-v2"
KIND = "radial-prescribed-current-equivalent"


def _complex_arrays(values: np.ndarray) -> dict:
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def _snapshot_data(
    result: RadializationResult, library: ScenarioLibrary, limit: float
) -> dict:
    report = evaluate_reduction(result, library, voltage_error_limit_pu=limit)
    pooled = pool_scenario_currents(result.selection, library)
    return {
        "report": {
            **asdict(report),
            "passed": report.passed,
            "max_voltage_magnitude_error_pu": report.max_voltage_magnitude_error_pu,
        },
        "provenance": {
            scenario.name: dict(scenario.provenance) for scenario in library.scenarios
        },
        "reference_voltage": _complex_arrays(library.voltage),
        "original_currents": _complex_arrays(library.currents),
        "pooled_retained_currents": _complex_arrays(
            pooled[result.equivalent.retained_indices]
        ),
    }


def build_result_bundle(
    result: RadializationResult,
    training: ScenarioLibrary,
    holdout: ScenarioLibrary,
    *,
    voltage_error_limit_pu: float = 0.001,
    timings_s: Mapping[str, float] | None = None,
) -> dict:
    """Capture a replayable radial matrix and separate training/holdout reports.

    No equipment model, LU cache, or independent nonlinear certificate is saved.
    The caller must keep the reference split independent of cluster selection.
    """
    timings = dict(timings_s or {})
    if any(
        not isinstance(name, str) or not np.isfinite(value) or value < 0
        for name, value in timings.items()
    ):
        raise ValueError(
            "Timings must have string names and finite nonnegative seconds."
        )
    versions = {}
    for package in (
        "gdm-reduce",
        "grid-data-models",
        "gdm-flow",
        "numpy",
        "scipy",
        "networkx",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    selection = result.selection
    network = result.equivalent.reduced_network
    matrix = network.admittance
    return {
        "schema": SPLIT_PHASE_SCHEMA if training.service_domains else SCHEMA,
        "kind": KIND,
        "gdm_model_exported": False,
        "configuration_fingerprint": selection.configuration_fingerprint,
        **(
            {
                "service_domains": dict(training.service_domains),
                "center_tap_grounding": training.scenarios[0].provenance.get(
                    "center_tap_grounding", "external-reference"
                ),
                "grounding_mode": "grounded-reference",
                "split_phase_contract": "grounded-center-tap-v1",
            }
            if training.service_domains
            else {}
        ),
        "units": {
            "admittance": "siemens",
            "voltage": "volt",
            "current": "ampere",
            "active_power": "watt",
            "reactive_power": "var",
            "angle": "degree",
        },
        "original_labels": training.network.labels,
        "voltage_bases": training.voltage_bases.tolist(),
        "original_to_retained": dict(selection.original_to_retained),
        "bus_uuid_to_retained_uuid": dict(selection.bus_uuid_to_retained_uuid),
        "component_buses": dict(training.component_buses),
        "clusters": dict(selection.clusters),
        "protected_buses": sorted(selection.protected_buses),
        "restored_buses": result.restored_buses,
        "radial_bus_edges": result.bus_edges,
        "search_reduction_fraction": selection.reduction_fraction,
        "radial_reduction_fraction": result.reduction_fraction,
        "max_selected_voltage_difference_v": result.max_selected_voltage_difference_v,
        "termination_reason": selection.termination_reason,
        "selection_history": [asdict(step) for step in selection.history],
        "equivalent": {
            "labels": network.labels,
            "source_voltages": [
                {"label": label, "real": voltage.real, "imag": voltage.imag}
                for label, voltage in network.source_voltages.items()
            ],
            "admittance": {
                "shape": list(matrix.shape),
                "indptr": matrix.indptr.tolist(),
                "indices": matrix.indices.tolist(),
                **_complex_arrays(matrix.data),
            },
        },
        "training": _snapshot_data(result, training, voltage_error_limit_pu),
        "holdout": _snapshot_data(result, holdout, voltage_error_limit_pu),
        "timings_s": {name: float(value) for name, value in timings.items()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "versions": versions,
            "thread_settings": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
    }


def network_from_result_bundle(bundle: Mapping) -> ElectricalNetwork:
    """Decode the numerical equivalent only; this does not load a GDM feeder."""
    if (
        bundle.get("schema") not in {SCHEMA, SPLIT_PHASE_SCHEMA}
        or bundle.get("kind") != KIND
    ):
        raise ValueError("Unsupported result bundle schema or kind.")
    try:
        equivalent = bundle["equivalent"]
        matrix = equivalent["admittance"]
        real = np.asarray(matrix["real"], dtype=float)
        imag = np.asarray(matrix["imag"], dtype=float)
        if real.ndim != 1 or real.shape != imag.shape:
            raise ValueError("Invalid complex CSR values.")
        values = real + 1j * imag
        indices = np.asarray(matrix["indices"])
        indptr = np.asarray(matrix["indptr"])
        if any(
            array.ndim != 1 or (array.size and array.dtype.kind not in "iu")
            for array in (indices, indptr)
        ):
            raise ValueError("CSR offsets and indices must be integer arrays.")
        admittance = csr_matrix((values, indices, indptr), shape=tuple(matrix["shape"]))
        admittance.check_format(full_check=True)
        records = equivalent["source_voltages"]
        sources = {
            tuple(record["label"]): complex(record["real"], record["imag"])
            for record in records
        }
        if len(sources) != len(records):
            raise ValueError("Duplicate sources.")
        network = ElectricalNetwork(
            admittance, tuple(tuple(label) for label in equivalent["labels"]), sources
        )
        split_buses = {bus for bus, phase in network.labels if phase in {"S1", "S2"}}
        if split_buses or bundle.get("schema") == SPLIT_PHASE_SCHEMA:
            domains = bundle["service_domains"]
            original_split = {
                bus for bus, phase in bundle["original_labels"] if phase in {"S1", "S2"}
            }
            if (
                bundle.get("schema") != SPLIT_PHASE_SCHEMA
                or bundle.get("grounding_mode") != "grounded-reference"
                or bundle.get("split_phase_contract") != "grounded-center-tap-v1"
                or not isinstance(domains, dict)
                or set(domains) != original_split
                or not split_buses <= set(domains)
                or any(
                    not isinstance(domain, str) or not domain
                    for domain in domains.values()
                )
                or any(
                    domains[bus] != domains[bundle["original_to_retained"][bus]]
                    for bus in original_split
                )
            ):
                raise ValueError("Invalid split-phase service/grounding metadata.")
        return network
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Invalid numerical equivalent in result bundle.") from exc


def save_result_bundle(
    path: str | Path, bundle: Mapping, *, overwrite: bool = False
) -> None:
    """Write validated JSON without overwriting an existing file by default."""
    network_from_result_bundle(bundle)
    content = json.dumps(bundle, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(content)
