"""Fixed-configuration reference libraries with explicit consistency checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .models import BusPhaseLabel, ElectricalNetwork


def split_phase_voltage_errors(
    library: ScenarioLibrary, represented: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-snapshot differential voltage and signed leg-imbalance diagnostics."""
    indices = {label: index for index, label in enumerate(library.network.labels)}
    pairs = [
        (indices[(bus, "S1")], indices[(bus, "S2")])
        for bus in sorted(library.service_domains)
        if (bus, "S1") in indices and (bus, "S2") in indices
    ]
    if not pairs:
        return tuple(np.zeros(reference.shape[1]) for _ in range(5))
    first, second = np.asarray(pairs).T
    bases = (library.voltage_bases[first] + library.voltage_bases[second])[:, None]
    reference_difference = reference[first] - reference[second]
    represented_difference = represented[first] - represented[second]
    reference_imbalance = 2 * (abs(reference[first]) - abs(reference[second])) / bases
    represented_imbalance = (
        2 * (abs(represented[first]) - abs(represented[second])) / bases
    )
    return (
        np.max(
            abs(abs(represented_difference) - abs(reference_difference)) / bases, axis=0
        ),
        np.max(abs(represented_difference - reference_difference) / bases, axis=0),
        np.max(abs(reference_imbalance), axis=0),
        np.max(abs(represented_imbalance), axis=0),
        np.max(abs(represented_imbalance - reference_imbalance), axis=0),
    )


def _bases(values: np.ndarray, size: int) -> np.ndarray:
    bases = np.array(values, dtype=float, copy=True)
    if bases.shape != (size,) or not np.all(np.isfinite(bases)) or np.any(bases <= 0):
        raise ValueError("Voltage bases must be finite positive terminal values.")
    bases.setflags(write=False)
    return bases


def configuration_fingerprint(
    network: ElectricalNetwork,
    voltage_bases: np.ndarray,
    *,
    model_signature: str = "",
    service_domains: Mapping[str, str] | None = None,
) -> str:
    """Hash canonical SI matrix data, indexing, bases, sources, and model identity."""
    bases = _bases(voltage_bases, len(network.labels))
    matrix = network.admittance.copy()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    descriptor = {
        "schema": "gdm-reduce-configuration-v1",
        "labels": network.labels,
        "sources": [
            [
                label,
                complex(network.source_voltages[label]).real,
                complex(network.source_voltages[label]).imag,
            ]
            for label in network.labels
            if label in network.source_voltages
        ],
        "model_signature": model_signature,
    }
    if service_domains:
        descriptor["service_domains"] = sorted(service_domains.items())
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
    for values, dtype in (
        (matrix.indptr, "<i8"),
        (matrix.indices, "<i8"),
        (matrix.data, "<c16"),
        (bases, "<f8"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceScenario:
    """One solved reference point: volts and amperes in original terminal order."""

    name: str
    labels: Sequence[BusPhaseLabel]
    voltage: np.ndarray
    currents: np.ndarray
    configuration_fingerprint: str
    reference_success: bool
    reference_mismatch_pu: float
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Scenario names must be nonempty strings.")
        labels = tuple(self.labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Scenario terminal labels must be nonempty and unique.")
        if self.reference_success is not True:
            raise ValueError("Reference solver must report successful convergence.")
        if (
            not np.isfinite(self.reference_mismatch_pu)
            or self.reference_mismatch_pu < 0
        ):
            raise ValueError("Reference mismatch must be finite and nonnegative.")
        if (
            not isinstance(self.configuration_fingerprint, str)
            or not self.configuration_fingerprint
        ):
            raise ValueError("Scenario configuration fingerprint is required.")
        for name in ("voltage", "currents"):
            values = np.array(getattr(self, name), dtype=complex, copy=True)
            if values.shape != (len(labels),) or not np.all(np.isfinite(values)):
                raise ValueError(f"Scenario {name} must have finite terminal values.")
            if name == "voltage" and np.any(values == 0):
                raise ValueError("Reference terminal voltages must be nonzero.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        provenance = dict(self.provenance)
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in provenance.items()
        ):
            raise ValueError("Scenario provenance must contain string keys and values.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))


@dataclass(frozen=True)
class ScenarioLibrary:
    """Validated references sharing one matrix, terminal order, and source state.

    KCL checks internal consistency, not independent feeder-model accuracy.
    Reference mismatch is the backend's convergence metric, not reduction error.
    """

    network: ElectricalNetwork
    voltage_bases: np.ndarray
    scenarios: Sequence[ReferenceScenario]
    model_signature: str = ""
    bus_uuids: Mapping[str, str] = field(default_factory=dict)
    component_buses: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    service_domains: Mapping[str, str] = field(default_factory=dict)
    max_reference_mismatch_pu: float = 1e-5
    kcl_atol_a: float = 1e-7
    kcl_rtol: float = 1e-8
    source_atol_v: float = 1e-6
    configuration_fingerprint: str = field(init=False)
    max_kcl_residual_a: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "max_reference_mismatch_pu",
            "kcl_atol_a",
            "kcl_rtol",
            "source_atol_v",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        network = ElectricalNetwork(
            self.network.admittance, self.network.labels, self.network.source_voltages
        )
        bases = _bases(self.voltage_bases, len(network.labels))
        domains = dict(self.service_domains)
        split_buses = {bus for bus, phase in network.labels if phase in {"S1", "S2"}}
        if set(domains) != split_buses or any(
            not isinstance(domain, str) or not domain for domain in domains.values()
        ):
            raise ValueError("Split-phase service domains must cover every S1/S2 bus.")
        fingerprint = configuration_fingerprint(
            network,
            bases,
            model_signature=self.model_signature,
            service_domains=domains,
        )
        scenarios = tuple(self.scenarios)
        if not scenarios or len({item.name for item in scenarios}) != len(scenarios):
            raise ValueError("A library requires nonempty, uniquely named scenarios.")
        residuals = []
        for scenario in scenarios:
            if scenario.labels != network.labels:
                raise ValueError("Scenario terminal indexing differs from the network.")
            if scenario.configuration_fingerprint != fingerprint:
                raise ValueError("Scenario electrical configuration is incompatible.")
            if scenario.reference_mismatch_pu > self.max_reference_mismatch_pu:
                raise ValueError("Reference convergence mismatch exceeds the limit.")
            for index, label in enumerate(network.labels):
                if (
                    label in network.source_voltages
                    and abs(scenario.voltage[index] - network.source_voltages[label])
                    > self.source_atol_v
                ):
                    raise ValueError(
                        "Reference scenario changed a fixed source voltage."
                    )
            residual = np.abs(network.admittance @ scenario.voltage - scenario.currents)
            if np.any(
                residual > self.kcl_atol_a + self.kcl_rtol * np.abs(scenario.currents)
            ):
                raise ValueError(
                    "Reference scenario violates the KCL residual tolerance."
                )
            residuals.append(float(np.max(residual)))
        buses = {label[0] for label in network.labels}
        bus_uuids = dict(self.bus_uuids)
        if bus_uuids and (
            set(bus_uuids) != buses
            or any(
                not isinstance(value, str) or not value for value in bus_uuids.values()
            )
            or len(set(bus_uuids.values())) != len(bus_uuids)
        ):
            raise ValueError("Bus UUID lineage must cover each bus exactly once.")
        lineage = {key: tuple(value) for key, value in self.component_buses.items()}
        if any(
            not isinstance(key, str) or not key or not set(value) <= buses
            for key, value in lineage.items()
        ):
            raise ValueError("Component lineage must refer to known original buses.")
        residuals = np.array(residuals)
        residuals.setflags(write=False)
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "voltage_bases", bases)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "configuration_fingerprint", fingerprint)
        object.__setattr__(self, "max_kcl_residual_a", residuals)
        object.__setattr__(self, "bus_uuids", MappingProxyType(bus_uuids))
        object.__setattr__(self, "component_buses", MappingProxyType(lineage))
        object.__setattr__(self, "service_domains", MappingProxyType(domains))

    @property
    def voltage(self) -> np.ndarray:
        """Terminal-by-scenario complex reference voltages in volts."""
        return np.column_stack([scenario.voltage for scenario in self.scenarios])

    @property
    def currents(self) -> np.ndarray:
        """Terminal-by-scenario injections in amperes, including source outputs."""
        return np.column_stack([scenario.currents for scenario in self.scenarios])
