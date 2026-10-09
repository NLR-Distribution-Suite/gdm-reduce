"""Evaluate frozen current-pooling assignments on compatible reference snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kron import solve_voltages
from .radialization import RadializationResult
from .scenarios import ScenarioLibrary, configuration_fingerprint
from .search import CandidateSearchResult


@dataclass(frozen=True)
class ScenarioValidation:
    name: str
    max_voltage_magnitude_error_pu: float
    max_complex_voltage_error_pu: float
    max_voltage_angle_error_deg: float | None
    full_network_max_complex_error_pu: float
    reference_mismatch_pu: float
    reference_kcl_residual_a: float
    source_active_power_error_w: float
    source_reactive_power_error_var: float
    passive_active_power_error_w: float
    within_magnitude_limit: bool


@dataclass(frozen=True)
class ReductionValidationReport:
    """Prescribed-current errors, not nonlinear reduced-device validation.

    Power differences are signed reduced-minus-reference values. Passive power
    includes all active absorption represented in Y, not just line losses.
    """

    configuration_fingerprint: str
    voltage_error_limit_pu: float
    scenarios: tuple[ScenarioValidation, ...]

    @property
    def passed(self) -> bool:
        return all(scenario.within_magnitude_limit for scenario in self.scenarios)

    @property
    def max_voltage_magnitude_error_pu(self) -> float:
        return max(
            scenario.max_voltage_magnitude_error_pu for scenario in self.scenarios
        )


def pool_scenario_currents(
    selection: CandidateSearchResult, library: ScenarioLibrary
) -> np.ndarray:
    """Apply existing whole-bus clusters without re-running candidate search."""
    fingerprint = configuration_fingerprint(
        selection.equivalent.network,
        library.voltage_bases,
        model_signature=library.model_signature,
    )
    if (
        library.network.labels != selection.equivalent.network.labels
        or library.configuration_fingerprint != selection.configuration_fingerprint
        or fingerprint != selection.configuration_fingerprint
    ):
        raise ValueError("Evaluation references must match the selected configuration.")
    indices = {label: index for index, label in enumerate(library.network.labels)}
    receivers = np.array(
        [
            indices[(selection.original_to_retained[bus], phase)]
            for bus, phase in library.network.labels
        ]
    )
    currents = np.zeros_like(library.currents)
    np.add.at(currents, receivers, library.currents)
    return currents


def evaluate_reduction(
    result: CandidateSearchResult | RadializationResult,
    library: ScenarioLibrary,
    *,
    voltage_error_limit_pu: float = 0.001,
) -> ReductionValidationReport:
    """Report errors on supplied snapshots with fixed clusters and sources.

    References may be training or held-out data; the caller owns that split.
    Magnitude/angle/complex errors use cluster representative voltages. Power
    diagnostics use electrically reconstructed voltages and actual YV currents,
    never the pooled bookkeeping entries at fixed sources.
    """
    if not np.isfinite(voltage_error_limit_pu) or voltage_error_limit_pu < 0:
        raise ValueError("voltage_error_limit_pu must be finite and nonnegative.")
    selection = result.selection if isinstance(result, RadializationResult) else result
    equivalent = result.equivalent
    pooled = pool_scenario_currents(selection, library)
    retained = equivalent.solve(pooled)
    indices = {label: index for index, label in enumerate(equivalent.retained_labels)}
    represented = retained[
        [
            indices[(selection.original_to_retained[bus], phase)]
            for bus, phase in library.network.labels
        ]
    ]
    electrical_voltage = equivalent.reconstruct(pooled, retained)
    full_voltage = solve_voltages(library.network, library.currents)
    if not np.all(np.isfinite(electrical_voltage)):
        raise ValueError("Evaluation produced nonfinite voltages.")
    reference = library.voltage
    bases = library.voltage_bases[:, None]
    magnitude_errors = np.abs(np.abs(represented) - np.abs(reference)) / bases
    complex_errors = np.abs(represented - reference) / bases
    baseline_errors = np.abs(full_voltage - reference) / bases
    angle_errors = np.abs(np.rad2deg(np.angle(represented * np.conj(reference))))
    actual_currents = library.network.admittance @ electrical_voltage
    reduced_power = electrical_voltage * np.conj(actual_currents)
    reference_power = reference * np.conj(library.currents)
    source_indices = [
        index
        for index, label in enumerate(library.network.labels)
        if label in library.network.source_voltages
    ]
    source_difference = (
        reduced_power[source_indices] - reference_power[source_indices]
    ).sum(axis=0)
    passive_difference = (reduced_power.real - reference_power.real).sum(axis=0)
    scenarios = tuple(
        ScenarioValidation(
            scenario.name,
            float(np.max(magnitude_errors[:, index])),
            float(np.max(complex_errors[:, index])),
            (
                None
                if np.any(represented[:, index] == 0)
                else float(np.max(angle_errors[:, index]))
            ),
            float(np.max(baseline_errors[:, index])),
            float(scenario.reference_mismatch_pu),
            float(library.max_kcl_residual_a[index]),
            float(source_difference[index].real),
            float(source_difference[index].imag),
            float(passive_difference[index]),
            bool(np.max(magnitude_errors[:, index]) <= voltage_error_limit_pu),
        )
        for index, scenario in enumerate(library.scenarios)
    )
    return ReductionValidationReport(
        library.configuration_fingerprint, voltage_error_limit_pu, scenarios
    )
