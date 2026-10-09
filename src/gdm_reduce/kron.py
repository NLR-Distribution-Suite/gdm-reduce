"""Exact source-conditioned Schur elimination for prescribed currents."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, hstack
from scipy.sparse.linalg import LinearOperator, SuperLU, norm, onenormest, splu

from .models import BusPhaseLabel, ElectricalNetwork


def _values(values: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim not in {1, 2} or result.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},) or ({size}, scenarios).")
    if result.ndim == 2 and result.shape[1] == 0:
        raise ValueError(f"{name} must include at least one scenario.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _factor(
    admittance: csr_matrix, max_condition_number: float
) -> tuple[SuperLU, float]:
    try:
        factor = splu(admittance.tocsc())
    except RuntimeError as exc:
        raise ValueError("Admittance is singular; check source boundaries.") from exc
    inverse = LinearOperator(
        admittance.shape,
        matvec=factor.solve,
        rmatvec=lambda vector: factor.solve(vector, trans="H"),
        matmat=factor.solve,
        dtype=np.complex128,
    )
    condition = float(norm(admittance, 1) * onenormest(inverse))
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError(
            f"Admittance is ill-conditioned (estimated condition {condition:.3g})."
        )
    return factor, condition


def _condition_limit(value: float) -> float:
    if not np.isfinite(value) or value < 1:
        raise ValueError("max_condition_number must be finite and at least one.")
    return float(value)


def _boundary(
    network: ElectricalNetwork, max_condition_number: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SuperLU | None, float]:
    sources = np.array(
        [
            index
            for index, label in enumerate(network.labels)
            if label in network.source_voltages
        ],
        dtype=int,
    )
    unknown = np.array(
        [
            index
            for index, label in enumerate(network.labels)
            if label not in network.source_voltages
        ],
        dtype=int,
    )
    voltage = np.array(
        [network.source_voltages[network.labels[index]] for index in sources],
        dtype=np.complex128,
    )
    if not len(unknown):
        return sources, unknown, voltage, None, 1.0
    factor, condition = _factor(
        network.admittance[unknown, :][:, unknown], max_condition_number
    )
    return sources, unknown, voltage, factor, condition


def _solve(
    network: ElectricalNetwork,
    currents: np.ndarray,
    sources: np.ndarray,
    unknown: np.ndarray,
    source_voltage: np.ndarray,
    factor: SuperLU | None,
) -> np.ndarray:
    voltage = np.empty_like(currents)
    voltage[sources] = source_voltage if currents.ndim == 1 else source_voltage[:, None]
    if factor is not None:
        source_term = network.admittance[unknown, :][:, sources] @ source_voltage
        rhs = currents[unknown] - (
            source_term if currents.ndim == 1 else source_term[:, None]
        )
        voltage[unknown] = factor.solve(rhs)
    return voltage


def solve_voltages(
    network: ElectricalNetwork,
    currents: np.ndarray,
    *,
    max_condition_number: float = 1e12,
) -> np.ndarray:
    """Solve fixed-current voltages; supplied source-current entries are ignored.

    Currents are in amperes, positive into the passive network. Inputs can be
    one scenario or a terminal-by-scenario matrix. This is not nonlinear PF.
    """
    currents = _values(currents, len(network.labels), "currents")
    limit = _condition_limit(max_condition_number)
    sources, unknown, source_voltage, factor, _ = _boundary(network, limit)
    return _solve(network, currents, sources, unknown, source_voltage, factor)


@dataclass(frozen=True)
class KronReductionResult:
    """Exact equivalent and cached solves, retaining the original input indexing."""

    network: ElectricalNetwork
    reduced_network: ElectricalNetwork
    retained_indices: np.ndarray
    eliminated_indices: np.ndarray
    estimated_condition_number: float
    _elimination_factor: SuperLU | None = field(repr=False)
    _sources: np.ndarray = field(repr=False)
    _unknown: np.ndarray = field(repr=False)
    _source_voltage: np.ndarray = field(repr=False)
    _retained_factor: SuperLU | None = field(repr=False)

    @property
    def admittance(self) -> csr_matrix:
        return self.reduced_network.admittance

    @property
    def retained_labels(self) -> tuple[BusPhaseLabel, ...]:
        return tuple(self.reduced_network.labels)

    def transform_currents(self, currents: np.ndarray) -> np.ndarray:
        """Return I_k - Y_kr solve(Y_rr, I_r), not cluster-pooled currents."""
        currents = _values(currents, len(self.network.labels), "currents")
        transformed = currents[self.retained_indices].copy()
        if self._elimination_factor is not None:
            coupling = self.network.admittance[self.retained_indices, :][
                :, self.eliminated_indices
            ]
            transformed -= coupling @ self._elimination_factor.solve(
                currents[self.eliminated_indices]
            )
        return transformed

    def solve(self, currents: np.ndarray) -> np.ndarray:
        """Return retained voltages in volts from currents in original indexing."""
        return _solve(
            self.reduced_network,
            self.transform_currents(currents),
            self._sources,
            self._unknown,
            self._source_voltage,
            self._retained_factor,
        )

    def reconstruct(
        self, currents: np.ndarray, retained_voltage: np.ndarray
    ) -> np.ndarray:
        """Recover all terminal voltages using the eliminated-node equations."""
        currents = _values(currents, len(self.network.labels), "currents")
        retained_voltage = _values(
            retained_voltage, len(self.retained_indices), "retained_voltage"
        )
        if currents.shape[1:] != retained_voltage.shape[1:]:
            raise ValueError("Currents and voltages need matching scenario shapes.")
        voltage = np.empty_like(currents)
        voltage[self.retained_indices] = retained_voltage
        if self._elimination_factor is not None:
            coupling = self.network.admittance[self.eliminated_indices, :][
                :, self.retained_indices
            ]
            voltage[self.eliminated_indices] = self._elimination_factor.solve(
                currents[self.eliminated_indices] - coupling @ retained_voltage
            )
        return voltage


def kron_reduce(
    network: ElectricalNetwork,
    retained_buses: Collection[str],
    *,
    block_size: int = 128,
    max_condition_number: float = 1e12,
) -> KronReductionResult:
    """Eliminate whole buses, keeping every source bus and existing phase coupling.

    Sparse LU solves evaluate the Schur complement in column batches without
    forming an inverse. The equivalent itself may have substantial fill-in.
    Loaded eliminated buses require the exact current transformation; this
    function does not aggregate devices or implement Opti-KRON selection.
    """
    if isinstance(retained_buses, str):
        raise ValueError("retained_buses must be a collection of bus names.")
    retained_buses = set(retained_buses)
    bus_names = {label[0] for label in network.labels}
    if not retained_buses or not retained_buses <= bus_names:
        raise ValueError("Retained buses must be a nonempty subset of known buses.")
    if any(label[0] not in retained_buses for label in network.source_voltages):
        raise ValueError("Every source bus must be retained.")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer.")
    limit = _condition_limit(max_condition_number)
    _, _, _, _, condition = _boundary(network, limit)
    retained = np.array(
        [
            index
            for index, label in enumerate(network.labels)
            if label[0] in retained_buses
        ],
        dtype=int,
    )
    eliminated = np.array(
        [
            index
            for index, label in enumerate(network.labels)
            if label[0] not in retained_buses
        ],
        dtype=int,
    )
    admittance = network.admittance
    reduced_admittance = admittance[retained, :][:, retained]
    elimination_factor = None
    if len(eliminated):
        elimination_factor, _ = _factor(admittance[eliminated, :][:, eliminated], limit)
        forward = admittance[retained, :][:, eliminated]
        backward = admittance[eliminated, :][:, retained]
        blocks = []
        for start in range(0, len(retained), block_size):
            columns = slice(start, start + block_size)
            correction = forward @ elimination_factor.solve(
                backward[:, columns].toarray()
            )
            blocks.append(
                csc_matrix(reduced_admittance[:, columns].toarray() - correction)
            )
        reduced_admittance = hstack(blocks, format="csr")
    reduced_network = ElectricalNetwork(
        reduced_admittance,
        tuple(network.labels[index] for index in retained),
        network.source_voltages,
    )
    sources, unknown, source_voltage, factor, _ = _boundary(reduced_network, limit)
    return KronReductionResult(
        network,
        reduced_network,
        retained,
        eliminated,
        condition,
        elimination_factor,
        sources,
        unknown,
        source_voltage,
        factor,
    )
