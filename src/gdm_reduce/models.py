"""Validated electrical inputs, indexed by existing bus-phase terminals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BusPhaseLabel = tuple[str, str]


@dataclass(frozen=True)
class ElectricalNetwork:
    """Passive admittance in siemens, with fixed source voltages in volts.

    Only existing A/B/C terminals are represented. Source currents are outputs
    of the voltage solve, not prescribed boundary conditions. Input matrices
    and source mappings are copied; no neutral or split-phase inference occurs.
    """

    admittance: csr_matrix | np.ndarray
    labels: Sequence[BusPhaseLabel]
    source_voltages: Mapping[BusPhaseLabel, complex]

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if not labels or any(
            not isinstance(label, tuple)
            or len(label) != 2
            or not isinstance(label[0], str)
            or not label[0]
            or label[1] not in {"A", "B", "C"}
            for label in labels
        ):
            raise ValueError("Labels must be nonempty (bus_name, A/B/C) tuples.")
        if len(set(labels)) != len(labels):
            raise ValueError("Bus-phase labels must be unique.")
        admittance = csr_matrix(self.admittance, dtype=np.complex128, copy=True)
        if admittance.shape != (len(labels), len(labels)):
            raise ValueError("Admittance shape must match the terminal labels.")
        admittance.sum_duplicates()
        admittance.eliminate_zeros()
        admittance.sort_indices()
        if not np.all(np.isfinite(admittance.data)):
            raise ValueError("Admittance must contain only finite values.")
        sources = dict(self.source_voltages)
        if not sources or any(label not in labels for label in sources):
            raise ValueError("At least one known source terminal is required.")
        if any(not np.isfinite(value) or value == 0 for value in sources.values()):
            raise ValueError("Source voltages must be finite and nonzero.")
        _, components = connected_components(admittance, directed=False)
        source_components = {components[labels.index(label)] for label in sources}
        if any(component not in source_components for component in components):
            raise ValueError("Every terminal must be connected to a fixed source.")
        object.__setattr__(self, "admittance", admittance)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "source_voltages", MappingProxyType(sources))
