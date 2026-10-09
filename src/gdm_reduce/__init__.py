"""Phase-aware electrical network reduction."""

from .adapters import build_gdm_scenario_library, from_gdm_flow_results
from .kron import KronReductionResult, kron_reduce, solve_voltages
from .models import BusPhaseLabel, ElectricalNetwork
from .scenarios import ReferenceScenario, ScenarioLibrary, configuration_fingerprint

__all__ = [
    "BusPhaseLabel",
    "ElectricalNetwork",
    "KronReductionResult",
    "ReferenceScenario",
    "ScenarioLibrary",
    "build_gdm_scenario_library",
    "configuration_fingerprint",
    "from_gdm_flow_results",
    "kron_reduce",
    "solve_voltages",
]
