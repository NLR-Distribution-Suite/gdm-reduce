"""Phase-aware electrical network reduction."""

from .adapters import (
    GDMSearchConstraints,
    build_gdm_scenario_library,
    from_gdm_flow_results,
    get_gdm_search_constraints,
)
from .kron import KronReductionResult, kron_reduce, solve_voltages
from .models import BusPhaseLabel, ElectricalNetwork
from .radialization import RadializationResult, radialize_reduction
from .scenarios import ReferenceScenario, ScenarioLibrary, configuration_fingerprint
from .search import CandidateSearchResult, ReductionStep, optimize_kron_reduction

__all__ = [
    "BusPhaseLabel",
    "CandidateSearchResult",
    "ElectricalNetwork",
    "GDMSearchConstraints",
    "KronReductionResult",
    "RadializationResult",
    "ReferenceScenario",
    "ReductionStep",
    "ScenarioLibrary",
    "build_gdm_scenario_library",
    "configuration_fingerprint",
    "from_gdm_flow_results",
    "get_gdm_search_constraints",
    "kron_reduce",
    "optimize_kron_reduction",
    "radialize_reduction",
    "solve_voltages",
]
