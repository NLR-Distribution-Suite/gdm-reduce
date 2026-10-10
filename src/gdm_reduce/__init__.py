"""Phase-aware electrical network reduction."""

from .adapters import (
    GDMSearchConstraints,
    audit_split_phase_grounding,
    build_gdm_scenario_library,
    from_gdm_flow_results,
    get_gdm_search_constraints,
)
from .artifacts import (
    build_result_bundle,
    network_from_result_bundle,
    save_result_bundle,
)
from .gdm_export import (
    GDMModelExport,
    build_reduced_gdm,
    save_reduced_gdm,
    validate_reduced_gdm,
)
from .kron import KronReductionResult, kron_reduce, solve_voltages
from .models import BusPhaseLabel, ElectricalNetwork
from .radialization import RadializationResult, radialize_reduction
from .scenarios import ReferenceScenario, ScenarioLibrary, configuration_fingerprint
from .search import CandidateSearchResult, ReductionStep, optimize_kron_reduction
from .validation import (
    ReductionValidationReport,
    ScenarioValidation,
    evaluate_reduction,
    pool_scenario_currents,
)

__all__ = [
    "BusPhaseLabel",
    "CandidateSearchResult",
    "ElectricalNetwork",
    "GDMModelExport",
    "GDMSearchConstraints",
    "KronReductionResult",
    "RadializationResult",
    "ReferenceScenario",
    "ReductionStep",
    "ReductionValidationReport",
    "ScenarioLibrary",
    "ScenarioValidation",
    "audit_split_phase_grounding",
    "build_gdm_scenario_library",
    "build_reduced_gdm",
    "build_result_bundle",
    "configuration_fingerprint",
    "evaluate_reduction",
    "from_gdm_flow_results",
    "get_gdm_search_constraints",
    "kron_reduce",
    "network_from_result_bundle",
    "optimize_kron_reduction",
    "pool_scenario_currents",
    "radialize_reduction",
    "save_result_bundle",
    "save_reduced_gdm",
    "solve_voltages",
    "validate_reduced_gdm",
]
