"""release_guard -- sequential, always-valid canary release decision engine."""
from .dashboard import DashboardRule
from .engine import DecisionEngine, DecisionResult, MetricResult, CellResult
from .eprocess import (
    msprt_harm, msprt_two_sided, confidence_sequence, harm_upper_bound,
    BernoulliSequential, TwoSampleBernoulliE,
)
from .generator import (
    TrafficGenerator, DegradationSpec, build_baseline, DIM_CATEGORIES,
)
from .metrics import MetricSpec, default_metrics
from .ramp import RampController, RampDecision

__all__ = [
    "DecisionEngine", "DecisionResult", "MetricResult", "CellResult",
    "msprt_harm", "msprt_two_sided", "confidence_sequence",
    "harm_upper_bound", "BernoulliSequential", "TwoSampleBernoulliE",
    "DashboardRule",
    "TrafficGenerator", "DegradationSpec", "build_baseline",
    "DIM_CATEGORIES", "MetricSpec", "default_metrics",
    "RampController", "RampDecision",
]
