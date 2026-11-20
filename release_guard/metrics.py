"""Metric definitions and their statistically correct per-user statistics.

Three metric families are supported, each reduced to *one scalar per user per
group*, because requests from the same user are dependent and treating them as
independent samples silently deflates the variance (requirement 5):

================  ============================  =================================
family            per-user summary x           harm direction
================  ============================  =================================
success rate      failure fraction 1-success    larger is worse
latency tail      fraction of requests above a  larger is worse (exceedance rate
                  long-tail threshold q0         of the pre-release p95/p99 -- a
                                                  distribution-free stand-in for
                                                  quantile comparison)
counts            requests per user             larger or smaller can be harmful;
                                                  two-sided by default
================  ============================  =================================

Why threshold exceedance instead of a raw quantile?  A quantile is a
non-smooth order statistic whose sequential distribution theory is awkward; the
event "latency > q0" is Bernoulli, and a shift of the long-tail mass is
exactly what the exceedance-rate test detects.  Several thresholds can be
tracked at once (p95 and p99).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str          # "success" | "latency" | "count"
    harm: str            # "one_sided" (larger is worse) | "two_sided"
    threshold: float | None = None   # latency family: tail cut-off q0
    # practical non-effect floor: harms smaller than this are considered
    # acceptable when deciding GO (keeps a trivially tiny but significant
    # regression from blocking a release); in the metric's native units
    effect_floor: float = 0.0
    description: str = ""

    @property
    def kind(self) -> str:
        if self.family == "success":
            return "proportion"
        if self.family == "latency":
            return f"tail-exceedance@q0={self.threshold:g}"
        return "count"


def default_metrics(latency_thresholds: tuple[float, ...] = ()) -> list[MetricSpec]:
    """Built-in dashboard metric set for the electronic-tax platform.

    ``latency_thresholds`` are the pre-release p95/p99 cut-offs estimated from
    baseline data; pass whatever the baseline generator produced.
    """
    metrics = [
        MetricSpec("success_rate", "success", "one_sided",
                   effect_floor=0.002,
                   description="申报请求成功率（按失败率检验，单侧），地板 +0.2pp"),
        MetricSpec("requests_per_user", "count", "two_sided",
                   # count dispersion is ~6 requests/user, so an absolute
                   # floor of 0.15 would be unreachable; 8% relative is the
                   # smallest operationally meaningful shift
                   effect_floor=0.08,
                   description="每用户每批请求数（计数，双侧，相对地板 8%）"),
    ]
    for i, q0 in enumerate(latency_thresholds):
        tag = "p95" if i == 0 else ("p99" if i == 1 else f"q{q0:g}")
        metrics.append(MetricSpec(
            f"latency_tail_{tag}", "latency", "one_sided",
            threshold=float(q0), effect_floor=0.02,
            description=f"时延长尾超标率（q0={q0:g}ms，单侧），地板 +2pp"))
    return metrics
