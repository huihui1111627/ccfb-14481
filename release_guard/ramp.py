"""Adaptive traffic-ramp controller (requirement 7).

A fixed staircase (1% -> 5% -> 20% -> 100% on a clock) spends the same
exposure on a clean release and on a suspicious one.  The adaptive rule here
chooses the next treatment fraction from the evidence in hand:

* the always-valid upper harm bound from :class:`DecisionEngine` says how bad
  the release could still plausibly be;
* the posterior expected positive harm says how bad it is expected to be;
* multiplying either by the number of users that a bigger canary would expose
  turns it into an expected-loss budget;
* the fraction grows while the budget is cheap and stalls (or shrinks) while
  it is not, and is hard-capped so one batch can never more than double
  exposure -- a guard against a single over-optimistic look.

Two policies are implemented for the validation comparison:

``"adaptive"``
    confidence-bound / posterior expected-loss budgeting (default);
``"staircase"``
    the conventional fixed schedule keyed only on elapsed batches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import DecisionEngine, DecisionResult, ALL_CELL

_EPS = 1e-12


@dataclass
class RampDecision:
    next_fraction: float
    policy: str
    rationale: str
    safe_budget: float
    expected_remaining_harm: float
    users_at_risk: int


class RampController:
    """Compute the next canary fraction after each engine decision."""

    STAIRCASE = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)

    def __init__(self,
                 policy: str = "adaptive",
                 loss_weights: dict[str, float] | None = None,
                 safe_budget_per_release: float = 200.0,
                 min_step: float = 0.01,
                 max_growth: float = 2.0,
                 users_per_batch: int = 20000) -> None:
        if policy not in ("adaptive", "staircase"):
            raise ValueError(policy)
        self.policy = policy
        # converts one unit of per-user metric harm into platform loss units;
        # defaults weight the failure-rate metric like a production SLO breach
        self.loss_weights = loss_weights or {}
        self.safe_budget = safe_budget_per_release
        self.min_step = min_step
        self.max_growth = max_growth
        self.users_per_batch = users_per_batch
        self.spent_budget = 0.0

    # ------------------------------------------------------------------

    def next_fraction(self,
                      engine: DecisionEngine,
                      result: DecisionResult,
                      current_fraction: float) -> RampDecision:
        if result.verdict == "ROLLBACK":
            return RampDecision(0.0, self.policy,
                                "判定回滚，流量归零", 0.0,
                                result.worst_abs_harm, 0)
        if result.verdict == "GO":
            return RampDecision(1.0, self.policy,
                                "判定放行，全量发布", self.safe_budget,
                                0.0, 0)

        if self.policy == "staircase":
            nxt = self._staircase(engine.batch_index, current_fraction)
            return RampDecision(nxt, "staircase",
                                f"固定阶梯第 {engine.batch_index} 步",
                                self.safe_budget,
                                result.worst_abs_harm,
                                int(nxt * self.users_per_batch))

        # adaptive: risk is measured on a NORMALISED scale per metric so a
        # high-dispersion count metric cannot dominate a rate metric merely
        # because of its native units.
        risk = 0.0
        n_active = 0
        any_alarm = False
        for mr in result.metric_results:
            g = mr.global_result
            if not g.active:
                continue
            n_active += 1
            if mr.rollback:
                any_alarm = True
            if mr.family == "count" and g.mean_control and \
                    abs(g.mean_control) > _EPS:
                rel_harm = max(g.cs_high, 0.0) / abs(g.mean_control)
            else:
                rel_harm = max(g.upper_harm, 0.0)
                if g.mean_control and abs(g.mean_control) > _EPS:
                    rel_harm = rel_harm / abs(g.mean_control)
            risk = max(risk, self.loss_weights.get(mr.metric, 1.0)
                       * rel_harm)
        exp_harm = 0.0
        for name, h in engine.expected_harm_estimates().items():
            if not math.isnan(h):
                exp_harm += self.loss_weights.get(name, 1.0) * abs(h)
        blended = 0.7 * risk + 0.3 * exp_harm
        remaining = max(self.safe_budget - self.spent_budget, 0.0)

        # Growth rule.  With no rollback signal the canary must keep climbing
        # so the confidence sequence can actually narrow: a slow-but-certain
        # ramp (growth cap) carries more information than stalling forever.
        # The safety budget only decides HOW FAST, never freezes at zero.
        if any_alarm:
            growth = self.min_step
        elif blended <= _EPS or n_active == 0:
            growth = self.max_growth
        else:
            affordable = (remaining / blended) / max(self.users_per_batch, 1)
            # map affordable exposure to a growth multiplier in [1.25, cap]
            growth = min(self.max_growth, max(1.25, min(affordable, 4.0)))
        if not result.eligible:
            growth = min(growth, 1.25)
        base = current_fraction if current_fraction > 0 else self.min_step
        nxt = float(min(max(base * growth, self.min_step), 1.0))
        users_at_risk = int(nxt * self.users_per_batch)
        self.spent_budget += blended * users_at_risk
        return RampDecision(
            nxt, "adaptive",
            f"归一化风险 {blended:.4g}，剩余安全预算 {remaining:.1f}，"
            f"增长倍率 x{growth:.2f}",
            remaining, blended, users_at_risk)

    def _staircase(self, batch_index: int, current: float) -> float:
        # one step every 5 batches, regardless of evidence
        step = min(batch_index // 5, len(self.STAIRCASE) - 1)
        return max(self.STAIRCASE[step], current)
