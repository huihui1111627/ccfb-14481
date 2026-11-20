"""Validation harness: thousands of simulated releases with scored outcomes.

For every scenario the harness runs many independent releases through both the
sequential engine (with an adaptive or staircase ramp) and the incumbent
fixed-threshold dashboard rule, then aggregates the acceptance metrics:

* **false rollback rate** on clean releases -- must stay inside the declared
  ``alpha_fwer``;
* **detection power** at each degradation magnitude;
* **time-to-decision** and the number / fraction of users already on the bad
  version when the rollback fires (the blast radius);
* **segment-localisation** success for degradations confined to one slice;
* head-to-head comparison against the aggregate-only dashboard rule.

Results are returned as plain dicts and also written to JSON/CSV so the report
notebook and dashboard can consume them without rerunning.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dashboard import DashboardRule
from .engine import DecisionEngine
from .generator import (
    DIM_CATEGORIES,
    DegradationSpec,
    TrafficGenerator,
    build_baseline,
)
from .metrics import default_metrics
from .ramp import RampController


@dataclass
class ScenarioConfig:
    name: str
    spec: DegradationSpec
    n_releases: int = 200
    seed_base: int = 1000


@dataclass
class ReleaseOutcome:
    verdict: str
    batches: int
    frac_at_stop: float
    treated_users: int
    treated_requests: int
    rollback_metric: str | None
    rollback_slices: list[str]
    target_slice: str | None
    localised: bool
    target_slice_armed: bool
    global_gate_first: bool
    dashboard_verdict: str
    dashboard_batches: int
    dashboard_treated_requests: int
    dashboard_z_verdict: str
    dashboard_z_batches: int
    dashboard_z_treated_requests: int


def _target_slice(spec: DegradationSpec) -> str | None:
    if spec.target_dim is None:
        return None
    return f"{spec.target_dim}:{spec.target_cat}"


def run_one_release(metrics, baseline, spec: DegradationSpec, seed: int,
                    max_batches: int, users_per_batch: int,
                    ramp_policy: str, dashboard: DashboardRule,
                    engine_kwargs: dict | None = None) -> ReleaseOutcome:
    gen = TrafficGenerator(seed=seed, users_per_batch=users_per_batch)
    eng = DecisionEngine(
        metrics, DIM_CATEGORIES, baseline["stats"],
        max_batches=max_batches,
        min_users_per_group=150, min_users_cell=40,
        min_users_slice_treat=300, **(engine_kwargs or {}))
    ramp = RampController(policy=ramp_policy,
                          users_per_batch=users_per_batch)
    dash = DashboardRule() if dashboard is None else dashboard
    dash_z = DashboardRule(use_zscore=True)
    frac = 0.01
    treated_users = 0
    treated_requests = 0
    dash_verdict = "CONTINUE"
    dash_batch = max_batches
    dash_exposed = 0
    dash_z_verdict = "CONTINUE"
    dash_z_batch = max_batches
    dash_z_exposed = 0
    target = _target_slice(spec)

    for b in range(1, max_batches + 1):
        batch = gen.sample_users(b, frac, spec)
        res = eng.update(batch, frac)
        tmask = batch["group"] == 1
        treated_requests += int(tmask.sum())
        treated_users += int(np.unique(batch["user_id"][tmask]).size)

        # both dashboard variants see the same batches; record first rollback
        if dash_verdict == "CONTINUE":
            dv, _ = dash.verdict(batch)
            if dv == "ROLLBACK":
                dash_verdict = "ROLLBACK"
                dash_batch = b
                dash_exposed = treated_requests
        if dash_z_verdict == "CONTINUE":
            zv, _ = dash_z.verdict(batch)
            if zv == "ROLLBACK":
                dash_z_verdict = "ROLLBACK"
                dash_z_batch = b
                dash_z_exposed = treated_requests

        if res.verdict in ("GO", "ROLLBACK"):
            frac = ramp.next_fraction(eng, res, frac).next_fraction
            break
        frac = ramp.next_fraction(eng, res, frac).next_fraction

    rollback_metric = res.rollback_metric()
    slices = []
    for mr in res.metric_results:
        if mr.rollback:
            slices.extend(mr.rollback_slices)
    localised = (target is None) or (target in slices)
    # whether the targeted slice had reached its testing exposure when the
    # decision was taken; and whether the global gate fired without any slice
    target_armed = False
    if target is not None:
        for mr in res.metric_results:
            for s in mr.slices:
                if s.cell == target:
                    target_armed = target_armed or s.active
    global_first = bool(rollback_metric) and not slices
    if dash_verdict == "CONTINUE":
        dash_exposed = treated_requests
    if dash_z_verdict == "CONTINUE":
        dash_z_exposed = treated_requests
    return ReleaseOutcome(
        verdict=res.verdict, batches=res.batch_index,
        frac_at_stop=frac, treated_users=treated_users,
        treated_requests=treated_requests,
        rollback_metric=rollback_metric, rollback_slices=slices,
        target_slice=target, localised=localised,
        target_slice_armed=target_armed, global_gate_first=global_first,
        dashboard_verdict=dash_verdict, dashboard_batches=dash_batch,
        dashboard_treated_requests=dash_exposed,
        dashboard_z_verdict=dash_z_verdict, dashboard_z_batches=dash_z_batch,
        dashboard_z_treated_requests=dash_z_exposed)


def summarise(outcomes: list[ReleaseOutcome], alpha: float) -> dict:
    n = len(outcomes)
    rb = [o for o in outcomes if o.verdict == "ROLLBACK"]
    go = [o for o in outcomes if o.verdict == "GO"]
    cont = [o for o in outcomes if o.verdict == "CONTINUE"]
    dash_rb = [o for o in outcomes if o.dashboard_verdict == "ROLLBACK"]
    dash_z_rb = [o for o in outcomes if o.dashboard_z_verdict == "ROLLBACK"]
    seg_outcomes = [o for o in outcomes if o.target_slice is not None]
    exposed = np.array([o.treated_requests for o in rb], dtype=float)
    batches = np.array([o.batches for o in rb], dtype=float)
    # localisation is only meaningful once the target slice is armed; a
    # strong effect may be caught by the global gate before that point --
    # still a correct rollback, just not slice-attributable yet
    armed_seg = [o for o in seg_outcomes if o.target_slice_armed]
    localised_when_armed = (
        float(np.mean([o.localised for o in armed_seg]))
        if armed_seg else None)
    global_first_rate = (
        float(np.mean([o.global_gate_first for o in seg_outcomes]))
        if seg_outcomes else None)
    return {
        "n_releases": n,
        "engine": {
            "rollback_rate": len(rb) / n,
            "go_rate": len(go) / n,
            "continue_rate": len(cont) / n,
            "declared_alpha": alpha,
            "alpha_respected": (len(rb) / n) <= alpha + 2.0 * math_sd(alpha, n),
            "mean_batches_to_rollback": float(batches.mean()) if len(batches) else None,
            "median_batches_to_rollback": float(np.median(batches)) if len(batches) else None,
            "mean_treated_requests_at_rollback":
                float(exposed.mean()) if len(exposed) else None,
            "median_treated_requests_at_rollback":
                float(np.median(exposed)) if len(exposed) else None,
            "segment_localisation_rate": (
                float(np.mean([o.localised for o in seg_outcomes]))
                if seg_outcomes else None),
            "segment_localisation_rate_when_armed": localised_when_armed,
            "global_gate_first_rate": global_first_rate,
        },
        "dashboard": {
            "rollback_rate": len(dash_rb) / n,
            "mean_batches_to_rollback": float(np.mean(
                [o.dashboard_batches for o in dash_rb])) if dash_rb else None,
            "mean_treated_requests_at_rollback": float(np.mean(
                [o.dashboard_treated_requests for o in dash_rb]))
            if dash_rb else None,
        },
        "dashboard_zscore": {
            "rollback_rate": len(dash_z_rb) / n,
            "mean_batches_to_rollback": float(np.mean(
                [o.dashboard_z_batches for o in dash_z_rb]))
            if dash_z_rb else None,
            "mean_treated_requests_at_rollback": float(np.mean(
                [o.dashboard_z_treated_requests for o in dash_z_rb]))
            if dash_z_rb else None,
        },
    }


def math_sd(p: float, n: int) -> float:
    """Normal-approx standard error of a binomial proportion."""
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def default_scenarios() -> list[ScenarioConfig]:
    """The acceptance scenario grid (clean calibration + magnitude sweep +
    shapes + segment-localised regressions)."""
    grid = [
        ScenarioConfig("clean", DegradationSpec(kind="none", name="clean"),
                       n_releases=400, seed_base=2000),
    ]
    for mag in (0.002, 0.005, 0.01, 0.02):
        grid.append(ScenarioConfig(
            f"failure_step_{mag}",
            DegradationSpec(kind="failure", magnitude=mag, shape="step",
                            name=f"fail+{mag}"),
            n_releases=200, seed_base=3000))
    for mag in (0.2, 0.5, 1.0):
        grid.append(ScenarioConfig(
            f"latency_step_{mag}",
            DegradationSpec(kind="latency", magnitude=mag, shape="step",
                            name=f"latx{1+mag}"),
            n_releases=200, seed_base=4000))
    for mag in (-0.3, 0.3):
        grid.append(ScenarioConfig(
            f"count_step_{mag}",
            DegradationSpec(kind="count", magnitude=mag, shape="step"),
            n_releases=200, seed_base=5000))
    # the exact failure that slipped through last year: one taxpayer type only
    grid.append(ScenarioConfig(
        "segment_failure_自然人",
        DegradationSpec(kind="failure", magnitude=0.03, shape="step",
                        target_dim="taxpayer_type", target_cat="自然人"),
        n_releases=200, seed_base=6000))
    grid.append(ScenarioConfig(
        "segment_latency_企业",
        DegradationSpec(kind="latency", magnitude=0.6, shape="step",
                        target_dim="taxpayer_type", target_cat="企业"),
        n_releases=200, seed_base=7000))
    grid.append(ScenarioConfig(
        "failure_ramp",
        DegradationSpec(kind="failure", magnitude=0.02, shape="ramp"),
        n_releases=150, seed_base=8000))
    grid.append(ScenarioConfig(
        "latency_burst",
        DegradationSpec(kind="latency", magnitude=0.8, shape="burst"),
        n_releases=150, seed_base=9000))
    return grid


def run_validation(out_dir: str | Path,
                   scenarios: list[ScenarioConfig] | None = None,
                   max_batches: int = 40,
                   users_per_batch: int = 4000,
                   alpha_fwer: float = 0.01,
                   ramp_policy: str = "adaptive",
                   baseline_seed: int = 1) -> dict:
    """Run the full acceptance grid and persist raw + summarised results."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline(seed=baseline_seed)
    metrics = default_metrics((baseline["p95"], baseline["p99"]))
    dashboard = DashboardRule()
    scenarios = scenarios or default_scenarios()

    report = {"alpha_fwer": alpha_fwer, "ramp_policy": ramp_policy,
              "max_batches": max_batches,
              "users_per_batch": users_per_batch, "scenarios": {}}
    t_start = time.time()
    for sc in scenarios:
        outcomes = []
        for i in range(sc.n_releases):
            outcomes.append(run_one_release(
                metrics, baseline, sc.spec,
                seed=sc.seed_base + i, max_batches=max_batches,
                users_per_batch=users_per_batch, ramp_policy=ramp_policy,
                dashboard=dashboard))
        summary = summarise(outcomes, alpha_fwer)
        summary["spec"] = {
            "kind": sc.spec.kind, "magnitude": sc.spec.magnitude,
            "shape": sc.spec.shape, "target_dim": sc.spec.target_dim,
            "target_cat": sc.spec.target_cat}
        report["scenarios"][sc.name] = summary
        # raw outcomes for downstream plotting
        with (out_path / f"{sc.name}.jsonl").open("w", encoding="utf-8") as f:
            for o in outcomes:
                f.write(json.dumps(o.__dict__, ensure_ascii=False) + "\n")
        eng_s = summary["engine"]
        print(f"{sc.name:24s} n={sc.n_releases:4d} "
              f"rollback={eng_s['rollback_rate']:.3f} "
              f"GO={eng_s['go_rate']:.3f} "
              f"med_batch={eng_s['median_batches_to_rollback']} "
              f"med_exposed={eng_s['median_treated_requests_at_rollback']} "
              f"dash_rb={summary['dashboard']['rollback_rate']:.3f} "
              f"dashZ_rb={summary['dashboard_zscore']['rollback_rate']:.3f}"
              + (f" localise={eng_s['segment_localisation_rate']:.2f}"
                 if eng_s['segment_localisation_rate'] is not None else ""))
    report["elapsed_seconds"] = time.time() - t_start
    with (out_path / "validation_report.json").open("w",
                                                     encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
