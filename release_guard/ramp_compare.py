"""Adaptive ramp vs fixed staircase head-to-head comparison.

Both policies drive the identical engine and see the identical per-release
traffic streams; only the next-fraction rule differs.  We compare time to
decision and the number of users exposed on a bad version (blast radius).

Run:  python3 -m release_guard.ramp_compare
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dashboard import DashboardRule
from .generator import DegradationSpec, build_baseline
from .metrics import default_metrics
from .simulation import run_one_release

SCENARIOS = [
    ("clean", DegradationSpec(kind="none")),
    ("fail_half_pp", DegradationSpec(kind="failure", magnitude=0.005)),
    ("fail_1pp", DegradationSpec(kind="failure", magnitude=0.01)),
    ("segment_natural",
     DegradationSpec(kind="failure", magnitude=0.03,
                     target_dim="taxpayer_type", target_cat="自然人")),
    ("latency_burst",
     DegradationSpec(kind="latency", magnitude=0.8, shape="burst")),
]


def compare(n_per: int = 120, out_dir: str = "outputs") -> dict:
    baseline = build_baseline(seed=1)
    metrics = default_metrics((baseline["p95"], baseline["p99"]))
    dash = DashboardRule()
    report = {}
    for name, spec in SCENARIOS:
        report[name] = {}
        for policy in ("adaptive", "staircase"):
            verdicts, batches, exposed = [], [], []
            for i in range(n_per):
                o = run_one_release(
                    metrics, baseline, spec, seed=12000 + i,
                    max_batches=40, users_per_batch=4000,
                    ramp_policy=policy, dashboard=dash)
                verdicts.append(o.verdict)
                batches.append(o.batches)
                exposed.append(o.treated_requests)
            report[name][policy] = {
                "rollback_rate": np.mean([v == "ROLLBACK" for v in verdicts]),
                "go_rate": np.mean([v == "GO" for v in verdicts]),
                "mean_batches": float(np.mean(batches)),
                "mean_treated_requests": float(np.mean(exposed)),
                # blast radius conditional on a bad release being rolled back
                "mean_treated_requests_if_rollback": float(np.mean(
                    [e for v, e in zip(verdicts, exposed)
                     if v == "ROLLBACK"]) or 0.0),
            }
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "ramp_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'scenario':18s} {'policy':10s} {'rollback':>9s} {'GO':>7s} "
          f"{'batches':>8s} {'exposed_req':>12s} {'exposed_if_RB':>14s}")
    for name, by_policy in report.items():
        for policy, r in by_policy.items():
            print(f"{name:18s} {policy:10s} {r['rollback_rate']:9.3f} "
                  f"{r['go_rate']:7.3f} {r['mean_batches']:8.1f} "
                  f"{r['mean_treated_requests']:12.0f} "
                  f"{r['mean_treated_requests_if_rollback']:14.0f}")
    return report


if __name__ == "__main__":
    compare()
