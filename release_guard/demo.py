"""Generate example release dashboards and a demo of the full pipeline.

Run:  python3 -m release_guard.demo
Writes outputs/dashboard_clean.html and outputs/dashboard_segment.html.
"""
from __future__ import annotations

from pathlib import Path

from .engine import DecisionEngine
from .generator import (
    DIM_CATEGORIES,
    DegradationSpec,
    TrafficGenerator,
    build_baseline,
)
from .metrics import default_metrics
from .ramp import RampController
from .viz import render_release_page

OUT = Path("outputs")


def drive(spec: DegradationSpec, seed: int, page_name: str,
          max_batches: int = 40) -> None:
    baseline = build_baseline(seed=1)
    metrics = default_metrics((baseline["p95"], baseline["p99"]))
    gen = TrafficGenerator(seed=seed, users_per_batch=4000)
    eng = DecisionEngine(
        metrics, DIM_CATEGORIES, baseline["stats"], max_batches=max_batches,
        min_users_per_group=150, min_users_cell=40,
        min_users_slice_treat=300)
    ramp = RampController(users_per_batch=4000)
    frac = 0.01
    for b in range(1, max_batches + 1):
        batch = gen.sample_users(b, frac, spec)
        res = eng.update(batch, frac)
        eng.record_snapshot(frac)
        frac = ramp.next_fraction(eng, res, frac).next_fraction
        if res.verdict in ("GO", "ROLLBACK"):
            break
    OUT.mkdir(exist_ok=True)
    path = render_release_page(eng, OUT / page_name,
                               title=f"发布轨迹 · {spec.name or spec.kind}")
    print(f"{page_name}: {res.verdict} at batch {res.batch_index}, "
          f"final fraction={frac:.3f} -> {path}")
    for reason in res.reasons:
        print("   ", reason)


def main() -> None:
    drive(DegradationSpec(kind="none", name="干净版本"),
          seed=2000, page_name="dashboard_clean.html")
    drive(DegradationSpec(kind="failure", magnitude=0.03,
                          target_dim="taxpayer_type", target_cat="自然人",
                          name="仅自然人失败率+3pp"),
          seed=6001, page_name="dashboard_segment.html")


if __name__ == "__main__":
    main()
