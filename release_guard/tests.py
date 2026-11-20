"""Self-contained unit tests (no pytest dependency).

Run with:  python3 -m release_guard.tests
Covers the statistical guarantees and engine mechanics the acceptance
validation depends on.
"""
from __future__ import annotations

import math

import numpy as np

from .eprocess import (
    BernoulliSequential,
    TwoSampleBernoulliE,
    confidence_sequence,
    harm_upper_bound,
    msprt_harm,
    msprt_two_sided,
)
from .engine import DecisionEngine, _CellStats
from .generator import DIM_CATEGORIES, DegradationSpec, TrafficGenerator, build_baseline
from .metrics import default_metrics

_FAIL = 0
_PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _FAIL, _PASS
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        print(f"  FAIL: {name} {detail}")


def test_bernoulli_sequential_null() -> None:
    """Optional stopping on a fixed null must keep P(E>=1/alpha) <= alpha."""
    rng = np.random.default_rng(0)
    alpha = 0.01
    threshold = 1.0 / alpha
    breaches = 0
    trials = 4000
    for _ in range(trials):
        ep = BernoulliSequential()
        for _ in range(30):
            m = 200
            k = rng.binomial(m, 0.05)
            ep.update(k, m, 0.05)
            if ep.e_value >= threshold:
                breaches += 1
                break
    rate = breaches / trials
    check("BernoulliSequential null calibration",
          rate <= alpha + 3 * math.sqrt(alpha / trials),
          f"rate={rate:.4f} alpha={alpha}")


def test_bernoulli_sequential_ramped_null() -> None:
    """Time-varying p0 (the ramp) must not break the martingale."""
    rng = np.random.default_rng(1)
    alpha = 0.01
    breaches = 0
    trials = 4000
    ramp = np.linspace(0.01, 0.5, 30)
    for _ in range(trials):
        ep = BernoulliSequential()
        for p0 in ramp:
            m = 300
            k = rng.binomial(m, p0)
            ep.update(k, m, p0)
            if ep.e_value >= 1.0 / alpha:
                breaches += 1
                break
    rate = breaches / trials
    check("BernoulliSequential ramped-p0 calibration",
          rate <= alpha + 3 * math.sqrt(alpha / trials),
          f"rate={rate:.4f}")


def test_two_sample_bernoulli_null() -> None:
    """Exact two-sample e-process respects the alpha promise under peeking."""
    rng = np.random.default_rng(2)
    alpha = 0.01
    breaches_h = breaches_2 = 0
    trials = 4000
    for _ in range(trials):
        ep = TwoSampleBernoulliE()
        k1 = n1 = k0 = n0 = 0
        for _ in range(20):
            b = rng.binomial(1, 0.02, 100)
            a = rng.binomial(1, 0.02, 100)
            k1 += int(b.sum()); n1 += 100
            k0 += int(a.sum()); n0 += 100
            ep.update(k1, n1, k0, n0)
            if ep.e_harm >= 1.0 / alpha:
                breaches_h += 1
                break
        if ep.e_two >= 1.0 / alpha:
            breaches_2 += 1
    check("TwoSampleBernoulliE one-sided null",
          breaches_h / trials <= alpha + 3 * math.sqrt(alpha / trials),
          f"rate={breaches_h/trials:.4f}")
    check("TwoSampleBernoulliE two-sided null",
          breaches_2 / trials <= alpha + 3 * math.sqrt(alpha / trials),
          f"rate={breaches_2/trials:.4f}")


def test_two_sample_bernoulli_power() -> None:
    """A genuine 2pp failure-rate rise should be detected most of the time."""
    rng = np.random.default_rng(3)
    detected = 0
    trials = 800
    for _ in range(trials):
        ep = TwoSampleBernoulliE()
        k1 = n1 = k0 = n0 = 0
        for _ in range(30):
            b = rng.binomial(1, 0.03, 200)
            a = rng.binomial(1, 0.01, 200)
            k1 += int(b.sum()); n1 += 200
            k0 += int(a.sum()); n0 += 200
            ep.update(k1, n1, k0, n0)
            if ep.e_harm >= 100:
                detected += 1
                break
    check("TwoSampleBernoulliE power on +2pp", detected / trials > 0.8,
          f"power={detected/trials:.2f}")


def test_confidence_sequence_coverage() -> None:
    """The always-valid CS must cover zero across repeated peeks.

    Tolerance uses a 4-SE binomial band around alpha (the asymptotic mSPRT CS
    is calibrated for large n, so a single small-sample look is not exact).
    """
    rng = np.random.default_rng(4)
    alpha = 0.02
    violations = 0
    trials = 3000
    s0 = 1.0
    for _ in range(trials):
        covered = True
        for n in (200, 500, 1000, 3000, 8000):
            a = rng.normal(0, 1, n)
            b = rng.normal(0, 1, n)
            diff = b.mean() - a.mean()
            v = a.var(ddof=1) / n + b.var(ddof=1) / n
            lo, hi = confidence_sequence(diff, v, s0, alpha)
            if not (lo <= 0.0 <= hi):
                covered = False
                break
        if not covered:
            violations += 1
    rate = violations / trials
    check("confidence sequence repeated-peek coverage",
          rate <= alpha + 4 * math.sqrt(alpha / trials),
          f"violation={rate:.4f} alpha={alpha}")


def test_cs_shrinks_with_n() -> None:
    """More data must narrow the interval monotonically in expectation."""
    rng = np.random.default_rng(5)
    widths = []
    for n in (100, 1000, 10000):
        a = rng.normal(0, 1, n)
        b = rng.normal(0, 1, n)
        diff = b.mean() - a.mean()
        v = a.var(ddof=1) / n + b.var(ddof=1) / n
        lo, hi = confidence_sequence(diff, v, 1.0, 0.01)
        widths.append(hi - lo)
    check("CS width shrinks with sample size",
          widths[0] > widths[1] > widths[2], f"widths={widths}")


def test_msprt_finite_on_extremes() -> None:
    """No overflow / NaN even with degenerate tiny-variance inputs."""
    for kwargs in (
        dict(diff=0.0, var_diff=1e-12, baseline_user_var=1e-12),
        dict(diff=5.0, var_diff=1e-10, baseline_user_var=1e-8),
        dict(diff=-5.0, var_diff=1e-10, baseline_user_var=1.0),
    ):
        e1 = msprt_harm(**kwargs)
        e2 = msprt_two_sided(**kwargs)
        u = harm_upper_bound(kwargs["diff"], kwargs["var_diff"],
                             kwargs["baseline_user_var"], 0.01)
        check("mSPRT finite on extreme inputs",
              all(math.isfinite(x) for x in (e1, e2, u)),
              f"{e1},{e2},{u}")


def test_cell_stats_incremental_matches_batch() -> None:
    """Incremental Chan updates equal a direct recomputation.

    A user can appear at most once per engine batch, so each synthetic batch
    is de-duplicated first (keeping the last in-batch value), matching the
    engine's calling convention.
    """
    rng = np.random.default_rng(6)
    st = _CellStats()
    last: dict[int, float] = {}
    for _ in range(8):
        u = rng.integers(0, 200, size=60)
        v = rng.normal(0, 1, 60)
        # de-duplicate within the batch, last value wins
        keep = {}
        for ui, vi in zip(u.tolist(), v.tolist()):
            keep[int(ui)] = float(vi)
        ub = np.array(list(keep.keys()), dtype=np.int64)
        vb = np.array(list(keep.values()), dtype=float)
        st.update_batch(ub, vb)
        last.update(keep)
    uniq = np.array(sorted(last.keys()), dtype=np.int64)
    direct = np.array([last[int(u)] for u in uniq.tolist()])
    check("CellStats n matches", st.n == len(uniq), f"{st.n} vs {len(uniq)}")
    check("CellStats mean matches",
          abs(st.mean - direct.mean()) < 1e-10,
          f"{st.mean} vs {direct.mean()}")
    check("CellStats var matches",
          abs(st.var() - direct.var(ddof=1)) < 1e-8,
          f"{st.var()} vs {direct.var(ddof=1)}")


def _make_engine(max_batches=20):
    baseline = build_baseline(seed=11)
    metrics = default_metrics((baseline["p95"], baseline["p99"]))
    return DecisionEngine(metrics, DIM_CATEGORIES, baseline["stats"],
                          max_batches=max_batches,
                          min_users_per_group=50, min_users_cell=20,
                          min_users_slice_treat=30,
                          srm_min_trials=500), baseline, metrics


def test_engine_clean_does_not_rollback() -> None:
    eng, _, _ = _make_engine()
    gen = TrafficGenerator(seed=12, users_per_batch=2000)
    for b in range(1, 21):
        batch = gen.sample_users(b, 0.1, DegradationSpec(kind="none"))
        res = eng.update(batch, 0.1)
        if res.verdict == "ROLLBACK":
            break
    check("clean release never rolls back", res.verdict != "ROLLBACK",
          str(res.reasons))


def test_engine_detects_segment_regression() -> None:
    eng, _, _ = _make_engine()
    gen = TrafficGenerator(seed=13, users_per_batch=2000)
    spec = DegradationSpec(kind="failure", magnitude=0.04,
                           target_dim="taxpayer_type", target_cat="自然人")
    verdict = "CONTINUE"
    for b in range(1, 21):
        batch = gen.sample_users(b, 0.2, spec)
        res = eng.update(batch, 0.2)
        if res.verdict == "ROLLBACK":
            verdict = "ROLLBACK"
            break
    check("segment-only regression is rolled back", verdict == "ROLLBACK")


def test_srm_detects_broken_randomisation() -> None:
    """A skewed assignment (30% instead of intended 10%) must trip SRM."""
    eng, _, _ = _make_engine()
    gen = TrafficGenerator(seed=14, users_per_batch=3000)
    tripped = False
    for b in range(1, 21):
        # actual 30% treated, engine told to expect 10%
        batch = gen.sample_users(b, 0.30, DegradationSpec(kind="none"))
        res = eng.update(batch, 0.10)
        if res.srm_flag:
            tripped = True
            break
    check("SRM catches mismatched treatment fraction", tripped,
          res.srm_detail)


def test_user_level_variance_exceeds_request_level() -> None:
    """Clustered users must inflate the variance of the group mean relative
    to treating requests as independent.

    The correct comparison is of the ESTIMATED VARIANCE OF THE MEAN: the
    per-user route uses var(user means)/n_users, while the naive request route
    divides the request-level Bernoulli variance by the request count -- which
    silently assumes independence and is too small under intra-user correlation.
    """
    rng = np.random.default_rng(15)
    n_users, req_per_user = 500, 10
    risk = rng.lognormal(0, 0.8, n_users)
    user_means = np.empty(n_users)
    request_vals = []
    for i, r in enumerate(risk):
        p = min(0.5, 0.02 * r)
        draws = (rng.random(req_per_user) < p).astype(float)
        user_means[i] = draws.mean()
        request_vals.extend(draws.tolist())
    request_vals = np.array(request_vals)
    # per-user estimate of Var(overall mean): independent units are users
    user_mean_var = user_means.var(ddof=1) / n_users
    # naive request-level estimate: pretends every request is independent
    request_mean_var = request_vals.var(ddof=1) / request_vals.size
    check("clustered per-user mean-variance > naive request estimate",
          user_mean_var > request_mean_var,
          f"{user_mean_var:.6f} vs {request_mean_var:.6f}")


def test_generator_segment_targeting() -> None:
    """Only the targeted slice should carry the degradation signal."""
    gen = TrafficGenerator(seed=16, users_per_batch=3000)
    spec = DegradationSpec(kind="failure", magnitude=0.05,
                           target_dim="taxpayer_type", target_cat="自然人")
    batch = gen.sample_users(1, 0.5, spec)
    codes = batch["dims"]["taxpayer_type"]
    treated = batch["group"] == 1
    target = codes == DIM_CATEGORIES["taxpayer_type"].index("自然人")
    fail = 1 - batch["success"]
    rate_target = fail[treated & target].mean()
    rate_other = fail[treated & ~target].mean()
    check("targeted slice degrades more than others",
          rate_target > rate_other + 0.02,
          f"{rate_target:.3f} vs {rate_other:.3f}")


def run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n{_PASS} passed, {_FAIL} failed, {_PASS + _FAIL} checks")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
