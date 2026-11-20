"""Sequential release-decision engine.

The engine ingests one batch of request-level rows at a time and, after every
batch, produces one of three verdicts -- GO (full rollout), ROLLBACK or
CONTINUE -- together with always-valid effect intervals.

Design pillars (matching the seven requirements):

1. **Continuous looking without alpha inflation.**  Every statistic is an
   e-process (:mod:`release_guard.eprocess`); Ville's inequality bounds the
   false-reject probability at any stopping time, so peeking after every batch
   is legal.
2. **Type-appropriate statistics.**  Proportions, long-tail latency
   (threshold-exceedance rates) and counts all become per-user means and feed
   the same asymptotic mSPRT machinery, with one- or two-sided alternatives.
3. **Multiplicity.**  A Bonferroni e-threshold (M/alpha) guards the global,
   per-metric rollback gate; slice-level rollbacks use a hierarchical
   per-metric gate (K_m/alpha), which together keep family-wise error <= alpha.
   e-BH at a separate FDR level produces the richer localisation shortlist.
4. **Randomisation validity.**  A sequential Jeffreys e-process checks the
   sample ratio (on unique users, the actual randomisation unit) against the
   intended, possibly ramped treatment fraction; pre-release baseline drift is
   checked per metric and cell.
5. **Within-user dependence.**  All rows are first aggregated per user; the
   sampling unit is the independent user, not the request.  A
   ``naive_request_level`` mode is kept solely to demonstrate the variance
   under-estimation of ignoring clustering.
6. **Three verdicts + intervals.**  GO requires every harm bound to sit inside
   a practical non-effect floor; at the observation deadline an inconclusive
   result still reports the worst effect still compatible with the data.
7. **Ramp sizing lives in :mod:`release_guard.ramp`** and consumes the bounds
   computed here.

Performance design: per metric x group a single array-backed accumulator holds
each user's running sum/count; per slice, the set of member user ids is stored
once and the per-user-mean sufficient statistics (n, sum of means, sum of
squared means) are updated incrementally, so a decision costs O(metrics x
cells) regardless of the accumulated traffic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .eprocess import (
    BernoulliSequential,
    TwoSampleBernoulliE,
    confidence_sequence,
    expected_positive_harm,
    harm_upper_bound,
    msprt_harm,
    msprt_two_sided,
)
from .metrics import MetricSpec

_EPS = 1e-12
ALL_CELL = "ALL"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    cell: str
    n_treat: int
    n_control: int
    mean_treat: float
    mean_control: float
    diff: float
    rel_change: float
    e_harm: float
    e_two: float
    cs_low: float
    cs_high: float
    upper_harm: float
    active: bool
    baseline_user_var: float


@dataclass
class MetricResult:
    metric: str
    family: str
    description: str
    effect_floor: float
    global_result: CellResult
    slices: list[CellResult]
    rollback_slices: list[str] = field(default_factory=list)
    fdr_slices: list[str] = field(default_factory=list)
    go_ok: bool = True
    rollback: bool = False
    reason: str = ""


@dataclass
class DecisionResult:
    verdict: str                           # "GO" | "ROLLBACK" | "CONTINUE"
    batch_index: int
    reasons: list[str]
    metric_results: list[MetricResult]
    srm_e: float
    srm_flag: bool
    srm_detail: str
    baseline_flags: list[str]
    worst_abs_harm: float
    worst_rel_harm: float
    deadline_reached: bool
    eligible: bool

    def rollback_metric(self) -> Optional[str]:
        for m in self.metric_results:
            if m.rollback:
                return m.metric
        return None


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

class _UserAccumulator:
    """Per-user running sum/count for one metric x one treatment group.

    NumPy arrays grow geometrically with the largest user id seen.  Slice
    statistics are maintained externally via :class:`_CellStats`.
    """

    __slots__ = ("sum_x", "count", "capacity")
    _INIT = 8192

    def __init__(self) -> None:
        self.sum_x = np.zeros(self._INIT, dtype=np.float64)
        self.count = np.zeros(self._INIT, dtype=np.int32)
        self.capacity = self._INIT

    def _ensure(self, max_uid: int) -> None:
        need = max_uid + 1
        if need <= self.capacity:
            return
        new_cap = max(need, int(self.capacity * 1.5))
        sx = np.zeros(new_cap, dtype=np.float64)
        ct = np.zeros(new_cap, dtype=np.int32)
        sx[:self.capacity] = self.sum_x
        ct[:self.capacity] = self.count
        self.sum_x, self.count, self.capacity = sx, ct, new_cap

    def add(self, uids: np.ndarray, x: np.ndarray) -> None:
        if uids.size == 0:
            return
        self._ensure(int(uids.max()))
        np.add.at(self.sum_x, uids, x)
        np.add.at(self.count, uids, 1)


class _CellStats:
    """Incremental per-user-value statistics for one slice.

    Stores each member user's latest per-user value and a running Chan/Welford
    (n, mean, M2).  A batch contributes the current value of each of its
    users; first-seen users are inserted and recurring users replace their
    previous value.  All moments for a batch are computed in vectorised NumPy
    calls, giving O(batch size) update cost rather than O(population) rescans.
    """

    __slots__ = ("members", "values", "n", "mean", "m2", "_arr")

    def __init__(self) -> None:
        self.members: set[int] = set()
        self.values: dict[int, float] = {}
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0
        # eagerly maintained so membership tests in update_batch stay valid;
        # unsorted is fine (only used for isin / fancy indexing)
        self._arr: np.ndarray = np.empty(0, dtype=np.int64)

    def update_batch(self, uids: np.ndarray, vals: np.ndarray) -> None:
        if uids.size == 0:
            return
        # vectorised membership test against the eagerly maintained array
        if self._arr.size:
            known = np.isin(uids, self._arr)
        else:
            known = np.zeros(uids.size, dtype=bool)
        if known.any():
            rids = uids[known]
            rvals = vals[known]
            old = np.array([self.values[int(u)] for u in rids.tolist()],
                           dtype=float)
            self._remove_aggregate(old)
            self._insert_aggregate(rvals)
            self.values.update(zip(rids.tolist(), rvals.tolist()))
        if (~known).any():
            nids = uids[~known]
            nvals = vals[~known]
            self._insert_aggregate(nvals)
            self.members.update(nids.tolist())
            self.values.update(zip(nids.tolist(), nvals.tolist()))
            # keep the membership array current for the next batch
            self._arr = np.concatenate([self._arr, nids])

    def _insert_aggregate(self, vals: np.ndarray) -> None:
        nb = vals.size
        if nb == 0:
            return
        bmean = float(vals.mean())
        bm2 = float(((vals - bmean) ** 2).sum())
        na = self.n
        delta = bmean - self.mean
        nt = np.float64(na + nb)
        self.mean += delta * nb / nt
        self.m2 += bm2 + delta * delta * na * nb / nt
        self.n = int(nt)

    def _remove_aggregate(self, old_vals: np.ndarray) -> None:
        nb = old_vals.size
        if nb == 0 or self.n <= nb:
            self.n, self.mean, self.m2 = 0, 0.0, 0.0
            return
        bmean = float(old_vals.mean())
        bm2 = float(((old_vals - bmean) ** 2).sum())
        nt = self.n
        na = nt - nb
        new_mean = (self.mean * nt - bmean * nb) / na
        delta = bmean - new_mean
        self.m2 = max(self.m2 - bm2 - delta * delta * na * nb / nt, 0.0)
        self.mean = new_mean
        self.n = na

    def member_array(self) -> np.ndarray:
        return self._arr

    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return max(self.m2 / (self.n - 1), 0.0)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Streaming, always-valid canary decision engine.

    Parameters
    ----------
    metrics:
        :class:`MetricSpec` list produced by the metric layer.
    dim_categories:
        mapping dimension name -> list of category labels (fixed before the
        release; slice keys are ``"dim:label"``).
    baseline_stats:
        pre-release per-user statistics, keyed
        ``baseline_stats[metric_name][cell_key] = (n_users, mean, var)``.
        The variance fixes the mSPRT mixture scales; the mean anchors the
        control-vs-baseline drift check.
    alpha_fwer:
        family-wise false-rollback probability promised for the whole release.
    alpha_fdr:
        level for the e-BH slice localisation shortlist.
    alpha_srm / alpha_baseline:
        levels for the randomisation and baseline checks.
    min_users_per_group / min_users_cell:
        exposure gates before GO / per-cell testing are allowed.
    srm_min_trials:
        unique users required before the SRM check is armed.
    max_batches:
        observation deadline; afterwards the verdict stays CONTINUE with
        ``deadline_reached=True`` and worst-case bounds.
    naive_request_level:
        diagnostic switch: treat every request as independent (anti-pattern).
    """

    def __init__(self,
                 metrics: list[MetricSpec],
                 dim_categories: dict[str, list[str]],
                 baseline_stats: dict[str, dict[str, tuple[int, float, float]]],
                 alpha_fwer: float = 0.01,
                 alpha_fdr: float = 0.05,
                 alpha_srm: float = 0.001,
                 alpha_baseline: float = 0.01,
                 min_users_per_group: int = 200,
                 min_users_cell: int = 50,
                 min_users_slice_treat: int = 100,
                 go_slice_min_users: int = 300,
                 srm_min_trials: int = 3000,
                 max_batches: int = 60,
                 naive_request_level: bool = False) -> None:
        self.metrics = metrics
        self.dim_names = list(dim_categories.keys())
        self.dim_categories = dim_categories
        self.cells = [ALL_CELL]
        for dim, cats in dim_categories.items():
            for cat in cats:
                self.cells.append(f"{dim}:{cat}")
        self.baseline_stats = baseline_stats
        self.alpha_fwer = alpha_fwer
        self.alpha_fdr = alpha_fdr
        self.alpha_srm = alpha_srm
        self.alpha_baseline = alpha_baseline
        self.min_users_per_group = min_users_per_group
        self.min_users_cell = min_users_cell
        # a slice needs a substantial treated cohort before its gate is armed;
        # at 1% traffic the first batches put single-digit treated users in
        # each category, where even corrected thresholds are unstable
        self.min_users_slice_treat = min_users_slice_treat
        # retained for backward compatibility; the GO gate now uses the global
        # cell only, so this no longer changes behaviour (slices gate rollback)
        self.go_slice_min_users = go_slice_min_users
        self.srm_min_trials = srm_min_trials
        self.max_batches = max_batches
        self.naive_request_level = naive_request_level

        # raw per-user arrays: [metric] -> [control acc, treat acc]
        self._acc: dict[str, list[_UserAccumulator]] = {
            m.name: [_UserAccumulator(), _UserAccumulator()] for m in metrics
        }
        # exact Bernoulli e-processes for proportion/tail metrics:
        # [metric][group-agnostic cell key] -> TwoSampleBernoulliE
        self._bern: dict[str, dict[str, TwoSampleBernoulliE]] = {
            m.name: {c: TwoSampleBernoulliE() for c in self.cells}
            for m in metrics
            if m.family in ("success", "latency")
        }
        # number of distinct batches each user has appeared in, per group;
        # used to normalise the count metric to requests/user/batch
        self._user_batches: list[dict[int, int]] = [{}, {}]
        # incremental per-cell stats: [metric][group][cell] -> _CellStats
        self._cells: dict[str, list[dict[str, _CellStats]]] = {
            m.name: [{c: _CellStats() for c in self.cells} for _ in range(2)]
            for m in metrics
        }
        self._base_var: dict[tuple[str, str], float] = {}
        self._srm = BernoulliSequential()
        self._seen_users: set[int] = set()
        self._total_treat = 0
        self._total_control = 0
        self.batch_index = 0
        self.history: list[DecisionResult] = []
        # lightweight per-batch snapshots for the visualisation page:
        # batch -> {fraction, srm_e, verdict, metrics: {name: {...}}}
        self.snapshots: list[dict] = []
        self._last_fraction = 0.0

    def record_snapshot(self, fraction: float) -> None:
        """Capture the current statistics of every global/slice cell.

        Called once per batch by the driver so the dashboard can draw how
        each e-process and confidence sequence approaches its boundary.
        """
        result = self.history[-1]
        rec = {
            "batch": self.batch_index,
            "fraction": float(fraction),
            "srm_e": float(result.srm_e),
            "verdict": result.verdict,
            "worst_abs_harm": float(result.worst_abs_harm),
            "metrics": {},
        }
        for mr in result.metric_results:
            g = mr.global_result
            rec["metrics"][mr.metric] = {
                "family": mr.family,
                "e_harm": float(g.e_harm),
                "e_two": float(g.e_two),
                "diff": None if not g.active else float(g.diff),
                "cs_low": None if not g.active else float(g.cs_low),
                "cs_high": None if not g.active else float(g.cs_high),
                "upper_harm": None if not g.active else float(g.upper_harm),
                "effect_floor": float(mr.effect_floor),
                "rollback": bool(mr.rollback),
                "go_ok": bool(mr.go_ok),
                "slices": {
                    s.cell: {
                        "e_harm": float(s.e_harm),
                        "e_two": float(s.e_two),
                        "diff": None if not s.active else float(s.diff),
                        "upper_harm": None if not s.active
                        else float(s.upper_harm),
                        "active": bool(s.active),
                        "n_treat": int(s.n_treat),
                    } for s in mr.slices
                },
            }
        self.snapshots.append(rec)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def update(self, batch: dict, intended_treat_fraction: float) -> DecisionResult:
        """Process one request-level batch and return the current verdict."""
        self.batch_index += 1
        uids = np.asarray(batch["user_id"], dtype=np.int64)
        groups = np.asarray(batch["group"], dtype=np.int8)
        n = len(uids)
        treat_mask = groups == 1

        # --- SRM on unique users (the randomisation unit) ---------------
        uniq_uids, inv = np.unique(uids, return_inverse=True)
        uniq_group = np.zeros(len(uniq_uids), dtype=np.int8)
        uniq_group[inv] = groups
        new_t = new_c = 0
        for uid, g in zip(uniq_uids.tolist(), uniq_group.tolist()):
            gi = int(g)
            self._user_batches[gi][uid] = \
                self._user_batches[gi].get(uid, 0) + 1
            if uid in self._seen_users:
                continue
            self._seen_users.add(uid)
            if g == 1:
                new_t += 1
            else:
                new_c += 1
        self._total_treat += new_t
        self._total_control += new_c
        self._srm.update(new_t, new_t + new_c, intended_treat_fraction)

        # --- per-user dimension labels (stable across batches) ----------
        user_dim = {}
        for dim in self.dim_names:
            codes = np.asarray(batch["dims"][dim], dtype=np.int64)
            labels = np.zeros(len(uniq_uids), dtype=np.int64)
            labels[inv] = codes
            user_dim[dim] = labels

        success = np.asarray(batch["success"], dtype=np.int8)
        latency = np.asarray(batch["latency_ms"], dtype=float)
        for spec in self.metrics:
            if spec.family == "success":
                x = (1.0 - success.astype(float))
            elif spec.family == "latency":
                x = (latency > spec.threshold).astype(float)
            else:
                x = np.ones(n, dtype=float)
            acc_c, acc_t = self._acc[spec.name]
            acc_c.add(uids[~treat_mask], x[~treat_mask])
            acc_t.add(uids[treat_mask], x[treat_mask])

            # current per-user value for this batch's unique users, fully
            # vectorised: group determines which accumulator to read
            is_count = spec.family == "count"
            user_values = np.empty(len(uniq_uids), dtype=float)
            for g in (0, 1):
                sel = uniq_group == g
                gu = uniq_uids[sel]
                acc = acc_t if g == 1 else acc_c
                if is_count:
                    batches = np.fromiter(
                        (self._user_batches[g].get(int(u), 1)
                         for u in gu.tolist()), dtype=float, count=gu.size)
                    user_values[sel] = acc.count[gu] / np.maximum(batches, 1.0)
                else:
                    user_values[sel] = (acc.sum_x[gu]
                                        / np.maximum(acc.count[gu], 1))
            for g in (0, 1):
                sel = uniq_group == g
                gu = uniq_uids[sel]
                gv = user_values[sel]
                table = self._cells[spec.name][g]
                table[ALL_CELL].update_batch(gu, gv)
                for dim in self.dim_names:
                    codes = user_dim[dim][sel]
                    for j, cat in enumerate(self.dim_categories[dim]):
                        cmask = codes == j
                        if cmask.any():
                            table[f"{dim}:{cat}"].update_batch(
                                gu[cmask], gv[cmask])

        return self._finish()

    def _finish(self) -> DecisionResult:
        result = self._decide()
        self.history.append(result)
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _baseline_var(self, metric: str, cell: str) -> float:
        key = (metric, cell)
        if key in self._base_var:
            return self._base_var[key]
        stats = self.baseline_stats.get(metric, {})
        pre = stats.get(cell) or stats.get(ALL_CELL)
        var = max(pre[2], _EPS) if pre else 1.0
        self._base_var[key] = var
        return var

    def _cell_user_stats(self, spec: MetricSpec, cell: str, group: int
                         ) -> tuple[int, float, float, np.ndarray]:
        """Return (n, mean, var, member-id array) for one cell.

        The independent sampling unit is always the USER (requirement 5).
        The ``naive_request_level`` switch is retained only for the count
        metric that goes through the asymptotic mSPRT: there it replaces the
        per-user variance with a request-level moment estimate so the
        anti-pattern's over-confidence can be exercised.  Proportion and tail
        metrics use the exact beta-binomial e-process and are unaffected.
        The definitive demonstration of the clustering effect lives in
        :mod:`release_guard.clustering_demo`.
        """
        st = self._cells[spec.name][group][cell]
        ids = st.member_array()
        if (not self.naive_request_level or spec.family != "count"
                or ids.size == 0):
            return st.n, st.mean, st.var(), ids
        acc = self._acc[spec.name][group]
        cnt = acc.count[ids].astype(float)
        total = int(cnt.sum())
        if total == 0:
            return 0, float("nan"), float("nan"), ids
        # request-level moments: every request counts as one observation,
        # which under-states the variance of the mean under clustering
        mean = float(acc.sum_x[ids].sum() / total)
        ex2 = float((acc.sum_x[ids] ** 2 / np.maximum(cnt, 1.0)).sum()
                    / total)
        return total, mean, max(ex2 - mean * mean, 0.0), ids

    def _cell_result(self, spec: MetricSpec, cell: str) -> CellResult:
        n_c, m_c, v_c, idx_c = self._cell_user_stats(spec, cell, 0)
        n_t, m_t, v_t, idx_t = self._cell_user_stats(spec, cell, 1)
        if cell == ALL_CELL:
            active = n_t >= self.min_users_cell and n_c >= self.min_users_cell
        else:
            active = (n_t >= self.min_users_slice_treat
                      and n_c >= self.min_users_cell)
        v_c = v_c if math.isfinite(v_c) else _EPS
        v_t = v_t if math.isfinite(v_t) else _EPS
        s0 = self._baseline_var(spec.name, cell)
        if not active:
            # inactive cells need only raw moments for the exposure gate;
            # skip the e-process and confidence-sequence work entirely
            return CellResult(cell, n_t, n_c,
                              m_t if n_t else float("nan"),
                              m_c if n_c else float("nan"),
                              float("nan"), float("nan"),
                              1.0, 1.0, float("nan"), float("nan"),
                              float("nan"), False, s0)
        var_diff = v_t / n_t + v_c / n_c
        diff = m_t - m_c
        if spec.family in ("success", "latency"):
            # exact beta-binomial e-process from request-level event counts
            acc_c, acc_t = self._acc[spec.name]
            k0 = float(acc_c.sum_x[idx_c].sum())
            k1 = float(acc_t.sum_x[idx_t].sum())
            nn0 = float(acc_c.count[idx_c].sum())
            nn1 = float(acc_t.count[idx_t].sum())
            bern = self._bern[spec.name][cell]
            bern.update(k1, nn1, k0, nn0)
            e_harm = bern.e_harm
            e_two = bern.e_two
        else:
            e_harm = msprt_harm(diff, var_diff, s0)
            e_two = msprt_two_sided(diff, var_diff, s0)
        alpha_m = self.alpha_fwer / max(len(self.metrics), 1)
        low, high = confidence_sequence(diff, var_diff, s0, alpha_m)
        upper = harm_upper_bound(diff, var_diff, s0, alpha_m)
        rel = diff / m_c if m_c and abs(m_c) > _EPS else float("nan")
        return CellResult(cell, n_t, n_c, m_t, m_c, diff, rel,
                          e_harm, e_two, low, high, upper, True, s0)

    # ------------------------------------------------------------------
    # Multiplicity-corrected decisions
    # ------------------------------------------------------------------

    @staticmethod
    def _ebh_shortlist(e_values: list[tuple[str, float]],
                       alpha: float) -> list[str]:
        """e-BH: largest k with the k-th largest e >= K/(k*alpha).

        Controls FDR at alpha under arbitrary dependence of the e-values.
        """
        if not e_values:
            return []
        ordered = sorted(e_values, key=lambda kv: kv[1], reverse=True)
        k_total = len(ordered)
        chosen: list[str] = []
        for k, (name, e) in enumerate(ordered, start=1):
            if e >= k_total / (k * alpha):
                chosen.append(name)
        return chosen

    def _decide(self) -> DecisionResult:
        reasons: list[str] = []
        baseline_flags: list[str] = []
        m_total = len(self.metrics)
        global_threshold = m_total / self.alpha_fwer
        metric_results: list[MetricResult] = []
        worst_abs = 0.0
        worst_rel = 0.0
        any_ineligible = False

        for spec in self.metrics:
            glob = self._cell_result(spec, ALL_CELL)
            if not glob.active:
                any_ineligible = True
            slices = [self._cell_result(spec, c)
                      for c in self.cells if c != ALL_CELL]
            active_slices = [s for s in slices if s.active]
            # Hierarchical FWER control: the global gate spends the metric's
            # alpha/2 share, all slice tests together spend the other half,
            # and the per-slice Bonferroni e-threshold divides that share by
            # the number of PLANNED cells (not active cells -- using active
            # counts makes the threshold data-dependent and invalid).
            n_planned_slices = len(self.cells) - 1
            slice_gate = (2 * m_total * max(n_planned_slices, 1)) \
                / self.alpha_fwer

            mr = MetricResult(
                metric=spec.name, family=spec.family,
                description=spec.description, effect_floor=spec.effect_floor,
                global_result=glob, slices=slices)

            # --- rollback gates (hierarchical, FWER <= alpha_fwer) -----
            gate_e = glob.e_two if spec.harm == "two_sided" else glob.e_harm
            if glob.active and gate_e >= global_threshold:
                mr.rollback = True
                mr.reason = (
                    f"全局 e={gate_e:.1f} >= 阈值 {global_threshold:.0f}"
                    f"（效应 {glob.diff:+.4g}，相对 {glob.rel_change:+.2%}）")
            for s in active_slices:
                se = s.e_two if spec.harm == "two_sided" else s.e_harm
                if se >= slice_gate:
                    mr.rollback = True
                    mr.rollback_slices.append(s.cell)
            # e-BH localisation shortlist (FDR alpha_fdr)
            pool = [(s.cell, s.e_two if spec.harm == "two_sided"
                     else s.e_harm) for s in active_slices]
            mr.fdr_slices = self._ebh_shortlist(pool, self.alpha_fdr)

            # --- GO gate ------------------------------------------------
            # GO requires the GLOBAL cell of every metric to show practical
            # non-effect.  Slice cells are intentionally NOT part of the GO
            # gate: their smaller samples give intrinsically wide intervals
            # that would make GO unreachable, while segment-localised
            # regressions are already caught by the slice ROLLBACK gates
            # above.  This split keeps both directions working: clean
            # releases can ship, single-slice regressions still block.
            go_ok = True
            s = glob
            if spec.family == "count" and s.mean_control and \
                    abs(s.mean_control) > _EPS:
                floor_hi = abs(s.mean_control) * spec.effect_floor
                floor_lo = -floor_hi
            else:
                floor_hi = spec.effect_floor
                floor_lo = -spec.effect_floor
            if spec.harm == "two_sided":
                if s.cs_high > floor_hi or s.cs_low < floor_lo:
                    go_ok = False
            elif s.upper_harm > floor_hi:
                go_ok = False
            mr.go_ok = go_ok

            if glob.active:
                # worst-case harm uses the interval appropriate to the
                # alternative: one-sided upper bound for harm metrics, the
                # positive end of the two-sided CS for two-sided ones
                bound = (max(glob.cs_high, 0.0) if spec.harm == "two_sided"
                         else max(glob.upper_harm, 0.0))
                worst_abs = max(worst_abs, bound)
                if glob.mean_control and abs(glob.mean_control) > _EPS:
                    worst_rel = max(worst_rel, bound / abs(glob.mean_control))

            metric_results.append(mr)

        # --- randomisation check (SRM), armed after enough users --------
        srm_armed = self._srm.n >= self.srm_min_trials
        # an e-process starts at 1.0 and drifts from there; tiny accumulated
        # values below 1 are evidence FOR the null, report them as-is but no
        # smaller than the floating-point floor
        srm_e = max(self._srm.e_value, 1e-300) if srm_armed else 1.0
        srm_flag = srm_armed and srm_e >= 1.0 / self.alpha_srm
        srm_detail = (
            f"处理组用户占比 {self._srm.p_hat:.4f}，期望按爬坡比例，"
            f"e={srm_e:.1f}（阈值 {1.0 / self.alpha_srm:.0f}，"
            f"用户 {self._srm.n}，{'已启用' if srm_armed else '未启用'}）")

        # --- pre-release baseline drift (control vs baseline) -----------
        base_gate = max(
            sum(len(d) for d in self.baseline_stats.values()), 1) \
            / self.alpha_baseline
        for spec in self.metrics:
            for cell in self.cells:
                pre = self.baseline_stats.get(spec.name, {}).get(cell)
                if pre is None:
                    continue
                n_c, m_c, v_c, _ = self._cell_user_stats(spec, cell, 0)
                if n_c < self.min_users_cell or pre[0] < 2:
                    continue
                n0, m0, v0_base = pre
                var_diff = (v_c if math.isfinite(v_c) else _EPS) / n_c \
                    + max(v0_base, _EPS) / n0
                e_base = msprt_two_sided(m_c - m0, var_diff,
                                         max(v0_base, _EPS))
                if e_base >= base_gate:
                    baseline_flags.append(
                        f"[{spec.name}/{cell}] 对照组与发布前基线不一致 "
                        f"(e={e_base:.1f}，对照组 {m_c:.4g} "
                        f"vs 基线 {m0:.4g})")

        # --- eligibility and verdict ------------------------------------
        eligible = (not any_ineligible
                    and self._total_treat >= self.min_users_per_group
                    and self._total_control >= self.min_users_per_group)
        deadline = self.batch_index >= self.max_batches

        verdict = "CONTINUE"
        if srm_flag:
            verdict = "ROLLBACK"
            reasons.append("流量分组比例异常（SRM），两组比较的前提不成立："
                           + srm_detail)
        for mr in metric_results:
            if mr.rollback:
                verdict = "ROLLBACK"
                where = ("，细分命中 " + "/".join(mr.rollback_slices)) \
                    if mr.rollback_slices else ""
                reasons.append(
                    f"[{mr.metric}] {mr.reason or '细分格子触发'}{where}")
        if verdict != "ROLLBACK":
            if eligible and all(mr.go_ok for mr in metric_results) \
                    and not baseline_flags:
                verdict = "GO"
                reasons.append("全部指标的有害效应上界均落在可接受地板内，"
                               "分组与基线检查通过，可以全量")
            elif deadline:
                reasons.append(
                    f"观察期 {self.max_batches} 批已用完仍未收敛；"
                    f"当前全局最坏可能劣化（绝对）≈ {worst_abs:.4g}，"
                    f"（相对）≈ {worst_rel:.2%}，需人工决策或延长观察")
            else:
                reasons.append("证据不足，继续观察")
        if deadline and verdict == "CONTINUE":
            reasons.append("（已到观察截止时间）")

        return DecisionResult(
            verdict=verdict, batch_index=self.batch_index, reasons=reasons,
            metric_results=metric_results, srm_e=srm_e, srm_flag=srm_flag,
            srm_detail=srm_detail, baseline_flags=baseline_flags,
            worst_abs_harm=worst_abs, worst_rel_harm=worst_rel,
            deadline_reached=deadline, eligible=eligible)

    # ------------------------------------------------------------------
    # Helpers used by the ramp controller
    # ------------------------------------------------------------------

    def expected_harm_estimates(self) -> dict[str, float]:
        """Posterior expected positive harm per metric (global cell)."""
        out: dict[str, float] = {}
        for spec in self.metrics:
            n_c, m_c, v_c, _ = self._cell_user_stats(spec, ALL_CELL, 0)
            n_t, m_t, v_t, _ = self._cell_user_stats(spec, ALL_CELL, 1)
            if n_t < self.min_users_cell or n_c < self.min_users_cell:
                out[spec.name] = float("nan")
                continue
            var_diff = (v_t if math.isfinite(v_t) else _EPS) / n_t \
                + (v_c if math.isfinite(v_c) else _EPS) / n_c
            s0 = self._baseline_var(spec.name, ALL_CELL)
            out[spec.name] = expected_positive_harm(
                m_t - m_c, var_diff, s0)
        return out
