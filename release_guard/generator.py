"""Synthetic traffic generator with injectable, parameterised degradations.

The generator models the electronic-tax platform's traffic at *request* level
but keeps every user's requests correlated through latent per-user effects,
which is what makes user-level (rather than request-level) variance estimation
necessary.

Dimensions (the slices the engine cuts by):
    taxpayer_type  -- 企业 / 个体工商户 / 自然人
    region         -- 华东 / 华南 / 华北 / 西部
    terminal       -- web / app / 自助终端

Users are drawn from a persistent pool with globally unique IDs and stable
dimension profiles, so the same user can recur across batches and the
within-user correlation is genuine.

Each release scenario configures a :class:`DegradationSpec`.  Degradations may
hit the whole population or a single (dimension, category) slice -- the
"only one taxpayer type got slower" failure from last year.  Shapes include
instant step changes, gradual ramps and intermittent bursts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

TAXPAYER_TYPES = ("企业", "个体工商户", "自然人")
REGIONS = ("华东", "华南", "华北", "西部")
TERMINALS = ("web", "app", "kiosk")
DIM_CATEGORIES = {
    "taxpayer_type": list(TAXPAYER_TYPES),
    "region": list(REGIONS),
    "terminal": list(TERMINALS),
}

# base failure probability and request volume differ by taxpayer type, so
# slices are genuinely heterogeneous rather than relabelled copies
BASE_FAILURE = np.array([0.006, 0.010, 0.015])
BASE_REQ_MEAN = np.array([14.0, 8.0, 4.0])
TERMINAL_LATENCY = np.array([0.0, 0.10, 0.25])   # multiplicative latency shift


@dataclass
class DegradationSpec:
    """What the bad release does.  ``magnitude`` is in native metric units.

    kind:
        ``"none"``        clean release (false-positive calibration)
        ``"failure"``     additive failure-probability increase
        ``"latency"``     multiplicative slowdown of the latency tail
        ``"count"``      fractional change in requests per user
                          (retries inflate counts, silent failures shrink)
    shape:
        ``"step"`` | ``"ramp"`` | ``"burst"``
    target_dim / target_cat:
        when set, only users in that slice are affected (segment regression).
    """
    kind: str = "none"
    magnitude: float = 0.0
    shape: str = "step"
    target_dim: Optional[str] = None
    target_cat: Optional[str] = None
    ramp_batches: int = 30
    burst_fraction: float = 0.25
    burst_seed: int = 7
    name: str = "clean"

    def strength(self, batch_index: int) -> float:
        """Time-varying multiplier on the configured magnitude."""
        if self.kind == "none":
            return 0.0
        if self.shape == "step":
            return 1.0
        if self.shape == "ramp":
            return min(batch_index / max(self.ramp_batches, 1), 1.0)
        # burst: deterministic pseudo-pattern from the batch index so runs
        # are reproducible without consuming scenario RNG draws
        h = np.sin(batch_index * 12.9898 + self.burst_seed * 78.233)
        return 1.0 if (h - np.floor(h)) < self.burst_fraction else 0.0


class TrafficGenerator:
    """Streaming request generator for one release + its pre-release period."""

    def __init__(self,
                 seed: int = 42,
                 users_per_batch: int = 4000,
                 base_p95: float = 800.0,
                 base_p99: float = 1500.0,
                 w_type: tuple = (0.30, 0.45, 0.25),
                 w_region: tuple = (0.32, 0.28, 0.22, 0.18),
                 w_terminal: tuple = (0.45, 0.40, 0.15),
                 user_effect_sd: float = 0.35,
                 recurring_user_fraction: float = 0.3) -> None:
        self.rng = np.random.default_rng(seed)
        self.users_per_batch = users_per_batch
        self.base_p95 = base_p95
        self.base_p99 = base_p99
        self.w_type = w_type
        self.w_region = w_region
        self.w_terminal = w_terminal
        self.user_effect_sd = user_effect_sd
        self.recurring = recurring_user_fraction
        self._next_uid = 0
        self._pool: dict[int, dict] = {}

    # ------------------------------------------------------------------

    def _draw_user_profile(self, n: int) -> list[dict]:
        """Draw ``n`` independent user profiles in one vectorised call."""
        types = self.rng.choice(3, size=n, p=self.w_type)
        regions = self.rng.choice(4, size=n, p=self.w_region)
        terms = self.rng.choice(3, size=n, p=self.w_terminal)
        risks = self.rng.lognormal(0.0, self.user_effect_sd, size=n)
        return [{"type": int(t), "region": int(r), "terminal": int(m),
                 "risk": float(k)}
                for t, r, m, k in zip(types, regions, terms, risks)]

    def _new_uids(self, n: int) -> np.ndarray:
        ids = np.arange(self._next_uid, self._next_uid + n, dtype=np.int64)
        self._next_uid += n
        return ids

    def _pick_users(self) -> tuple[np.ndarray, dict[str, np.ndarray],
                                   np.ndarray]:
        """Return (uids, dim codes per dim, per-user latent risk)."""
        n = self.users_per_batch
        pool_ids = np.fromiter(self._pool.keys(), dtype=np.int64,
                               count=len(self._pool))
        n_recur = min(len(pool_ids),
                      int(round(self.recurring * n))) if len(pool_ids) else 0
        if n_recur:
            recur = self.rng.choice(pool_ids, size=n_recur, replace=False)
        else:
            recur = np.empty(0, dtype=np.int64)
        n_fresh = n - n_recur
        fresh = self._new_uids(n_fresh)
        for uid, prof in zip(fresh.tolist(),
                             self._draw_user_profile(n_fresh)):
            self._pool[int(uid)] = prof
        uids = np.concatenate([recur, fresh])
        self.rng.shuffle(uids)
        profiles = [self._pool[int(u)] for u in uids]
        dims = {
            "taxpayer_type": np.array([p["type"] for p in profiles], np.int64),
            "region": np.array([p["region"] for p in profiles], np.int64),
            "terminal": np.array([p["terminal"] for p in profiles], np.int64),
        }
        risk = np.array([p["risk"] for p in profiles], dtype=float)
        return uids, dims, risk

    # ------------------------------------------------------------------

    def sample_users(self, batch_index: int, treat_fraction: float,
                     spec: DegradationSpec) -> dict:
        """Generate one batch of request rows for both release groups."""
        rng = self.rng
        uids, dims, risk = self._pick_users()
        n = len(uids)
        type_code = dims["taxpayer_type"]

        # requests per user: over-dispersed, correlated with the latent effect
        lam = BASE_REQ_MEAN[type_code] * np.exp(
            self.user_effect_sd * rng.standard_normal(n))
        req_counts = np.maximum(1, rng.poisson(lam)).astype(np.int64)

        group = (rng.random(n) < treat_fraction).astype(np.int8)
        strength = spec.strength(batch_index)
        if spec.target_dim is not None:
            cat_idx = DIM_CATEGORIES[spec.target_dim].index(spec.target_cat)
            target = dims[spec.target_dim] == cat_idx
        else:
            target = np.ones(n, dtype=bool)
        affected_user = (group == 1) & target

        # expand to request level ------------------------------------------------
        rep_user = np.repeat(uids, req_counts)
        rep_group = np.repeat(group, req_counts)
        rep_affected = np.repeat(affected_user, req_counts)
        rep_type = np.repeat(type_code, req_counts)
        rep_term = np.repeat(dims["terminal"], req_counts)
        rep_risk = np.repeat(risk, req_counts)
        req_dims = {k: np.repeat(v, req_counts) for k, v in dims.items()}

        # --- success / failure --------------------------------------------------
        p_fail = BASE_FAILURE[rep_type] * rep_risk
        if spec.kind == "failure" and strength > 0:
            p_fail = p_fail + rep_affected * spec.magnitude * strength
        p_fail = np.clip(p_fail, 1e-6, 0.5)
        success = (rng.random(len(rep_user)) > p_fail).astype(np.int8)

        # --- latency: log-normal body with a heavy tail ------------------------
        log_mu = np.log(320.0) + 0.45 * TERMINAL_LATENCY[rep_term] \
            + 0.15 * (rep_risk - 1.0)
        latency = rng.lognormal(mean=log_mu, sigma=0.55, size=len(rep_user))
        tail = rng.random(len(rep_user)) < 0.08
        latency[tail] = rng.lognormal(
            mean=np.log(1400.0) + 0.3 * TERMINAL_LATENCY[rep_term[tail]],
            sigma=0.35, size=tail.sum())
        if spec.kind == "latency" and strength > 0:
            latency[rep_affected] *= (1.0 + spec.magnitude * strength)

        # --- count degradation: retries add rows, silent failure removes -------
        if spec.kind == "count" and strength > 0:
            rep_user, rep_group, req_dims, success, latency = self._apply_count(
                rep_user, rep_group, req_dims, success, latency,
                rep_affected, spec.magnitude * strength, rng)

        return {
            "user_id": rep_user, "group": rep_group, "dims": req_dims,
            "success": success, "latency_ms": latency,
        }

    @staticmethod
    def _apply_count(user_id, group, dims, success, latency,
                     affected, magnitude, rng) -> tuple:
        frac = abs(magnitude)
        sel = (rng.random(len(user_id)) < frac) & affected
        if magnitude > 0:                       # retries: duplicate rows
            idx = np.flatnonzero(sel)
            user_id = np.concatenate([user_id, user_id[idx]])
            group = np.concatenate([group, group[idx]])
            success = np.concatenate([success, success[idx]])
            latency = np.concatenate([latency, latency[idx]])
            for k in dims:
                dims[k] = np.concatenate([dims[k], dims[k][idx]])
        else:                                   # silent loss: drop rows
            keep = ~sel
            user_id, group, success, latency = (
                user_id[keep], group[keep], success[keep], latency[keep])
            for k in dims:
                dims[k] = dims[k][keep]
        return user_id, group, dims, success, latency


# ---------------------------------------------------------------------------
# Pre-release baseline: clean multi-batch period, per-user summaries
# ---------------------------------------------------------------------------

def build_baseline(seed: int = 1,
                   n_batches: int = 10,
                   users_per_batch: int = 4000) -> dict:
    """Generate a clean pre-release period and summarise per metric x cell.

    The baseline spans ``n_batches`` with the same recurring-user pool and
    accumulation rules as the online engine, so count-metric baselines are
    cumulative requests per observed user rather than single-window counts
    (the two distributions differ in their tails and would fake a drift).
    Returns empirical tail cut-offs and per-user (n, mean, variance) stats for
    every metric and cell.
    """
    gen = TrafficGenerator(seed=seed, users_per_batch=users_per_batch,
                           recurring_user_fraction=0.3)
    clean = DegradationSpec(kind="none")
    fail_store: dict[int, float] = {}
    req_store: dict[int, float] = {}
    window_count: dict[int, int] = {}
    batch_seen: dict[int, int] = {}
    dim_of_user: dict[int, dict[str, int]] = {}
    lat_all = []
    for b in range(n_batches):
        batch = gen.sample_users(b, 0.0, clean)
        lat_all.append(batch["latency_ms"])
        uids = batch["user_id"]
        uniq, inv = np.unique(uids, return_inverse=True)
        # first-seen dimension labels (profiles are stable per user)
        for dim in ("taxpayer_type", "region", "terminal"):
            codes = batch["dims"][dim]
            labels = np.zeros(len(uniq), dtype=np.int64)
            labels[inv] = codes
            for uid, lab in zip(uniq.tolist(), labels.tolist()):
                dim_of_user.setdefault(int(uid), {})[dim] = int(lab)
        for uid in uniq.tolist():
            batch_seen[int(uid)] = batch_seen.get(int(uid), 0) + 1
        fail = 1.0 - batch["success"].astype(float)
        for uid, xi in zip(uids.tolist(), fail.tolist()):
            fail_store[uid] = fail_store.get(uid, 0.0) + xi
            window_count[uid] = window_count.get(uid, 0) + 1
        for uid in uids.tolist():
            req_store[uid] = req_store.get(uid, 0.0) + 1.0

    lat = np.concatenate(lat_all)
    cut95 = float(np.quantile(lat, 0.95))
    cut99 = float(np.quantile(lat, 0.99))

    # second identical-seed pass for tail exceedance per-user means
    gen2 = TrafficGenerator(seed=seed, users_per_batch=users_per_batch,
                            recurring_user_fraction=0.3)
    t95_store: dict[int, float] = {}
    t99_store: dict[int, float] = {}
    for b in range(n_batches):
        batch = gen2.sample_users(b, 0.0, clean)
        uids = batch["user_id"]
        t95 = (batch["latency_ms"] > cut95).astype(float)
        t99 = (batch["latency_ms"] > cut99).astype(float)
        for uid, xi in zip(uids.tolist(), t95.tolist()):
            t95_store[uid] = t95_store.get(uid, 0.0) + xi
        for uid, xi in zip(uids.tolist(), t99.tolist()):
            t99_store[uid] = t99_store.get(uid, 0.0) + xi

    metric_users: dict[str, np.ndarray] = {}
    metric_values: dict[str, np.ndarray] = {}
    for name, store in (("success_rate", fail_store),
                        ("latency_tail_p95", t95_store),
                        ("latency_tail_p99", t99_store)):
        u = np.fromiter(store.keys(), dtype=np.int64, count=len(store))
        v = np.array([store[int(uid)] / window_count[int(uid)]
                      for uid in u.tolist()], dtype=float)
        metric_users[name] = u
        metric_values[name] = v
    cu = np.fromiter(req_store.keys(), dtype=np.int64, count=len(req_store))
    metric_users["requests_per_user"] = cu
    # requests per user per batch -- same window-free intensity as the engine
    metric_values["requests_per_user"] = np.array(
        [req_store[int(uid)] / batch_seen[int(uid)]
         for uid in cu.tolist()], dtype=float)

    cells = [("ALL", None)]
    for dim, cats in DIM_CATEGORIES.items():
        for j, cat in enumerate(cats):
            cells.append((f"{dim}:{cat}", (dim, j)))

    stats: dict[str, dict[str, tuple[int, float, float]]] = {}
    for name, vals_all in metric_values.items():
        uids_arr = metric_users[name]
        stats[name] = {}
        for cell, rule in cells:
            if rule is None:
                vals = vals_all
            else:
                dim, j = rule
                mask = np.array([dim_of_user[int(u)][dim] == j
                                 for u in uids_arr.tolist()], dtype=bool)
                vals = vals_all[mask]
            if len(vals) < 5:
                continue
            stats[name][cell] = (
                int(len(vals)), float(vals.mean()),
                float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return {"stats": stats, "p95": cut95, "p99": cut99}
