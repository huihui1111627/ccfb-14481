"""The incumbent practice the engine is measured against: dashboard thresholds.

An on-call engineer watches aggregate ("大盘") metrics and rolls back only when
a raw rate crosses a fixed relative threshold -- the workflow that missed the
single-taxpayer-type slowdown because the global average barely moved.

Two weaknesses are deliberately reproduced:

* no multiplicity handling and no optional-stopping correction: the threshold
  is applied to every batch, so the look-again-every-batch rule inflates the
  false-alarm rate (calibration reports quantify it);
* only the global aggregate is watched, so slice-localised regressions are
  diluted by unaffected traffic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generator import DIM_CATEGORIES


@dataclass
class DashboardRule:
    """Fixed-threshold, aggregate-only rollback rule.

    Two flavours are supported via ``use_zscore``:

    * ``use_zscore=False`` (default) reproduces the incumbent practice: a raw
      relative threshold applied to every batch -- no optional-stopping
      correction, global aggregate only.  Its false-alarm rate under repeated
      peeking is measured by the validation harness.
    * ``use_zscore=True`` is the strongest reasonable fixed-threshold rule: a
      pooled two-proportion z-test at a Bonferroni-corrected per-batch level.
      It is still aggregate-only (blind to segment regressions) and still
      ignores optional stopping, but gives the engine a fairer benchmark.
    """
    rel_failure_threshold: float = 0.20
    rel_latency_threshold: float = 0.20
    count_threshold: float = 0.15
    min_requests: int = 1000
    use_zscore: bool = False
    zscore_level: float = 3.29      # ~0.001 two-sided nominal per look
    n_metrics: int = 4

    def verdict(self, batch: dict) -> tuple[str, str]:
        groups = batch["group"]
        t = groups == 1
        c = ~t
        if t.sum() < self.min_requests or c.sum() < self.min_requests:
            return "CONTINUE", "样本不足"

        fail_t = 1.0 - batch["success"][t].mean()
        fail_c = 1.0 - batch["success"][c].mean()
        rel_fail = (fail_t - fail_c) / max(fail_c, 1e-9)
        if self.use_zscore:
            # pooled two-proportion z on the failure rate
            k1 = int((1 - batch["success"][t]).sum())
            k0 = int((1 - batch["success"][c]).sum())
            n1, n0 = int(t.sum()), int(c.sum())
            p = (k1 + k0) / (n1 + n0)
            se = (p * (1 - p) * (1.0 / n1 + 1.0 / n0)) ** 0.5
            z = (fail_t - fail_c) / max(se, 1e-12)
            if z > self.zscore_level:
                return "ROLLBACK", f"大盘失败率 z={z:.2f} 超阈值"
        elif rel_fail > self.rel_failure_threshold:
            return "ROLLBACK", f"大盘失败率相对上升 {rel_fail:.1%}"

        lat_t = np.quantile(batch["latency_ms"][t], 0.95)
        lat_c = np.quantile(batch["latency_ms"][c], 0.95)
        rel_lat = (lat_t - lat_c) / max(lat_c, 1e-9)
        if not self.use_zscore and rel_lat > self.rel_latency_threshold:
            return "ROLLBACK", f"大盘 p95 时延相对上升 {rel_lat:.1%}"

        # requests per user, aggregate only
        ut, ct = np.unique(batch["user_id"][t], return_counts=True)
        uc, cc = np.unique(batch["user_id"][c], return_counts=True)
        rel_cnt = (ct.mean() - cc.mean()) / max(cc.mean(), 1e-9)
        if abs(rel_cnt) > self.count_threshold:
            return "ROLLBACK", f"大盘人均请求数偏移 {rel_cnt:.1%}"
        return "CONTINUE", "大盘指标正常"
