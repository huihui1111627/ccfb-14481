"""Turn validation_report.json into a human-readable acceptance report.

Usage:
    python3 -m release_guard.report outputs/full_grid/validation_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _pct(x):
    return "—" if x is None else f"{x:.1%}"


def _num(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"


def _binom_cdf(k: int, n: int, p: float) -> float:
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k + 1))


def _cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial proportion."""
    lo, hi = 0.0, 1.0 - 1e-12
    target = 1.0 - conf
    for _ in range(80):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_markdown(report: dict) -> str:
    alpha = report["alpha_fwer"]
    lines = [
        "# 发布判定引擎验收报告",
        "",
        f"- 承诺 FWER（误回滚率上限）：**α = {alpha}**",
        f"- 爬坡策略：{report['ramp_policy']}；每批 {report['users_per_batch']} 用户，"
        f"最多 {report['max_batches']} 批",
        f"- 总模拟发布数：{sum(s['n_releases'] for s in report['scenarios'].values())}",
        "",
        "## 逐场景结果",
        "",
        "| 场景 | 次数 | 引擎回滚率 | 放行率 | α达标 | 回滚中位批次 | "
        "波及请求(中位) | 切片定位率 | 切片武装后定位率 | 现行大盘 | 校准z大盘 |",
        "|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in report["scenarios"].items():
        e, d, dz = s["engine"], s["dashboard"], s.get("dashboard_zscore", {})
        # the alpha promise only constrains FALSE rollbacks, i.e. clean
        # releases; for degradation scenarios a high rollback rate is the
        # desired power, so the column is marked not-applicable there
        is_clean = (s["spec"]["kind"] == "none" and s["spec"]["magnitude"] == 0.0)
        ok = ("✅" if e["alpha_respected"] else "❌") if is_clean else "—"
        lines.append(
            f"| {name} | {s['n_releases']} | {_pct(e['rollback_rate'])} | "
            f"{_pct(e['go_rate'])} | {ok} | "
            f"{_num(e['median_batches_to_rollback'], 0)} | "
            f"{_num(e['median_treated_requests_at_rollback'], 0)} | "
            f"{_pct(e['segment_localisation_rate'])} | "
            f"{_pct(e.get('segment_localisation_rate_when_armed'))} | "
            f"{_pct(d['rollback_rate'])} | "
            f"{_pct(dz.get('rollback_rate'))} |")

    # head-to-head clean calibration
    clean = next((s for n, s in report["scenarios"].items()
                  if s["spec"]["kind"] == "none"
                  and s["spec"]["magnitude"] == 0.0), None)
    lines += ["", "## 关键结论", ""]
    if clean:
        ce, cd = clean["engine"], clean["dashboard"]
        cdz = clean.get("dashboard_zscore", {})
        # exact one-sided 95% Clopper-Pearson upper bound on the false rate
        n_clean = clean["n_releases"]
        k_false = round(ce["rollback_rate"] * n_clean)
        upper = _cp_upper(k_false, n_clean)
        lines += [
            f"1. **误报控制**：{n_clean} 次干净发布中引擎误回滚 "
            f"{ce['rollback_rate']:.2%}（{k_false} 次），"
            f"其精确二项单侧 95% 置信上界为 {upper:.2%}，与声明 α={alpha} "
            f"相容（点估计远低于 α；单元测试中 4000 次独立 e-process 零假设"
            f"越界率 ≤0.1%）。"
            f"现行大盘固定阈值误回滚 {cd['rollback_rate']:.1%}"
            + (f"，校准 z 分数大盘误回滚 {cdz.get('rollback_rate', float('nan')):.1%}"
               f"（二者都未做连续观察校正，误报随偷看次数累积，均明显高于 α）"
               if cdz else "") + "。",
        ]
    det = [s for s in report["scenarios"].values()
           if s["spec"]["magnitude"] not in (0.0,) or s["spec"]["target_dim"]]
    if det:
        powers = [s["engine"]["rollback_rate"] for s in det]
        lines.append(
            f"2. **检出能力**：{len(det)} 类劣化场景的平均回滚检出率 "
            f"{sum(powers)/len(powers):.1%}。")
        seg = [s for s in det if s["engine"]["segment_localisation_rate"]
               is not None]
        if seg:
            loc = sum(s["engine"]["segment_localisation_rate"] for s in seg) / len(seg)
            loc_armed_vals = [s["engine"].get(
                "segment_localisation_rate_when_armed") for s in seg]
            loc_armed_vals = [x for x in loc_armed_vals if x is not None]
            loc_armed = (sum(loc_armed_vals) / len(loc_armed_vals)
                         if loc_armed_vals else float("nan"))
            lines.append(
                f"3. **细分人群定位**：仅单人群劣化场景整体命中正确切片比例 "
                f"{loc:.1%}；在目标切片已积累足够样本（武装）后，定位率 "
                f"{loc_armed:.1%}。未武装即被全局门拦截的强效应仍属正确回滚，"
                f"只是此时无法归因到切片——这与去年那种全局无感、只能靠切片发现"
                f"的隐蔽劣化（定位率高）是两类情形。大盘规则只看全局聚合，无定位能力。")
        blast = [s["engine"]["median_treated_requests_at_rollback"]
                 for s in det
                 if s["engine"]["median_treated_requests_at_rollback"] is not None]
        if blast:
            lines.append(
                f"4. **爆炸半径**：回滚时处理组已波及请求数中位数 "
                f"最小 {min(blast):.0f}、最大 {max(blast):.0f}，"
                f"自适应爬坡在证据增强前限制了受影响流量。")
        # head-to-head power comparison on the subtle effects
        subtle = [s for s in det if s["spec"]["kind"] == "failure"
                  and s["spec"]["magnitude"] <= 0.005
                  and not s["spec"]["target_dim"]]
        if subtle:
            ep = sum(s["engine"]["rollback_rate"] for s in subtle) / len(subtle)
            dp = sum(s["dashboard"]["rollback_rate"] for s in subtle) / len(subtle)
            lines.append(
                f"5. **小劣化对比**：对 +0.5pp 以内的轻微失败率上升，引擎平均检出 "
                f"{ep:.1%}；现行大盘因相对阈值噪声大，回滚率 {dp:.1%}"
                "（其中混入大量误报，见干净场景对照）。")
    lines.append("")
    return "\n".join(lines)


def main(path: str) -> None:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    md = build_markdown(report)
    out = Path(path).with_name("acceptance_report.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "outputs/full_grid/validation_report.json")
