"""Self-contained release visualisation (no matplotlib).

Given an engine that has run a release, :func:`render_release_page` emits a
single HTML file with inline SVG charts showing, for every metric,

* how the harm / two-sided e-process climbs toward the rollback boundary
  (the multiplicity-corrected e-threshold is drawn as a horizontal line);
* how the always-valid confidence sequence narrows batch by batch and how its
  upper harm bound approaches (or stays inside) the practical effect floor;
* the adaptive traffic-ramp fraction and the SRM e-process on a shared time
  axis;
* a slice heat-map of per-cell e-values, so a segment-localised regression is
  visible as one bright row while the global aggregate stays quiet -- exactly
  the failure mode the old aggregate dashboard missed.

Everything is string-built, so the output opens in any browser with no JS
library or network dependency.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

from .engine import DecisionEngine

_W, _H = 720, 240
_MARGIN = (56, 16, 30, 44)     # left, right, top, bottom


def _scale(values, lo=None, hi=None):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo = min(finite) if lo is None else lo
    hi = max(finite) if hi is None else hi
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def _axes(x_lo, x_hi, y_lo, y_hi):
    l, r, t, b = _MARGIN
    iw, ih = _W - l - r, _H - t - b

    def X(x):
        return l + (x - x_lo) / max(x_hi - x_lo, 1e-9) * iw

    def Y(y):
        return t + ih - (y - y_lo) / max(y_hi - y_lo, 1e-9) * ih
    return X, Y, l, r, t, b, iw, ih


def _line_series(X, Y, points, color, width=2.0, dash=None):
    if not points:
        return ""
    d = "M " + " L ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{extra}/>')


def _hline(X, Y, yval, x_lo, x_hi, color, label=None, dash="6,4"):
    out = (f'<line x1="{X(x_lo):.1f}" y1="{Y(yval):.1f}" '
           f'x2="{X(x_hi):.1f}" y2="{Y(yval):.1f}" '
           f'stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
    if label:
        out += (f'<text x="{X(x_hi):.1f}" y="{Y(yval)-4:.1f}" '
                f'text-anchor="end" font-size="11" fill="{color}">'
                f'{html.escape(label)}</text>')
    return out


def _frame(title, x_label, y_label, x_lo, x_hi, y_lo, y_hi):
    X, Y, l, r, t, b, iw, ih = _axes(x_lo, x_hi, y_lo, y_hi)
    grid = []
    for i in range(5):
        gy = t + ih - i * ih / 4
        yv = y_lo + (y_hi - y_lo) * i / 4
        grid.append(f'<line x1="{l}" y1="{gy:.1f}" x2="{l+iw:.1f}" y2="{gy:.1f}" '
                    f'stroke="#eee" stroke-width="1"/>')
        grid.append(f'<text x="{l-6}" y="{gy+4:.1f}" text-anchor="end" '
                    f'font-size="10" fill="#888">{yv:g}</text>')
    head = (f'<text x="{l}" y="{t-10}" font-size="13" font-weight="600" '
            f'fill="#222">{html.escape(title)}</text>')
    xlab = (f'<text x="{l+iw/2}" y="{t+ih+28}" text-anchor="middle" '
            f'font-size="11" fill="#666">{html.escape(x_label)}</text>')
    ylab = (f'<text x="14" y="{t+ih/2}" font-size="11" fill="#666" '
            f'transform="rotate(-90 14,{t+ih/2})" text-anchor="middle">'
            f'{html.escape(y_label)}</text>')
    return X, Y, "".join(grid) + head + xlab + ylab


def _e_process_chart(snapshots, metric_name, info, gate_threshold):
    batches = [s["batch"] for s in snapshots]
    x_lo, x_hi = 1, max(batches)
    eh = [(s["batch"], s["metrics"][metric_name]["e_harm"]) for s in snapshots]
    e2 = [(s["batch"], s["metrics"][metric_name]["e_two"]) for s in snapshots]
    y_hi = max(gate_threshold * 1.1,
               max(y for _, y in eh + e2) * 1.1, 2.0)
    X, Y, frame = _frame(
        f"{metric_name} — e-process vs 回滚边界",
        "批次", "e 值（对数刻度）", x_lo, x_hi, 0.9, y_hi)
    # log-scale y: transform values to log before plotting
    log_gate = math.log10(gate_threshold)
    log_eh = [(x, math.log10(max(y, 1.0))) for x, y in eh]
    log_e2 = [(x, math.log10(max(y, 1.0))) for x, y in e2]
    Xl, Yl, *_ = _axes(x_lo, x_hi, 0.0, math.log10(y_hi))
    body = frame
    body += _hline(Xl, Yl, log_gate, x_lo, x_hi, "#d9534f",
                   f"门限 e={gate_threshold:g}")
    body += _line_series(Xl, Yl, log_eh, "#d9534f", 2.2)
    body += _line_series(Xl, Yl, log_e2, "#337ab7", 1.8, dash="5,3")
    body += (f'<text x="{_MARGIN[0]}" y="{_H-8}" font-size="11" fill="#d9534f">'
             f'— 单侧有害 e</text>')
    body += (f'<text x="{_MARGIN[0]+120}" y="{_H-8}" font-size="11" '
             f'fill="#337ab7">--- 双侧 e</text>')
    return f'<svg viewBox="0 0 {_W} {_H}" class="chart">{body}</svg>'


def _interval_chart(snapshots, metric_name, info):
    batches = [s["batch"] for s in snapshots]
    x_lo, x_hi = 1, max(batches)
    rows = [s["metrics"][metric_name] for s in snapshots]
    lows = [r["cs_low"] for r in rows if r["cs_low"] is not None]
    highs = [r["cs_high"] for r in rows if r["cs_high"] is not None]
    diffs = [r["diff"] for r in rows if r["diff"] is not None]
    if not lows:
        return ""
    floor = info.effect_floor
    y_lo = min(min(lows), -floor) * 1.2
    y_hi = max(max(highs), floor) * 1.2
    X, Y, frame = _frame(
        f"{metric_name} — 置信序列如何收窄",
        "批次", "处理-对照 差值", x_lo, x_hi, y_lo, y_hi)
    body = frame
    body += _hline(X, Y, 0.0, x_lo, x_hi, "#999", "零效应")
    if info.harm == "two_sided":
        body += _hline(X, Y, floor, x_lo, x_hi, "#f0ad4e", f"+地板 {floor:g}")
        body += _hline(X, Y, -floor, x_lo, x_hi, "#f0ad4e", f"-地板 {floor:g}")
    else:
        body += _hline(X, Y, floor, x_lo, x_hi, "#f0ad4e",
                       f"实际地板 {floor:g}")
    # confidence band as filled area
    active = [(s["batch"], s["metrics"][metric_name]) for s in snapshots
              if s["metrics"][metric_name]["cs_low"] is not None]
    band_pts = ([(b, r["cs_low"]) for b, r in active]
                + [(b, r["cs_high"]) for b, r in reversed(active)])
    d = "M " + " L ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in band_pts) + " Z"
    body += f'<path d="{d}" fill="#337ab7" fill-opacity="0.12"/>'
    body += _line_series(
        X, Y, [(b, r["cs_high"]) for b, r in active], "#337ab7", 1.5)
    body += _line_series(
        X, Y, [(b, r["cs_low"]) for b, r in active], "#337ab7", 1.5)
    body += _line_series(
        X, Y, [(b, r["diff"]) for b, r in active], "#222", 2.0)
    return f'<svg viewBox="0 0 {_W} {_H}" class="chart">{body}</svg>'


def _ramp_srm_chart(snapshots):
    batches = [s["batch"] for s in snapshots]
    x_lo, x_hi = 1, max(batches)
    frac = [(s["batch"], s["fraction"]) for s in snapshots]
    Xf, Yf, l, r, t, b, iw, ih = _axes(x_lo, x_hi, 0.0, 1.0)
    frame_parts = []
    for i in range(5):
        gy = t + ih - i * ih / 4
        frame_parts.append(
            f'<line x1="{l}" y1="{gy:.1f}" x2="{l+iw:.1f}" y2="{gy:.1f}" '
            f'stroke="#eee"/>')
        frame_parts.append(
            f'<text x="{l-6}" y="{gy+4:.1f}" text-anchor="end" font-size="10" '
            f'fill="#888">{i/4:.2f}</text>')
    frame = "".join(frame_parts)
    frame += (f'<text x="{l}" y="{t-10}" font-size="13" font-weight="600" '
              f'fill="#222">自适应爬坡比例 与 SRM e-process</text>')
    body = frame + _line_series(Xf, Yf, frac, "#5cb85c", 2.4)
    body += (f'<text x="{l}" y="{_H-8}" font-size="11" fill="#5cb85c">'
             f'— 处理组流量比例</text>')
    # SRM on a secondary log scale
    srm = [(s["batch"], math.log10(max(s["srm_e"], 1e-3)))
           for s in snapshots]
    srm_gate = math.log10(1000.0)
    ymax = max(srm_gate + 0.5, max(y for _, y in srm) + 0.5, 2.0)
    Xs, Ys, *_ = _axes(x_lo, x_hi, 0.0, ymax)
    body += _hline(Xs, Ys, srm_gate, x_lo, x_hi, "#d9534f",
                   "SRM 门限 1000")
    body += _line_series(Xs, Ys, srm, "#d9534f", 1.8, dash="5,3")
    body += (f'<text x="{l+160}" y="{_H-8}" font-size="11" fill="#d9534f">'
             f'--- SRM log10(e)</text>')
    return f'<svg viewBox="0 0 {_W} {_H}" class="chart">{body}</svg>'


def _slice_heatmap(snapshots, metric_name):
    """Rows = slice cells, columns = batches; colour = log10 e-value."""
    rows = snapshots[-1]["metrics"][metric_name]["slices"]
    cells = list(rows.keys())
    if not cells:
        return ""
    batches = [s["batch"] for s in snapshots]
    matrix = []
    vmax = 1.0
    for cell in cells:
        row = []
        for s in snapshots:
            sl = s["metrics"][metric_name]["slices"].get(cell)
            if sl is None or not sl["active"]:
                row.append(None)
            else:
                lv = math.log10(max(sl["e_harm"], 1.0))
                row.append(lv)
                vmax = max(vmax, lv)
        matrix.append(row)
    cell_w = min(16.0, 640.0 / max(len(batches), 1))
    cell_h = 22.0
    width = 120 + cell_w * len(batches)
    height = 40 + cell_h * len(cells)
    out = [f'<svg viewBox="0 0 {width} {height}" class="heatmap">']
    for j, cell in enumerate(cells):
        y = 30 + j * cell_h
        out.append(f'<text x="112" y="{y+15:.1f}" text-anchor="end" '
                   f'font-size="10.5" fill="#333">{html.escape(cell)}</text>')
        for i, lv in enumerate(matrix[j]):
            x = 118 + i * cell_w
            if lv is None:
                color = "#f4f4f4"
            else:
                # white -> red intensity by log e-value relative to max
                inten = min(lv / vmax, 1.0)
                r = 255
                g = int(245 - 170 * inten)
                b = int(240 - 200 * inten)
                color = f"rgb({r},{g},{b})"
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-1:.1f}" '
                       f'height="{cell_h-2:.1f}" fill="{color}"/>')
    out.append(f'<text x="118" y="20" font-size="11" fill="#666">'
               f'格子 e 值（越红越强），共 {len(batches)} 批</text>')
    out.append("</svg>")
    return "".join(out)


def render_release_page(engine: DecisionEngine, out_path: str | Path,
                        title: str = "发布过程判定轨迹") -> Path:
    """Write a standalone HTML dashboard for one completed release."""
    snaps = engine.snapshots
    if not snaps:
        raise RuntimeError("engine has no snapshots; call record_snapshot() "
                           "after each update()")
    m_total = len(engine.metrics)
    global_gate = m_total / engine.alpha_fwer
    sections = [f"<h2>{html.escape(title)}</h2>",
                f"<p class='meta'>最终判定：<b>{snaps[-1]['verdict']}</b> · "
                f"批次数：{snaps[-1]['batch']} · "
                f"承诺 FWER α={engine.alpha_fwer} · "
                f"全局 e 门限={global_gate:g} · "
                f"切片层级门限="
                f"{2*m_total*(len(engine.cells)-1)/engine.alpha_fwer:g}</p>"]
    sections.append("<div class='row'>" + _ramp_srm_chart(snaps) + "</div>")
    for spec in engine.metrics:
        info = next(m for m in engine.metrics if m.name == spec.name)
        sections.append(
            "<div class='row'>"
            + _e_process_chart(snaps, spec.name, info, global_gate)
            + _interval_chart(snaps, spec.name, info) + "</div>")
        sections.append(
            f"<details open><summary>{html.escape(spec.name)} "
            f"细分格子热力图</summary>"
            + _slice_heatmap(snaps, spec.name) + "</details>")
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  margin: 24px; color: #222; background: #fafafa; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 8px; }}
.meta {{ color: #555; font-size: 13px; }}
.row {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0; }}
.chart {{ background: #fff; border: 1px solid #e6e6e6; border-radius: 8px;
  width: min({_W}px, 48%); min-width: 380px; }}
.heatmap {{ background: #fff; border: 1px solid #e6e6e6; border-radius: 8px;
  padding: 8px; max-width: 100%; }}
details {{ margin: 6px 0 18px; }}
summary {{ cursor: pointer; font-weight: 600; margin-bottom: 6px; }}
</style></head><body>
<h1>电子税务局版本发布 · 连续判定引擎</h1>
{''.join(sections)}
</body></html>"""
    path = Path(out_path)
    path.write_text(page, encoding="utf-8")
    return path
