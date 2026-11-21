"""공급계획 결과를 한 장짜리 HTML 보고서로 (브라우저에서 열어 인쇄 → PDF 저장 가능)"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body { font-family: 'Malgun Gothic', 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif; color: #1f2937;
       margin: 0 auto; max-width: 960px; padding: 18px; font-size: 13px; line-height: 1.5; background: #fff; }
h1 { font-size: 22px; margin: 0 0 2px 0; color: #1f3f8f; }
h2 { font-size: 15px; margin: 18px 0 6px 0; color: #1f3f8f; border-bottom: 2px solid #dbe3f4; padding-bottom: 3px; }
.sub { color: #4b5563; font-size: 12px; }
.badge { display: inline-block; padding: 1px 10px; border-radius: 8px; color: #fff; font-weight: 700; }
.cards { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.card { flex: 1; min-width: 150px; border-left: 6px solid #1f3f8f; background: #f4f6fa; border-radius: 10px; padding: 8px 12px; }
.card .t { font-size: 11px; color: #4b5563; } .card .v { font-size: 24px; font-weight: 800; line-height: 1.2; }
.card .s { font-size: 11px; color: #6b7280; }
.box { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 8px 12px; white-space: pre-line; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 4px 0; }
th { background: #1f3f8f; color: #fff; padding: 4px 6px; font-weight: 600; }
td { border: 1px solid #d1d5db; padding: 3px 6px; text-align: right; }
td:first-child { text-align: center; }
td:last-child { white-space: nowrap; }
.cond td { text-align: left; } .cond td:first-child { width: 28%; background: #f8fafc; text-align: left; }
.note { color: #6b7280; font-size: 11px; }
.fig { page-break-inside: avoid; }
.printbtn { float: right; padding: 6px 14px; border: 0; border-radius: 8px; background: #1f3f8f; color: #fff; cursor: pointer; }
@media print { .printbtn { display: none; } body { padding: 0; } h2 { page-break-after: avoid; } }
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _table(df: pd.DataFrame) -> str:
    head = "<tr><th></th>" + "".join(f"<th>{_esc(c)}</th>" for c in df.columns) + "</tr>"
    rows = []
    for idx, r in df.iterrows():
        cells = []
        for c, v in r.items():
            if isinstance(v, (int, float)) and pd.notna(v):
                s = f"{v:+.1f}" if "오차" in str(c) else (f"{v:,.2f}" if abs(v) < 10 and v != int(v) else f"{v:,.1f}")
            elif v is None or (isinstance(v, float) and pd.isna(v)):
                s = "-"
            else:
                s = _esc(v)
            cells.append(f"<td>{s}</td>")
        rows.append(f"<tr><td>{_esc(idx)}</td>{''.join(cells)}</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def plan_report_html(name: str, location: str, now: dict, stage_color: str, cards: list[tuple[str, str, str, str]],
                     summary: str, conditions: list[tuple[str, str]], monthly: pd.DataFrame,
                     figs: list[tuple[str, object, str]], rule_text: str, notes: list[str]) -> str:
    """name·location: 저수지, now: 현재 상태(pct·stage·pct_of_normal·source·time), cards: (제목, 값, 설명, 색),
    conditions: (항목, 값) 입력조건, figs: (제목, plotly Figure, 설명)"""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    pn = now.get("pct_of_normal")
    pn_txt = f"평년 대비 {pn:.0f}%" if pn is not None and not pd.isna(pn) else "평년 자료 없음"
    card_html = "".join(
        f'<div class="card" style="border-left-color:{c}"><div class="t">{_esc(t)}</div>'
        f'<div class="v" style="color:{c}">{_esc(v)}</div><div class="s">{_esc(s)}</div></div>' for t, v, s, c in cards)
    cond_html = "<table class='cond'>" + "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in conditions) + "</table>"
    fig_html, first = [], True
    for title, fig, desc in figs:
        if fig is None:
            continue
        f = fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                        config={"displayModeBar": False, "staticPlot": False})
        first = False
        fig_html.append(f'<div class="fig"><h2>{_esc(title)}</h2>{f}<div class="note">{_esc(desc)}</div></div>')
    notes_html = "".join(f"<li>{_esc(n)}</li>" for n in notes)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{_esc(name)} 공급계획 보고서</title>
<style>{CSS}</style></head><body>
<button class="printbtn" onclick="window.print()">🖨 인쇄 / PDF 저장</button>
<h1>💧 {_esc(name)} 저수지 — 가뭄 예측·공급계획 보고서</h1>
<div class="sub">{_esc(location)} · 작성 {stamp} · sungju-water.streamlit.app</div>
<h2>1. 현재 상태</h2>
<div>유효저수율 <b style="font-size:18px">{now.get('pct', float('nan')):.1f}%</b> · {_esc(pn_txt)} · 농업가뭄
 <span class="badge" style="background:{stage_color}">{_esc(now.get('stage', '-'))}</span>
 <span class="note">({_esc(now.get('source', ''))} {_esc(str(now.get('time', ''))[:16])})</span></div>
<h2>2. 입력 조건</h2>{cond_html}
<h2>3. 핵심 결과</h2><div class="cards">{card_html}</div>
<div class="box">{_esc(summary)}</div>
<h2>4. 월별 공급계획</h2>{_table(monthly)}
<div class="note">{_esc(rule_text)}</div>
{''.join(fig_html)}
<h2>읽는 법</h2><ul class="note">{notes_html}</ul>
</body></html>"""


def simple_report_html(title: str, subtitle: str, sections: list[tuple], notes: list[str] | None = None) -> str:
    """일반 결과 보고서. sections 항목:
    ("kv", 제목, [(항목, 값), …]) · ("table", 제목, DataFrame) · ("fig", 제목, plotly Figure, 설명)
    ("text", 제목, 글) · ("cards", 제목, [(제목, 값, 설명, 색), …])"""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out, first, n = [], True, 0
    for sec in sections:
        kind, head = sec[0], sec[1]
        if kind == "fig" and sec[2] is None:
            continue
        n += 1
        h = f"<h2>{n}. {_esc(head)}</h2>"
        if kind == "kv":
            out.append(h + "<table class='cond'>" + "".join(
                f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in sec[2]) + "</table>")
        elif kind == "table":
            out.append(h + _table(sec[2]))
        elif kind == "text":
            out.append(h + f"<div class='box'>{_esc(sec[2])}</div>")
        elif kind == "cards":
            out.append(h + "<div class='cards'>" + "".join(
                f'<div class="card" style="border-left-color:{c}"><div class="t">{_esc(t)}</div>'
                f'<div class="v" style="color:{c}">{_esc(v)}</div><div class="s">{_esc(s)}</div></div>'
                for t, v, s, c in sec[2]) + "</div>")
        elif kind == "fig":
            f = sec[2].to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                               config={"displayModeBar": False})
            first = False
            desc = sec[3] if len(sec) > 3 else ""
            out.append(f'<div class="fig">{h}{f}<div class="note">{_esc(desc)}</div></div>')
    notes_html = ("<h2>읽는 법</h2><ul class='note'>" + "".join(f"<li>{_esc(x)}</li>" for x in notes) + "</ul>") if notes else ""
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{_esc(title)}</title>
<style>{CSS}</style></head><body>
<button class="printbtn" onclick="window.print()">🖨 인쇄 / PDF 저장</button>
<h1>💧 {_esc(title)}</h1><div class="sub">{_esc(subtitle)} · 작성 {stamp} · sungju-water.streamlit.app</div>
{''.join(out)}{notes_html}
</body></html>"""
