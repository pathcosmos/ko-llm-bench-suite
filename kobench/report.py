"""
HTML/Markdown 통합 리포트 생성 — 모든 트랙 결과를 대시보드로
"""

import json
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from . import config

# 한국어 폰트 설정
_KR_FONTS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for fpath in _KR_FONTS:
    if Path(fpath).exists():
        fm.fontManager.addfont(fpath)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fpath).get_name()
        break
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── 차트 생성 ─────────────────────────────────────────────────────────────────

def chart_bar(data: dict[str, float], title: str, ylabel: str = "Score") -> str:
    """수평 막대 차트"""
    models = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5)))
    colors = ["#4A90D9" if "frankenstallm" in m else "#E8853D" for m in models]
    bars = ax.barh(models, values, color=colors)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01 * max(values), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    return _fig_to_base64(fig)


def chart_grouped_bar(data: dict[str, dict[str, float]], title: str) -> str:
    """카테고리별 그룹 막대 차트"""
    models = list(data.keys())
    if not models:
        return ""
    categories = list(data[models[0]].keys())
    x = np.arange(len(categories))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        vals = [data[model].get(c, 0) for c in categories]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model[:20])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return _fig_to_base64(fig)


def chart_radar(data: dict[str, dict[str, float]], title: str) -> str:
    """레이더 차트 — 다면적 비교"""
    models = list(data.keys())
    if not models:
        return ""
    categories = list(data[models[0]].keys())
    n = len(categories)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model in models:
        vals = [data[model].get(c, 0) for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, label=model[:20])
        ax.fill(angles, vals, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=7)
    return _fig_to_base64(fig)


def chart_elo(elo_data: dict[str, dict]) -> str:
    """Elo 스코어 차트 (에러바 포함)"""
    sorted_models = sorted(elo_data.items(), key=lambda x: x[1].get("elo", 0), reverse=True)
    models = [m for m, _ in sorted_models]
    elos = [d.get("elo", 1000) for _, d in sorted_models]
    ci_low = [d.get("elo", 1000) - d.get("ci_lower", 1000) for _, d in sorted_models]
    ci_high = [d.get("ci_upper", 1000) - d.get("elo", 1000) for _, d in sorted_models]

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5)))
    colors = ["#4A90D9" if "frankenstallm" in m else "#E8853D" for m in models]
    ax.barh(models, elos, xerr=[ci_low, ci_high], color=colors, capsize=3)
    ax.set_xlabel("Elo Rating")
    ax.set_title("Pairwise Elo Rankings (95% CI)")
    ax.invert_yaxis()
    ax.axvline(x=1000, color="gray", linestyle="--", alpha=0.5)

    return _fig_to_base64(fig)


def chart_performance_line(perf_data: dict[str, dict], metric: str, title: str) -> str:
    """성능 라인 차트"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, data in perf_data.items():
        if metric in data:
            vals = data[metric]
            if isinstance(vals, dict):
                ax.bar(model[:15], vals.get("mean", 0))
            else:
                ax.bar(model[:15], vals)
    ax.set_title(title)
    ax.set_ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    return _fig_to_base64(fig)


# ── HTML 리포트 ───────────────────────────────────────────────────────────────

def generate_html_report(
    track_results: dict,
    scorecard: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """전체 HTML 대시보드 리포트 생성"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if output_path is None:
        output_path = config.REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    charts = {}

    # Track 1: 한국어 벤치마크
    if "track1" in track_results:
        t1 = track_results["track1"].get("summary", {})
        # 벤치마크별 정확도
        bench_data = {}
        for model, benchmarks in t1.items():
            if isinstance(benchmarks, dict):
                bench_data[model] = benchmarks
        if bench_data:
            charts["track1_radar"] = chart_radar(bench_data, "Track 1: 한국어 벤치마크 정확도")
            overall = {m: np.mean(list(v.values())) for m, v in bench_data.items() if v}
            charts["track1_bar"] = chart_bar(overall, "Track 1: 평균 벤치마크 정확도", "Accuracy")

    # Track 2: Ko-Bench
    if "track2" in track_results:
        t2 = track_results["track2"].get("summary", {})
        cat_data = {}
        for model, cats in t2.items():
            if isinstance(cats, dict):
                cat_data[model] = {c: v.get("overall_mean", 0) if isinstance(v, dict) else v for c, v in cats.items()}
        if cat_data:
            charts["track2_radar"] = chart_radar(cat_data, "Track 2: Ko-Bench 카테고리별 점수")

    # Track 3: 한국어 심화
    if "track3" in track_results:
        t3 = track_results["track3"].get("summary", {})
        cat_data = {}
        for model, cats in t3.items():
            if isinstance(cats, dict):
                cat_data[model] = {c: v.get("accuracy", v.get("score", 0)) if isinstance(v, dict) else v for c, v in cats.items()}
        if cat_data:
            charts["track3_radar"] = chart_radar(cat_data, "Track 3: 한국어 심화 평가")

    # Track 6: 성능
    if "track6" in track_results:
        t6 = track_results["track6"].get("summary", {})
        if t6:
            tok_s = {m: v.get("tokens_per_sec", {}).get("mean", 0) for m, v in t6.items() if isinstance(v, dict)}
            if any(tok_s.values()):
                charts["track6_speed"] = chart_bar(tok_s, "Track 6: 생성 속도 (tok/s)", "Tokens/sec")

    # Track 7: Elo
    if "track7" in track_results:
        t7_res = track_results["track7"].get("results", {})
        elo = t7_res.get("elo_scores", {})
        if elo:
            charts["track7_elo"] = chart_elo(elo)

    # HTML 조립
    chart_sections = []
    section_titles = {
        "track1_radar": "Track 1: 한국어 표준 벤치마크",
        "track1_bar": "Track 1: 평균 정확도",
        "track2_radar": "Track 2: Ko-Bench 멀티턴 평가",
        "track3_radar": "Track 3: 한국어 언어 능력 심화",
        "track6_speed": "Track 6: 성능 프로파일링",
        "track7_elo": "Track 7: Elo 랭킹",
    }
    for key, b64 in charts.items():
        title = section_titles.get(key, key)
        chart_sections.append(f"""
        <div class="chart-section">
            <h2>{title}</h2>
            <img src="data:image/png;base64,{b64}" alt="{title}">
        </div>""")

    # 스코어카드 테이블
    scorecard_html = ""
    if scorecard:
        headers = set()
        for v in scorecard.values():
            headers.update(k for k in v if k != "model")
        headers = sorted(headers)
        rows = []
        for model in config.ALL_MODELS:
            if model in scorecard:
                vals = scorecard[model]
                cells = "".join(f"<td>{vals.get(h, '-'):.3f}</td>" if isinstance(vals.get(h), (int, float)) else f"<td>{vals.get(h, '-')}</td>" for h in headers)
                cls = "frankenstallm" if "frankenstallm" in model else "comparison"
                rows.append(f'<tr class="{cls}"><td>{model}</td>{cells}</tr>')
        header_row = "".join(f"<th>{h}</th>" for h in headers)
        scorecard_html = f"""
        <div class="chart-section">
            <h2>종합 스코어카드</h2>
            <table><tr><th>모델</th>{header_row}</tr>{"".join(rows)}</table>
        </div>"""

    # 트랙별 상세 결과 테이블
    detail_sections = []
    for track_name, track_data in sorted(track_results.items()):
        summary = track_data.get("summary", {})
        if summary:
            detail_sections.append(f"""
            <div class="chart-section">
                <h2>{track_name} 상세</h2>
                <pre>{json.dumps(summary, ensure_ascii=False, indent=2, default=str)[:5000]}</pre>
            </div>""")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>FRANKENSTALLM 3B 심화 평가 리포트</title>
    <style>
        body {{ font-family: 'Nanum Gothic', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .chart-section {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-section img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background: #3498db; color: white; }}
        tr.frankenstallm {{ background: #ebf5fb; }}
        tr.comparison {{ background: #fef9e7; }}
        tr:hover {{ background: #d5e8d4; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 12px; }}
        .meta {{ color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>FRANKENSTALLM 3B 심화 평가 리포트</h1>
    <p class="meta">생성 시간: {timestamp} | 평가 모델: {len(config.ALL_MODELS)}개 | 트랙: {len(track_results)}개</p>

    {scorecard_html}
    {"".join(chart_sections)}
    {"".join(detail_sections)}

    <div class="chart-section">
        <h2>평가 설정</h2>
        <pre>{json.dumps({
            "models": config.ALL_MODELS,
            "sampling": config.SAMPLING_PARAMS,
            "benchmark_sampling": config.BENCHMARK_SAMPLING,
        }, ensure_ascii=False, indent=2)}</pre>
    </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📊 HTML 리포트 생성: {output_path}")
    return output_path


# ── Markdown 리포트 (상세) ────────────────────────────────────────────────────

def _fmt(v, precision=3) -> str:
    """숫자 포맷 헬퍼"""
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v) if v is not None else "-"


def _rank_models(data: dict[str, float], reverse: bool = True) -> list[tuple[str, float, int]]:
    """모델을 점수로 정렬하고 순위 부여"""
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=reverse)
    return [(m, v, i + 1) for i, (m, v) in enumerate(sorted_items)]


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """마크다운 테이블 생성"""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "---|" * len(headers))
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_markdown_report(
    track_results: dict,
    scorecard: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """상세 Markdown 리포트 — 트랙별 세부 분석 포함"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if output_path is None:
        output_path = config.REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    L = []  # lines accumulator
    a = L.append

    a("# FRANKENSTALLM 3B 심화 평가 리포트")
    a("")
    a(f"> 생성: {timestamp}")
    a(f"> 평가 대상: {len(config.ALL_MODELS)}개 모델 | {len(track_results)}개 트랙")
    a("")
    a("## 목차")
    a("")
    a("1. [평가 개요](#평가-개요)")
    a("2. [종합 스코어카드](#종합-스코어카드)")
    a("3. [Track 6: 성능 프로파일링](#track-6-성능-프로파일링)")
    a("4. [Track 1: 한국어 표준 벤치마크](#track-1-한국어-표준-벤치마크)")
    a("5. [Track 3: 한국어 언어 능력 심화](#track-3-한국어-언어-능력-심화)")
    a("6. [Track 2: Ko-Bench 멀티턴 평가](#track-2-ko-bench-멀티턴-평가)")
    a("7. [Track 4: 코드 생성 & 수학 추론](#track-4-코드-생성--수학-추론)")
    a("8. [Track 5: 일관성 & 강건성](#track-5-일관성--강건성)")
    a("9. [Track 7: 쌍대비교 Elo 랭킹](#track-7-쌍대비교-elo-랭킹)")
    a("10. [종합 분석](#종합-분석)")
    a("11. [평가 설정](#평가-설정)")
    a("")

    # ── 평가 개요 ──
    a("---")
    a("## 평가 개요")
    a("")
    a("### 평가 모델")
    a("")
    a("**Frankenstallm 변형 (6종)**")
    a("")
    for m in config.FRANKENSTALLM_MODELS:
        quant = "f16" if "f16" in m else "Q8_0" if "Q8_0" in m else "Q4_K_M"
        ver = "v2" if "v2" in m else "v1"
        a(f"- `{m}` — {ver}, {quant} 양자화")
    a("")
    a("**비교 모델 (5종)**")
    a("")
    for m in config.COMPARISON_MODELS:
        a(f"- `{m}`")
    a("")
    a("### 하드웨어")
    a("")
    a("- **GPU:** NVIDIA RTX 5060 Ti 16GB VRAM")
    a("- **추론 엔진:** Ollama")
    a(f"- **모델 저장:** `/var/ollama/models`")
    a("")

    # ── 종합 스코어카드 ──
    a("---")
    a("## 종합 스코어카드")
    a("")
    if scorecard:
        headers_set = set()
        for v in scorecard.values():
            headers_set.update(k for k in v if k != "model")
        headers = sorted(headers_set)
        header_names = ["모델"] + headers
        rows = []
        for model in config.ALL_MODELS:
            if model in scorecard:
                vals = scorecard[model]
                row = [f"`{model}`"] + [_fmt(vals.get(h)) for h in headers]
                rows.append(row)
        a(_md_table(header_names, rows))
    else:
        a("*스코어카드 데이터 없음*")
    a("")

    # ── Track 6: 성능 프로파일링 ──
    a("---")
    a("## Track 6: 성능 프로파일링")
    a("")
    if "track6" in track_results:
        t6 = track_results["track6"]
        summary = t6.get("summary", {})
        results = t6.get("results", [])

        if summary:
            a("### 모델별 성능 요약")
            a("")
            perf_headers = ["모델", "Decode (tok/s)", "Prefill (tok/s)", "TTFT (s)", "VRAM (MB)", "최대 컨텍스트"]
            perf_rows = []
            for model in config.ALL_MODELS:
                s = summary.get(model, {})
                if s:
                    perf_rows.append([
                        f"`{model}`",
                        _fmt(s.get("avg_decode_tok_s", 0), 1),
                        _fmt(s.get("avg_prefill_tok_s", 0), 1),
                        _fmt(s.get("avg_ttft_s", 0), 4),
                        _fmt(s.get("vram_used_mb", 0), 0),
                        _fmt(s.get("max_context_reached", "-"), 0),
                    ])
            a(_md_table(perf_headers, perf_rows))
            a("")

            # 속도 랭킹
            decode_speeds = {m: s.get("avg_decode_tok_s", 0) for m, s in summary.items() if s}
            if decode_speeds:
                a("### Decode 속도 랭킹")
                a("")
                for model, speed, rank in _rank_models(decode_speeds):
                    marker = "**" if "frankenstallm" in model else ""
                    a(f"{rank}. {marker}`{model}`{marker} — {speed:.1f} tok/s")
                a("")

        # 양자화별 비교
        quant_groups = {}
        for r in results:
            if r.get("test_type") == "quantization":
                base = r["model"].replace("-Q4_K_M", "").replace("-Q8_0", "").replace("-f16", "")
                if base not in quant_groups:
                    quant_groups[base] = {}
                quant = "f16" if "f16" in r["model"] else "Q8_0" if "Q8_0" in r["model"] else "Q4_K_M"
                if quant not in quant_groups[base]:
                    quant_groups[base][quant] = []
                quant_groups[base][quant].append(r.get("tokens_per_sec", 0))

        if quant_groups:
            a("### 양자화별 속도 비교")
            a("")
            q_headers = ["모델 베이스", "f16", "Q8_0", "Q4_K_M"]
            q_rows = []
            for base, quants in sorted(quant_groups.items()):
                row = [f"`{base}`"]
                for q in ["f16", "Q8_0", "Q4_K_M"]:
                    vals = quants.get(q, [])
                    row.append(f"{np.mean(vals):.1f}" if vals else "-")
                q_rows.append(row)
            a(_md_table(q_headers, q_rows))
            a("")

        # 동시 요청 성능
        concurrent = {}
        for r in results:
            if r.get("test_type") == "concurrent":
                m = r["model"]
                if m not in concurrent:
                    concurrent[m] = {}
                level = r.get("output_length", r.get("concurrency", "?"))
                concurrent[m][level] = r.get("tokens_per_sec", 0)

        if concurrent:
            a("### 동시 요청 처리 성능")
            a("")
            levels = sorted(set(lv for d in concurrent.values() for lv in d))
            c_headers = ["모델"] + [f"동시 {lv}개" for lv in levels]
            c_rows = []
            for model in config.ALL_MODELS:
                if model in concurrent:
                    row = [f"`{model}`"] + [_fmt(concurrent[model].get(lv, 0), 1) for lv in levels]
                    c_rows.append(row)
            a(_md_table(c_headers, c_rows))
            a("")
    else:
        a("*Track 6 데이터 없음*")
    a("")

    # ── Track 1: 한국어 표준 벤치마크 ──
    a("---")
    a("## Track 1: 한국어 표준 벤치마크")
    a("")
    if "track1" in track_results:
        t1 = track_results["track1"]
        summary = t1.get("summary", {})

        if summary:
            # 벤치마크 목록 추출
            benchmarks = set()
            for model_data in summary.values():
                if isinstance(model_data, dict):
                    benchmarks.update(model_data.keys())
            benchmarks = sorted(benchmarks)

            a("### 벤치마크별 정확도")
            a("")
            b_headers = ["모델"] + benchmarks + ["평균"]
            b_rows = []
            model_avgs = {}
            for model in config.ALL_MODELS:
                s = summary.get(model, {})
                if isinstance(s, dict) and s:
                    vals = [s.get(b, 0) for b in benchmarks]
                    avg = np.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else 0
                    model_avgs[model] = avg
                    row = [f"`{model}`"] + [f"{v:.1%}" if isinstance(v, float) else str(v) for v in vals] + [f"{avg:.1%}"]
                    b_rows.append(row)
            a(_md_table(b_headers, b_rows))
            a("")

            # 랭킹
            if model_avgs:
                a("### 종합 정확도 랭킹")
                a("")
                for model, acc, rank in _rank_models(model_avgs):
                    marker = "**" if "frankenstallm" in model else ""
                    a(f"{rank}. {marker}`{model}`{marker} — {acc:.1%}")
                a("")

            # 벤치마크별 1위
            a("### 벤치마크별 최고 성능 모델")
            a("")
            for bench in benchmarks:
                bench_scores = {m: summary[m].get(bench, 0) for m in summary if isinstance(summary[m], dict)}
                if bench_scores:
                    best_model = max(bench_scores, key=bench_scores.get)
                    a(f"- **{bench}**: `{best_model}` ({bench_scores[best_model]:.1%})")
            a("")

        # 개별 문항 결과 샘플
        results = t1.get("results", [])
        if results:
            a("### 문항 샘플 (오답 분석)")
            a("")
            wrong = [r for r in results if r.get("correct") is False][:20]
            if wrong:
                w_headers = ["모델", "벤치마크", "문항 (요약)", "모델 답변", "정답"]
                w_rows = []
                for r in wrong:
                    q = r.get("question", r.get("prompt", ""))[:40] + "..."
                    w_rows.append([
                        f"`{r.get('model', '?')}`",
                        r.get("benchmark", "?"),
                        q,
                        str(r.get("model_answer", "?"))[:20],
                        str(r.get("correct_answer", "?"))[:20],
                    ])
                a(_md_table(w_headers, w_rows))
            else:
                a("*모든 문항 정답 또는 오답 데이터 없음*")
            a("")
    else:
        a("*Track 1 데이터 없음*")
    a("")

    # ── Track 3: 한국어 심화 ──
    a("---")
    a("## Track 3: 한국어 언어 능력 심화")
    a("")
    if "track3" in track_results:
        t3 = track_results["track3"]
        summary = t3.get("summary", {})

        if summary:
            categories = set()
            for model_data in summary.values():
                if isinstance(model_data, dict):
                    categories.update(k for k in model_data if not k.startswith("_"))
            categories = sorted(categories)

            a("### 카테고리별 성적")
            a("")
            c_headers = ["모델"] + categories + ["종합"]
            c_rows = []
            for model in config.ALL_MODELS:
                s = summary.get(model, {})
                if isinstance(s, dict) and s:
                    vals = []
                    for cat in categories:
                        cv = s.get(cat, {})
                        if isinstance(cv, dict):
                            vals.append(cv.get("accuracy", cv.get("avg_score", 0)))
                        else:
                            vals.append(cv if isinstance(cv, (int, float)) else 0)
                    overall = s.get("_overall", {})
                    avg = overall.get("accuracy", overall.get("avg_score", np.mean(vals) if vals else 0)) if isinstance(overall, dict) else np.mean(vals) if vals else 0
                    row = [f"`{model}`"] + [_fmt(v) for v in vals] + [_fmt(avg)]
                    c_rows.append(row)
            a(_md_table(c_headers, c_rows))
            a("")

            # 카테고리별 분석
            a("### 카테고리별 상세 분석")
            a("")
            for cat in categories:
                cat_scores = {}
                for model in config.ALL_MODELS:
                    s = summary.get(model, {})
                    if isinstance(s, dict):
                        cv = s.get(cat, {})
                        if isinstance(cv, dict):
                            cat_scores[model] = cv.get("accuracy", cv.get("avg_score", 0))
                        elif isinstance(cv, (int, float)):
                            cat_scores[model] = cv
                if cat_scores:
                    best = max(cat_scores, key=cat_scores.get)
                    worst = min(cat_scores, key=cat_scores.get)
                    a(f"#### {cat}")
                    a(f"- 최고: `{best}` ({_fmt(cat_scores[best])})")
                    a(f"- 최저: `{worst}` ({_fmt(cat_scores[worst])})")
                    spread = max(cat_scores.values()) - min(cat_scores.values())
                    a(f"- 편차: {_fmt(spread)} (모델 간 차이)")
                    a("")

        # 응답 샘플
        results = t3.get("results", [])
        if results:
            a("### 응답 샘플")
            a("")
            seen = set()
            for r in results[:100]:
                qid = r.get("question_id", r.get("id", ""))
                if qid not in seen and len(seen) < 5:
                    seen.add(qid)
                    a(f"**Q: {r.get('question', '?')[:80]}**")
                    a("")
                    a(f"- `{r.get('model', '?')}`: {r.get('response', '')[:150]}...")
                    score = r.get("score", r.get("judge_score", ""))
                    if score:
                        a(f"  - 점수: {score}")
                    a("")
    else:
        a("*Track 3 데이터 없음*")
    a("")

    # ── Track 2: Ko-Bench ──
    a("---")
    a("## Track 2: Ko-Bench 멀티턴 평가")
    a("")
    if "track2" in track_results:
        t2 = track_results["track2"]
        summary = t2.get("summary", {})

        if summary:
            categories = set()
            for model_data in summary.values():
                if isinstance(model_data, dict):
                    categories.update(model_data.keys())
            categories = sorted(categories)

            # Turn 1 vs Turn 2 비교
            a("### Turn별 평균 점수")
            a("")
            t_headers = ["모델"] + [f"{c} T1" for c in categories] + [f"{c} T2" for c in categories] + ["종합"]
            t_rows = []
            model_overalls = {}
            for model in config.ALL_MODELS:
                s = summary.get(model, {})
                if isinstance(s, dict) and s:
                    t1_vals = []
                    t2_vals = []
                    for cat in categories:
                        cv = s.get(cat, {})
                        if isinstance(cv, dict):
                            t1_vals.append(cv.get("turn1_mean", 0))
                            t2_vals.append(cv.get("turn2_mean", 0))
                        else:
                            t1_vals.append(0)
                            t2_vals.append(0)
                    all_vals = t1_vals + t2_vals
                    avg = np.mean([v for v in all_vals if v > 0]) if any(v > 0 for v in all_vals) else 0
                    model_overalls[model] = avg
                    row = [f"`{model}`"] + [_fmt(v, 1) for v in t1_vals] + [_fmt(v, 1) for v in t2_vals] + [_fmt(avg, 2)]
                    t_rows.append(row)
            a(_md_table(t_headers, t_rows))
            a("")

            # 카테고리별 요약 (간결)
            a("### 카테고리별 평균 (Turn 1 + 2)")
            a("")
            cat_headers = ["모델"] + categories + ["평균"]
            cat_rows = []
            for model in config.ALL_MODELS:
                s = summary.get(model, {})
                if isinstance(s, dict) and s:
                    vals = []
                    for cat in categories:
                        cv = s.get(cat, {})
                        vals.append(cv.get("overall_mean", 0) if isinstance(cv, dict) else 0)
                    avg = np.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else 0
                    cat_rows.append([f"`{model}`"] + [_fmt(v, 1) for v in vals] + [_fmt(avg, 2)])
            a(_md_table(cat_headers, cat_rows))
            a("")

            # 랭킹
            if model_overalls:
                a("### Ko-Bench 종합 랭킹")
                a("")
                for model, score, rank in _rank_models(model_overalls):
                    marker = "**" if "frankenstallm" in model else ""
                    a(f"{rank}. {marker}`{model}`{marker} — {score:.2f}/10")
                a("")
    else:
        a("*Track 2 데이터 없음*")
    a("")

    # ── Track 4: 코드/수학 ──
    a("---")
    a("## Track 4: 코드 생성 & 수학 추론")
    a("")
    if "track4" in track_results:
        t4 = track_results["track4"]
        summary = t4.get("summary", {})

        if summary:
            a("### 모델별 정확도")
            a("")
            s_headers = ["모델", "Python Pass@1", "수학 정확도", "SQL 정확도", "디버깅 정확도", "종합"]
            s_rows = []
            for model in config.ALL_MODELS:
                s = summary.get(model, {})
                if isinstance(s, dict) and s:
                    py = s.get("python_pass1", 0)
                    math_acc = s.get("math_accuracy", 0)
                    sql = s.get("sql_accuracy", 0)
                    dbg = s.get("debug_accuracy", 0)
                    avg = np.mean([py, math_acc, sql, dbg])
                    s_rows.append([f"`{model}`", f"{py:.1%}", f"{math_acc:.1%}", f"{sql:.1%}", f"{dbg:.1%}", f"{avg:.1%}"])
            a(_md_table(s_headers, s_rows))
            a("")

            # 세부 분석
            a("### 영역별 최고 성능")
            a("")
            for field, label in [("python_pass1", "Python"), ("math_accuracy", "수학"), ("sql_accuracy", "SQL"), ("debug_accuracy", "디버깅")]:
                scores = {m: summary[m].get(field, 0) for m in summary if isinstance(summary[m], dict)}
                if scores:
                    best = max(scores, key=scores.get)
                    a(f"- **{label}**: `{best}` ({scores[best]:.1%})")
            a("")

        # 실패 사례
        results = t4.get("results", [])
        failed = [r for r in results if r.get("correct") is False or r.get("passed") is False][:10]
        if failed:
            a("### 주요 실패 사례")
            a("")
            for r in failed[:10]:
                a(f"- `{r.get('model', '?')}` | {r.get('problem_type', '?')} | `{r.get('problem_id', '?')}`: {r.get('error_detail', r.get('response', ''))[:80]}")
            a("")
    else:
        a("*Track 4 데이터 없음*")
    a("")

    # ── Track 5: 일관성 ──
    a("---")
    a("## Track 5: 일관성 & 강건성")
    a("")
    if "track5" in track_results:
        t5 = track_results["track5"]
        summary = t5.get("summary", {})

        if summary:
            dims = ["repetition_consistency", "paraphrase_robustness", "length_sensitivity",
                     "language_consistency", "instruction_following", "hallucination_detection"]
            dim_labels = {
                "repetition_consistency": "반복 일관성",
                "paraphrase_robustness": "패러프레이즈 강건성",
                "length_sensitivity": "길이 민감도",
                "language_consistency": "언어 일관성",
                "instruction_following": "지시 준수",
                "hallucination_detection": "환각 탐지",
            }

            a("### 6차원 평가 결과")
            a("")
            d_headers = ["모델"] + [dim_labels.get(d, d) for d in dims] + ["평균"]
            d_rows = []
            model_avgs = {}
            for model in config.ALL_MODELS:
                s = summary.get(model, {})
                if isinstance(s, dict) and s:
                    vals = [s.get(d, 0) for d in dims]
                    # 각 값을 float로 안전 변환
                    float_vals = []
                    for v in vals:
                        if isinstance(v, dict):
                            float_vals.append(v.get("mean", v.get("score", 0)))
                        else:
                            float_vals.append(float(v) if v else 0)
                    avg = np.mean([v for v in float_vals if v > 0]) if any(v > 0 for v in float_vals) else 0
                    model_avgs[model] = avg
                    d_rows.append([f"`{model}`"] + [_fmt(v) for v in float_vals] + [_fmt(avg)])
            a(_md_table(d_headers, d_rows))
            a("")

            # 차원별 분석
            a("### 차원별 분석")
            a("")
            for dim in dims:
                label = dim_labels.get(dim, dim)
                dim_scores = {}
                for m in config.ALL_MODELS:
                    s = summary.get(m, {})
                    if isinstance(s, dict):
                        v = s.get(dim, 0)
                        if isinstance(v, dict):
                            v = v.get("mean", v.get("score", 0))
                        dim_scores[m] = float(v) if v else 0
                if dim_scores and any(dim_scores.values()):
                    best = max(dim_scores, key=dim_scores.get)
                    worst = min(dim_scores, key=dim_scores.get)
                    a(f"#### {label}")
                    a(f"- 최고: `{best}` ({_fmt(dim_scores[best])})")
                    a(f"- 최저: `{worst}` ({_fmt(dim_scores[worst])})")
                    a("")
    else:
        a("*Track 5 데이터 없음*")
    a("")

    # ── Track 7: 쌍대비교 ──
    a("---")
    a("## Track 7: 쌍대비교 Elo 랭킹")
    a("")
    if "track7" in track_results:
        t7 = track_results["track7"]
        t7_res = t7.get("results", {})
        elo_scores = t7_res.get("elo_scores", t7.get("summary", {}))

        if elo_scores:
            a("### Elo 점수 (95% 신뢰구간)")
            a("")
            sorted_elo = sorted(elo_scores.items(),
                                key=lambda x: x[1].get("elo", 0) if isinstance(x[1], dict) else 0,
                                reverse=True)
            e_headers = ["순위", "모델", "Elo", "95% CI", "승", "패", "승률"]
            e_rows = []
            for rank, (model, data) in enumerate(sorted_elo, 1):
                if isinstance(data, dict):
                    elo = data.get("elo", 1000)
                    ci_l = data.get("ci_lower", elo)
                    ci_u = data.get("ci_upper", elo)
                    wins = data.get("wins", 0)
                    losses = data.get("losses", 0)
                    total = wins + losses
                    winrate = f"{wins/total:.1%}" if total > 0 else "-"
                    marker = "**" if "frankenstallm" in model else ""
                    e_rows.append([
                        str(rank),
                        f"{marker}`{model}`{marker}",
                        _fmt(elo, 1),
                        f"[{_fmt(ci_l, 1)}, {_fmt(ci_u, 1)}]",
                        str(wins),
                        str(losses),
                        winrate,
                    ])
            a(_md_table(e_headers, e_rows))
            a("")

            # 승패 매트릭스 (상위 모델)
            comparisons = t7_res.get("comparisons", [])
            if comparisons:
                a("### 상위 모델 직접 대결 결과")
                a("")
                top_models = [m for m, _ in sorted_elo]
                win_matrix = {}
                for c in comparisons:
                    ma, mb = c.get("model_a", ""), c.get("model_b", "")
                    w = c.get("winner", "TIE")
                    if ma in top_models and mb in top_models:
                        key = (ma, mb)
                        if key not in win_matrix:
                            win_matrix[key] = {"A": 0, "B": 0, "TIE": 0}
                        win_matrix[key][w] = win_matrix[key].get(w, 0) + 1

                if win_matrix:
                    a("| 대결 | 결과 |")
                    a("|---|---|")
                    for (ma, mb), results in sorted(win_matrix.items()):
                        a_wins = results.get("A", 0)
                        b_wins = results.get("B", 0)
                        ties = results.get("TIE", 0)
                        if a_wins > b_wins:
                            a(f"| `{ma}` vs `{mb}` | **{ma}** 승 ({a_wins}-{b_wins}-{ties}) |")
                        elif b_wins > a_wins:
                            a(f"| `{ma}` vs `{mb}` | **{mb}** 승 ({b_wins}-{a_wins}-{ties}) |")
                        else:
                            a(f"| `{ma}` vs `{mb}` | 무승부 ({a_wins}-{b_wins}-{ties}) |")
                    a("")
    else:
        a("*Track 7 데이터 없음*")
    a("")

    # ── 종합 분석 ──
    a("---")
    a("## 종합 분석")
    a("")
    a("### Frankenstallm vs 비교 모델")
    a("")

    # 자동 분석 — 각 트랙에서 frankenstallm이 비교 모델보다 높은 트랙 수 등
    frank_better = 0
    comp_better = 0
    track_winners = {}

    for track_name, track_data in track_results.items():
        summary = track_data.get("summary", {})
        if not summary:
            continue

        frank_scores = []
        comp_scores = []
        for model, data in summary.items():
            score = 0
            if isinstance(data, dict):
                # 다양한 스코어 키에서 추출 시도
                for key in ["mean", "accuracy", "avg_score", "elo", "overall_mean",
                           "python_pass1", "avg_decode_tok_s"]:
                    if key in data:
                        score = float(data[key])
                        break
                if score == 0:
                    # 값들의 평균
                    nums = [float(v) for v in data.values() if isinstance(v, (int, float))]
                    score = np.mean(nums) if nums else 0
            elif isinstance(data, (int, float)):
                score = float(data)

            if any(f in model for f in ["frankenstallm"]):
                frank_scores.append(score)
            else:
                comp_scores.append(score)

        if frank_scores and comp_scores:
            frank_avg = np.mean(frank_scores)
            comp_avg = np.mean(comp_scores)
            if frank_avg > comp_avg:
                frank_better += 1
                track_winners[track_name] = f"Frankenstallm ({frank_avg:.3f} vs {comp_avg:.3f})"
            else:
                comp_better += 1
                track_winners[track_name] = f"비교 모델 ({comp_avg:.3f} vs {frank_avg:.3f})"

    if track_winners:
        a(f"- Frankenstallm 우세 트랙: **{frank_better}개**")
        a(f"- 비교 모델 우세 트랙: **{comp_better}개**")
        a("")
        for track, winner in sorted(track_winners.items()):
            a(f"- **{track}**: {winner}")
        a("")

    a("### v1 vs v2 비교")
    a("")
    a("*각 양자화 수준에서 v1과 v2의 성능 차이를 확인하세요 (스코어카드 참조)*")
    a("")

    a("### 양자화 영향")
    a("")
    a("*f16 → Q8_0 → Q4_K_M으로 양자화할 때 정확도 저하를 Track 1, Track 4에서 확인하세요*")
    a("")

    # ── 평가 설정 ──
    a("---")
    a("## 평가 설정")
    a("")
    a("```json")
    a(json.dumps({
        "models": config.ALL_MODELS,
        "sampling_params": config.SAMPLING_PARAMS,
        "benchmark_sampling": config.BENCHMARK_SAMPLING,
        "timeouts": {m: config.MODEL_TIMEOUTS.get(m) for m in config.ALL_MODELS},
        "cooldown_between_models": config.COOLDOWN_BETWEEN_MODELS,
        "max_retries": config.MAX_RETRIES,
    }, ensure_ascii=False, indent=2))
    a("```")
    a("")
    a("---")
    a(f"*Generated by FRANKENSTALLM Evaluation Framework v1.0 — {timestamp}*")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"📝 Markdown 상세 리포트 생성: {output_path}")
    return output_path
