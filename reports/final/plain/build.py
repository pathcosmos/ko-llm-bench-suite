"""일반인용 보고서(간단판·상세판) 생성기. data.json(final_data) → 정적 HTML 조각 2개. 사용: python reports/final/plain/build.py"""
import json, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from kobench.standard import final_report as fr

d = json.load(open(HERE.parent / "data.json"))
M = fr.uni_scores(d)["models"]
perf = {}
for r in d["perf"]:
    if not r.get("error") and r.get("engine") in ("vllm", "evafrill") and r.get("run") in ("prefix_cache_on", "static_batch") \
            and r.get("workload") == "w1" and r["concurrency"] in (1, 64):
        perf.setdefault(r["model"], {})[r["concurrency"]] = r["output_tok_per_s"]
rows = []
for m, o in M.items():
    if "comp" not in o:
        continue
    meta = d["models"].get(m, {})
    rows.append(dict(id=m, name=fr.display(m), kind=o["kind"], params=meta.get("params_b"), active=meta.get("active_b"),
                     user=meta.get("owner") == "self", comp=o["comp"][0], areas={a: v[0] for a, v in o["areas"].items()},
                     c1=perf.get(m, {}).get(1), c64=perf.get(m, {}).get(64)))
rows.sort(key=lambda x: -x["comp"])
for i, r in enumerate(rows, 1):
    r["rank"] = i
N = len(rows)
AREAS = ["지식", "전문·의료", "사실성·편향", "수학", "코드", "지시 수행", "대화·생성 품질", "다턴", "언어 모델링"]
PLAIN = {"지식": "상식·학과 지식 문제", "전문·의료": "의료 국가시험 문제", "사실성·편향": "사실 맞히기·편견 피하기", "수학": "수학 문장제",
         "코드": "프로그래밍", "지시 수행": "시킨 대로 따라 하기", "대화·생성 품질": "대화·글쓰기 품질", "다턴": "여러 번 주고받는 대화",
         "언어 모델링": "한국어 글 이어쓰기"}
pub = [r for r in rows if not r["user"]]
def group(r):
    if r["user"]: return "user"
    return "moe" if (r["params"] or 0) >= 15 else "small"
GL = {"user": "우리가 만든 모델", "small": "공개 소형 모델(약 2~8B)", "moe": "공개 대형 모델(전체 21~35B, 계산에는 약 3B만 사용)"}

CSS = """
:root{--bg:#f5f7f8;--surface:#ffffff;--ink:#16212b;--ink2:#46535f;--mute:#6b7884;--line:#dde3e8;--user:#d4480f;--small:#2f6f9f;--moe:#9db3c4;--good:#1f7a4d;--soft:#fdeee6;--heat:#2f6f9f;--track:#e8edf1}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f161c;--surface:#17212a;--ink:#e8eef2;--ink2:#b3c0ca;--mute:#8a98a4;--line:#2a3742;--user:#ff8a52;--small:#6fb0e0;--moe:#4a6073;--good:#5fd09a;--soft:#2a1d17;--heat:#6fb0e0;--track:#22303b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f161c;--surface:#17212a;--ink:#e8eef2;--ink2:#b3c0ca;--mute:#8a98a4;--line:#2a3742;--user:#ff8a52;--small:#6fb0e0;--moe:#4a6073;--good:#5fd09a;--soft:#2a1d17;--heat:#6fb0e0;--track:#22303b;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:"Noto Sans KR",system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:16px;line-height:1.7;padding-inline:max(16px,env(safe-area-inset-left,0px)) max(16px,env(safe-area-inset-right,0px));word-break:keep-all}
*{box-sizing:border-box}
.wrap{max-width:860px;margin:0 auto;padding-block:32px 64px;display:flex;flex-direction:column;gap:40px}
h1,h2,h3{font-family:"Noto Sans KR",system-ui,sans-serif;line-height:1.3;text-wrap:balance;margin:0}
h1{font-size:clamp(1.7rem,5vw,2.4rem);font-weight:800;letter-spacing:-.02em}
h2{font-size:1.4rem;font-weight:700;letter-spacing:-.01em}
h3{font-size:1.05rem;font-weight:700}
p{margin:0}
.eyebrow{font-size:.78rem;letter-spacing:.08em;color:var(--mute);font-weight:600;text-transform:uppercase}
.lead{font-size:1.15rem;color:var(--ink2)}
.sec{display:flex;flex-direction:column;gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px}
.hero{display:flex;flex-direction:column;gap:14px}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.verdict .card h3{margin-bottom:6px}
.big{font-size:2.2rem;font-weight:800;line-height:1.1;font-variant-numeric:tabular-nums}
.good .big{color:var(--good)} .bad .big{color:var(--user)}
.note{font-size:.9rem;color:var(--mute)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.85rem;color:var(--ink2)}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.rank{display:flex;flex-direction:column;gap:2px}
.rk{display:grid;grid-template-columns:2rem minmax(7rem,12rem) 1fr 3.2rem;gap:8px;align-items:center;font-size:.88rem;padding:2px 0}
.rk .n{color:var(--mute);text-align:right;font-variant-numeric:tabular-nums}
.rk .nm{overflow-wrap:anywhere;line-height:1.25}
.rk.u .nm{font-weight:700;color:var(--user)}
.rk .tr{height:14px;background:var(--track);border-radius:3px;position:relative}
.rk .b{height:100%;border-radius:3px}
.rk .v{font-variant-numeric:tabular-nums;text-align:right;color:var(--ink2)}
.rk.u{background:var(--soft);border-radius:6px}
.rk.gap{color:var(--mute);justify-content:center;padding-left:2.5rem;font-size:.8rem}
.ar{display:grid;grid-template-columns:minmax(8.5rem,11rem) 1fr;gap:6px 12px;align-items:center;padding:10px 0;border-top:1px solid var(--line)}
.ar:first-of-type{border-top:0}
.ar .lab b{display:block;font-size:.95rem}.ar .lab span{font-size:.78rem;color:var(--mute)}
.bars{display:flex;flex-direction:column;gap:3px}
.bar{display:grid;grid-template-columns:1fr 2.8rem;gap:8px;align-items:center;font-size:.8rem}
.bar .tr{height:10px;background:var(--track);border-radius:3px}
.bar .b{height:100%;border-radius:3px}
.bar em{font-style:normal;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink2)}
.tag{font-size:.7rem;color:var(--mute)}
.speed .rk{grid-template-columns:minmax(7rem,12rem) 1fr 4.5rem}
.faq{display:flex;flex-direction:column;gap:10px}
.faq b{display:block}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--ink2);font-size:.78rem;vertical-align:bottom}
td.h{background:color-mix(in srgb,var(--heat) calc(var(--v)*1%),var(--surface));color:var(--ink)}
tr.u td{font-weight:700}
tr.u td:first-child{color:var(--user)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
.scroll table{min-width:760px}
.callout{border-left:4px solid var(--user);background:var(--soft);padding:12px 16px;border-radius:0 8px 8px 0}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.foot{font-size:.85rem;color:var(--mute);display:flex;flex-direction:column;gap:6px}
ul{margin:0;padding-left:1.2rem;display:flex;flex-direction:column;gap:6px}
.kv{display:grid;grid-template-columns:minmax(9rem,13rem) 1fr;gap:4px 14px;font-size:.92rem}
.kv dt{color:var(--mute)} .kv dd{margin:0}
.nav{font-size:.85rem;color:var(--mute)}
@media (max-width:560px){.rk{grid-template-columns:1.6rem 1fr 3rem}.rk .tr{grid-column:2/4}.rk .v{grid-row:1;grid-column:3}.rk .nm{grid-column:2}.ar{grid-template-columns:1fr}.speed .rk{grid-template-columns:1fr 4.5rem}.speed .rk .tr{grid-column:1/3}.kv{grid-template-columns:1fr}.kv dt{margin-top:6px}}
"""
HEAD = '<meta name="claude:viewport-fit-sides" content="cover">\n<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&display=swap">\n'

def col(r): return {"user": "var(--user)", "small": "var(--small)", "moe": "var(--moe)"}[group(r)]
def legend():
    return '<div class="legend">' + "".join(f'<span><i style="background:var(--{k})"></i>{GL[k]}</span>' for k in ("user", "small", "moe")) + "</div>"
def rk_row(r, mx=0.6):
    w = 100 * r["comp"] / mx
    return (f'<div class="rk{" u" if r["user"] else ""}"><span class="n">{r["rank"]}</span><span class="nm">{r["name"]}</span>'
            f'<span class="tr"><span class="b" style="display:block;width:{w:.1f}%;background:{col(r)}"></span></span><span class="v">{100*r["comp"]:.0f}</span></div>')
def rank_full():
    return '<div class="rank">' + "".join(rk_row(r) for r in rows) + "</div>"
def rank_short():
    keep = [1, 2, 8, 10, 17, 24, 30] + [r["rank"] for r in rows if r["user"]]
    keep = sorted(set(keep)); out = []; prev = 0
    for r in rows:
        if r["rank"] in keep:
            if r["rank"] - prev > 1:
                out.append(f'<div class="rk gap">… {r["rank"]-prev-1}개 모델 생략 …</div>')
            out.append(rk_row(r)); prev = r["rank"]
    return '<div class="rank">' + "".join(out) + "</div>"
def med(a): return statistics.median(r["areas"][a] for r in pub)
def top(a): return max(r["areas"][a] for r in pub)
def jan(): return next(r for r in rows if r["id"] == "jansori-bf16")
def area_block(users):
    out = []
    for a in AREAS:
        bars = []
        for r in users:
            bars.append((r["name"], r["areas"][a], "var(--user)"))
        bars += [("공개 모델 중간값", med(a), "var(--small)"), ("공개 모델 최고", top(a), "var(--moe)")]
        bs = "".join(f'<div class="bar"><span><span class="tag">{n}</span><span class="tr" style="display:block"><span class="b" style="display:block;width:{100*v:.1f}%;background:{c}"></span></span></span><em>{100*v:.0f}</em></div>' for n, v, c in bars)
        out.append(f'<div class="ar"><div class="lab"><b>{PLAIN[a]}</b><span>{a}</span></div><div class="bars">{bs}</div></div>')
    return "".join(out)
def speed_rows(ids):
    mx = max(r["c64"] for r in rows if r["c64"])
    out = []
    for i in ids:
        r = next(x for x in rows if x["id"] == i)
        out.append(f'<div class="rk{" u" if r["user"] else ""}"><span class="nm">{r["name"]}</span><span class="tr"><span class="b" style="display:block;width:{100*r["c64"]/mx:.1f}%;background:{col(r)}"></span></span><span class="v">{r["c64"]/1000:.1f}k</span></div>')
    return '<div class="rank speed">' + "".join(out) + "</div>"
USERS = [next(r for r in rows if r["id"] == i) for i in ("jansori-bf16", "frankenstallm-bf16", "evafrill-mo-3b-slerp", "evafrill-mo-3b-pretrain") if any(x["id"] == i for x in rows)]
if len(USERS) < 4: USERS = [r for r in rows if r["user"]]
u31 = {r["name"]: r["rank"] for r in rows if r["user"]}
SPEED_IDS = ["kanana-1.5-2.1b", "jansori-bf16", "frankenstallm-bf16", "qwen2.5-3b-instruct", "llama-3.2-3b-instruct", "qwen3-4b-2507", "gemma-4-e4b-it", "evafrill-mo-3b-slerp", "qwen3.6-35b-a3b"]
SPEED_IDS = [i for i in SPEED_IDS if any(x["id"] == i for x in rows)]
J = jan(); lm = J["areas"]["언어 모델링"]
nav = '<p class="nav">{}</p>'

# ---------------- 간단판 ----------------
simple = f"""<title>한국어 소형 AI 성적표</title>
<style>{CSS}</style>
{HEAD}
<main class="wrap">
<header class="hero">
<p class="eyebrow">직접 만든 한국어 AI 3종 · 2026-11-24</p>
<h1>우리가 만든 AI는 {N}개 중 {min(u31.values())}~{max(u31.values())}위였습니다</h1>
<p class="lead">글을 자연스럽게 이어 쓰는 능력과 답하는 속도는 상위권입니다. 문제를 풀고, 시킨 일을 하고, 대화하는 능력은 아직 많이 부족합니다.</p>
</header>

<section class="verdict">
<div class="card good"><h3>잘하는 것</h3><p class="big">상위권</p><p>한국어 글 이어쓰기와 답변 속도. JANSORI는 글 이어쓰기에서 같은 종류(base) 15개 중 4위입니다.</p></div>
<div class="card bad"><h3>부족한 것</h3><p class="big">최하위권</p><p>지식 문제, 수학, 프로그래밍, 지시 따르기, 여러 번 주고받는 대화에서 비교한 모델 중 거의 꼴찌입니다.</p></div>
</section>

<section class="sec">
<h2>무엇을 비교했나요</h2>
<p>직접 만든 AI 3종(JANSORI, frankenstallm, EVAFRILL-Mo)을 인터넷에 공개된 AI 30개와 같은 문제, 같은 방식으로 풀게 했습니다. EVAFRILL-Mo는 학습 단계가 다른 두 버전(pretrain, slerp)을 따로 재서 우리 쪽 항목은 4개입니다.</p>
<p>공개 모델에는 우리 것과 비슷한 크기(약 30억 개 매개변수)의 소형 모델과, 전체 크기는 크지만 실제 계산에는 약 30억 개만 쓰는 대형 모델이 함께 들어 있습니다.</p>
<p class="note">매개변수는 AI 안의 조절 나사 수입니다. 많을수록 대체로 더 많이 배울 수 있습니다.</p>
</section>

<section class="sec">
<h2>종합 순위</h2>
<p>9개 영역(지식, 의료, 사실성, 수학, 프로그래밍, 지시 수행, 대화 품질, 여러 번 주고받는 대화, 글 이어쓰기)을 같은 비중으로 평균한 100점 만점 점수입니다.</p>
{legend()}
{rank_short()}
<p class="note">1위 qwen3.6-35b-a3b는 58점, 우리 모델은 15~18점입니다. 공개 모델 중 가장 낮은 llama-3.2-3b도 25점입니다.</p>
</section>

<section class="sec">
<h2>영역별로 보면</h2>
<p>주황색이 JANSORI, 파란색이 공개 모델 중간값, 회색이 공개 모델 최고점입니다. 숫자는 100점 만점입니다.</p>
<div class="card">{area_block([J])}</div>
<p class="note">글 이어쓰기는 JANSORI({100*lm:.0f}점)가 공개 모델 중간값(76점)보다 높습니다. 나머지 영역은 1~15점으로, 공개 모델 중간값(11~67점)에 크게 못 미칩니다.</p>
</section>

<section class="sec">
<h2>속도는 강점입니다</h2>
<p>한 번에 64명이 질문할 때 1초에 만들어내는 글자 조각(토큰) 수입니다. 높을수록 많은 사람에게 동시에 빨리 답할 수 있습니다.</p>
{legend()}
{speed_rows(SPEED_IDS)}
<p class="note">JANSORI와 frankenstallm은 비슷한 크기의 공개 모델 12개 중 1위와 3위입니다. EVAFRILL-Mo는 구조가 달라 직접 만든 실행기로 측정했고 약 4분의 1 속도입니다. 같은 컴퓨터(H200 1장)에서 쟀습니다.</p>
</section>

<section class="sec">
<h2>설정을 바꾸면 나아지나요</h2>
<p>같은 말을 반복하는 문제는 설정으로 크게 줄었습니다(frankenstallm 9.1%에서 1.2%, EVAFRILL 31.6%에서 2.2%). 하지만 문제를 맞히는 비율은 오르지 않았고 일부는 오히려 내려갔습니다. 설정 조정만으로 실력 차이를 메우기는 어렵습니다.</p>
</section>

<section class="sec">
<h2>읽을 때 알아둘 점</h2>
<ul>
<li>순위는 모든 모델에 같은 형식의 문제를 낸 조건입니다. 대화 전용 모델에는 불리한 조건이라, 모델마다 최선의 조건으로 잰 표와는 순위가 다를 수 있습니다.</li>
<li>JANSORI는 대화용이 아니라 글을 이어 쓰는 데만 학습된 기초 모델(base)입니다. 시킨 일을 하는 능력은 원래 목표가 아닙니다.</li>
<li>프로그래밍은 정해진 형식을 따르지 못하면 0점이 되어, 공개 모델 일부도 0에 가깝습니다.</li>
<li>왜 점수가 낮은지, 학습 데이터나 크기의 영향은 이번 평가에서 따로 확인하지 않았습니다.</li>
</ul>
</section>

<footer class="foot">
<p>자세한 방법과 모델별 전체 수치는 상세판에 있습니다. 차트를 직접 눌러 볼 수 있는 전체 보고서는 https://pathcosmos.github.io/ko-llm-bench-suite/ 에 있습니다.</p>
<p>점수는 영역별 정규화 점수(무작위로 찍어도 나오는 점수를 뺀 값)의 평균이며, 오차 범위(95% 신뢰구간)는 약 ±0.01입니다.</p>
</footer>
</main>
"""

# ---------------- 상세판 ----------------
def heat():
    head = "<tr><th>순위 · 모델</th><th>종합</th>" + "".join(f"<th>{PLAIN[a]}</th>" for a in AREAS) + "</tr>"
    body = ""
    for r in rows:
        cells = "".join(f'<td class="h" style="--v:{min(100,100*r["areas"][a]*1.0):.0f}">{100*r["areas"][a]:.0f}</td>' for a in AREAS)
        body += f'<tr class="{"u" if r["user"] else ""}"><td>{r["rank"]}. {r["name"]}</td><td><b>{100*r["comp"]:.0f}</b></td>{cells}</tr>'
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{body}</tbody></table></div>'
def speed_table():
    ids = [r for r in rows if r["c64"]]
    ids.sort(key=lambda r: -r["c64"])
    body = "".join(f'<tr class="{"u" if r["user"] else ""}"><td>{r["name"]}</td><td>{r["params"]}B / {r["active"]}B</td><td>{r["c1"]:.0f}</td><td>{r["c64"]/1000:.1f}k</td></tr>' for r in ids)
    return f'<div class="scroll"><table style="min-width:520px"><thead><tr><th>모델</th><th>전체 / 계산에 쓰는 크기</th><th>혼자 쓸 때 (토큰/초)</th><th>64명 동시 (토큰/초)</th></tr></thead><tbody>{body}</tbody></table></div>'

detail = f"""<title>한국어 소형 AI 성적표 상세판</title>
<style>{CSS}</style>
{HEAD}
<main class="wrap">
<header class="hero">
<p class="eyebrow">상세판 · JANSORI · frankenstallm · EVAFRILL-Mo · 2026-11-24</p>
<h1>직접 만든 한국어 AI 3종을 공개 모델 30개와 비교했습니다</h1>
<p class="lead">무엇을 어떤 조건에서 쟀고, 어디가 강하고 어디가 약한지, 숫자와 함께 풀어서 설명합니다. 용어는 맨 아래 해설에 모았습니다.</p>
<p class="nav">간단판은 「한국어 소형 AI 성적표」입니다. 클릭해서 볼 수 있는 전체 보고서: https://pathcosmos.github.io/ko-llm-bench-suite/</p>
</header>

<section class="sec">
<h2>한눈에 보기</h2>
<div class="verdict">
<div class="card"><h3>종합 순위</h3><p class="big">{min(u31.values())}~{max(u31.values())}위 / {N}</p><p>우리 모델 4개 항목의 종합 점수는 15~18점입니다. 1위는 58점, 공개 모델 최하위는 25점입니다.</p></div>
<div class="card good"><h3>강점</h3><p class="big">글 이어쓰기 · 속도</p><p>JANSORI의 한국어 글 이어쓰기는 같은 종류 15개 중 4위. 64명 동시 처리량은 비슷한 크기 12개 중 1위와 3위.</p></div>
<div class="card bad"><h3>약점</h3><p class="big">풀기 · 대화</p><p>지식 문제, 수학, 코드, 지시 수행, 대화, 여러 번 주고받는 대화는 모두 비교군 최하위권.</p></div>
</div>
</section>

<section class="sec">
<h2>1. 대상과 비교군</h2>
<dl class="kv">
<dt>JANSORI</dt><dd>30억 매개변수 일반 구조(Llama 호환). 글을 이어 쓰도록만 학습한 기초 모델(base)이라 대화 틀이 없습니다.</dd>
<dt>frankenstallm</dt><dd>30억 매개변수 일반 구조. 지시를 따르도록 추가 학습한 대화형(instruct) 모델입니다.</dd>
<dt>EVAFRILL-Mo</dt><dd>29.4억 매개변수의 하이브리드 구조(Mamba-2와 Transformer 혼합). 학습 초기(pretrain)와 최종 합성본(slerp) 두 버전을 따로 측정했습니다. 일반 도구로는 돌릴 수 없어 자체 실행기를 만들었습니다.</dd>
<dt>공개 소형 모델</dt><dd>Qwen, kanana, EXAONE, SmolLM3, Gemma, Llama, Phi 등 약 2~8B급.</dd>
<dt>공개 대형 모델</dt><dd>전체는 21~35B이지만 한 번에 계산하는 부분은 약 3B인 MoE(전문가 혼합) 모델. 같은 계산량에서 얼마나 잘하는지 보려고 넣었습니다.</dd>
</dl>
<p class="note">비교는 base끼리, instruct끼리가 원칙입니다. 같은 컴퓨터(H200 1장), 같은 정밀도(BF16), 같은 길이 제한(4096 토큰)에서 쟀고, 공개 모델은 버전(revision)을 고정했습니다.</p>
</section>

<section class="sec">
<h2>2. 어떻게 쟀나요</h2>
<ul>
<li><b>객관식</b>: AI가 보기마다 얼마나 그럴듯하다고 보는지(확률)를 비교해 가장 높은 보기를 답으로 봅니다. 글자 파싱 실수나 위치 편향이 끼지 않습니다.</li>
<li><b>글 이어쓰기</b>: 실제 한국어 문서를 얼마나 잘 예측하는지 재는 지표(bits-per-byte)입니다. 낮을수록 좋고, 토크나이저가 달라도 공정하게 비교하려고 글자 바이트 기준으로 환산합니다.</li>
<li><b>생성(직접 쓰게 하기)</b>: 수학, 지시 따르기, 의료 문제는 정답 규칙으로 채점합니다. 대화와 글쓰기 품질은 큰 AI(Qwen3.6-35B-A3B)가 채점자(judge)로 점수를 매겼습니다.</li>
<li><b>코드</b>: 만들어진 코드를 격리된 곳에서 실제로 실행해 테스트를 통과하는지 봅니다.</li>
<li><b>다턴 시나리오</b>: 3~5번 주고받는 대화 300개(수학 조건 변경, 일정 기억, 지시 유지, 모름 인정 등 10범주)를 규칙으로 채점합니다.</li>
<li><b>속도</b>: 서버를 띄워 질문을 보내며 첫 글자까지 걸린 시간, 초당 생성량, 메모리를 재었습니다.</li>
</ul>
<div class="callout"><p><b>설정은 미리 정한 규칙으로만 골랐습니다.</b> 별도의 연습용(dev) 문제에서 설정을 고르고 시험(test) 결과를 보고 바꾸지 않았습니다. 규칙은 측정 전에 파일로 등록했습니다.</p></div>
</section>

<section class="sec">
<h2>3. 모든 모델을 같은 조건으로 세운 종합 순위</h2>
<p>대화 틀(chat 템플릿) 없이 모든 모델에 같은 원문 프롬프트를 넣은 통일 비교입니다. 대화 모델이 유리하거나 불리해지지 않게 하려는 조건입니다. 9개 영역을 같은 비중으로 평균했고, 객관식은 찍어서 맞히는 점수를 뺀 값입니다.</p>
{legend()}
{rank_full()}
<p class="note">1~4위는 모두 대형(MoE) 모델입니다. 소형 중 최고는 qwen3-4b-2507(49점, 8위)이고 exaone-3.5-2.4b(47점)가 뒤를 잇습니다. 이 조건에서는 base와 instruct 점수 차이가 작습니다. gemma-4-e4b-it(29위)는 대화 틀이 없는 조건에 약해 모델별 최선 조건 순위와 크게 다릅니다. 종합 점수의 오차 범위는 약 ±0.01입니다.</p>
</section>

<section class="sec">
<h2>4. 영역별 점수 전체표</h2>
<p>100점 만점, 칸이 진할수록 높은 점수입니다. 표는 옆으로 밀어 볼 수 있습니다.</p>
{heat()}
<p class="note">사실성·편향 영역(KoBBQ, TruthfulQA, KoSimpleQA)은 모든 모델이 2~26점으로 낮아 순위를 가르는 힘이 작습니다.</p>
</section>

<section class="sec">
<h2>5. 우리 모델 4항목, 영역별로</h2>
<p>주황색이 각 우리 모델, 파란색이 공개 모델 중간값, 회색이 공개 모델 최고점입니다.</p>
<div class="card">{area_block(USERS)}</div>
</section>

<section class="sec">
<h2>6. 영역별 해설</h2>
<div class="two">
<div class="card"><h3>객관식 지식</h3><p>KMMLU, HAE-RAE, Belebele 등 지식 문제는 우리 모델 3종 모두 무작위로 찍는 수준 근처입니다. 신호가 있는 곳은 KoBEST(JANSORI 58.7)와 영어 HellaSwag·ARC(JANSORI 52.6, 38.0. 나머지 둘은 약 30, 23)뿐입니다.</p></div>
<div class="card"><h3>KMMLU 20.4가 무작위(25)보다 낮은 이유</h3><p>정답은 A 9.9%, B 29.7%, C 29.9%, D 30.5%로 A가 드뭅니다. 그런데 JANSORI는 A 50.5%, B 40.8%로 글자에 쏠려 답했습니다. 정답 분포와 상관없이 이렇게 답하면 기대 정확도가 19.7%이고 관측값과 같습니다. 지식 신호가 없고 글자 편향만 있다는 뜻입니다.</p></div>
<div class="card"><h3>글 이어쓰기</h3><p>한국어 bits-per-byte는 JANSORI 0.752, frankenstallm 0.795, EVAFRILL 0.82입니다(낮을수록 좋음). 공개 최상위 kanana-2-30b-a3b-base는 0.662입니다. 통일 비교의 환산 점수는 JANSORI {100*lm:.0f}점입니다.</p></div>
<div class="card"><h3>직접 쓰게 하는 과제</h3><p>지시 따르기(IFEval-Ko)는 frankenstallm 18.1%, EVAFRILL 14.6%로 공개 소형 모델 35~71%에 못 미칩니다. 의료 문제는 14.7%와 9.6%, 수학(HRM8K)은 거의 0%(요구된 \\boxed 답 형식을 99% 지키지 못함), 코드는 0%입니다.</p></div>
<div class="card"><h3>대화·글쓰기(judge)</h3><p>KoMT 점수는 두 모델 모두 1.1점(10점 만점)이고 공개 모델은 2.4~8.0점입니다. 기준 모델(llama-3.2-3b-instruct)과 맞붙은 승률은 0.16과 0.11입니다(0.5가 동률).</p></div>
<div class="card"><h3>여러 번 주고받는 대화</h3><p>300개 시나리오를 끝까지 통과한 비율은 우리 모델 0%입니다(턴 통과 0.10, 0.09). 공개 소형 모델은 턴 0.53~0.84, 시나리오 0.30~0.63입니다. frankenstallm은 이전 답을 그대로 복사하는 비율이 16~22%입니다.</p></div>
</div>
</section>

<section class="sec">
<h2>7. 설정을 바꾸면 나아지나요</h2>
<p>모델마다 연습용 문제로 디코딩 설정(반복 억제, 샘플링, 한국어 시스템 지시 등)을 골랐습니다. 반복 루프는 크게 줄었습니다.</p>
<div class="two">
<div class="card"><h3>frankenstallm</h3><p class="big">9.1% → 1.2%</p><p>반복 루프 비율(repetition penalty 1.1)</p></div>
<div class="card"><h3>EVAFRILL-Mo</h3><p class="big">31.6% → 2.2%</p><p>반복 루프 비율(공식 권장 샘플링)</p></div>
</div>
<p>그러나 시험 문제에서 품질은 그대로이거나 떨어졌습니다(frankenstallm TruthfulQA-ko −6.6%p, EVAFRILL KoBBQ −24.3%p). EVAFRILL과 공개 llama-3.2-3b-instruct는 오작동 제한을 모두 만족하는 설정이 없었습니다. 설정으로는 모자란 부분을 메우지 못합니다.</p>
</section>

<section class="sec">
<h2>8. 속도와 메모리</h2>
<p>JANSORI와 frankenstallm은 표준 구조라 vLLM에서 혼자 쓸 때 371 토큰/초, 64명 동시에 약 16.2~16.5k 토큰/초입니다. EVAFRILL 자체 실행기는 279와 4.2k, 대형(MoE) 모델은 64명 동시에 3.5~3.7k입니다. MoE가 느린 것은 계산은 3B만 하지만 가중치 57 GiB 전체를 단계마다 읽어야 하기 때문입니다.</p>
{legend()}
{speed_table()}
<h3>GGUF 양자화(파일을 줄인 버전)</h3>
<p>Q8_0은 원본(f16)과 차이가 오차 범위 안입니다. Q4_K_M은 한국어 bpb가 JANSORI +0.005, frankenstallm +0.006 늘고 KoBEST가 각각 −0.9%p, −0.4%p 내려가 손실이 작습니다. EVAFRILL은 구조상 GGUF로 만들 수 없습니다.</p>
</section>

<section class="sec">
<h2>9. 읽을 때 주의</h2>
<ul>
<li>JANSORI는 base라 대화 평가가 원래 목표 밖입니다. 비교는 가능하지만 해석에 이 점을 반영해야 합니다.</li>
<li>코드는 base 이어쓰기 형식으로 통일해 형식을 못 맞춘 모델의 MBPP+가 0에 가깝습니다(ERNIE-4.5 instruct는 코드 대신 설명문, nemotron-3.5는 378개 중 230개가 문법 오류). 형식 규칙은 측정 전에 정해 바꾸지 않았습니다.</li>
<li>공개 벤치마크 문제가 모델 학습에 섞였을 가능성(오염)은 통제하지 못했습니다.</li>
<li>점수가 낮은 원인(학습 데이터 양, 모델 크기, 학습 방법 등)은 이번 평가로 구분할 수 없습니다.</li>
<li>Qwen3.5-4B의 사고(thinking) 모드는 처음 greedy로 실행되어 생각이 잘리는 일이 많았고, 공식 권장 샘플링으로 다시 쟀더니 수학 75.0%에서 79.9%, 지시 수행 31.9%에서 79.2%로 올랐습니다. 반대로 LogicKor와 K2 생성은 내려갔습니다.</li>
</ul>
</section>

<section class="sec">
<h2>용어 해설</h2>
<dl class="kv">
<dt>base / instruct</dt><dd>글만 이어 쓰게 학습한 기초 모델 / 지시를 따르도록 추가 학습한 대화형 모델</dd>
<dt>MoE (A3B)</dt><dd>여러 전문가 중 일부만 골라 계산하는 구조. A3B는 한 번에 쓰는 크기가 약 30억이라는 뜻</dd>
<dt>토큰</dt><dd>AI가 글을 자르는 단위. 한글 한두 글자 정도</dd>
<dt>bits-per-byte</dt><dd>글을 얼마나 잘 예측하는지. 낮을수록 좋음</dd>
<dt>judge</dt><dd>답변 품질을 점수로 매기는 채점용 AI</dd>
<dt>dev / test</dt><dd>설정을 고르는 연습 문제 / 최종 성적을 내는 시험 문제</dd>
<dt>토큰/초</dt><dd>1초에 만들어내는 글자 조각 수. 높을수록 빠름</dd>
<dt>95% 신뢰구간</dt><dd>같은 측정을 다시 하면 점수가 대부분 들어올 범위</dd>
</dl>
</section>

<footer class="foot">
<p>측정 환경: H200 1장, vLLM 0.33 BF16(GGUF는 llama.cpp b12084), 문맥 4096(thinking은 16384). 모든 코드, 설정, 요약 결과는 GitHub pathcosmos/ko-llm-bench-suite에 있습니다.</p>
</footer>
</main>
"""
(HERE / "simple.html").write_text(simple, encoding="utf-8")
(HERE / "detail.html").write_text(detail, encoding="utf-8")
print("ok", len(simple), len(detail), N, [ (r['name'],r['rank']) for r in USERS])
