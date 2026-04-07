#!/home/lanco/ai-env/bin/python3
"""
FRANKENSTALLM 평가 — 보고서용 추가 시각화 (Chart 19~26)
"""

import json
import itertools
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

for font in ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.fontManager.addfont(
            matplotlib.font_manager.findfont(font))
        plt.rcParams['font.family'] = font
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

RESULTS = Path("results/full_results_20260407_173207.json")
OUTPUT = Path("reports/visualizations")
OUTPUT.mkdir(parents=True, exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)

t6_results = []
for ckpt in ['track6_performance_checkpoint.json', 'track6_v2quant_checkpoint.json']:
    p = Path('results') / ckpt
    if p.exists():
        with open(p) as f:
            t6_results.extend(json.load(f).get('results', []))

SHORT = {
    'frankenstallm-3b:latest': 'FS v1', 'frankenstallm-3b:Q8_0': 'FS v1:Q8',
    'frankenstallm-3b-v2:latest': 'FS v2', 'frankenstallm-3b-v2:Q8_0': 'FS v2:Q8',
    'frankenstallm-3b-v2-Q4_K_M': 'FS v2:Q4', 'frankenstallm-3b-v2-Q8_0': 'FS v2:Q8g',
    'frankenstallm-3b-v2-f16': 'FS v2:f16', 'evafrill-mo-3b-slerp': 'EVAFRILL',
    'qwen2.5:3b': 'Qwen2.5', 'gemma3:4b': 'Gemma3', 'phi4-mini': 'Phi4',
    'exaone3.5:2.4b': 'EXAONE3.5', 'llama3.2:3b': 'Llama3.2',
    'llama3.1:8b-instruct-q8_0': 'Llama3.1-8B', 'ingu627/exaone4.0:1.2b': 'EXAONE4',
    'deepseek-r1:1.5b': 'DeepSeek',
}
sn = lambda m: SHORT.get(m, m)
OUR = [m for m in SHORT if 'frankenstallm' in m or 'evafrill' in m]
ALL = sorted(SHORT.keys())
OUR_C, BASE_C = '#e74c3c', '#3498db'
mc = lambda m: OUR_C if m in OUR else BASE_C

# ── 공통 데이터 추출 ────────────────────────────────────────────────────────
# T1
t1_scores = {}
for r in data['track1']['results']:
    s = r.get('scores', {})
    if s:
        t1_scores[r['model']] = np.mean(list(s.values()))

# T2 turn1/turn2
t2_t1_avg, t2_t2_avg = {}, {}
for r in data['track2']['results']:
    m = r['model']
    v1, v2 = r.get('turn1_mean', 0) or 0, r.get('turn2_mean', 0) or 0
    t2_t1_avg.setdefault(m, []).append(v1)
    t2_t2_avg.setdefault(m, []).append(v2)
t2_t1 = {m: np.mean(v) for m, v in t2_t1_avg.items()}
t2_t2 = {m: np.mean(v) for m, v in t2_t2_avg.items()}

# T3
t3_raw = {}
for r in data['track3']['results']:
    s = r.get('judge_score_raw')
    if s is not None:
        t3_raw.setdefault(r['model'], []).append(s)
t3_scores = {m: np.mean(v) for m, v in t3_raw.items()}

# T4
t4_scores = {}
for r in data['track4']['results']:
    s = r.get('scores', {})
    if s:
        t4_scores[r['model']] = np.mean(list(s.values()))

# T5
t5_raw = {}
for r in data['track5']['results']:
    v = r.get('avg_jaccard_similarity', 0) or 0
    t5_raw.setdefault(r['model'], []).append(v)
t5_scores = {m: np.mean(v) for m, v in t5_raw.items()}

# T6 decode
t6_dec = {}
for r in t6_results:
    if r.get('test_type') == 'decode_speed' and r.get('tokens_per_sec', 0) > 0:
        t6_dec.setdefault(r['model'], []).append(r['tokens_per_sec'])
t6_avg = {m: np.mean(v) for m, v in t6_dec.items()}

# T7
t7_elo = {m: s['elo'] for m, s in data['track7']['results']['elo_scores'].items()}

# Composite
def composite(m):
    return (
        t1_scores.get(m, 0) * 15 +
        t2_t1.get(m, 0) / 10 * 20 +
        t3_scores.get(m, 0) / 10 * 15 +
        t4_scores.get(m, 0) * 15 +
        t5_scores.get(m, 0) * 5 * 10 +
        (t7_elo.get(m, 500) - 500) / 1100 * 25
    )

# ══════════════════════════════════════════════════════════════════════════════
# CHART 19: T2 Turn1 vs Turn2 산점도
# ══════════════════════════════════════════════════════════════════════════════
print("[19/26] T2 Turn1 vs Turn2 산점도...")

fig, ax = plt.subplots(figsize=(10, 10))
# y=x reference line
lim = max(max(t2_t1.values()), max(t2_t2.values())) + 0.5
ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Turn1 = Turn2')
ax.fill_between([0, lim], [0, lim], [0, 0], alpha=0.05, color='red')

for m in ALL:
    x, y = t2_t1.get(m, 0), t2_t2.get(m, 0)
    c = mc(m)
    marker = 's' if m in OUR else 'o'
    ax.scatter(x, y, c=c, s=120, marker=marker, edgecolors='black', linewidth=0.5, zorder=3)
    offset = (5, 5) if y < x else (5, -10)
    ax.annotate(sn(m), (x, y), fontsize=8, ha='left', va='center',
               xytext=offset, textcoords='offset points', fontweight='bold')

ax.set_xlabel('Turn 1 평균 점수', fontsize=12)
ax.set_ylabel('Turn 2 평균 점수', fontsize=12)
ax.set_title('T2: Turn1 vs Turn2 성능 (대각선 아래 = 하락)', fontsize=14, fontweight='bold')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

our_patch = mpatches.Patch(color=OUR_C, label='자체 모델')
base_patch = mpatches.Patch(color=BASE_C, label='베이스라인')
ax.legend(handles=[our_patch, base_patch, plt.Line2D([0],[0], linestyle='--', color='k', alpha=0.3, label='Turn1=Turn2')],
         fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT / '19_t2_turn_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 20: T4 판별력 높은 문제 히트맵 (정답률 <50%)
# ══════════════════════════════════════════════════════════════════════════════
print("[20/26] T4 판별 문제 히트맵...")

t4_matrix = {}
all_problems = set()
for r in data['track4']['results']:
    m = r['model']
    t4_matrix[m] = {}
    for dk in ['python_details', 'sql_details', 'debug_details', 'math_details']:
        for d in r.get(dk, []):
            pid = d.get('id', '')
            if dk == 'python_details':
                correct = d.get('pass_at_1', 0)
            else:
                correct = 1.0 if d.get('correct', False) else 0.0
            t4_matrix[m][pid] = correct
            all_problems.add(pid)

# Calculate solve rate per problem
solve_rate = {}
for p in all_problems:
    solved = sum(t4_matrix.get(m, {}).get(p, 0) for m in t4_matrix)
    solve_rate[p] = solved / len(t4_matrix)

# Filter: problems with <50% solve rate AND >0% (exclude impossible)
hard_problems = sorted([p for p, r in solve_rate.items() if 0 < r < 0.5],
                       key=lambda p: solve_rate[p])

# Only models that solve at least 1 hard problem
active_models = sorted(
    [m for m in t4_matrix if any(t4_matrix[m].get(p, 0) > 0 for p in hard_problems)],
    key=lambda m: sum(t4_matrix[m].get(p, 0) for p in hard_problems), reverse=True
)

if hard_problems and active_models:
    matrix = [[t4_matrix[m].get(p, 0) for p in hard_problems] for m in active_models]

    fig, ax = plt.subplots(figsize=(max(14, len(hard_problems) * 0.5), max(6, len(active_models) * 0.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(active_models)))
    ax.set_yticklabels([sn(m) for m in active_models], fontsize=9)
    ax.set_xticks(range(len(hard_problems)))
    ax.set_xticklabels(hard_problems, rotation=60, ha='right', fontsize=7)

    # Add solve rate on top
    for i, p in enumerate(hard_problems):
        ax.text(i, -0.7, f'{solve_rate[p]:.0%}', ha='center', fontsize=7, color='navy')
    ax.text(len(hard_problems)/2, -1.2, '← 문제별 정답률', ha='center', fontsize=9, color='navy')

    ax.set_title('T4: 판별력 높은 문제 (정답률 0%~50%)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6, label='정답(1) / 오답(0)')
    plt.tight_layout()
    plt.savefig(OUTPUT / '20_t4_discriminating_problems.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("  ⚠ 판별 문제 없음, 스킵")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 21: T6 컨텍스트 한계 + 속도 저하
# ══════════════════════════════════════════════════════════════════════════════
print("[21/26] T6 컨텍스트 한계...")

ctx_data = {}
for r in t6_results:
    if r.get('test_type') == 'max_context':
        m = r['model']
        ctx = r.get('input_length', 0) or r.get('context_length', 0)
        err = r.get('error')
        spd = r.get('prefill_tok_s', 0) or 0
        ctx_data.setdefault(m, []).append({'ctx': ctx, 'speed': spd, 'error': err})

# Max successful context per model
max_ctx = {}
speed_degrad = {}
for m, records in ctx_data.items():
    successful = [r for r in records if not r['error'] and r['speed'] > 0]
    if successful:
        max_ctx[m] = max(r['ctx'] for r in successful)
        speeds = sorted(successful, key=lambda r: r['ctx'])
        if len(speeds) >= 2:
            first_speed = speeds[0]['speed']
            last_speed = speeds[-1]['speed']
            speed_degrad[m] = (last_speed - first_speed) / first_speed * 100 if first_speed > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: Max context bar chart with tier coloring
if max_ctx:
    sorted_ctx = sorted(max_ctx.items(), key=lambda x: x[1], reverse=True)
    names = [sn(m) for m, _ in sorted_ctx]
    vals = [v for _, v in sorted_ctx]

    tier_colors = []
    for _, v in sorted_ctx:
        if v >= 4200: tier_colors.append('#2ecc71')    # Tier 1
        elif v >= 4100: tier_colors.append('#3498db')   # Tier 2
        elif v >= 3700: tier_colors.append('#f39c12')   # Tier 3
        else: tier_colors.append('#e74c3c')              # Tier 4

    bars = axes[0].barh(names, vals, color=tier_colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        axes[0].text(v + 20, bar.get_y() + bar.get_height()/2, f'{v}',
                    va='center', fontsize=8)
    axes[0].set_xlabel('최대 컨텍스트 (tokens)')
    axes[0].set_title('T6: 최대 컨텍스트 길이 (계층별)', fontsize=14, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Tier 1 (≥4200)'),
        mpatches.Patch(color='#3498db', label='Tier 2 (≥4100)'),
        mpatches.Patch(color='#f39c12', label='Tier 3 (≥3700)'),
        mpatches.Patch(color='#e74c3c', label='Tier 4 (<3700)'),
    ]
    axes[0].legend(handles=legend_patches, fontsize=9, loc='lower right')

# Right: Speed degradation
if speed_degrad:
    sorted_deg = sorted(speed_degrad.items(), key=lambda x: x[1])
    names_d = [sn(m) for m, _ in sorted_deg]
    vals_d = [v for _, v in sorted_deg]
    colors_d = ['#e74c3c' if v < -20 else '#f39c12' if v < 0 else '#2ecc71' for v in vals_d]

    axes[1].barh(names_d, vals_d, color=colors_d, alpha=0.85)
    axes[1].axvline(x=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('속도 변화 % (장문 vs 단문)')
    axes[1].set_title('T6: 장문 처리 시 속도 변화', fontsize=14, fontweight='bold')
    for i, v in enumerate(vals_d):
        axes[1].text(v + (2 if v >= 0 else -2), i, f'{v:+.1f}%',
                    va='center', fontsize=8, ha='left' if v >= 0 else 'right')

plt.tight_layout()
plt.savefig(OUTPUT / '21_t6_context_limits.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 22: v1 vs v2 쌍대비교 (7트랙)
# ══════════════════════════════════════════════════════════════════════════════
print("[22/26] v1 vs v2 쌍대비교...")

pairs = [
    ('frankenstallm-3b:latest', 'frankenstallm-3b-v2:latest', 'Q4_K_M'),
    ('frankenstallm-3b:Q8_0', 'frankenstallm-3b-v2:Q8_0', 'Q8_0'),
]

tracks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']

def get_track_vals(m):
    return [
        t1_scores.get(m, 0),
        t2_t1.get(m, 0) / 10,
        t3_scores.get(m, 0) / 10,
        t4_scores.get(m, 0),
        t5_scores.get(m, 0) * 10,
        min(t6_avg.get(m, 0) / 100, 1.0),
        (t7_elo.get(m, 500) - 500) / 1100,
    ]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (v1_m, v2_m, qlabel) in enumerate(pairs):
    ax = axes[idx]
    v1_vals = get_track_vals(v1_m)
    v2_vals = get_track_vals(v2_m)

    x = np.arange(len(tracks))
    width = 0.35
    bars1 = ax.bar(x - width/2, v1_vals, width, label=f'v1 ({qlabel})', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + width/2, v2_vals, width, label=f'v2 ({qlabel})', color='#3498db', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tracks)
    ax.set_ylabel('정규화 점수')
    ax.set_title(f'v1 vs v2 비교 ({qlabel})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(v1_vals), max(v2_vals)) * 1.2 + 0.05)

    # Mark winner per track
    for i, (v1, v2) in enumerate(zip(v1_vals, v2_vals)):
        if v1 > v2 + 0.01:
            ax.text(i - width/2, v1 + 0.01, '★', ha='center', fontsize=10, color='#e74c3c')
        elif v2 > v1 + 0.01:
            ax.text(i + width/2, v2 + 0.01, '★', ha='center', fontsize=10, color='#3498db')

fig.suptitle('Frankenstallm v1 vs v2 — 동일 양자화에서의 트랙별 비교', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT / '22_v1_vs_v2_paired.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 23: 한국어 특화 지수 (T4 vs T1+T3)
# ══════════════════════════════════════════════════════════════════════════════
print("[23/26] 한국어 특화 지수...")

fig, ax = plt.subplots(figsize=(10, 10))

for m in ALL:
    x_val = t4_scores.get(m, 0)  # General capability (Code/Math)
    y_val = (t1_scores.get(m, 0) + t3_scores.get(m, 0) / 10) / 2  # Korean capability
    c = mc(m)
    marker = 's' if m in OUR else 'o'
    ax.scatter(x_val, y_val, c=c, s=150, marker=marker, edgecolors='black',
              linewidth=0.5, zorder=3)
    ax.annotate(sn(m), (x_val, y_val), fontsize=9, ha='center', va='bottom',
               xytext=(0, 8), textcoords='offset points', fontweight='bold')

# Quadrant labels
ax.axhline(y=0.3, color='gray', linestyle=':', alpha=0.3)
ax.axvline(x=0.3, color='gray', linestyle=':', alpha=0.3)
ax.text(0.6, 0.05, '코드 특화\n한국어 약', ha='center', fontsize=9, color='gray', alpha=0.5)
ax.text(0.05, 0.6, '한국어 특화\n코드 약', ha='center', fontsize=9, color='gray', alpha=0.5)
ax.text(0.6, 0.6, '범용 우수', ha='center', fontsize=9, color='gray', alpha=0.5)
ax.text(0.05, 0.05, '전반적 약', ha='center', fontsize=9, color='gray', alpha=0.5)

ax.set_xlabel('T4 코드/수학 (범용 능력)', fontsize=12)
ax.set_ylabel('(T1 + T3) / 2 (한국어 능력)', fontsize=12)
ax.set_title('한국어 특화 vs 범용 능력', fontsize=14, fontweight='bold')
ax.set_xlim(-0.05, 0.9)
ax.set_ylim(-0.05, 0.7)

our_patch = mpatches.Patch(color=OUR_C, label='자체 모델')
base_patch = mpatches.Patch(color=BASE_C, label='베이스라인')
ax.legend(handles=[our_patch, base_patch], fontsize=10, loc='upper left')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT / '23_korean_specialization.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 24: 품질-속도 트레이드오프 (양자화 라인)
# ══════════════════════════════════════════════════════════════════════════════
print("[24/26] 품질-속도 트레이드오프...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot all models
for m in ALL:
    x_val = t6_avg.get(m, 0)
    y_val = composite(m)
    c = mc(m)
    marker = 's' if m in OUR else 'o'
    ax.scatter(x_val, y_val, c=c, s=120, marker=marker, edgecolors='black',
              linewidth=0.5, zorder=3, alpha=0.8)
    ax.annotate(sn(m), (x_val, y_val), fontsize=7, ha='center', va='bottom',
               xytext=(0, 6), textcoords='offset points')

# Connect quantization line for v2
quant_models = [
    ('frankenstallm-3b-v2-f16', 'f16'),
    ('frankenstallm-3b-v2-Q8_0', 'Q8'),
    ('frankenstallm-3b-v2-Q4_K_M', 'Q4'),
]
qx = [t6_avg.get(m, 0) for m, _ in quant_models]
qy = [composite(m) for m, _ in quant_models]
if all(x > 0 for x in qx):
    ax.plot(qx, qy, 'r--', linewidth=2, alpha=0.6, zorder=2)
    ax.annotate('양자화\n(f16→Q8→Q4)', xy=(np.mean(qx), np.mean(qy)),
               fontsize=8, color='red', ha='center',
               xytext=(30, 20), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

ax.set_xlabel('디코딩 속도 (tok/s)', fontsize=12)
ax.set_ylabel('종합 품질 점수 (가중 합산)', fontsize=12)
ax.set_title('품질 vs 속도 트레이드오프', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT / '24_quality_speed_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 25: T5 일관성 6차원 스택 바
# ══════════════════════════════════════════════════════════════════════════════
print("[25/26] T5 일관성 스택 바...")

t5_dim = {}
for r in data['track5']['results']:
    m = r['model']
    tt = r.get('test_type', '?')
    val = r.get('avg_jaccard_similarity', 0) or r.get('avg_edit_distance_ratio', 0) or 0
    t5_dim.setdefault(m, {}).setdefault(tt, []).append(val)

test_types = sorted(set(tt for v in t5_dim.values() for tt in v))
t5_by_dim = {}
for m in ALL:
    t5_by_dim[m] = {}
    for tt in test_types:
        vals = t5_dim.get(m, {}).get(tt, [0])
        t5_by_dim[m][tt] = np.mean(vals)

# Sort models by total consistency
totals = {m: sum(t5_by_dim[m].values()) for m in ALL}
sorted_models = sorted(ALL, key=lambda m: totals[m], reverse=True)

fig, ax = plt.subplots(figsize=(16, 9))
colors_dim = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
bottom = np.zeros(len(sorted_models))

for i, tt in enumerate(test_types):
    vals = [t5_by_dim[m].get(tt, 0) for m in sorted_models]
    short_tt = tt.replace('_', '\n')
    ax.barh(range(len(sorted_models)), vals, left=bottom, label=short_tt,
           color=colors_dim[i % len(colors_dim)], alpha=0.85)
    bottom += vals

ax.set_yticks(range(len(sorted_models)))
ax.set_yticklabels([sn(m) for m in sorted_models], fontsize=9)
ax.set_xlabel('누적 일관성 점수')
ax.set_title('T5: 6차원 일관성 프로파일 (스택)', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, ncol=3, loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT / '25_t5_stacked_consistency.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 26: 효율성 버블 (파라미터 × 품질 × 속도)
# ══════════════════════════════════════════════════════════════════════════════
print("[26/26] 효율성 버블 차트...")

model_params = {
    'frankenstallm-3b:latest': 3.2, 'frankenstallm-3b:Q8_0': 3.2,
    'frankenstallm-3b-v2:latest': 3.0, 'frankenstallm-3b-v2:Q8_0': 3.0,
    'frankenstallm-3b-v2-Q4_K_M': 3.0, 'frankenstallm-3b-v2-Q8_0': 3.0,
    'frankenstallm-3b-v2-f16': 3.0, 'evafrill-mo-3b-slerp': 2.94,
    'qwen2.5:3b': 3.09, 'gemma3:4b': 4.3, 'phi4-mini': 3.84,
    'exaone3.5:2.4b': 2.4, 'llama3.2:3b': 3.21,
    'llama3.1:8b-instruct-q8_0': 8.03, 'ingu627/exaone4.0:1.2b': 1.2,
    'deepseek-r1:1.5b': 1.5,
}

fig, ax = plt.subplots(figsize=(14, 10))

for m in ALL:
    params = model_params.get(m, 3)
    quality = composite(m)
    speed = t6_avg.get(m, 5)
    c = mc(m)

    # Bubble size proportional to speed (min 50, max 400)
    size = max(50, min(400, speed * 5))

    ax.scatter(params, quality, s=size, c=c, alpha=0.6, edgecolors='black', linewidth=0.5, zorder=3)
    ax.annotate(sn(m), (params, quality), fontsize=8, ha='center', va='bottom',
               xytext=(0, 8), textcoords='offset points', fontweight='bold')

# Legend for bubble size
for spd, label in [(10, '10 tok/s'), (50, '50 tok/s'), (100, '100 tok/s')]:
    ax.scatter([], [], s=spd*5, c='gray', alpha=0.3, edgecolors='black',
              linewidth=0.5, label=label)

ax.set_xlabel('파라미터 수 (Billion)', fontsize=12)
ax.set_ylabel('종합 품질 점수', fontsize=12)
ax.set_title('모델 효율성: 파라미터 × 품질 × 속도(버블 크기)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, title='디코딩 속도', loc='upper left')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT / '26_efficiency_bubble.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("추가 시각화 생성 완료!")
print(f"{'='*60}")
for f in sorted(OUTPUT.glob('*.png')):
    print(f"  📊 {f.name} ({f.stat().st_size / 1024:.0f}KB)")
print(f"\n총 {len(list(OUTPUT.glob('*.png')))}개 차트")
