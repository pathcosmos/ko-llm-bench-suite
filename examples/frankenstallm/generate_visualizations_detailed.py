#!/home/lanco/ai-env/bin/python3
"""
FRANKENSTALLM 평가 — 상세 시각화 (Chart 9~18)
기존 8개 차트에 추가되는 심층 분석 차트
"""

import json
import itertools
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# ── 한글 폰트 ───────────────────────────────────────────────────────────────
for font in ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.fontManager.addfont(
            matplotlib.font_manager.findfont(font))
        plt.rcParams['font.family'] = font
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ─────────────────────────────────────────────────────────────
RESULTS = Path("results/full_results_20260407_173207.json")
OUTPUT = Path("reports/visualizations")
OUTPUT.mkdir(parents=True, exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)

# T6 checkpoints
t6_results = []
for ckpt in ['track6_performance_checkpoint.json', 'track6_v2quant_checkpoint.json']:
    p = Path('results') / ckpt
    if p.exists():
        with open(p) as f:
            t6_results.extend(json.load(f).get('results', []))

# T7 checkpoint
with open('results/track7_pairwise_checkpoint.json') as f:
    t7_ckpt = json.load(f)

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
BASE = [m for m in SHORT if m not in OUR]
ALL = sorted(SHORT.keys())

OUR_C = '#e74c3c'
BASE_C = '#3498db'

def mc(m):
    return OUR_C if m in OUR else BASE_C


# ══════════════════════════════════════════════════════════════════════════════
# CHART 9: T2 Turn1→Turn2 성능 변화 히트맵
# ══════════════════════════════════════════════════════════════════════════════
print("[9/18] T2 Turn 성능 변화 히트맵...")

t2_turn = {}
for r in data['track2']['results']:
    m, cat = r['model'], r.get('category', '?')
    t1, t2 = r.get('turn1_mean', 0) or 0, r.get('turn2_mean', 0) or 0
    t2_turn.setdefault(m, {}).setdefault(cat, {'t1': [], 't2': []})
    t2_turn[m][cat]['t1'].append(t1)
    t2_turn[m][cat]['t2'].append(t2)

cats_t2 = sorted(set(c for v in t2_turn.values() for c in v))
degrad_data = {}
for m in ALL:
    degrad_data[sn(m)] = {}
    for cat in cats_t2:
        d = t2_turn.get(m, {}).get(cat, {'t1': [0], 't2': [0]})
        avg1, avg2 = np.mean(d['t1']), np.mean(d['t2'])
        degrad_data[sn(m)][cat] = avg2 - avg1  # positive = improved

df_deg = pd.DataFrame(degrad_data).T
# Sort by average degradation
df_deg['_avg'] = df_deg.mean(axis=1)
df_deg = df_deg.sort_values('_avg', ascending=True)
df_deg = df_deg.drop('_avg', axis=1)

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df_deg, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Turn2 - Turn1 (양수=향상)'})
ax.set_title('T2: Turn1→Turn2 성능 변화 (카테고리별)', fontsize=15, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(OUTPUT / '09_t2_turn_degradation.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 10: T3 한국어 심층이해 카테고리별 히트맵
# ══════════════════════════════════════════════════════════════════════════════
print("[10/18] T3 한국어 심층이해 카테고리 히트맵...")

t3_cat = {}
for r in data['track3']['results']:
    m = r['model']
    cat = r.get('category', '기타')
    s = r.get('judge_score_raw')
    if s is not None:
        t3_cat.setdefault(m, {}).setdefault(cat, []).append(s)

cats_t3 = sorted(set(c for v in t3_cat.values() for c in v))
t3_heat = {}
for m in ALL:
    t3_heat[sn(m)] = {}
    for cat in cats_t3:
        vals = t3_cat.get(m, {}).get(cat, [0])
        t3_heat[sn(m)][cat] = np.mean(vals)

df_t3 = pd.DataFrame(t3_heat).T
df_t3['_avg'] = df_t3.mean(axis=1)
df_t3 = df_t3.sort_values('_avg', ascending=True)
df_t3 = df_t3.drop('_avg', axis=1)

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df_t3, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
            ax=ax, vmin=0, cbar_kws={'label': 'Judge Score (0~10)'})
ax.set_title('T3: 한국어 심층이해 — 카테고리별 Judge 점수', fontsize=15, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(OUTPUT / '10_t3_korean_deep_categories.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 11: T4 문제별 정답 히트맵
# ══════════════════════════════════════════════════════════════════════════════
print("[11/18] T4 문제별 정답 히트맵...")

t4_matrix = {}
all_problems = set()
for r in data['track4']['results']:
    m = r['model']
    t4_matrix[m] = {}
    for detail_key in ['python_details', 'sql_details', 'debug_details', 'math_details']:
        for d in r.get(detail_key, []):
            pid = d.get('id', '')
            if detail_key == 'python_details':
                correct = d.get('pass_at_1', 0)
            elif detail_key == 'math_details':
                correct = 1.0 if d.get('correct', False) else 0.0
            else:
                correct = 1.0 if d.get('correct', False) else 0.0
            t4_matrix[m][pid] = correct
            all_problems.add(pid)

problems_sorted = sorted(all_problems)
# Sort models by total correct
model_totals = {m: sum(t4_matrix[m].values()) for m in t4_matrix}
models_sorted_t4 = sorted(model_totals.keys(), key=lambda m: model_totals[m], reverse=True)

# Build matrix
matrix = []
for m in models_sorted_t4:
    row = [t4_matrix[m].get(p, 0) for p in problems_sorted]
    matrix.append(row)

fig, ax = plt.subplots(figsize=(22, 10))
im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')

ax.set_yticks(range(len(models_sorted_t4)))
ax.set_yticklabels([sn(m) for m in models_sorted_t4], fontsize=8)
ax.set_xticks(range(len(problems_sorted)))
ax.set_xticklabels(problems_sorted, rotation=90, fontsize=5)
ax.set_title('T4: 문제별 정답 현황 (녹색=정답, 적색=오답)', fontsize=15, fontweight='bold')

# Problem difficulty bar on top
difficulty = [sum(t4_matrix.get(m, {}).get(p, 0) for m in t4_matrix) / max(len(t4_matrix), 1)
              for p in problems_sorted]
ax2 = ax.twiny()
ax2.bar(range(len(problems_sorted)), difficulty, color='steelblue', alpha=0.3, width=0.8)
ax2.set_xlim(-0.5, len(problems_sorted) - 0.5)
ax2.set_ylabel('정답률')
ax2.set_xticks([])

plt.colorbar(im, ax=ax, shrink=0.6, label='정답(1) / 오답(0)')
plt.tight_layout()
plt.savefig(OUTPUT / '11_t4_problem_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 12: T5 6차원 일관성 레이더
# ══════════════════════════════════════════════════════════════════════════════
print("[12/18] T5 일관성 레이더...")

t5_dim = {}
for r in data['track5']['results']:
    m = r['model']
    tt = r.get('test_type', '?')
    # Use appropriate metric per test type
    val = r.get('avg_jaccard_similarity', 0) or r.get('compliance_rate', 0) or \
          r.get('korean_ratio', 0) or r.get('keyword_hit_rate', 0) or \
          r.get('refusal_rate', 0) or 0
    t5_dim.setdefault(m, {}).setdefault(tt, []).append(val)

test_types_t5 = sorted(set(tt for v in t5_dim.values() for tt in v))
t5_avg = {}
for m in ALL:
    t5_avg[m] = {}
    for tt in test_types_t5:
        vals = t5_dim.get(m, {}).get(tt, [0])
        t5_avg[m][tt] = np.mean(vals)

# Radar chart for top models
key_models_t5 = ['gemma3:4b', 'qwen2.5:3b', 'exaone3.5:2.4b', 'llama3.1:8b-instruct-q8_0',
                 'frankenstallm-3b:latest', 'evafrill-mo-3b-slerp', 'phi4-mini', 'deepseek-r1:1.5b']

n_dims = len(test_types_t5)
if n_dims >= 3:
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors_cycle = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22', '#1abc9c', '#34495e']

    for i, m in enumerate(key_models_t5):
        vals = [t5_avg.get(m, {}).get(tt, 0) for tt in test_types_t5]
        # Normalize per dimension
        max_vals = [max(t5_avg.get(mm, {}).get(tt, 0) for mm in ALL) for tt in test_types_t5]
        vals_norm = [v / mx if mx > 0 else 0 for v, mx in zip(vals, max_vals)]
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, color=colors_cycle[i % len(colors_cycle)],
                linewidth=2, label=sn(m))
        ax.fill(angles, vals_norm, color=colors_cycle[i % len(colors_cycle)], alpha=0.05)

    short_labels = [tt.replace('_', '\n') for tt in test_types_t5]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title('T5: 일관성/안전성 6차원 프로파일', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT / '12_t5_consistency_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("  ⚠ T5 차원이 3개 미만, 레이더 차트 생략")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 13: T6 성능 곡선 (입력/출력 길이 vs 속도)
# ══════════════════════════════════════════════════════════════════════════════
print("[13/18] T6 성능 곡선...")

prefill_data = {}  # {model: {input_len: tok/s}}
decode_data = {}   # {model: {output_len: tok/s}}

for r in t6_results:
    m = r['model']
    tt = r.get('test_type', '')
    if tt == 'prefill_speed':
        inp = r.get('input_length', 0)
        spd = r.get('prefill_tok_s', 0)
        if spd > 0:
            prefill_data.setdefault(m, {})[inp] = spd
    elif tt == 'decode_speed':
        out = r.get('output_length', 0)
        spd = r.get('tokens_per_sec', 0)
        if spd > 0:
            decode_data.setdefault(m, {})[out] = spd

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
colors_list = plt.cm.tab20(np.linspace(0, 1, 16))

# Prefill speed curves
for i, m in enumerate(sorted(prefill_data.keys())):
    lengths = sorted(prefill_data[m].keys())
    speeds = [prefill_data[m][l] for l in lengths]
    ls = '--' if m in OUR else '-'
    lw = 2.5 if m in OUR else 1.5
    axes[0].plot(lengths, speeds, marker='o', label=sn(m), linestyle=ls,
                linewidth=lw, color=colors_list[i], markersize=5)

axes[0].set_xlabel('입력 토큰 수')
axes[0].set_ylabel('Prefill Speed (tok/s)')
axes[0].set_title('T6: 입력 길이별 Prefill 속도', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=7, ncol=2)
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Decode speed curves
for i, m in enumerate(sorted(decode_data.keys())):
    lengths = sorted(decode_data[m].keys())
    speeds = [decode_data[m][l] for l in lengths]
    ls = '--' if m in OUR else '-'
    lw = 2.5 if m in OUR else 1.5
    axes[1].plot(lengths, speeds, marker='s', label=sn(m), linestyle=ls,
                linewidth=lw, color=colors_list[i], markersize=5)

axes[1].set_xlabel('출력 토큰 수')
axes[1].set_ylabel('Decode Speed (tok/s)')
axes[1].set_title('T6: 출력 길이별 Decode 속도', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=7, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / '13_t6_speed_curves.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 14: T6 양자화 비교 (Q4 vs Q8 vs f16)
# ══════════════════════════════════════════════════════════════════════════════
print("[14/18] T6 양자화 비교...")

quant_models = ['frankenstallm-3b-v2-Q4_K_M', 'frankenstallm-3b-v2-Q8_0', 'frankenstallm-3b-v2-f16']
quant_labels = ['Q4_K_M', 'Q8_0', 'f16']

quant_data = {q: {} for q in quant_labels}
for r in t6_results:
    m = r['model']
    if m not in quant_models:
        continue
    idx = quant_models.index(m)
    ql = quant_labels[idx]
    tt = r.get('test_type', '')

    if tt == 'decode_speed':
        quant_data[ql].setdefault('decode', []).append(r.get('tokens_per_sec', 0))
    elif tt == 'prefill_speed':
        quant_data[ql].setdefault('prefill', []).append(r.get('prefill_tok_s', 0))
    elif tt == 'ttft':
        quant_data[ql].setdefault('ttft', []).append(r.get('ttft_s', 0) or r.get('wall_time_s', 0))
    elif tt == 'concurrent':
        conc = r.get('concurrency', 1)
        quant_data[ql][f'conc_{conc}'] = r.get('aggregate_tok_s', 0) or r.get('tokens_per_sec', 0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_q = ['#3498db', '#e74c3c', '#2ecc71']

# Decode speed
ax = axes[0, 0]
for i, ql in enumerate(quant_labels):
    vals = quant_data[ql].get('decode', [0])
    ax.bar(i, np.mean(vals), color=colors_q[i], label=ql, alpha=0.85)
    ax.text(i, np.mean(vals) + 5, f'{np.mean(vals):.1f}', ha='center', fontsize=10)
ax.set_xticks(range(3))
ax.set_xticklabels(quant_labels)
ax.set_ylabel('tok/s')
ax.set_title('Decode Speed', fontweight='bold')

# Prefill speed
ax = axes[0, 1]
for i, ql in enumerate(quant_labels):
    vals = quant_data[ql].get('prefill', [0])
    ax.bar(i, np.mean(vals), color=colors_q[i], alpha=0.85)
    ax.text(i, np.mean(vals) + 500, f'{np.mean(vals):.0f}', ha='center', fontsize=10)
ax.set_xticks(range(3))
ax.set_xticklabels(quant_labels)
ax.set_ylabel('tok/s')
ax.set_title('Prefill Speed', fontweight='bold')

# TTFT
ax = axes[1, 0]
for i, ql in enumerate(quant_labels):
    vals = quant_data[ql].get('ttft', [0])
    ax.bar(i, np.mean(vals) * 1000, color=colors_q[i], alpha=0.85)
    ax.text(i, np.mean(vals) * 1000 + 1, f'{np.mean(vals)*1000:.1f}ms', ha='center', fontsize=10)
ax.set_xticks(range(3))
ax.set_xticklabels(quant_labels)
ax.set_ylabel('ms')
ax.set_title('Time To First Token', fontweight='bold')

# Concurrency scaling
ax = axes[1, 1]
conc_levels = [1, 2, 4]
x = np.arange(len(conc_levels))
width = 0.25
for i, ql in enumerate(quant_labels):
    vals = [quant_data[ql].get(f'conc_{c}', 0) for c in conc_levels]
    ax.bar(x + i * width, vals, width, label=ql, color=colors_q[i], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels([f'{c} thread' for c in conc_levels])
ax.set_ylabel('Aggregate tok/s')
ax.set_title('Concurrent Scaling', fontweight='bold')
ax.legend()

fig.suptitle('T6: Frankenstallm v2 양자화 비교 (Q4_K_M vs Q8_0 vs f16)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT / '14_t6_quantization.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 15: T7 Head-to-Head 승률 히트맵 (16×16)
# ══════════════════════════════════════════════════════════════════════════════
print("[15/18] T7 Head-to-Head 승률 히트맵...")

# Build win matrix from comparisons
h2h = {}  # {(a, b): {'a_wins': n, 'b_wins': n, 'ties': n}}
for comp in t7_ckpt.get('comparisons', []):
    a, b = comp['model_a'], comp['model_b']
    key = (a, b)
    if key not in h2h:
        h2h[key] = {'a_wins': 0, 'b_wins': 0, 'ties': 0}
    w = comp.get('winner', 'TIE')
    if w == 'A':
        h2h[key]['a_wins'] += 1
    elif w == 'B':
        h2h[key]['b_wins'] += 1
    else:
        h2h[key]['ties'] += 1

# Build win rate matrix
models_elo_order = sorted(data['track7']['results']['elo_scores'].keys(),
                          key=lambda m: data['track7']['results']['elo_scores'][m]['elo'],
                          reverse=True)

n = len(models_elo_order)
win_matrix = np.full((n, n), np.nan)

for i, mi in enumerate(models_elo_order):
    for j, mj in enumerate(models_elo_order):
        if i == j:
            win_matrix[i, j] = 0.5
            continue
        # Check both orderings
        if (mi, mj) in h2h:
            d = h2h[(mi, mj)]
            total = d['a_wins'] + d['b_wins'] + d['ties']
            win_matrix[i, j] = (d['a_wins'] + d['ties'] * 0.5) / total if total > 0 else 0.5
        elif (mj, mi) in h2h:
            d = h2h[(mj, mi)]
            total = d['a_wins'] + d['b_wins'] + d['ties']
            win_matrix[i, j] = (d['b_wins'] + d['ties'] * 0.5) / total if total > 0 else 0.5

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.isnan(win_matrix)
sns.heatmap(win_matrix, annot=True, fmt='.0%', cmap='RdYlGn', center=0.5,
            linewidths=0.5, ax=ax, mask=mask, vmin=0, vmax=1,
            xticklabels=[sn(m) for m in models_elo_order],
            yticklabels=[sn(m) for m in models_elo_order],
            cbar_kws={'label': '행 모델이 열 모델을 이긴 비율'})
ax.set_title('T7: Head-to-Head 승률 매트릭스 (16×16)', fontsize=15, fontweight='bold')
ax.set_xlabel('상대 모델')
ax.set_ylabel('기준 모델')
plt.tight_layout()
plt.savefig(OUTPUT / '15_t7_h2h_winrate.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 16: 트랙 간 상관관계 매트릭스
# ══════════════════════════════════════════════════════════════════════════════
print("[16/18] 트랙 간 상관관계...")

# Collect per-model scores for each track
track_scores = {}
for m in ALL:
    # T1
    t1 = sum(data['track1'].get('summary', {}).get(m, {}).values()) / max(len(data['track1'].get('summary', {}).get(m, {})), 1) if m in data['track1'].get('summary', {}) else 0
    # Fallback to results
    if t1 == 0:
        for r in data['track1']['results']:
            if r['model'] == m and r.get('scores'):
                t1 = np.mean(list(r['scores'].values()))
                break

    # T2
    t2_vals = []
    for r in data['track2']['results']:
        if r['model'] == m:
            v = r.get('turn1_mean', 0)
            if v:
                t2_vals.append(v)
    t2 = np.mean(t2_vals) if t2_vals else 0

    # T3
    t3_vals = [r.get('judge_score_raw', 0) or 0 for r in data['track3']['results'] if r['model'] == m]
    t3 = np.mean(t3_vals) if t3_vals else 0

    # T4
    t4 = 0
    for r in data['track4']['results']:
        if r['model'] == m and r.get('scores'):
            t4 = np.mean(list(r['scores'].values()))
            break

    # T5
    t5_vals = [r.get('avg_jaccard_similarity', 0) or 0 for r in data['track5']['results'] if r['model'] == m]
    t5 = np.mean(t5_vals) if t5_vals else 0

    # T6
    t6_vals = [r.get('tokens_per_sec', 0) for r in t6_results
               if r['model'] == m and r.get('test_type') == 'decode_speed' and r.get('tokens_per_sec', 0) > 0]
    t6 = np.mean(t6_vals) if t6_vals else 0

    # T7
    t7 = data['track7']['results']['elo_scores'].get(m, {}).get('elo', 0)

    track_scores[m] = {'T1 한국어': t1, 'T2 Ko-Bench': t2, 'T3 심층이해': t3,
                       'T4 코드/수학': t4, 'T5 일관성': t5, 'T6 속도': t6, 'T7 Elo': t7}

df_corr = pd.DataFrame(track_scores).T
corr = df_corr.corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})

# Correlation heatmap
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=1, ax=axes[0], vmin=-1, vmax=1, square=True)
axes[0].set_title('트랙 간 상관관계 (피어슨)', fontsize=14, fontweight='bold')

# Key scatter plots
ax = axes[1]
# T1 vs T7 Elo
for m in ALL:
    x = track_scores[m]['T1 한국어']
    y = track_scores[m]['T7 Elo']
    c = mc(m)
    marker = 's' if m in OUR else 'o'
    ax.scatter(x, y, c=c, s=80, marker=marker, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.annotate(sn(m), (x, y), fontsize=6, ha='center', va='bottom',
               xytext=(0, 4), textcoords='offset points')

# Add trend line
x_all = [track_scores[m]['T1 한국어'] for m in ALL]
y_all = [track_scores[m]['T7 Elo'] for m in ALL]
if len(x_all) > 2:
    slope, intercept, r_val, _, _ = stats.linregress(x_all, y_all)
    x_line = np.linspace(min(x_all), max(x_all), 100)
    ax.plot(x_line, slope * x_line + intercept, '--', color='gray', alpha=0.5,
           label=f'r={r_val:.2f}')
    ax.legend(fontsize=10)

ax.set_xlabel('T1 한국어 벤치 정확도')
ax.set_ylabel('T7 Elo')
ax.set_title('T1 한국어 능력 vs T7 Elo (r={:.2f})'.format(r_val if len(x_all) > 2 else 0),
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT / '16_cross_track_correlation.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 17: Frankenstallm 계보 분석
# ══════════════════════════════════════════════════════════════════════════════
print("[17/18] Frankenstallm 계보 분석...")

fs_order = [
    'frankenstallm-3b:latest', 'frankenstallm-3b:Q8_0',
    'frankenstallm-3b-v2:latest', 'frankenstallm-3b-v2:Q8_0',
    'frankenstallm-3b-v2-Q4_K_M', 'frankenstallm-3b-v2-Q8_0',
    'frankenstallm-3b-v2-f16', 'evafrill-mo-3b-slerp',
]

track_labels = ['T1 한국어', 'T2 Ko-Bench', 'T3 심층이해', 'T4 코드', 'T5 일관성', 'T7 Elo']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes_flat = axes.flatten()

for ti, tl in enumerate(track_labels):
    ax = axes_flat[ti]
    vals = [track_scores.get(m, {}).get(tl, 0) for m in fs_order]
    names = [sn(m) for m in fs_order]
    colors = ['#e74c3c' if 'v1' in sn(m) else '#ff6b6b' if 'v2' in sn(m) else '#fdcb6e'
              for m in fs_order]

    bars = ax.bar(range(len(fs_order)), vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(fs_order)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_title(tl, fontweight='bold', fontsize=12)

    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + ax.get_ylim()[1]*0.02,
                   f'{v:.2f}' if v < 10 else f'{v:.0f}', ha='center', fontsize=7)

# Legend
legend_patches = [
    mpatches.Patch(color='#e74c3c', label='v1 (원본)'),
    mpatches.Patch(color='#ff6b6b', label='v2 (재빌드)'),
    mpatches.Patch(color='#fdcb6e', label='EVAFRILL (SLERP)'),
]
fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 0.98))
fig.suptitle('Frankenstallm 계보 — 버전/양자화별 트랙 성능', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT / '17_frankenstallm_lineage.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 18: 효율성 파레토 프론티어
# ══════════════════════════════════════════════════════════════════════════════
print("[18/18] 효율성 파레토 프론티어...")

model_sizes = {
    'frankenstallm-3b:latest': 2.0, 'frankenstallm-3b:Q8_0': 3.4,
    'frankenstallm-3b-v2:latest': 2.4, 'frankenstallm-3b-v2:Q8_0': 1.3,
    'frankenstallm-3b-v2-Q4_K_M': 0.8, 'frankenstallm-3b-v2-Q8_0': 1.3,
    'frankenstallm-3b-v2-f16': 2.4, 'evafrill-mo-3b-slerp': 5.6,
    'qwen2.5:3b': 1.9, 'gemma3:4b': 3.3, 'phi4-mini': 2.5,
    'exaone3.5:2.4b': 1.6, 'llama3.2:3b': 2.0,
    'llama3.1:8b-instruct-q8_0': 8.5, 'ingu627/exaone4.0:1.2b': 0.8,
    'deepseek-r1:1.5b': 1.1,
}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: Size vs Elo with Pareto frontier
points = []
for m in ALL:
    size = model_sizes.get(m, 2)
    elo = data['track7']['results']['elo_scores'].get(m, {}).get('elo', 500)
    points.append((m, size, elo))
    c = mc(m)
    marker = 's' if m in OUR else 'o'
    axes[0].scatter(size, elo, c=c, s=150, marker=marker, alpha=0.8,
                   edgecolors='black', linewidth=0.5, zorder=3)
    axes[0].annotate(sn(m), (size, elo), fontsize=8, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', fontweight='bold')

# Pareto frontier (smaller size + higher Elo is better)
sorted_pts = sorted(points, key=lambda p: p[1])  # sort by size
pareto = []
max_elo = -1
for m, s, e in sorted_pts:
    if e > max_elo:
        pareto.append((s, e))
        max_elo = e

if len(pareto) > 1:
    px, py = zip(*pareto)
    axes[0].plot(px, py, 'g--', linewidth=2, alpha=0.6, label='Pareto Frontier')
    axes[0].fill_between(px, py, min(py) - 50, alpha=0.05, color='green')
    axes[0].legend(fontsize=11)

axes[0].set_xlabel('모델 크기 (GB)', fontsize=12)
axes[0].set_ylabel('T7 Elo', fontsize=12)
axes[0].set_title('크기 vs 품질 — 파레토 프론티어', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.2)

# Right: Elo/GB efficiency ranking
efficiency = [(m, data['track7']['results']['elo_scores'].get(m, {}).get('elo', 0) / model_sizes.get(m, 1))
              for m in ALL]
efficiency.sort(key=lambda x: x[1], reverse=True)

names = [sn(m) for m, _ in efficiency]
vals = [v for _, v in efficiency]
colors = [mc(m) for m, _ in efficiency]
bars = axes[1].barh(names, vals, color=colors, alpha=0.85)
axes[1].set_xlabel('Elo / GB (높을수록 효율적)')
axes[1].set_title('모델 효율성 (Elo per GB)', fontsize=14, fontweight='bold')
for bar, v in zip(bars, vals):
    axes[1].text(v + 5, bar.get_y() + bar.get_height()/2, f'{v:.0f}',
                va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT / '18_efficiency_frontier.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("추가 시각화 생성 완료!")
print(f"{'='*60}")
for f in sorted(OUTPUT.glob('*.png')):
    print(f"  📊 {f.name} ({f.stat().st_size / 1024:.0f}KB)")
print(f"\n총 {len(list(OUTPUT.glob('*.png')))}개 차트")
