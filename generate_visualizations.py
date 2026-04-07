#!/home/lanco/ai-env/bin/python3
"""
FRANKENSTALLM 평가 결과 종합 시각화 생성기
16모델 × 7트랙 전체 결과를 다양한 차트로 시각화
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
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
RESULTS_FILE = Path("results/full_results_20260407_173207.json")
OUTPUT_DIR = Path("reports/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

# ── 모델 분류 + 짧은 이름 ───────────────────────────────────────────────────
SHORT_NAMES = {
    'frankenstallm-3b:latest': 'FS-3B v1',
    'frankenstallm-3b:Q8_0': 'FS-3B v1:Q8',
    'frankenstallm-3b-v2:latest': 'FS-3B v2',
    'frankenstallm-3b-v2:Q8_0': 'FS-3B v2:Q8',
    'frankenstallm-3b-v2-Q4_K_M': 'FS-3B v2:Q4',
    'frankenstallm-3b-v2-Q8_0': 'FS-3B v2:Q8(g)',
    'frankenstallm-3b-v2-f16': 'FS-3B v2:f16',
    'evafrill-mo-3b-slerp': 'EVAFRILL-3B',
    'qwen2.5:3b': 'Qwen2.5-3B',
    'gemma3:4b': 'Gemma3-4B',
    'phi4-mini': 'Phi4-Mini',
    'exaone3.5:2.4b': 'EXAONE-2.4B',
    'llama3.2:3b': 'Llama3.2-3B',
    'llama3.1:8b-instruct-q8_0': 'Llama3.1-8B',
    'ingu627/exaone4.0:1.2b': 'EXAONE4-1.2B',
    'deepseek-r1:1.5b': 'DeepSeek-1.5B',
}

OUR_MODELS = [m for m in SHORT_NAMES if 'frankenstallm' in m or 'evafrill' in m]
BASELINE_MODELS = [m for m in SHORT_NAMES if m not in OUR_MODELS]

# 색상 팔레트
OUR_COLOR = '#e74c3c'
OUR_COLORS = ['#e74c3c', '#c0392b', '#ff6b6b', '#ff8787', '#d63031', '#fab1a0', '#e17055', '#fdcb6e']
BASE_COLORS = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e', '#16a085']

def sn(model):
    return SHORT_NAMES.get(model, model)

def model_color(model):
    if model in OUR_MODELS:
        idx = OUR_MODELS.index(model) % len(OUR_COLORS)
        return OUR_COLORS[idx]
    idx = BASELINE_MODELS.index(model) % len(BASE_COLORS)
    return BASE_COLORS[idx]

# ── 1. 데이터 추출 ──────────────────────────────────────────────────────────
print("데이터 추출 중...")

# T1: Korean Bench
t1_scores = {}
for r in data['track1']['results']:
    scores = r.get('scores', {})
    if scores:
        t1_scores[r['model']] = {
            'avg': sum(scores.values()) / len(scores),
            **scores
        }

# T2: Ko-Bench (turn1_mean per category per model)
t2_raw = {}
for r in data['track2']['results']:
    m = r['model']
    cat = r.get('category', 'unknown')
    s = r.get('turn1_mean', 0) or 0
    t2_raw.setdefault(m, {}).setdefault(cat, []).append(s)
t2_scores = {}
for m, cats in t2_raw.items():
    t2_scores[m] = {cat: np.mean(vs) for cat, vs in cats.items()}
    t2_scores[m]['avg'] = np.mean([v for vs in cats.values() for v in vs])

# T3: Korean Deep (judge_score_raw)
t3_raw = {}
for r in data['track3']['results']:
    m = r['model']
    s = r.get('judge_score_raw', 0) or 0
    t3_raw.setdefault(m, []).append(s)
t3_scores = {m: np.mean(vs) for m, vs in t3_raw.items()}

# T4: Code & Math
t4_scores = {}
for r in data['track4']['results']:
    scores = r.get('scores', {})
    if scores:
        t4_scores[r['model']] = scores

# T5: Consistency
t5_raw = {}
for r in data['track5']['results']:
    m = r['model']
    s = r.get('avg_jaccard_similarity', 0) or 0
    t5_raw.setdefault(m, []).append(s)
t5_scores = {m: np.mean(vs) for m, vs in t5_raw.items()}

# T6: Performance (decode speed)
t6_scores = {}
# From multiple checkpoint sources
for ckpt_name in ['track6_performance_checkpoint.json', 'track6_v2quant_checkpoint.json']:
    ckpt_path = Path('results') / ckpt_name
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            t6_data = json.load(f)
        for r in t6_data.get('results', []):
            if r.get('test_type') == 'decode_speed':
                m = r['model']
                tps = r.get('tokens_per_sec', 0) or 0
                if tps > 0:
                    t6_scores.setdefault(m, []).append(tps)
t6_avg = {m: np.mean(vs) for m, vs in t6_scores.items()}

# T7: Elo
t7_elo = data['track7']['results']['elo_scores']

all_models = sorted(SHORT_NAMES.keys())

print(f"  T1: {len(t1_scores)} models, T2: {len(t2_scores)}, T3: {len(t3_scores)}")
print(f"  T4: {len(t4_scores)}, T5: {len(t5_scores)}, T6: {len(t6_avg)}, T7: {len(t7_elo)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: 종합 스코어카드 히트맵
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] 종합 히트맵 생성...")

# Normalize all scores to 0-1 for heatmap
heatmap_data = {}
for m in all_models:
    heatmap_data[sn(m)] = {
        'T1 한국어벤치': t1_scores.get(m, {}).get('avg', 0),
        'T2 Ko-Bench': t2_scores.get(m, {}).get('avg', 0) / 10,  # 0-10 → 0-1
        'T3 심층이해': t3_scores.get(m, 0) / 10,  # 0-10 → 0-1
        'T4 코드/수학': np.mean(list(t4_scores.get(m, {}).values())) if t4_scores.get(m) else 0,
        'T5 일관성': t5_scores.get(m, 0),
        'T6 속도(정규화)': min(t6_avg.get(m, 0) / 100, 1.0),  # cap at 100 tok/s
        'T7 Elo(정규화)': (t7_elo.get(m, {}).get('elo', 500) - 500) / 1100,  # 500-1600 → 0-1
    }

df_heat = pd.DataFrame(heatmap_data).T
# Sort by T7 Elo
df_heat = df_heat.sort_values('T7 Elo(정규화)', ascending=True)

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df_heat, annot=True, fmt='.2f', cmap='RdYlGn', linewidths=0.5,
            ax=ax, vmin=0, vmax=1, cbar_kws={'label': '정규화 점수 (0~1)'})
ax.set_title('16모델 × 7트랙 종합 성능 히트맵', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('')

# Highlight our models
for i, label in enumerate(ax.get_yticklabels()):
    if any(tag in label.get_text() for tag in ['FS-', 'EVAFRILL']):
        label.set_color(OUR_COLOR)
        label.set_fontweight('bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_heatmap_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: T1 한국어 벤치마크 상세
# ══════════════════════════════════════════════════════════════════════════════
print("[2/8] T1 한국어 벤치마크 차트...")

t1_metrics = ['kmmlu', 'kobest_boolq', 'kobest_copa', 'kobest_hellaswag', 'kobest_sentineg']
t1_labels = ['KMMLU', 'BoolQ', 'COPA', 'HellaSwag', 'SentiNeg']

# Sort by average
sorted_models = sorted(t1_scores.keys(), key=lambda m: t1_scores[m]['avg'], reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: Grouped bar chart
x = np.arange(len(sorted_models))
width = 0.15
for i, (metric, label) in enumerate(zip(t1_metrics, t1_labels)):
    vals = [t1_scores[m].get(metric, 0) for m in sorted_models]
    axes[0].bar(x + i * width, vals, width, label=label, alpha=0.85)

axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels([sn(m) for m in sorted_models], rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('정확도')
axes[0].set_title('T1: 한국어 벤치마크 세부 점수', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=8, ncol=3)
axes[0].set_ylim(0, 1.1)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# Right: Average ranking
avgs = [t1_scores[m]['avg'] for m in sorted_models]
colors = [model_color(m) for m in sorted_models]
bars = axes[1].barh([sn(m) for m in sorted_models], avgs, color=colors)
axes[1].set_xlabel('평균 정확도')
axes[1].set_title('T1: 평균 점수 순위', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 1.1)
for bar, v in zip(bars, avgs):
    axes[1].text(v + 0.02, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_t1_korean_bench.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: T2 Ko-Bench 카테고리별
# ══════════════════════════════════════════════════════════════════════════════
print("[3/8] T2 Ko-Bench 카테고리 차트...")

categories = sorted(set(c for m in t2_scores.values() for c in m if c != 'avg'))
sorted_t2 = sorted(t2_scores.keys(), key=lambda m: t2_scores[m].get('avg', 0), reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: Radar-style grouped bar
cat_data = []
for m in sorted_t2[:8]:  # Top 8 models
    for cat in categories:
        cat_data.append({'Model': sn(m), 'Category': cat, 'Score': t2_scores[m].get(cat, 0)})

df_cat = pd.DataFrame(cat_data)
if not df_cat.empty:
    pivot = df_cat.pivot(index='Model', columns='Category', values='Score')
    pivot.plot(kind='bar', ax=axes[0], width=0.8, alpha=0.85)
    axes[0].set_title('T2: Ko-Bench 카테고리별 점수 (Top 8)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Judge 점수 (1~10)')
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].tick_params(axis='x', rotation=45)

# Right: Overall ranking
all_avgs = [(m, t2_scores[m].get('avg', 0)) for m in sorted_t2]
names = [sn(m) for m, _ in all_avgs]
vals = [v for _, v in all_avgs]
colors = [model_color(m) for m, _ in all_avgs]
bars = axes[1].barh(names, vals, color=colors)
axes[1].set_xlabel('평균 Judge 점수 (1~10)')
axes[1].set_title('T2: Ko-Bench 전체 순위', fontsize=13, fontweight='bold')
for bar, v in zip(bars, vals):
    axes[1].text(v + 0.1, bar.get_y() + bar.get_height()/2, f'{v:.2f}',
                va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_t2_ko_bench.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4: T4 Code & Math 상세
# ══════════════════════════════════════════════════════════════════════════════
print("[4/8] T4 코드/수학 차트...")

t4_metrics = ['python_pass1', 'sql_accuracy', 'debug_accuracy', 'math_accuracy']
t4_labels = ['Python', 'SQL', 'Debug', 'Math']

sorted_t4 = sorted(t4_scores.keys(),
                    key=lambda m: np.mean(list(t4_scores[m].values())) if t4_scores[m] else 0,
                    reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: Stacked/grouped bar
x = np.arange(len(sorted_t4))
width = 0.18
for i, (metric, label) in enumerate(zip(t4_metrics, t4_labels)):
    vals = [t4_scores[m].get(metric, 0) for m in sorted_t4]
    axes[0].bar(x + i * width, vals, width, label=label, alpha=0.85)

axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels([sn(m) for m in sorted_t4], rotation=55, ha='right', fontsize=7)
axes[0].set_ylabel('정확도')
axes[0].set_title('T4: 코드 & 수학 세부 점수', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1.05)

# Right: Average ranking
avgs = [(m, np.mean(list(t4_scores[m].values())) if t4_scores[m] else 0) for m in sorted_t4]
names = [sn(m) for m, _ in avgs]
vals = [v for _, v in avgs]
colors = [model_color(m) for m, _ in avgs]
bars = axes[1].barh(names, vals, color=colors)
axes[1].set_xlabel('평균 정확도')
axes[1].set_title('T4: 코드/수학 종합 순위', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 1.0)
for bar, v in zip(bars, vals):
    if v > 0:
        axes[1].text(v + 0.01, bar.get_y() + bar.get_height()/2, f'{v:.0%}',
                    va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_t4_code_math.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5: T7 Elo 랭킹 + Win/Loss
# ══════════════════════════════════════════════════════════════════════════════
print("[5/8] T7 Elo 랭킹 차트...")

sorted_elo = sorted(t7_elo.items(), key=lambda x: x[1]['elo'], reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: Elo bar chart with CI
names = [sn(m) for m, _ in sorted_elo]
elos = [s['elo'] for _, s in sorted_elo]
ci_low = [s.get('ci_lower', s['elo']) for _, s in sorted_elo]
ci_high = [s.get('ci_upper', s['elo']) for _, s in sorted_elo]
errors = [[e - l for e, l in zip(elos, ci_low)], [h - e for e, h in zip(elos, ci_high)]]
colors = [model_color(m) for m, _ in sorted_elo]

bars = axes[0].barh(names, elos, xerr=errors, color=colors, alpha=0.85,
                    capsize=3, error_kw={'linewidth': 1})
axes[0].set_xlabel('Elo Rating')
axes[0].set_title('T7: Pairwise Elo 랭킹 (95% CI)', fontsize=14, fontweight='bold')
axes[0].axvline(x=1000, color='gray', linestyle='--', alpha=0.5, label='Base=1000')
for bar, v in zip(bars, elos):
    axes[0].text(v + 15, bar.get_y() + bar.get_height()/2, f'{v:.0f}',
                va='center', fontsize=8)

# Right: Win rate pie-like horizontal
wins = [s['wins'] for _, s in sorted_elo]
losses = [s['losses'] for _, s in sorted_elo]
total = [w + l for w, l in zip(wins, losses)]
win_rates = [w / t * 100 if t > 0 else 0 for w, t in zip(wins, total)]

y = np.arange(len(names))
axes[1].barh(y, win_rates, color=colors, alpha=0.85)
axes[1].barh(y, [100 - wr for wr in win_rates], left=win_rates, color='lightgray', alpha=0.4)
axes[1].set_yticks(y)
axes[1].set_yticklabels(names)
axes[1].set_xlabel('승률 (%)')
axes[1].set_title('T7: 승률 (Win Rate)', fontsize=14, fontweight='bold')
axes[1].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
for i, (wr, w, l) in enumerate(zip(win_rates, wins, losses)):
    axes[1].text(wr + 1, i, f'{wr:.0f}% ({w}W-{l}L)', va='center', fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_t7_elo_ranking.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6: T6 성능 (토큰/초) + T5 일관성
# ══════════════════════════════════════════════════════════════════════════════
print("[6/8] T6 성능 + T5 일관성 차트...")

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: T6 decode speed
sorted_t6 = sorted(t6_avg.items(), key=lambda x: x[1], reverse=True)
names_t6 = [sn(m) for m, _ in sorted_t6]
vals_t6 = [v for _, v in sorted_t6]
colors_t6 = [model_color(m) for m, _ in sorted_t6]
bars = axes[0].barh(names_t6, vals_t6, color=colors_t6, alpha=0.85)
axes[0].set_xlabel('Decode Speed (tokens/sec)')
axes[0].set_title('T6: 디코딩 속도 (높을수록 좋음)', fontsize=14, fontweight='bold')
for bar, v in zip(bars, vals_t6):
    axes[0].text(v + 1, bar.get_y() + bar.get_height()/2, f'{v:.1f}',
                va='center', fontsize=8)

# Right: T5 consistency
sorted_t5 = sorted(t5_scores.items(), key=lambda x: x[1], reverse=True)
names_t5 = [sn(m) for m, _ in sorted_t5]
vals_t5 = [v for _, v in sorted_t5]
colors_t5 = [model_color(m) for m, _ in sorted_t5]
bars = axes[1].barh(names_t5, vals_t5, color=colors_t5, alpha=0.85)
axes[1].set_xlabel('Jaccard Similarity (높을수록 일관적)')
axes[1].set_title('T5: 응답 일관성', fontsize=14, fontweight='bold')
for bar, v in zip(bars, vals_t5):
    axes[1].text(v + 0.002, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_t6_speed_t5_consistency.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 7: 자체모델 vs 베이스라인 레이더 차트
# ══════════════════════════════════════════════════════════════════════════════
print("[7/8] 레이더 차트 (자체모델 vs 베이스라인)...")

radar_metrics = ['T1 한국어', 'T2 Ko-Bench', 'T3 심층이해', 'T4 코드/수학', 'T5 일관성', 'T7 Elo']
n_metrics = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]

def get_radar_values(model):
    vals = [
        t1_scores.get(model, {}).get('avg', 0),
        t2_scores.get(model, {}).get('avg', 0) / 10,
        t3_scores.get(model, 0) / 10,
        np.mean(list(t4_scores.get(model, {}).values())) if t4_scores.get(model) else 0,
        min(t5_scores.get(model, 0) * 5, 1.0),  # scale up for visibility
        (t7_elo.get(model, {}).get('elo', 500) - 500) / 1100,
    ]
    vals += vals[:1]
    return vals

# Select key models for comparison
key_models = [
    ('frankenstallm-3b:latest', OUR_COLORS[0], '-', 2.5),
    ('frankenstallm-3b-v2:latest', OUR_COLORS[2], '--', 2),
    ('evafrill-mo-3b-slerp', OUR_COLORS[5], '-.', 2),
    ('gemma3:4b', BASE_COLORS[1], '-', 2),
    ('qwen2.5:3b', BASE_COLORS[0], '-', 2),
    ('exaone3.5:2.4b', BASE_COLORS[3], '-', 2),
    ('llama3.1:8b-instruct-q8_0', BASE_COLORS[6], '--', 1.5),
]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for model, color, ls, lw in key_models:
    vals = get_radar_values(model)
    ax.plot(angles, vals, color=color, linestyle=ls, linewidth=lw, label=sn(model))
    ax.fill(angles, vals, color=color, alpha=0.05)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('자체모델 vs 베이스라인 종합 비교', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_radar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 8: 종합 대시보드 (6-in-1)
# ══════════════════════════════════════════════════════════════════════════════
print("[8/8] 종합 대시보드...")

fig = plt.figure(figsize=(24, 16))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Top-left: T1+T2 scatter (한국어 능력 종합)
ax1 = fig.add_subplot(gs[0, 0])
for m in all_models:
    x_val = t1_scores.get(m, {}).get('avg', 0)
    y_val = t2_scores.get(m, {}).get('avg', 0)
    c = model_color(m)
    marker = 's' if m in OUR_MODELS else 'o'
    ax1.scatter(x_val, y_val, c=c, s=100, marker=marker, edgecolors='white', linewidth=0.5, zorder=3)
    ax1.annotate(sn(m), (x_val, y_val), fontsize=6, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')
ax1.set_xlabel('T1 한국어 벤치 (정확도)')
ax1.set_ylabel('T2 Ko-Bench (Judge 1~10)')
ax1.set_title('한국어 능력: T1 vs T2', fontweight='bold')
ax1.axhline(y=5, color='gray', linestyle=':', alpha=0.3)
ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)

# Top-center: T4 코드/수학 vs T7 Elo
ax2 = fig.add_subplot(gs[0, 1])
for m in all_models:
    x_val = np.mean(list(t4_scores.get(m, {}).values())) if t4_scores.get(m) else 0
    y_val = t7_elo.get(m, {}).get('elo', 500)
    c = model_color(m)
    marker = 's' if m in OUR_MODELS else 'o'
    ax2.scatter(x_val, y_val, c=c, s=100, marker=marker, edgecolors='white', linewidth=0.5, zorder=3)
    ax2.annotate(sn(m), (x_val, y_val), fontsize=6, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')
ax2.set_xlabel('T4 코드/수학 (평균 정확도)')
ax2.set_ylabel('T7 Elo')
ax2.set_title('코딩 능력 vs 종합 Elo', fontweight='bold')

# Top-right: T6 속도 vs T7 Elo (속도-품질 트레이드오프)
ax3 = fig.add_subplot(gs[0, 2])
for m in all_models:
    x_val = t6_avg.get(m, 0)
    y_val = t7_elo.get(m, {}).get('elo', 500)
    c = model_color(m)
    marker = 's' if m in OUR_MODELS else 'o'
    ax3.scatter(x_val, y_val, c=c, s=100, marker=marker, edgecolors='white', linewidth=0.5, zorder=3)
    ax3.annotate(sn(m), (x_val, y_val), fontsize=6, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')
ax3.set_xlabel('T6 디코딩 속도 (tok/s)')
ax3.set_ylabel('T7 Elo')
ax3.set_title('속도 vs 품질 트레이드오프', fontweight='bold')

# Bottom-left: Frankenstallm 버전별 비교
ax4 = fig.add_subplot(gs[1, 0])
fs_models = [m for m in all_models if 'frankenstallm' in m]
fs_data = []
for m in fs_models:
    fs_data.append({
        'Model': sn(m),
        'T1': t1_scores.get(m, {}).get('avg', 0),
        'T2': t2_scores.get(m, {}).get('avg', 0) / 10,
        'T4': np.mean(list(t4_scores.get(m, {}).values())) if t4_scores.get(m) else 0,
        'T7': (t7_elo.get(m, {}).get('elo', 500) - 500) / 1100,
    })
df_fs = pd.DataFrame(fs_data).set_index('Model')
df_fs.plot(kind='bar', ax=ax4, width=0.8, alpha=0.85)
ax4.set_title('Frankenstallm 버전별 비교', fontweight='bold')
ax4.set_ylabel('정규화 점수')
ax4.tick_params(axis='x', rotation=45)
ax4.legend(fontsize=8)

# Bottom-center: 모델 크기별 T7 Elo (bubble chart)
ax5 = fig.add_subplot(gs[1, 1])
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
for m in all_models:
    size = model_sizes.get(m, 2)
    elo = t7_elo.get(m, {}).get('elo', 500)
    c = model_color(m)
    marker = 's' if m in OUR_MODELS else 'o'
    ax5.scatter(size, elo, c=c, s=size*40, marker=marker, alpha=0.7,
               edgecolors='white', linewidth=0.5, zorder=3)
    ax5.annotate(sn(m), (size, elo), fontsize=6, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')
ax5.set_xlabel('모델 크기 (GB)')
ax5.set_ylabel('T7 Elo')
ax5.set_title('모델 크기 vs Elo (효율성)', fontweight='bold')

# Bottom-right: 종합 랭킹 테이블
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

# Compute composite score
composite = {}
for m in all_models:
    score = (
        t1_scores.get(m, {}).get('avg', 0) * 15 +
        t2_scores.get(m, {}).get('avg', 0) / 10 * 20 +
        t3_scores.get(m, 0) / 10 * 15 +
        (np.mean(list(t4_scores.get(m, {}).values())) if t4_scores.get(m) else 0) * 15 +
        t5_scores.get(m, 0) * 5 * 10 +
        (t7_elo.get(m, {}).get('elo', 500) - 500) / 1100 * 25
    )
    composite[m] = score

sorted_comp = sorted(composite.items(), key=lambda x: x[1], reverse=True)

table_data = []
for i, (m, score) in enumerate(sorted_comp, 1):
    table_data.append([
        str(i),
        sn(m),
        f"{t1_scores.get(m, {}).get('avg', 0):.2f}",
        f"{t2_scores.get(m, {}).get('avg', 0):.1f}",
        f"{t3_scores.get(m, 0):.1f}",
        f"{np.mean(list(t4_scores.get(m, {}).values())):.0%}" if t4_scores.get(m) else "0%",
        f"{t7_elo.get(m, {}).get('elo', 0):.0f}",
        f"{score:.1f}",
    ])

table = ax6.table(
    cellText=table_data,
    colLabels=['#', 'Model', 'T1', 'T2', 'T3', 'T4', 'T7 Elo', 'Score'],
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.3)
# Color header
for j in range(8):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Color our models
for i, (m, _) in enumerate(sorted_comp, 1):
    if m in OUR_MODELS:
        for j in range(8):
            table[i, j].set_facecolor('#ffebee')

ax6.set_title('종합 랭킹 (가중 합산)', fontweight='bold', fontsize=12, pad=15)

fig.suptitle('FRANKENSTALLM 3B 심화 평가 — 16모델 × 7트랙 종합 대시보드',
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_DIR / '08_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 완료 + 요약 출력
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"시각화 생성 완료!")
print(f"출력 경로: {OUTPUT_DIR}/")
print(f"{'='*60}")
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  📊 {f.name} ({f.stat().st_size / 1024:.0f}KB)")

print(f"\n종합 랭킹 Top 5:")
for i, (m, score) in enumerate(sorted_comp[:5], 1):
    print(f"  {i}. {sn(m):20s} — {score:.1f}점 (Elo: {t7_elo.get(m, {}).get('elo', 0):.0f})")
