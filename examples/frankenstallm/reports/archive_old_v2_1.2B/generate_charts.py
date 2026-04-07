#!/usr/bin/env python3
"""
FrankenStaLLM Benchmark Chart Generator
Generates 14 matplotlib charts (300dpi PNG) comparing 8 models across 7 tracks.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ---------------------------------------------------------------------------
# Korean font setup
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
CHARTS_DIR = BASE_DIR / 'reports' / 'charts'
SUMMARY_JSON = BASE_DIR / 'reports' / 'summary_data.json'
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    'frankenstallm-3b-v2-Q4_K_M',
    'frankenstallm-3b-v2-Q8_0',
    'frankenstallm-3b-v2-f16',
    'qwen2.5:3b',
    'gemma3:4b',
    'phi4-mini',
    'exaone3.5:2.4b',
    'llama3.2:3b',
]

SHORT_NAMES = {
    'frankenstallm-3b-v2-Q4_K_M': 'F-Q4',
    'frankenstallm-3b-v2-Q8_0': 'F-Q8',
    'frankenstallm-3b-v2-f16': 'F-F16',
    'qwen2.5:3b': 'Qwen',
    'gemma3:4b': 'Gemma',
    'phi4-mini': 'Phi4',
    'exaone3.5:2.4b': 'EXAONE',
    'llama3.2:3b': 'LLaMA',
}

# Colors: frankenstallm variants in red/orange, comparison models in blue/green
COLORS = {
    'frankenstallm-3b-v2-Q4_K_M': '#E53935',   # red
    'frankenstallm-3b-v2-Q8_0': '#FF7043',      # orange
    'frankenstallm-3b-v2-f16': '#FFB300',        # amber
    'qwen2.5:3b': '#1E88E5',                     # blue
    'gemma3:4b': '#43A047',                       # green
    'phi4-mini': '#5E35B1',                       # deep purple (blue family)
    'exaone3.5:2.4b': '#00897B',                  # teal
    'llama3.2:3b': '#039BE5',                     # light blue
}

DPI = 300

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load data from summary_data.json or directly from result JSONs."""
    data = {}

    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON) as f:
            raw = json.load(f)
        # summary_data.json has gpu_results containing track1/4/5/6
        # and track2/3/7 at top level
        if 'gpu_results' in raw:
            gpu = raw['gpu_results']
            for tk in ('track1', 'track4', 'track5', 'track6'):
                if tk in gpu:
                    data[tk] = gpu[tk]
        for tk in ('track2', 'track3', 'track7'):
            if tk in raw:
                data[tk] = raw[tk]
        # Load track7 comparisons from individual file if available
        t7_path = RESULTS_DIR / 'track7_pairwise_20260312_195322.json'
        if t7_path.exists():
            try:
                with open(t7_path) as f:
                    t7 = json.load(f)
                results = t7.get('results', {})
                if isinstance(results, dict):
                    data['track7_comparisons'] = results.get('comparisons', [])
            except Exception as e:
                print(f"  Warning: Could not load Track 7 comparisons: {e}")
        return data

    # Track 1
    t1_path = RESULTS_DIR / 'track1_korean_bench_20260312_181703.json'
    if t1_path.exists():
        with open(t1_path) as f:
            t1 = json.load(f)
        data['track1'] = t1.get('summary', {})

    # Track 4
    t4_path = RESULTS_DIR / 'track4_code_math_20260312_183644.json'
    if t4_path.exists():
        with open(t4_path) as f:
            t4 = json.load(f)
        data['track4'] = t4.get('summary', {})

    # Track 5
    t5_path = RESULTS_DIR / 'track5_consistency_20260312_192305.json'
    if t5_path.exists():
        with open(t5_path) as f:
            t5 = json.load(f)
        data['track5'] = t5.get('summary', {})

    # Track 6
    t6_path = RESULTS_DIR / 'track6_performance_20260312_192550.json'
    if t6_path.exists():
        with open(t6_path) as f:
            t6 = json.load(f)
        data['track6'] = t6.get('summary', {})

    # Track 7
    t7_path = RESULTS_DIR / 'track7_pairwise_20260312_195322.json'
    if t7_path.exists():
        with open(t7_path) as f:
            t7 = json.load(f)
        data['track7'] = t7.get('summary', {})
        # Also store comparisons for win matrix
        results = t7.get('results', {})
        if isinstance(results, dict):
            data['track7_comparisons'] = results.get('comparisons', [])
        else:
            data['track7_comparisons'] = []

    # Track 2 & 3 from full_results (load last since it's large)
    full_path = RESULTS_DIR / 'full_results_20260312_195323.json'
    if full_path.exists():
        try:
            with open(full_path) as f:
                full = json.load(f)
            if 'track2' in full:
                data['track2'] = full['track2'].get('summary', {})
            if 'track3' in full:
                data['track3'] = full['track3'].get('summary', {})
        except Exception as e:
            print(f"  Warning: Could not load full_results: {e}")

    return data


def short(model):
    return SHORT_NAMES.get(model, model)


def get_color(model):
    return COLORS.get(model, '#888888')


def get_models_present(summary_dict):
    """Return models from MODEL_ORDER that exist in the summary."""
    return [m for m in MODEL_ORDER if m in summary_dict]


# ---------------------------------------------------------------------------
# Chart 1: Overall ranking (normalized 0-1 average across all tracks)
# ---------------------------------------------------------------------------
def chart_01_overall_ranking(data):
    print("  Generating chart 01: Overall ranking...")
    track_scores = {}

    # Track 1: average of 5 benchmarks (values are 0-1 already)
    if 'track1' in data:
        for m in MODEL_ORDER:
            if m in data['track1']:
                vals = list(data['track1'][m].values())
                track_scores.setdefault(m, []).append(np.mean(vals))

    # Track 2: overall_mean across categories (scale is 1-10, normalize to 0-1)
    if 'track2' in data:
        for m in MODEL_ORDER:
            if m in data['track2']:
                cats = data['track2'][m]
                means = [cats[c]['overall_mean'] for c in cats if isinstance(cats[c], dict) and 'overall_mean' in cats[c]]
                if means:
                    track_scores.setdefault(m, []).append(np.mean(means) / 10.0)

    # Track 3: _overall avg_score (0-1 already)
    if 'track3' in data:
        for m in MODEL_ORDER:
            if m in data['track3']:
                overall = data['track3'][m].get('_overall', {})
                if 'avg_score' in overall:
                    track_scores.setdefault(m, []).append(overall['avg_score'])

    # Track 4: average of 4 metrics (0-1)
    if 'track4' in data:
        for m in MODEL_ORDER:
            if m in data['track4']:
                vals = list(data['track4'][m].values())
                track_scores.setdefault(m, []).append(np.mean(vals))

    # Track 5: average of 6 dimensions (0-1)
    if 'track5' in data:
        for m in MODEL_ORDER:
            if m in data['track5']:
                vals = list(data['track5'][m].values())
                track_scores.setdefault(m, []).append(np.mean(vals))

    # Track 6: normalize decode tok/s (higher is better, normalize by max)
    if 'track6' in data:
        decode_vals = {}
        for m in MODEL_ORDER:
            if m in data['track6']:
                decode_vals[m] = data['track6'][m].get('avg_decode_tok_s', 0)
        max_decode = max(decode_vals.values()) if decode_vals else 1
        for m, v in decode_vals.items():
            track_scores.setdefault(m, []).append(v / max_decode)

    # Track 7: normalize Elo (min-max across models)
    if 'track7' in data:
        elos = {}
        for m in MODEL_ORDER:
            if m in data['track7']:
                elos[m] = data['track7'][m].get('elo', 1000)
        if elos:
            min_elo = min(elos.values())
            max_elo = max(elos.values())
            rng = max_elo - min_elo if max_elo != min_elo else 1
            for m, e in elos.items():
                track_scores.setdefault(m, []).append((e - min_elo) / rng)

    # Compute average
    models_with_scores = [(m, np.mean(track_scores[m])) for m in MODEL_ORDER if m in track_scores]
    models_with_scores.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [short(m) for m, _ in models_with_scores]
    values = [v for _, v in models_with_scores]
    colors = [get_color(m) for m, _ in models_with_scores]

    bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.set_xlabel('정규화 평균 점수 (0-1)')
    ax.set_title('전체 트랙 종합 순위 (정규화 평균)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '01_overall_ranking.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: Track 1 heatmap
# ---------------------------------------------------------------------------
def chart_02_track1_heatmap(data):
    print("  Generating chart 02: Track 1 heatmap...")
    if 'track1' not in data:
        print("    SKIP: No Track 1 data")
        return

    t1 = data['track1']
    benchmarks = ['kmmlu', 'kobest_boolq', 'kobest_copa', 'kobest_sentineg', 'kobest_hellaswag']
    bench_labels = ['KMMLU', 'BoolQ', 'COPA', 'SentiNeg', 'HellaSwag']
    models = get_models_present(t1)

    matrix = []
    for m in models:
        row = [t1[m].get(b, 0) for b in benchmarks]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(bench_labels)))
    ax.set_xticklabels(bench_labels, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short(m) for m in models], fontsize=10)

    for i in range(len(models)):
        for j in range(len(benchmarks)):
            val = matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('정확도')
    ax.set_title('트랙 1: 한국어 벤치마크 정확도', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '02_track1_heatmap.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: Track 2 radar chart
# ---------------------------------------------------------------------------
def chart_03_track2_radar(data):
    print("  Generating chart 03: Track 2 radar...")
    if 'track2' not in data:
        print("    SKIP: No Track 2 data")
        return

    t2 = data['track2']
    categories = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
    cat_labels = ['작문', '역할극', '추론', '수학', '코딩', '추출', 'STEM', '인문학']
    models = get_models_present(t2)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    for m in models:
        vals = []
        for c in categories:
            if c in t2[m] and isinstance(t2[m][c], dict):
                vals.append(t2[m][c].get('overall_mean', 0))
            else:
                vals.append(0)
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', label=short(m), color=get_color(m), linewidth=1.5, markersize=4)
        ax.fill(angles, vals, alpha=0.05, color=get_color(m))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_title('트랙 2: Ko-Bench 카테고리별 점수 (레이더)', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '03_track2_radar.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Track 2 Turn1 vs Turn2
# ---------------------------------------------------------------------------
def chart_04_track2_turns(data):
    print("  Generating chart 04: Track 2 turns...")
    if 'track2' not in data:
        print("    SKIP: No Track 2 data")
        return

    t2 = data['track2']
    models = get_models_present(t2)

    turn1_means = []
    turn2_means = []
    for m in models:
        t1_vals = []
        t2_vals = []
        for cat in t2[m]:
            if isinstance(t2[m][cat], dict):
                if 'turn1_mean' in t2[m][cat]:
                    t1_vals.append(t2[m][cat]['turn1_mean'])
                if 'turn2_mean' in t2[m][cat]:
                    t2_vals.append(t2[m][cat]['turn2_mean'])
        turn1_means.append(np.mean(t1_vals) if t1_vals else 0)
        turn2_means.append(np.mean(t2_vals) if t2_vals else 0)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, turn1_means, width, label='Turn 1', color='#1E88E5')
    bars2 = ax.bar(x + width/2, turn2_means, width, label='Turn 2', color='#FF7043')

    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], fontsize=10)
    ax.set_ylabel('평균 점수 (1-10)')
    ax.set_title('트랙 2: Turn 1 vs Turn 2 평균 점수', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '04_track2_turns.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5: Track 3 horizontal bar
# ---------------------------------------------------------------------------
def chart_05_track3_scores(data):
    print("  Generating chart 05: Track 3 scores...")
    if 'track3' not in data:
        print("    SKIP: No Track 3 data")
        return

    t3 = data['track3']
    models = get_models_present(t3)

    scores = []
    for m in models:
        overall = t3[m].get('_overall', {})
        scores.append(overall.get('avg_score', 0))

    # Sort by score
    paired = sorted(zip(models, scores), key=lambda x: x[1])
    models_s, scores_s = zip(*paired)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [get_color(m) for m in models_s]
    bars = ax.barh([short(m) for m in models_s], scores_s, color=colors, edgecolor='white')

    for bar, val in zip(bars, scores_s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlim(0, max(scores_s) * 1.15 if scores_s else 1)
    ax.set_xlabel('평균 점수')
    ax.set_title('트랙 3: 한국어 심화 평가 평균 점수', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '05_track3_scores.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 6: Track 4 grouped bar
# ---------------------------------------------------------------------------
def chart_06_track4_grouped(data):
    print("  Generating chart 06: Track 4 grouped bar...")
    if 'track4' not in data:
        print("    SKIP: No Track 4 data")
        return

    t4 = data['track4']
    models = get_models_present(t4)
    metrics = ['python_pass1', 'sql_accuracy', 'debug_accuracy', 'math_accuracy']
    metric_labels = ['Python', 'SQL', 'Debug', 'Math']

    x = np.arange(len(models))
    width = 0.18
    offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2
    metric_colors = ['#1E88E5', '#43A047', '#FF7043', '#AB47BC']

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [t4[m].get(metric, 0) for m in models]
        bars = ax.bar(x + offsets[i] * width, vals, width, label=label, color=metric_colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], fontsize=10)
    ax.set_ylabel('정확도')
    ax.set_ylim(0, 1.1)
    ax.set_title('트랙 4: 코드/수학 벤치마크 정확도', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '06_track4_grouped.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 7: Track 5 radar
# ---------------------------------------------------------------------------
def chart_07_track5_radar(data):
    print("  Generating chart 07: Track 5 radar...")
    if 'track5' not in data:
        print("    SKIP: No Track 5 data")
        return

    t5 = data['track5']
    dims = ['repetition_consistency', 'paraphrase_robustness', 'length_sensitivity',
            'language_consistency', 'instruction_following', 'hallucination_detection']
    dim_labels = ['반복 일관성', '패러프레이즈', '길이 민감도', '언어 일관성', '지시 따르기', '환각 감지']
    models = get_models_present(t5)

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    for m in models:
        vals = [t5[m].get(d, 0) for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', label=short(m), color=get_color(m), linewidth=1.5, markersize=4)
        ax.fill(angles, vals, alpha=0.05, color=get_color(m))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('트랙 5: 일관성 평가 (레이더)', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '07_track5_radar.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 8: Track 6 decode speed bar
# ---------------------------------------------------------------------------
def chart_08_track6_speed(data):
    print("  Generating chart 08: Track 6 speed...")
    if 'track6' not in data:
        print("    SKIP: No Track 6 data")
        return

    t6 = data['track6']
    models = [m for m in MODEL_ORDER if m in t6]
    speeds = [t6[m].get('avg_decode_tok_s', 0) for m in models]

    # Sort by speed
    paired = sorted(zip(models, speeds), key=lambda x: x[1])
    models_s, speeds_s = zip(*paired) if paired else ([], [])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [get_color(m) for m in models_s]
    bars = ax.barh([short(m) for m in models_s], speeds_s, color=colors, edgecolor='white')

    for bar, val in zip(bars, speeds_s):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

    ax.set_xlabel('Decode tok/s')
    ax.set_title('트랙 6: 디코드 속도 (GPU)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '08_track6_speed.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 9: Track 6 concurrent throughput line
# ---------------------------------------------------------------------------
def chart_09_track6_concurrent(data):
    print("  Generating chart 09: Track 6 concurrent...")
    if 'track6' not in data:
        print("    SKIP: No Track 6 data")
        return

    t6 = data['track6']
    models = [m for m in MODEL_ORDER if m in t6]
    levels = ['1', '2', '4']

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        concurrent = t6[m].get('concurrent_aggregate_tok_s', {})
        vals = [concurrent.get(l, 0) for l in levels]
        ax.plot(levels, vals, 'o-', label=short(m), color=get_color(m), linewidth=2, markersize=6)

    ax.set_xlabel('동시 요청 수')
    ax.set_ylabel('총 처리량 (tok/s)')
    ax.set_title('트랙 6: 동시 요청 수별 총 처리량', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '09_track6_concurrent.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 10: Track 7 Elo ratings
# ---------------------------------------------------------------------------
def chart_10_track7_elo(data):
    print("  Generating chart 10: Track 7 Elo...")
    if 'track7' not in data:
        print("    SKIP: No Track 7 data")
        return

    t7 = data['track7']
    models = [m for m in MODEL_ORDER if m in t7]

    elos = [(m, t7[m]['elo'], t7[m].get('ci_lower', t7[m]['elo']),
             t7[m].get('ci_upper', t7[m]['elo'])) for m in models]
    elos.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [short(m) for m, _, _, _ in elos]
    values = [e for _, e, _, _ in elos]
    err_low = [e - cl for _, e, cl, _ in elos]
    err_high = [cu - e for _, e, _, cu in elos]
    colors = [get_color(m) for m, _, _, _ in elos]

    bars = ax.barh(labels, values, color=colors, edgecolor='white', xerr=[err_low, err_high],
                   capsize=3, error_kw={'linewidth': 1})

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

    ax.set_xlabel('Elo 등급')
    ax.set_title('트랙 7: Elo 등급 (오차 범위 포함)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '10_track7_elo.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 11: Track 7 win matrix heatmap
# ---------------------------------------------------------------------------
def chart_11_track7_winmatrix(data):
    print("  Generating chart 11: Track 7 win matrix...")
    if 'track7_comparisons' not in data and 'track7' not in data:
        print("    SKIP: No Track 7 data")
        return

    comparisons = data.get('track7_comparisons', [])
    if not comparisons:
        print("    SKIP: No Track 7 comparisons data")
        return

    models = [m for m in MODEL_ORDER if m in data.get('track7', {})]
    n = len(models)
    model_idx = {m: i for i, m in enumerate(models)}
    win_matrix = np.zeros((n, n))

    for r in comparisons:
        a = r.get('model_a', '')
        b = r.get('model_b', '')
        winner = r.get('winner', '')
        if a not in model_idx or b not in model_idx:
            continue
        ia, ib = model_idx[a], model_idx[b]
        if winner == a:
            win_matrix[ia, ib] += 1
        elif winner == b:
            win_matrix[ib, ia] += 1
        elif winner == 'TIE':
            win_matrix[ia, ib] += 0.5
            win_matrix[ib, ia] += 0.5

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(win_matrix, cmap='Blues', aspect='auto')

    labels = [short(m) for m in models]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('상대 모델')
    ax.set_ylabel('모델')

    for i in range(n):
        for j in range(n):
            val = win_matrix[i, j]
            if val > 0:
                color = 'white' if val > win_matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('승리 횟수')
    ax.set_title('트랙 7: 승리 매트릭스', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '11_track7_winmatrix.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 12: Quantization accuracy line
# ---------------------------------------------------------------------------
def chart_12_quantization_accuracy(data):
    print("  Generating chart 12: Quantization accuracy...")
    quant_models = [
        'frankenstallm-3b-v2-Q4_K_M',
        'frankenstallm-3b-v2-Q8_0',
        'frankenstallm-3b-v2-f16',
    ]
    quant_labels = ['Q4_K_M', 'Q8_0', 'F16']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Track 1 average
    if 'track1' in data:
        vals = []
        for m in quant_models:
            if m in data['track1']:
                vals.append(np.mean(list(data['track1'][m].values())))
            else:
                vals.append(0)
        ax.plot(quant_labels, vals, 'o-', label='트랙 1 평균', color='#E53935', linewidth=2, markersize=8)

    # Track 4 average
    if 'track4' in data:
        vals = []
        for m in quant_models:
            if m in data['track4']:
                vals.append(np.mean(list(data['track4'][m].values())))
            else:
                vals.append(0)
        ax.plot(quant_labels, vals, 's--', label='트랙 4 평균', color='#FF7043', linewidth=2, markersize=8)

    ax.set_xlabel('양자화 수준')
    ax.set_ylabel('정확도')
    ax.set_title('양자화 수준별 정확도 변화 (FrankenStaLLM)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '12_quantization_accuracy.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 13: Quantization speed bar
# ---------------------------------------------------------------------------
def chart_13_quantization_speed(data):
    print("  Generating chart 13: Quantization speed...")
    if 'track6' not in data:
        print("    SKIP: No Track 6 data")
        return

    quant_models = [
        'frankenstallm-3b-v2-Q4_K_M',
        'frankenstallm-3b-v2-Q8_0',
        'frankenstallm-3b-v2-f16',
    ]
    quant_labels = ['Q4_K_M', 'Q8_0', 'F16']
    quant_colors = ['#E53935', '#FF7043', '#FFB300']

    speeds = [data['track6'].get(m, {}).get('avg_decode_tok_s', 0) for m in quant_models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(quant_labels, speeds, color=quant_colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Decode tok/s')
    ax.set_title('양자화 수준별 디코드 속도 (FrankenStaLLM)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '13_quantization_speed.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 14: FrankenStaLLM overview radar vs best comparison model
# ---------------------------------------------------------------------------
def chart_14_frankenstallm_overview(data):
    print("  Generating chart 14: FrankenStaLLM overview...")
    frank = 'frankenstallm-3b-v2-Q4_K_M'
    comparison_models = ['qwen2.5:3b', 'gemma3:4b', 'phi4-mini', 'exaone3.5:2.4b', 'llama3.2:3b']

    track_names = ['트랙 1\n한국어 벤치', '트랙 2\nKo-Bench', '트랙 3\n심화 평가',
                   '트랙 4\n코드/수학', '트랙 5\n일관성', '트랙 6\n속도', '트랙 7\nElo']

    def get_track_score(model, data):
        scores = []
        # Track 1
        if 'track1' in data and model in data['track1']:
            scores.append(np.mean(list(data['track1'][model].values())))
        else:
            scores.append(0)
        # Track 2
        if 'track2' in data and model in data['track2']:
            cats = data['track2'][model]
            means = [cats[c]['overall_mean'] for c in cats if isinstance(cats[c], dict) and 'overall_mean' in cats[c]]
            scores.append(np.mean(means) / 10.0 if means else 0)
        else:
            scores.append(0)
        # Track 3
        if 'track3' in data and model in data['track3']:
            scores.append(data['track3'][model].get('_overall', {}).get('avg_score', 0))
        else:
            scores.append(0)
        # Track 4
        if 'track4' in data and model in data['track4']:
            scores.append(np.mean(list(data['track4'][model].values())))
        else:
            scores.append(0)
        # Track 5
        if 'track5' in data and model in data['track5']:
            scores.append(np.mean(list(data['track5'][model].values())))
        else:
            scores.append(0)
        # Track 6 (normalize decode speed)
        if 'track6' in data:
            all_speeds = [data['track6'][m].get('avg_decode_tok_s', 0) for m in MODEL_ORDER if m in data['track6']]
            max_speed = max(all_speeds) if all_speeds else 1
            if model in data['track6']:
                scores.append(data['track6'][model].get('avg_decode_tok_s', 0) / max_speed)
            else:
                scores.append(0)
        else:
            scores.append(0)
        # Track 7 (normalize Elo)
        if 'track7' in data:
            elos = {m: data['track7'][m]['elo'] for m in MODEL_ORDER if m in data['track7']}
            min_elo = min(elos.values()) if elos else 1000
            max_elo = max(elos.values()) if elos else 1000
            rng = max_elo - min_elo if max_elo != min_elo else 1
            if model in elos:
                scores.append((elos[model] - min_elo) / rng)
            else:
                scores.append(0)
        else:
            scores.append(0)
        return scores

    frank_scores = get_track_score(frank, data)

    # Find best comparison model (highest average)
    best_model = None
    best_avg = -1
    for m in comparison_models:
        s = get_track_score(m, data)
        avg = np.mean(s)
        if avg > best_avg:
            best_avg = avg
            best_model = m
            best_scores = s

    angles = np.linspace(0, 2 * np.pi, len(track_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    frank_vals = frank_scores + frank_scores[:1]
    best_vals = best_scores + best_scores[:1]

    ax.plot(angles, frank_vals, 'o-', label=f'{short(frank)} (FrankenStaLLM)',
            color='#E53935', linewidth=2.5, markersize=7)
    ax.fill(angles, frank_vals, alpha=0.15, color='#E53935')

    ax.plot(angles, best_vals, 's-', label=f'{short(best_model)} (최고 비교 모델)',
            color='#1E88E5', linewidth=2.5, markersize=7)
    ax.fill(angles, best_vals, alpha=0.15, color='#1E88E5')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(track_names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('FrankenStaLLM Q4 vs 최고 비교 모델 (전 트랙)', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '14_frankenstallm_overview.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    data = load_data()
    print(f"  Tracks loaded: {[k for k in data.keys() if k.startswith('track')]}")

    chart_funcs = [
        chart_01_overall_ranking,
        chart_02_track1_heatmap,
        chart_03_track2_radar,
        chart_04_track2_turns,
        chart_05_track3_scores,
        chart_06_track4_grouped,
        chart_07_track5_radar,
        chart_08_track6_speed,
        chart_09_track6_concurrent,
        chart_10_track7_elo,
        chart_11_track7_winmatrix,
        chart_12_quantization_accuracy,
        chart_13_quantization_speed,
        chart_14_frankenstallm_overview,
    ]

    success = 0
    failed = []
    for func in chart_funcs:
        try:
            func(data)
            success += 1
        except Exception as e:
            name = func.__name__
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print(f"\nDone: {success}/{len(chart_funcs)} charts generated successfully.")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    else:
        print(f"All charts saved to {CHARTS_DIR}/")


if __name__ == '__main__':
    main()
