#!/usr/bin/env python3
"""
FrankenStaLLM GPU Benchmark Chart Generator
GPU 모드 평가 결과 시각화 — 10개 모델, 14개 차트 (300dpi PNG)
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
CHARTS_DIR = BASE_DIR / 'reports' / 'charts_gpu'
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model definitions (10개 모델)
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
    'llama3.1:8b-instruct-q8_0',
    'ingu627/exaone4.0:1.2b',
]

SHORT_NAMES = {
    'frankenstallm-3b-v2-Q4_K_M': 'F-Q4',
    'frankenstallm-3b-v2-Q8_0': 'F-Q8',
    'frankenstallm-3b-v2-f16': 'F-F16',
    'qwen2.5:3b': 'Qwen2.5',
    'gemma3:4b': 'Gemma3',
    'phi4-mini': 'Phi4-mini',
    'exaone3.5:2.4b': 'EXAONE3.5',
    'llama3.2:3b': 'LLaMA3.2',
    'llama3.1:8b-instruct-q8_0': 'LLaMA3.1-8B',
    'ingu627/exaone4.0:1.2b': 'EXAONE4.0',
}

COLORS = {
    'frankenstallm-3b-v2-Q4_K_M': '#E53935',
    'frankenstallm-3b-v2-Q8_0': '#FF7043',
    'frankenstallm-3b-v2-f16': '#FFB300',
    'qwen2.5:3b': '#1E88E5',
    'gemma3:4b': '#43A047',
    'phi4-mini': '#5E35B1',
    'exaone3.5:2.4b': '#00897B',
    'llama3.2:3b': '#039BE5',
    'llama3.1:8b-instruct-q8_0': '#6D4C41',
    'ingu627/exaone4.0:1.2b': '#F06292',
}

DPI = 300


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    data = {}

    # Track 1 (10모델)
    p = RESULTS_DIR / 'track1_korean_bench_20260312_193138.json'
    if p.exists():
        with open(p) as f:
            data['track1'] = json.load(f).get('summary', {})

    # Track 2 (7모델 — 3모델 Judge 오류)
    p = RESULTS_DIR / 'track2_ko_bench_20260313_163926.json'
    if p.exists():
        with open(p) as f:
            data['track2'] = json.load(f).get('summary', {})

    # Track 3 (10모델)
    p = RESULTS_DIR / 'track3_korean_deep_20260313_191425.json'
    if p.exists():
        with open(p) as f:
            data['track3'] = json.load(f).get('summary', {})

    # Track 4 — 두 파일 합산 (8모델 + 2모델)
    t4_combined = {}
    for fname in ['track4_code_math_20260312_183644.json', 'track4_code_math_20260312_193725.json']:
        p = RESULTS_DIR / fname
        if p.exists():
            with open(p) as f:
                t4_combined.update(json.load(f).get('summary', {}))
    if t4_combined:
        data['track4'] = t4_combined

    # Track 5 (10모델 — 두 번째 파일)
    p = RESULTS_DIR / 'track5_consistency_20260312_194920.json'
    if p.exists():
        with open(p) as f:
            data['track5'] = json.load(f).get('summary', {})

    # Track 6 (10모델)
    p = RESULTS_DIR / 'track6_performance_20260312_192550.json'
    if p.exists():
        with open(p) as f:
            data['track6'] = json.load(f).get('summary', {})

    # Track 7 (10모델)
    p = RESULTS_DIR / 'track7_pairwise_20260314_014024.json'
    if p.exists():
        with open(p) as f:
            t7 = json.load(f)
        data['track7'] = t7.get('summary', {})
        results = t7.get('results', {})
        data['track7_comparisons'] = results.get('comparisons', []) if isinstance(results, dict) else []

    return data


def short(model):
    return SHORT_NAMES.get(model, model)


def get_color(model):
    return COLORS.get(model, '#888888')


def get_models_present(summary_dict):
    return [m for m in MODEL_ORDER if m in summary_dict]


# ---------------------------------------------------------------------------
# Chart 01: Overall ranking
# ---------------------------------------------------------------------------
def chart_01_overall_ranking(data):
    print("  [01] Overall ranking...")
    track_scores = {}

    if 'track1' in data:
        for m in MODEL_ORDER:
            if m in data['track1']:
                vals = list(data['track1'][m].values())
                track_scores.setdefault(m, []).append(np.mean(vals))

    if 'track2' in data:
        for m in MODEL_ORDER:
            if m in data['track2']:
                cats = data['track2'][m]
                means = [cats[c]['overall_mean'] for c in cats
                         if isinstance(cats[c], dict) and 'overall_mean' in cats[c]]
                if means:
                    track_scores.setdefault(m, []).append(np.mean(means) / 10.0)

    if 'track3' in data:
        for m in MODEL_ORDER:
            if m in data['track3']:
                score = data['track3'][m].get('_overall', {}).get('avg_score', 0)
                track_scores.setdefault(m, []).append(score)

    if 'track4' in data:
        for m in MODEL_ORDER:
            if m in data['track4']:
                vals = list(data['track4'][m].values())
                track_scores.setdefault(m, []).append(np.mean(vals))

    if 'track5' in data:
        for m in MODEL_ORDER:
            if m in data['track5']:
                vals = list(data['track5'][m].values())
                track_scores.setdefault(m, []).append(np.mean(vals))

    if 'track7' in data:
        elos = {m: data['track7'][m]['elo'] for m in MODEL_ORDER if m in data['track7']}
        if elos:
            min_elo, max_elo = min(elos.values()), max(elos.values())
            rng = max_elo - min_elo if max_elo != min_elo else 1
            for m, e in elos.items():
                track_scores.setdefault(m, []).append((e - min_elo) / rng)

    models_with_scores = [(m, np.mean(track_scores[m])) for m in MODEL_ORDER if m in track_scores]
    models_with_scores.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [short(m) for m, _ in models_with_scores]
    values = [v for _, v in models_with_scores]
    colors = [get_color(m) for m, _ in models_with_scores]

    bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.set_xlabel('정규화 평균 점수 (0-1)')
    ax.set_title('전체 트랙 종합 순위 (GPU 평가, 10모델)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '01_overall_ranking.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 02: Track 1 heatmap
# ---------------------------------------------------------------------------
def chart_02_track1_heatmap(data):
    print("  [02] Track 1 heatmap...")
    if 'track1' not in data:
        print("    SKIP: No Track 1 data"); return

    t1 = data['track1']
    benchmarks = ['kmmlu', 'kobest_boolq', 'kobest_copa', 'kobest_sentineg', 'kobest_hellaswag']
    bench_labels = ['KMMLU', 'BoolQ', 'COPA', 'SentiNeg', 'HellaSwag']
    models = get_models_present(t1)

    matrix = np.array([[t1[m].get(b, 0) for b in benchmarks] for m in models])

    fig, ax = plt.subplots(figsize=(11, 8))
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

    fig.colorbar(im, ax=ax, shrink=0.8).set_label('정확도')
    ax.set_title('Track 1: 한국어 벤치마크 정확도 (GPU, 10모델)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '02_track1_heatmap.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 03: Track 2 radar
# ---------------------------------------------------------------------------
def chart_03_track2_radar(data):
    print("  [03] Track 2 radar...")
    if 'track2' not in data:
        print("    SKIP: No Track 2 data"); return

    t2 = data['track2']
    categories = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
    cat_labels = ['작문', '역할극', '추론', '수학', '코딩', '추출', 'STEM', '인문학']
    models = get_models_present(t2)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    for m in models:
        vals = [t2[m][c]['overall_mean'] if c in t2[m] and isinstance(t2[m][c], dict) else 0
                for c in categories] + [0]
        vals[-1] = vals[0]
        ax.plot(angles, vals, 'o-', label=short(m), color=get_color(m), linewidth=1.5, markersize=4)
        ax.fill(angles, vals, alpha=0.05, color=get_color(m))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_title('Track 2: Ko-Bench 카테고리별 점수 (GPU, 7모델)', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '03_track2_radar.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 04: Track 2 Turn1 vs Turn2
# ---------------------------------------------------------------------------
def chart_04_track2_turns(data):
    print("  [04] Track 2 turns...")
    if 'track2' not in data:
        print("    SKIP: No Track 2 data"); return

    t2 = data['track2']
    models = get_models_present(t2)

    turn1_means, turn2_means = [], []
    for m in models:
        t1v = [t2[m][c]['turn1_mean'] for c in t2[m] if isinstance(t2[m][c], dict) and 'turn1_mean' in t2[m][c]]
        t2v = [t2[m][c]['turn2_mean'] for c in t2[m] if isinstance(t2[m][c], dict) and 'turn2_mean' in t2[m][c]]
        turn1_means.append(np.mean(t1v) if t1v else 0)
        turn2_means.append(np.mean(t2v) if t2v else 0)

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width / 2, turn1_means, width, label='Turn 1', color='#1E88E5')
    b2 = ax.bar(x + width / 2, turn2_means, width, label='Turn 2', color='#FF7043')

    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('평균 점수 (1-10)')
    ax.set_title('Track 2: Turn 1 vs Turn 2 평균 점수 (GPU)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in list(b1) + list(b2):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '04_track2_turns.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 05: Track 3 horizontal bar
# ---------------------------------------------------------------------------
def chart_05_track3_scores(data):
    print("  [05] Track 3 scores...")
    if 'track3' not in data:
        print("    SKIP: No Track 3 data"); return

    t3 = data['track3']
    models = get_models_present(t3)
    scores = [t3[m].get('_overall', {}).get('avg_score', 0) for m in models]

    paired = sorted(zip(models, scores), key=lambda x: x[1])
    models_s, scores_s = zip(*paired)

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh([short(m) for m in models_s], scores_s,
                   color=[get_color(m) for m in models_s], edgecolor='white')
    for bar, val in zip(bars, scores_s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlim(0, max(scores_s) * 1.15)
    ax.set_xlabel('평균 점수 (0-1)')
    ax.set_title('Track 3: 한국어 심화 평균 점수 (GPU, 10모델)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '05_track3_scores.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 06: Track 4 grouped bar
# ---------------------------------------------------------------------------
def chart_06_track4_grouped(data):
    print("  [06] Track 4 grouped bar...")
    if 'track4' not in data:
        print("    SKIP: No Track 4 data"); return

    t4 = data['track4']
    models = get_models_present(t4)
    metrics = ['python_pass1', 'sql_accuracy', 'debug_accuracy', 'math_accuracy']
    metric_labels = ['Python', 'SQL', 'Debug', 'Math']
    metric_colors = ['#1E88E5', '#43A047', '#FF7043', '#AB47BC']

    x = np.arange(len(models))
    width = 0.18
    offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [t4[m].get(metric, 0) for m in models]
        ax.bar(x + offsets[i] * width, vals, width, label=label, color=metric_colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], fontsize=9, rotation=10, ha='right')
    ax.set_ylabel('정확도')
    ax.set_ylim(0, 1.1)
    ax.set_title('Track 4: 코드/수학 벤치마크 정확도 (GPU, 10모델)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '06_track4_grouped.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 07: Track 5 radar
# ---------------------------------------------------------------------------
def chart_07_track5_radar(data):
    print("  [07] Track 5 radar...")
    if 'track5' not in data:
        print("    SKIP: No Track 5 data"); return

    t5 = data['track5']
    dims = ['repetition_consistency', 'paraphrase_robustness', 'length_sensitivity',
            'language_consistency', 'instruction_following', 'hallucination_detection']
    dim_labels = ['반복 일관성', '패러프레이즈', '길이 민감도', '언어 일관성', '지시 준수', '환각 탐지']
    models = get_models_present(t5)

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw=dict(polar=True))
    for m in models:
        vals = [t5[m].get(d, 0) for d in dims] + [t5[m].get(dims[0], 0)]
        ax.plot(angles, vals, 'o-', label=short(m), color=get_color(m), linewidth=1.5, markersize=4)
        ax.fill(angles, vals, alpha=0.04, color=get_color(m))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('Track 5: 일관성 & 강건성 (GPU, 10모델)', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '07_track5_radar.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 08: Track 6 decode speed
# ---------------------------------------------------------------------------
def chart_08_track6_speed(data):
    print("  [08] Track 6 speed...")
    if 'track6' not in data:
        print("    SKIP: No Track 6 data"); return

    t6 = data['track6']
    models = [m for m in MODEL_ORDER if m in t6]
    speeds = [t6[m].get('avg_decode_tok_s', 0) for m in models]
    paired = sorted(zip(models, speeds), key=lambda x: x[1])
    models_s, speeds_s = zip(*paired)

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh([short(m) for m in models_s], speeds_s,
                   color=[get_color(m) for m in models_s], edgecolor='white')
    for bar, val in zip(bars, speeds_s):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=9)
    ax.set_xlabel('Decode tok/s')
    ax.set_title('Track 6: GPU 디코드 속도 (10모델)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '08_track6_speed.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 09: Track 6 concurrent throughput
# ---------------------------------------------------------------------------
def chart_09_track6_concurrent(data):
    print("  [09] Track 6 concurrent...")
    if 'track6' not in data:
        print("    SKIP: No Track 6 data"); return

    t6 = data['track6']
    models = [m for m in MODEL_ORDER if m in t6]
    levels = ['1', '2', '4']

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        conc = t6[m].get('concurrent_aggregate_tok_s', {})
        vals = [conc.get(l, 0) for l in levels]
        ax.plot(levels, vals, 'o-', label=short(m), color=get_color(m), linewidth=2, markersize=6)

    ax.set_xlabel('동시 요청 수')
    ax.set_ylabel('총 처리량 (tok/s)')
    ax.set_title('Track 6: 동시 요청 수별 총 처리량 (GPU)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
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
    print("  [10] Track 7 Elo...")
    if 'track7' not in data:
        print("    SKIP: No Track 7 data"); return

    t7 = data['track7']
    models = [m for m in MODEL_ORDER if m in t7]
    elos = [(m, t7[m]['elo'], t7[m].get('ci_lower', t7[m]['elo']), t7[m].get('ci_upper', t7[m]['elo']))
            for m in models]
    elos.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [short(m) for m, *_ in elos]
    values = [e for _, e, *_ in elos]
    err_low = [e - cl for _, e, cl, _ in elos]
    err_high = [cu - e for _, e, _, cu in elos]
    colors = [get_color(m) for m, *_ in elos]

    bars = ax.barh(labels, values, color=colors, edgecolor='white',
                   xerr=[err_low, err_high], capsize=3, error_kw={'linewidth': 1})
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 25, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=9)

    ax.set_xlabel('Elo 점수')
    ax.set_title('Track 7: Pairwise Elo 랭킹 (GPU, 10모델, 1800 판정)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '10_track7_elo.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 11: Track 7 win matrix
# ---------------------------------------------------------------------------
def chart_11_track7_winmatrix(data):
    print("  [11] Track 7 win matrix...")
    comparisons = data.get('track7_comparisons', [])
    if not comparisons:
        print("    SKIP: No comparisons data"); return

    models = [m for m in MODEL_ORDER if m in data.get('track7', {})]
    n = len(models)
    model_idx = {m: i for i, m in enumerate(models)}
    win_matrix = np.zeros((n, n))

    for r in comparisons:
        a, b, winner = r.get('model_a', ''), r.get('model_b', ''), r.get('winner', '')
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

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(win_matrix, cmap='Blues', aspect='auto')
    labels = [short(m) for m in models]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('상대 모델'); ax.set_ylabel('모델')

    for i in range(n):
        for j in range(n):
            val = win_matrix[i, j]
            if val > 0:
                color = 'white' if val > win_matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8).set_label('승리 횟수')
    ax.set_title('Track 7: 승리 매트릭스 (GPU, 10모델)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '11_track7_winmatrix.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 12: Quantization accuracy
# ---------------------------------------------------------------------------
def chart_12_quantization_accuracy(data):
    print("  [12] Quantization accuracy...")
    quant_models = ['frankenstallm-3b-v2-Q4_K_M', 'frankenstallm-3b-v2-Q8_0', 'frankenstallm-3b-v2-f16']
    quant_labels = ['Q4_K_M', 'Q8_0', 'F16']

    fig, ax = plt.subplots(figsize=(8, 5))
    if 'track1' in data:
        vals = [np.mean(list(data['track1'][m].values())) if m in data['track1'] else 0 for m in quant_models]
        ax.plot(quant_labels, vals, 'o-', label='Track 1 평균', color='#E53935', linewidth=2, markersize=8)
    if 'track5' in data:
        vals = [np.mean(list(data['track5'][m].values())) if m in data['track5'] else 0 for m in quant_models]
        ax.plot(quant_labels, vals, 's--', label='Track 5 일관성', color='#1E88E5', linewidth=2, markersize=8)

    ax.set_xlabel('양자화 수준')
    ax.set_ylabel('점수 (0-1)')
    ax.set_title('FRANKENSTALLM: 양자화별 정확도/일관성 (GPU)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '12_quantization_accuracy.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 13: Quantization speed
# ---------------------------------------------------------------------------
def chart_13_quantization_speed(data):
    print("  [13] Quantization speed...")
    if 'track6' not in data:
        print("    SKIP: No Track 6 data"); return

    quant_models = ['frankenstallm-3b-v2-Q4_K_M', 'frankenstallm-3b-v2-Q8_0', 'frankenstallm-3b-v2-f16']
    quant_labels = ['Q4_K_M', 'Q8_0', 'F16']
    quant_colors = ['#E53935', '#FF7043', '#FFB300']
    speeds = [data['track6'].get(m, {}).get('avg_decode_tok_s', 0) for m in quant_models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(quant_labels, speeds, color=quant_colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Decode tok/s')
    ax.set_title('FRANKENSTALLM: 양자화별 GPU 디코드 속도', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '13_quantization_speed.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 14: FrankenStaLLM overview radar vs best model
# ---------------------------------------------------------------------------
def chart_14_frankenstallm_overview(data):
    print("  [14] FrankenStaLLM overview...")
    frank = 'frankenstallm-3b-v2-Q4_K_M'
    comparison_models = [m for m in MODEL_ORDER if 'frankenstallm' not in m]
    track_names = ['T1\n한국어 벤치', 'T2\nKo-Bench', 'T3\n심화 평가',
                   'T4\n코드/수학', 'T5\n일관성', 'T6\n속도', 'T7\nElo']

    def get_scores(model):
        scores = []
        # T1
        scores.append(np.mean(list(data['track1'][model].values())) if 'track1' in data and model in data['track1'] else 0)
        # T2
        if 'track2' in data and model in data['track2']:
            cats = data['track2'][model]
            means = [cats[c]['overall_mean'] for c in cats if isinstance(cats[c], dict) and 'overall_mean' in cats[c]]
            scores.append(np.mean(means) / 10.0 if means else 0)
        else:
            scores.append(0)
        # T3
        scores.append(data['track3'][model].get('_overall', {}).get('avg_score', 0) if 'track3' in data and model in data['track3'] else 0)
        # T4
        scores.append(np.mean(list(data['track4'][model].values())) if 'track4' in data and model in data['track4'] else 0)
        # T5
        scores.append(np.mean(list(data['track5'][model].values())) if 'track5' in data and model in data['track5'] else 0)
        # T6
        if 'track6' in data:
            all_spd = [data['track6'][m].get('avg_decode_tok_s', 0) for m in MODEL_ORDER if m in data['track6']]
            mx = max(all_spd) if all_spd else 1
            scores.append(data['track6'].get(model, {}).get('avg_decode_tok_s', 0) / mx)
        else:
            scores.append(0)
        # T7
        if 'track7' in data:
            elos = {m: data['track7'][m]['elo'] for m in MODEL_ORDER if m in data['track7']}
            mn, mx = min(elos.values()), max(elos.values())
            rng = mx - mn if mx != mn else 1
            scores.append((elos.get(model, mn) - mn) / rng)
        else:
            scores.append(0)
        return scores

    frank_scores = get_scores(frank)
    best_model = max(comparison_models, key=lambda m: np.mean(get_scores(m)))
    best_scores = get_scores(best_model)

    angles = np.linspace(0, 2 * np.pi, len(track_names), endpoint=False).tolist() + [0]
    frank_vals = frank_scores + frank_scores[:1]
    best_vals = best_scores + best_scores[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, frank_vals, 'o-', label=f'FRANKENSTALLM Q4 (v2)', color='#E53935', linewidth=2.5, markersize=7)
    ax.fill(angles, frank_vals, alpha=0.15, color='#E53935')
    ax.plot(angles, best_vals, 's-', label=f'{short(best_model)} (최고 비교 모델)', color='#1E88E5', linewidth=2.5, markersize=7)
    ax.fill(angles, best_vals, alpha=0.15, color='#1E88E5')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(track_names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('FRANKENSTALLM Q4 vs 최고 비교 모델 (GPU, 전 트랙)', fontsize=13, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / '14_frankenstallm_overview.png', dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading GPU evaluation data...")
    data = load_data()
    print(f"  Tracks loaded: {[k for k in data if k.startswith('track')]}")
    print(f"  Models per track: ", {k: len(v) for k, v in data.items() if k.startswith('track') and isinstance(v, dict)})

    chart_funcs = [
        chart_01_overall_ranking, chart_02_track1_heatmap,
        chart_03_track2_radar,    chart_04_track2_turns,
        chart_05_track3_scores,   chart_06_track4_grouped,
        chart_07_track5_radar,    chart_08_track6_speed,
        chart_09_track6_concurrent, chart_10_track7_elo,
        chart_11_track7_winmatrix,  chart_12_quantization_accuracy,
        chart_13_quantization_speed, chart_14_frankenstallm_overview,
    ]

    success, failed = 0, []
    for func in chart_funcs:
        try:
            func(data)
            success += 1
        except Exception as e:
            print(f"  ERROR in {func.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(func.__name__)

    print(f"\nDone: {success}/{len(chart_funcs)} charts → {CHARTS_DIR}/")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == '__main__':
    main()
