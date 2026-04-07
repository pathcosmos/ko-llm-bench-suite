"""
점수 집계 및 Bradley-Terry Elo 모델 피팅
"""

import math
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Optional

from scipy.optimize import minimize

from . import config


# ── 트랙별 점수 집계 ─────────────────────────────────────────────────────────

def aggregate_accuracy(results: list[dict], model_key: str = "model") -> dict[str, float]:
    """정확도 기반 트랙 (Track 1, 4) 집계 — 모델별 평균 정확도"""
    model_scores = defaultdict(list)
    for r in results:
        if r.get("correct") is not None:
            model_scores[r[model_key]].append(1.0 if r["correct"] else 0.0)
    return {m: sum(s) / len(s) if s else 0.0 for m, s in model_scores.items()}


def aggregate_judge_scores(results: list[dict], model_key: str = "model") -> dict[str, dict]:
    """LLM Judge 점수 트랙 (Track 2, 3) 집계"""
    model_scores = defaultdict(list)
    model_by_cat = defaultdict(lambda: defaultdict(list))

    for r in results:
        score = r.get("judge_score") or r.get("score", 0)
        if score > 0:
            model_scores[r[model_key]].append(score)
            cat = r.get("category", "general")
            model_by_cat[r[model_key]][cat].append(score)

    summary = {}
    for model in model_scores:
        scores = model_scores[model]
        cats = model_by_cat[model]
        summary[model] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": float(np.median(scores)),
            "n": len(scores),
            "by_category": {
                cat: {"mean": np.mean(s), "n": len(s)}
                for cat, s in cats.items()
            },
        }
    return summary


def aggregate_performance(results: list[dict]) -> dict[str, dict]:
    """Track 6 성능 메트릭 집계"""
    model_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        m = r["model"]
        for key in ["tokens_per_sec", "prefill_tok_s", "ttft_s", "vram_used_mb"]:
            if key in r and r[key]:
                model_data[m][key].append(r[key])

    summary = {}
    for model, metrics in model_data.items():
        summary[model] = {}
        for key, vals in metrics.items():
            summary[model][key] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary


# ── Bradley-Terry Elo ─────────────────────────────────────────────────────────

def fit_bradley_terry(
    comparisons: list[dict],
    models: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Bradley-Terry 모델로 Elo 점수 산출

    Args:
        comparisons: [{"model_a": str, "model_b": str, "winner": "A"|"B"|"TIE"}, ...]
        models: 모델 목록 (None이면 자동 추출)

    Returns:
        {model_name: {"elo": float, "ci_lower": float, "ci_upper": float, "wins": int, "losses": int}}
    """
    if models is None:
        model_set = set()
        for c in comparisons:
            model_set.add(c["model_a"])
            model_set.add(c["model_b"])
        models = sorted(model_set)

    n = len(models)
    if n < 2:
        return {m: {"elo": 1000, "ci_lower": 1000, "ci_upper": 1000, "wins": 0, "losses": 0} for m in models}

    model_idx = {m: i for i, m in enumerate(models)}

    # 승패 매트릭스 구성
    wins = np.zeros((n, n))
    for c in comparisons:
        i = model_idx.get(c["model_a"])
        j = model_idx.get(c["model_b"])
        if i is None or j is None:
            continue
        if c["winner"] == "A":
            wins[i][j] += 1
        elif c["winner"] == "B":
            wins[j][i] += 1
        else:  # TIE
            wins[i][j] += 0.5
            wins[j][i] += 0.5

    # BT 모델 피팅: log-likelihood 최대화
    def neg_log_likelihood(params):
        nll = 0
        for i in range(n):
            for j in range(i + 1, n):
                total = wins[i][j] + wins[j][i]
                if total == 0:
                    continue
                p_i = 1.0 / (1.0 + math.exp(params[j] - params[i]))
                p_i = max(min(p_i, 1 - 1e-10), 1e-10)
                nll -= wins[i][j] * math.log(p_i)
                nll -= wins[j][i] * math.log(1 - p_i)
        # 정규화: 첫 번째 파라미터 고정 (anchor)
        nll += 100 * params[0] ** 2
        return nll

    result = minimize(neg_log_likelihood, np.zeros(n), method="L-BFGS-B")
    strengths = result.x

    # 부트스트랩으로 신뢰구간
    n_bootstrap = 1000
    bootstrap_elos = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        indices = np.random.choice(len(comparisons), size=len(comparisons), replace=True)
        boot_wins = np.zeros((n, n))
        for idx in indices:
            c = comparisons[idx]
            i = model_idx.get(c["model_a"])
            j = model_idx.get(c["model_b"])
            if i is None or j is None:
                continue
            if c["winner"] == "A":
                boot_wins[i][j] += 1
            elif c["winner"] == "B":
                boot_wins[j][i] += 1
            else:
                boot_wins[i][j] += 0.5
                boot_wins[j][i] += 0.5

        def boot_nll(params):
            nll = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total = boot_wins[i][j] + boot_wins[j][i]
                    if total == 0:
                        continue
                    p_i = 1.0 / (1.0 + math.exp(params[j] - params[i]))
                    p_i = max(min(p_i, 1 - 1e-10), 1e-10)
                    nll -= boot_wins[i][j] * math.log(p_i)
                    nll -= boot_wins[j][i] * math.log(1 - p_i)
            nll += 100 * params[0] ** 2
            return nll

        boot_result = minimize(boot_nll, strengths, method="L-BFGS-B")
        bootstrap_elos[b] = boot_result.x

    # Elo 스케일 변환 (강도 → Elo: base=1000, scale=400)
    base_elo = 1000
    scale = 400 / math.log(10)

    scores = {}
    for i, model in enumerate(models):
        elo = base_elo + scale * strengths[i]
        boot_elo = base_elo + scale * bootstrap_elos[:, i]
        ci_lower = float(np.percentile(boot_elo, 2.5))
        ci_upper = float(np.percentile(boot_elo, 97.5))

        total_wins = int(np.sum(wins[i]))
        total_losses = int(np.sum(wins[:, i]))

        scores[model] = {
            "elo": round(elo, 1),
            "ci_lower": round(ci_lower, 1),
            "ci_upper": round(ci_upper, 1),
            "wins": total_wins,
            "losses": total_losses,
        }

    return scores


# ── 종합 스코어 카드 ─────────────────────────────────────────────────────────

def _extract_representative_score(track_name: str, val) -> float | None:
    """트랙별 summary 값에서 대표 점수 하나를 추출한다."""
    if not isinstance(val, dict):
        return val

    # mean / elo 키가 있으면 그대로
    if "mean" in val:
        return val["mean"]
    if "elo" in val:
        return val["elo"]

    # Track 1: 카테고리별 accuracy → 전체 평균
    # e.g. {"kmmlu": 0.86, "kobest_boolq": 1.0, ...}
    if track_name == "track1":
        nums = [v for v in val.values() if isinstance(v, (int, float))]
        return round(sum(nums) / len(nums), 4) if nums else None

    # Track 3: 카테고리별 {accuracy, avg_score, n} → n 가중 avg_score 평균
    if track_name == "track3":
        total_n, weighted = 0, 0.0
        for cat_data in val.values():
            if isinstance(cat_data, dict) and "avg_score" in cat_data:
                n = cat_data.get("n", 1)
                weighted += cat_data["avg_score"] * n
                total_n += n
        return round(weighted / total_n, 4) if total_n else None

    # Track 5: 메트릭별 float → 평균
    if track_name == "track5":
        nums = [v for v in val.values() if isinstance(v, (int, float))]
        return round(sum(nums) / len(nums), 4) if nums else None

    # Track 6: 성능 → decode tok/s 대표값
    if track_name == "track6":
        return val.get("avg_decode_tok_s")

    # Track 2: {category: {turn1_mean, turn2_mean, overall_mean}} → overall_mean 평균
    if track_name == "track2":
        means = []
        for cat_data in val.values():
            if isinstance(cat_data, dict) and "overall_mean" in cat_data:
                means.append(cat_data["overall_mean"])
        return round(sum(means) / len(means), 4) if means else None

    # 기타: 숫자 값들의 평균
    nums = [v for v in val.values() if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else 0


def build_scorecard(track_results: dict[str, dict]) -> dict[str, dict]:
    """
    모든 트랙 결과를 하나의 스코어카드로 통합

    Args:
        track_results: {"track1": {model: score, ...}, "track2": {model: {mean: ...}}, ...}
    """
    scorecard = {}
    for model in config.ALL_MODELS:
        scorecard[model] = {"model": model}
        for track_name, data in track_results.items():
            if model in data:
                score = _extract_representative_score(track_name, data[model])
                if score is not None:
                    scorecard[model][track_name] = score

    return scorecard


def save_scorecard(scorecard: dict, path: Optional[Path] = None) -> Path:
    if path is None:
        path = config.RESULTS_DIR / "scorecard.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, ensure_ascii=False, indent=2, default=str)
    return path
