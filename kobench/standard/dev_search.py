"""dev 분할에서 설정 변형을 실행하고(configs/standard/dev_selection.yaml) 규칙대로 최적 설정을 고른다.

python -m kobench.standard.dev_search run --models jansori-bf16 frankenstallm-bf16
python -m kobench.standard.dev_search run --evafrill-checkpoints          # 1차: 체크포인트 7개 A
python -m kobench.standard.dev_search run --evafrill-variants evafrill-mo-3b-dpo-r2
python -m kobench.standard.dev_search run --public                         # 공개 모델(base/instruct 변형)
python -m kobench.standard.dev_search select                               # best_settings.yaml 갱신
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from kobench.standard.registry import CONFIG_DIR, load_models, load_suites
from kobench.standard.run import Variant, _out_paths, prefetch_datasets, run_model

RULE = CONFIG_DIR / "dev_selection.yaml"
BEST = CONFIG_DIR / "best_settings.yaml"


def load_rule() -> dict:
    return yaml.safe_load(RULE.read_text(encoding="utf-8"))


def variants_for(model_id: str, rule: dict, models) -> list[str]:
    sp = rule["search_space"]
    if model_id in sp["user_models"]:
        return sp["user_models"][model_id]
    ev = sp["user_models"]["evafrill_checkpoints"]
    if model_id in ev["checkpoints"]:
        vs = ev["variants_after_selection"]
        return vs if models[model_id].kind == "instruct" else [v for v in vs if "chat" not in v]
    return sp["public_models"][models[model_id].kind]


def dev_score(model_id: str, variant: Variant, tasks) -> dict | None:
    scores = {}
    for t in tasks:
        n = t.num_fewshot if variant.shots is None else variant.shots
        p, _ = _out_paths(model_id, t.task, n, None, variant)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        block = (d.get("groups") or {}).get(t.task) or d["results"][t.task]
        scores[t.task] = block.get("acc,none")
    return {"tasks": scores, "mean": sum(scores.values()) / len(scores)}


def run(model_variants: list[tuple[str, str]], gpu_mem: float) -> None:
    models, suites = load_models(), load_suites()
    tasks = suites[load_rule()["suite"]]["tasks"]
    prefetch_datasets(tasks)
    for mid, v in model_variants:
        run_model(models[mid], tasks, None, False, gpu_mem, "auto", Variant.parse(v))


def select(model_ids: list[str] | None = None) -> dict:
    rule = load_rule()
    models, suites = load_models(), load_suites()
    tasks = suites[rule["suite"]]["tasks"]
    best = yaml.safe_load(BEST.read_text(encoding="utf-8")) if BEST.exists() else {}
    best.setdefault("rule_file", str(RULE.name))
    best.setdefault("models", {})
    for mid in model_ids or list(models):
        cands = {}
        for v in variants_for(mid, rule, models):
            s = dev_score(mid, Variant.parse(v), tasks)
            if s is not None:
                cands[Variant.parse(v).name] = s
        if "A" not in cands or len(cands) < len(variants_for(mid, rule, models)):
            continue                                # 모든 변형이 끝난 모델만 선택
        top = max(cands, key=lambda k: cands[k]["mean"])
        chosen = top if cands[top]["mean"] - cands["A"]["mean"] > 0.001 else "A"
        best["models"][mid] = {"chosen": chosen, "dev": cands,
                               "gain_over_A_dev": round(cands[chosen]["mean"] - cands["A"]["mean"], 5),
                               "selected_at": datetime.now(timezone.utc).isoformat()}
    BEST.write_text(yaml.safe_dump(best, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="*", default=[])
    r.add_argument("--evafrill-checkpoints", action="store_true")
    r.add_argument("--evafrill-variants", default=None, help="1차에서 고른 EVAFRILL 체크포인트 id")
    r.add_argument("--public", action="store_true")
    r.add_argument("--gpu-mem", type=float, default=0.85)
    s = sub.add_parser("select")
    s.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args(argv)
    rule, models = load_rule(), load_models()
    if a.cmd == "select":
        print(yaml.safe_dump(select(a.models), allow_unicode=True, sort_keys=False)[:4000])
        return 0
    plan: list[tuple[str, str]] = []
    for mid in a.models:
        plan += [(mid, v) for v in variants_for(mid, rule, models)]
    if a.evafrill_checkpoints:
        plan += [(c, "A") for c in rule["search_space"]["user_models"]["evafrill_checkpoints"]["checkpoints"]]
    if a.evafrill_variants:
        plan += [(a.evafrill_variants, v) for v in variants_for(a.evafrill_variants, rule, models)]
    if a.public:
        for mid, spec in models.items():
            if spec.owner != "self":
                plan += [(mid, v) for v in variants_for(mid, rule, models)]
    print(f"dev 실행 계획 {len(plan)}건")
    run(plan, a.gpu_mem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
