"""표준 벤치 결과 집계 — base/instruct별 표, 문항 bootstrap 95% CI, 기준 모델 대비 paired 차이.

python -m kobench.standard.report --suite ko_mc_v1 [--ref jansori-bf16] [--table A|B] [--out reports/standard]

지표 정의
- lm-eval: lm-evaluation-harness가 보고한 태스크/그룹 점수 (그룹은 lm-eval의 집계 규칙을 따름)
- micro:   문항별 샘플(.samples.jsonl.gz)에서 하위 태스크를 모두 합쳐 낸 값. CI·paired 차이는 이 값 기준.
           정확도류는 문항 평균, bits_per_byte는 코퍼스 단위(−Σloglik / ln2 / Σbytes, lm-eval과 같은 정의)이며
           bootstrap도 문서 단위로 재표집해 같은 비율을 다시 계산한다.
- Δref:    같은 문항 집합에서 (모델 − 기준 모델) micro 차이, paired bootstrap 95% 구간
- ΔA:      (B 표만) 같은 모델의 (선택 설정 − A) micro 차이, paired bootstrap 95% 구간

A 표 = 공통 프로토콜(모든 모델 같은 프롬프트·shot·채점). B 표 = configs/standard/best_settings.yaml에서 dev로 고른 설정.
best_settings에 없는 모델은 A로 둔다. 로그우도 PPL(bits_per_byte)은 chat·문서 시작 변형의 영향을 받지 않으므로
(A에서 이미 문서 앞 EOS로 계산) B 표에서도 A 결과를 쓴다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import yaml

from kobench.standard.registry import CONFIG_DIR, load_models, load_suites
from kobench.standard.run import Variant, _out_paths

ROOT = Path(__file__).resolve().parents[2]
N_BOOT = 2000
SEED = 1234
RATIO_METRICS = {"bits_per_byte"}          # 문항값이 (loglik, bytes) 쌍인 코퍼스 비율 지표
LOWER_BETTER = {"bits_per_byte"}


def _load(model_id: str, task: str, nshot: int, limit=None, variant: Variant | None = None):
    j, s = _out_paths(model_id, task, nshot, limit, variant)
    if not j.exists():
        return None, None
    doc = json.loads(j.read_text(encoding="utf-8"))
    items = {}
    if s.exists():
        with gzip.open(s, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                items[(r["subtask"], r["doc_id"])] = r
    return doc, items


def _headline(doc: dict, task: str, metric: str) -> float | None:
    for block in (doc.get("groups") or {}, doc.get("results") or {}):
        if task in block:
            v = block[task].get(f"{metric},none")
            if v is None:
                v = block[task].get("acc,none")
            return v if isinstance(v, (int, float)) else None
    return None


def _item_scores(items: dict, metric: str) -> dict:
    """문항 키 → (분자, 분모). 정확도류는 (점수, 1), bits_per_byte는 (−loglik/ln2, bytes)."""
    out = {}
    for k, r in items.items():
        v = r.get(metric)
        if metric in RATIO_METRICS:
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[k] = (-float(v[0]) / math.log(2), float(v[1]))
            continue
        if v is None:
            v = r.get("acc")
        if v is not None:
            out[k] = (float(v), 1.0)
    return out


def _as_pairs(x) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(x, dict):
        x = list(x.values())
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        return a, np.ones_like(a)
    return a[:, 0], a[:, 1]


def micro(x) -> float:
    num, den = _as_pairs(x)
    return float(num.sum() / den.sum()) if len(num) else float("nan")


def bootstrap_ci(x, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """x: 1차원 점수 배열 또는 (분자, 분모) 쌍 배열. 문항 단위 재표집으로 Σ분자/Σ분모의 95% 구간."""
    num, den = _as_pairs(x)
    if len(num) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    m = num[idx].sum(1) / den[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_diff(a: dict, b: dict) -> tuple[float, float, float, int]:
    """공통 문항에서 micro(a) − micro(b)와 같은 재표집 인덱스를 쓰는 paired bootstrap 95% 구간."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return (float("nan"),) * 3 + (0,)
    an, ad = _as_pairs([a[k] if isinstance(a[k], tuple) else (a[k], 1.0) for k in keys])
    bn, bd = _as_pairs([b[k] if isinstance(b[k], tuple) else (b[k], 1.0) for k in keys])
    d = an.sum() / ad.sum() - bn.sum() / bd.sum()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    boot = an[idx].sum(1) / ad[idx].sum(1) - bn[idx].sum(1) / bd[idx].sum(1)
    return float(d), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(keys)


def chosen_variants(path: Path | None = None) -> dict[str, Variant]:
    p = path or CONFIG_DIR / "best_settings.yaml"
    if not p.exists():
        return {}
    best = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {m: Variant.from_name(v.get("chosen")) for m, v in (best.get("models") or {}).items()}


def _task_entry(mid, t, limit, variant):
    nshot = t.num_fewshot if variant.shots is None else variant.shots
    doc, items = _load(mid, t.task, nshot, limit, variant)
    if doc is None:
        return None
    sc = _item_scores(items, t.metric)
    return {"lmeval": _headline(doc, t.task, t.metric),
            "micro": micro(sc) if sc else None,
            "ci": bootstrap_ci(list(sc.values())) if sc else None,
            "n": len(sc), "nshot": nshot, "_items": sc}


def build(suite: str, ref: str | None, limit=None, table: str = "A", best_path: Path | None = None) -> dict:
    models, suites = load_models(), load_suites()
    tasks = suites[suite]["tasks"]
    chosen = chosen_variants(best_path) if table == "B" else {}
    rows = []
    for mid, spec in models.items():
        var = chosen.get(mid, Variant())
        per_task = {}
        for t in tasks:
            v = var if t.metric not in RATIO_METRICS else Variant()
            e = _task_entry(mid, t, limit, v)
            if e is None:
                continue
            if table == "B" and v.name != "A":
                a = _task_entry(mid, t, limit, Variant())
                if a is not None:
                    e["delta_A"] = paired_diff(e["_items"], a["_items"])
            per_task[t.task] = e
        if per_task:
            rows.append({"model": mid, "spec": spec, "variant": var.name, "tasks": per_task})
    if ref:
        ref_row = next((r for r in rows if r["model"] == ref), None)
        for r in rows:
            for tname, v in r["tasks"].items():
                if ref_row and tname in ref_row["tasks"] and r is not ref_row:
                    v["delta_ref"] = paired_diff(v["_items"], ref_row["tasks"][tname]["_items"])
    return {"suite": suite, "tasks": tasks, "rows": rows, "ref": ref, "table": table}


def _fmt(v, metric="acc"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.3f}" if metric in RATIO_METRICS else f"{v * 100:.1f}"


def _fmt_d(d, metric):
    m, lo, hi, _ = d
    if metric in RATIO_METRICS:
        return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return f"{m * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"


def to_markdown(rep: dict) -> str:
    tasks = rep["tasks"]
    ratio = any(t.metric in RATIO_METRICS for t in tasks)
    title = f"# 표준 벤치 결과 — {rep['suite']} ({rep['table']} 표)"
    lines = [title, ""]
    if rep["table"] == "A":
        lines.append("A 표: 공통 프로토콜(모든 모델 같은 프롬프트·shot·채점, chat 템플릿 미적용).")
    else:
        lines.append("B 표: dev에서 고른 설정(configs/standard/best_settings.yaml). 설정 열이 A가 아닌 모델에는 "
                     "같은 모델의 A 대비 ΔA [paired 95% CI]를 붙인다. test 결과는 설정 선택에 쓰지 않았다.")
    unit = "bits-per-byte(낮을수록 좋음)" if ratio else "%"
    lines.append(f"단위 {unit}. 셀 = lm-eval 점수 / micro [95% CI]"
                 + (f" / Δ{rep['ref']} [paired 95% CI]" if rep["ref"] else "") + ".")
    lines.append("")
    lines.append("| 태스크 | shot | 지표 | 출처 |")
    lines.append("|---|---:|---|---|")
    for t in tasks:
        lines.append(f"| {t.task} | {t.num_fewshot} | {t.metric} | {t.source} |")
    for kind in ("base", "instruct"):
        rows = [r for r in rep["rows"] if r["spec"].kind == kind]
        if not rows:
            continue
        rows.sort(key=lambda r: (r["spec"].owner != "self", r["spec"].active_b, r["model"]))
        lines += ["", f"## {kind}", ""]
        head = "| 모델 | 활성/전체 B | " + ("설정 | " if rep["table"] == "B" else "")
        lines.append(head + " | ".join(t.task for t in tasks) + " | 평균(lm-eval) |")
        lines.append("|---|---:|" + ("---|" if rep["table"] == "B" else "") + "---:|" * (len(tasks) + 1))
        for r in rows:
            cells, means = [], []
            for t in tasks:
                v = r["tasks"].get(t.task)
                if not v:
                    cells.append("–")
                    continue
                c = f"{_fmt(v['lmeval'], t.metric)} / {_fmt(v['micro'], t.metric)}"
                if v["ci"]:
                    c += f" [{_fmt(v['ci'][0], t.metric)}–{_fmt(v['ci'][1], t.metric)}]"
                if "delta_ref" in v:
                    c += " / " + _fmt_d(v["delta_ref"], t.metric)
                if "delta_A" in v:
                    c += " / ΔA " + _fmt_d(v["delta_A"], t.metric)
                cells.append(c)
                if v["lmeval"] is not None:
                    means.append(v["lmeval"])
            name = ("**" + r["model"] + "**") if r["spec"].owner == "self" else r["model"]
            avg = _fmt(float(np.mean(means)), tasks[0].metric) if len(means) == len(tasks) else f"({len(means)}/{len(tasks)})"
            var = f"{r['variant']} | " if rep["table"] == "B" else ""
            lines.append(f"| {name} | {r['spec'].active_b}/{r['spec'].params_b} | {var}" + " | ".join(cells) + f" | {avg} |")
    return "\n".join(lines) + "\n"


def to_csv_rows(rep: dict) -> list[dict]:
    out = []
    for r in rep["rows"]:
        for tname, v in r["tasks"].items():
            row = {"suite": rep["suite"], "table": rep["table"], "model": r["model"], "variant": r["variant"],
                   "kind": r["spec"].kind, "owner": r["spec"].owner,
                   "active_b": r["spec"].active_b, "params_b": r["spec"].params_b, "task": tname, "nshot": v["nshot"],
                   "lmeval": v["lmeval"], "micro": v["micro"], "ci_lo": v["ci"][0] if v["ci"] else None,
                   "ci_hi": v["ci"][1] if v["ci"] else None, "n": v["n"]}
            if "delta_ref" in v:
                row.update({"ref": rep["ref"], "delta": v["delta_ref"][0], "delta_lo": v["delta_ref"][1],
                            "delta_hi": v["delta_ref"][2], "n_paired": v["delta_ref"][3]})
            if "delta_A" in v:
                row.update({"delta_A": v["delta_A"][0], "delta_A_lo": v["delta_A"][1], "delta_A_hi": v["delta_A"][2]})
            out.append(row)
    return out


CSV_KEYS = ["suite", "table", "model", "variant", "kind", "owner", "active_b", "params_b", "task", "nshot",
            "lmeval", "micro", "ci_lo", "ci_hi", "n", "ref", "delta", "delta_lo", "delta_hi", "n_paired",
            "delta_A", "delta_A_lo", "delta_A_hi"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="ko_mc_v1")
    ap.add_argument("--ref", default=None, help="paired 차이 기준 모델 id")
    ap.add_argument("--table", choices=["A", "B"], default="A")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "standard"))
    a = ap.parse_args(argv)
    rep = build(a.suite, a.ref, a.limit, a.table)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = a.suite + ("__B" if a.table == "B" else "") + (f"__limit{a.limit}" if a.limit else "")
    md = to_markdown(rep)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    import csv

    rows = to_csv_rows(rep)
    if rows:
        with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_KEYS)
            w.writeheader()
            w.writerows(rows)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
