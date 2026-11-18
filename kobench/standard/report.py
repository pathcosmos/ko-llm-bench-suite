"""표준 벤치 결과 집계 — base/instruct별 표, 문항 bootstrap 95% CI, 기준 모델 대비 paired 차이.

python -m kobench.standard.report --suite ko_mc_v1 [--ref jansori-bf16] [--out reports/standard]

지표 정의
- lm-eval: lm-evaluation-harness가 보고한 태스크/그룹 점수 (그룹은 lm-eval의 집계 규칙을 따름)
- micro:   문항별 샘플(.samples.jsonl.gz)에서 하위 태스크를 모두 합쳐 낸 정확도. CI·paired 차이는 이 값 기준
- Δref:    같은 문항 집합에서 (모델 − 기준 모델) micro 차이, paired bootstrap 95% 구간
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kobench.standard.registry import load_models, load_suites
from kobench.standard.run import RESULTS

ROOT = Path(__file__).resolve().parents[2]
N_BOOT = 2000
SEED = 1234


def _load(model_id: str, task: str, nshot: int, limit=None):
    stem = f"{task}__{nshot}shot" + (f"__limit{limit}" if limit else "")
    j = RESULTS / model_id / f"{stem}.json"
    s = RESULTS / model_id / f"{stem}.samples.jsonl.gz"
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
    out = {}
    for k, r in items.items():
        v = r.get(metric)
        if v is None:
            v = r.get("acc")
        if v is not None:
            out[k] = float(v)
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_diff(a: dict, b: dict) -> tuple[float, float, float, int]:
    keys = sorted(set(a) & set(b))
    if not keys:
        return (float("nan"),) * 3 + (0,)
    d = np.array([a[k] - b[k] for k in keys])
    lo, hi = bootstrap_ci(d)
    return float(d.mean()), lo, hi, len(keys)


def build(suite: str, ref: str | None, limit=None) -> dict:
    models, suites = load_models(), load_suites()
    tasks = suites[suite]["tasks"]
    rows = []
    for mid, spec in models.items():
        per_task = {}
        for t in tasks:
            doc, items = _load(mid, t.task, t.num_fewshot, limit)
            if doc is None:
                continue
            sc = _item_scores(items, t.metric)
            arr = np.array(list(sc.values()))
            per_task[t.task] = {
                "lmeval": _headline(doc, t.task, t.metric),
                "micro": float(arr.mean()) if len(arr) else None,
                "ci": bootstrap_ci(arr) if len(arr) else None,
                "n": len(arr),
                "_items": sc,
            }
        if per_task:
            rows.append({"model": mid, "spec": spec, "tasks": per_task})
    if ref:
        ref_row = next((r for r in rows if r["model"] == ref), None)
        for r in rows:
            for tname, v in r["tasks"].items():
                if ref_row and tname in ref_row["tasks"] and r is not ref_row:
                    v["delta_ref"] = paired_diff(v["_items"], ref_row["tasks"][tname]["_items"])
    return {"suite": suite, "tasks": tasks, "rows": rows, "ref": ref}


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v * 100:.1f}" if pct else f"{v:.3f}"


def to_markdown(rep: dict) -> str:
    tasks = rep["tasks"]
    lines = [f"# 표준 벤치 결과 — {rep['suite']}", ""]
    lines.append("점수는 %. 셀 = lm-eval 점수 / micro [95% CI]" + (f" / Δ{rep['ref']} [paired 95% CI]" if rep["ref"] else "") + ". "
                 "MC는 모두 로그우도 채점, chat 템플릿 미적용.")
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
        lines.append("| 모델 | 활성/전체 B | " + " | ".join(t.task for t in tasks) + " | 평균(lm-eval) |")
        lines.append("|---|---:|" + "---:|" * (len(tasks) + 1))
        for r in rows:
            cells, means = [], []
            for t in tasks:
                v = r["tasks"].get(t.task)
                if not v:
                    cells.append("–")
                    continue
                c = f"{_fmt(v['lmeval'])} / {_fmt(v['micro'])} [{_fmt(v['ci'][0])}–{_fmt(v['ci'][1])}]"
                if "delta_ref" in v:
                    d, lo, hi, _ = v["delta_ref"]
                    c += f" / {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
                cells.append(c)
                if v["lmeval"] is not None:
                    means.append(v["lmeval"])
            name = ("**" + r["model"] + "**") if r["spec"].owner == "self" else r["model"]
            avg = _fmt(float(np.mean(means))) if len(means) == len(tasks) else f"({len(means)}/{len(tasks)})"
            lines.append(f"| {name} | {r['spec'].active_b}/{r['spec'].params_b} | " + " | ".join(cells) + f" | {avg} |")
    return "\n".join(lines) + "\n"


def to_csv_rows(rep: dict) -> list[dict]:
    out = []
    for r in rep["rows"]:
        for tname, v in r["tasks"].items():
            row = {"suite": rep["suite"], "model": r["model"], "kind": r["spec"].kind, "owner": r["spec"].owner,
                   "active_b": r["spec"].active_b, "params_b": r["spec"].params_b, "task": tname,
                   "lmeval": v["lmeval"], "micro": v["micro"], "ci_lo": v["ci"][0] if v["ci"] else None,
                   "ci_hi": v["ci"][1] if v["ci"] else None, "n": v["n"]}
            if "delta_ref" in v:
                row.update({"ref": rep["ref"], "delta": v["delta_ref"][0], "delta_lo": v["delta_ref"][1],
                            "delta_hi": v["delta_ref"][2], "n_paired": v["delta_ref"][3]})
            out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="ko_mc_v1")
    ap.add_argument("--ref", default=None, help="paired 차이 기준 모델 id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "standard"))
    a = ap.parse_args(argv)
    rep = build(a.suite, a.ref, a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = a.suite + (f"__limit{a.limit}" if a.limit else "")
    md = to_markdown(rep)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    import csv

    rows = to_csv_rows(rep)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
