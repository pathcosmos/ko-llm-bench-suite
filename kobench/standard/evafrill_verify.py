"""EVAFRILL 증분 디코딩 엔진 검증 (Goal 1-B). 허용 오차: runs/evafrill_engine_v1/tolerance_preregistered.json

python -m kobench.standard.evafrill_verify --out ../runs/evafrill_engine_v1 [--ckpts pretrain sft-v2 slerp] [--only V1V2|V3|V4]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from kobench.standard import formats
from kobench.standard.evafrill_engine import EvafrillEngine, original_greedy
from kobench.standard.evafrill_lm import _resolve_snapshot, load_evafrill

REPO = "pathcosmos/EVAFRILL-Mo-3B"
REV = "e95edece955af9df9a6215210db594a9984fddf5"
QUESTIONS = [
    "한국의 수도는 어디인가요?",
    "파이썬에서 리스트를 정렬하는 방법을 설명해 주세요.",
    "지구 온난화의 주요 원인 세 가지를 알려 주세요.",
    "What is the difference between a list and a tuple in Python?",
    "좋은 수면 습관을 만들기 위한 팁을 알려 주세요.",
    "삼각형의 내각의 합은 몇 도인가요? 이유도 설명해 주세요.",
    "김치를 처음 담그는 사람에게 필요한 재료를 정리해 주세요.",
    "Explain photosynthesis in two sentences.",
]


def _src():
    return os.environ.get("EVAFRILL_SRC") or str(Path(os.environ["LT_SRC"]) / "EVAFRILL-Mo")


def load(ckpt: str, dtype):
    sub = "" if ckpt == "root" else ckpt
    d = _resolve_snapshot(REPO, REV, sub)
    m, tok, info = load_evafrill(d, _src(), "cuda", dtype)
    return m, tok


class fp32_reference_attention:
    """FP32 기준선: 원본 MultiHeadAttention.forward에서 bf16 강제 변환만 뺀다 (나머지 동일)."""

    def __enter__(self):
        import model.attention as am

        self.am, self.orig = am, am.MultiHeadAttention.forward

        def fwd(this, x, cos, sin):
            B, T, C = x.shape
            qkv = this.qkv_proj(x)
            q, k, v = qkv.split([this._q_dim, this._kv_dim, this._kv_dim], dim=-1)
            q = q.view(B, T, this.n_heads, this.head_dim)
            k = k.view(B, T, this.n_kv_heads, this.head_dim)
            v = v.view(B, T, this.n_kv_heads, this.head_dim)
            q = am.apply_rotary_emb(q, cos, sin)
            k = am.apply_rotary_emb(k, cos, sin)
            return this.out_proj(this._standard_attention(q, k, v, B, T))

        am.MultiHeadAttention.forward = fwd
        return self

    def __exit__(self, *a):
        self.am.MultiHeadAttention.forward = self.orig


def long_text_ids(n: int) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset("skt/kobest_v1", "boolq", split="test")
    text = "\n".join(r["paragraph"] for r in ds.select(range(400)))
    ids = formats.encode_text("evafrill-mo-3b", text)
    assert len(ids) >= n, len(ids)
    return ids


@torch.no_grad()
def full_logits(model, seq: list[int]) -> torch.Tensor:
    logits, _ = model(torch.tensor([seq], device="cuda"))
    return logits[0]


def pos_stats(E: torch.Tensor, B: torch.Tensor, R: torch.Tensor) -> dict:
    """위치별 지표 (E,B,R: (P,V))."""
    E, B, R = E.float(), B.float(), R.float()
    eB = (B - R).abs()
    eE = (E - R).abs()
    top2 = B.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    agree = E.argmax(-1) == B.argmax(-1)
    return {
        "eB_max": eB.amax(-1).cpu(), "eE_max": eE.amax(-1).cpu(),
        "rB": eB.pow(2).mean(-1).sqrt().cpu(), "rE": eE.pow(2).mean(-1).sqrt().cpu(),
        "margin": margin.cpu(), "agree": agree.cpu(),
        "EB_max": (E - B).abs().amax(-1).cpu(), "bitwise_equal": bool(torch.equal(E, B)),
    }


def judge(st: dict) -> dict:
    max_eB, max_eE = float(st["eB_max"].max()), float(st["eE_max"].max())
    mean_rB, mean_rE = float(st["rB"].mean()), float(st["rE"].mean())
    sel = st["margin"] >= 0.5
    n_dis = int((~st["agree"] & sel).sum())
    t1 = max_eE <= 1.5 * max_eB + 0.05
    t2 = mean_rE <= 1.25 * mean_rB + 0.01
    t3 = n_dis == 0
    return {"positions": int(st["eB_max"].numel()), "max_eB": max_eB, "max_eE": max_eE, "mean_rB": mean_rB,
            "mean_rE": mean_rE, "max_abs_E_minus_B": float(st["EB_max"].max()), "bitwise_equal_E_B": st["bitwise_equal"],
            "argmax_disagree_margin_ge_0.5": n_dis, "argmax_disagree_any": int((~st["agree"]).sum()),
            "T1": t1, "T2": t2, "T3": t3, "pass": t1 and t2 and t3}


def run_v1v2(ckpt, m16, m32, eng, ids_long, lengths=(1, 17, 512, 4000), n_decode=256) -> list[dict]:
    out = []
    for L in lengths:
        prompt = ids_long[:L]
        # V1: prefill 전 위치
        E_pre = eng.prefill(prompt, 0, all_logits=True)
        B_pre = full_logits(m16, prompt)
        with fp32_reference_attention():
            R_pre = full_logits(m32, prompt)
        v1 = judge(pos_stats(E_pre, B_pre, R_pre))
        del E_pre, B_pre, R_pre
        # V2: 이후 n_decode 단계 (토큰 n_decode+1개 생성 → decode 단계 n_decode개)
        res = eng.generate([prompt], n_decode + 1, eos_ids=(), record_logits=True)[0]
        seq = prompt + res.tokens
        E_dec = torch.stack(res.logits[1:])                     # 단계 1..n_decode
        B_full = full_logits(m16, seq)
        with fp32_reference_attention():
            R_full = full_logits(m32, seq)
        idx = slice(L, L + n_decode)                            # 위치 L .. L+n_decode-1
        v2 = judge(pos_stats(E_dec, B_full[idx], R_full[idx]))
        out.append({"ckpt": ckpt, "len": L, "V1_prefill": v1, "V2_decode": v2})
        print(f"[{ckpt}] L={L} V1 pass={v1['pass']} maxE={v1['max_eE']:.3f} maxB={v1['max_eB']:.3f} bitwise={v1['bitwise_equal_E_B']} | "
              f"V2 pass={v2['pass']} maxE={v2['max_eE']:.3f} maxB={v2['max_eB']:.3f} dis={v2['argmax_disagree_any']}", flush=True)
        del B_full, R_full, E_dec
        torch.cuda.empty_cache()
    return out


def v3_prompts(ckpt, ids_long) -> list[list[int]]:
    if ckpt == "pretrain":
        return [ids_long[i * 300: i * 300 + 40 + 10 * i] for i in range(8)]
    return [formats.encode_prompt("evafrill-mo-3b", [{"role": "user", "content": q}]) for q in QUESTIONS]


def run_v3(ckpt, m16, eng, ids_long, n=256) -> dict:
    rows = []
    for p in v3_prompts(ckpt, ids_long):
        e = eng.generate([p], n, eos_ids=(2,))[0].tokens
        o, ologs = original_greedy(m16, p, n, eos_ids=(2,), record_logits=True)
        same = e == o
        row = {"prompt_len": len(p), "engine_len": len(e), "orig_len": len(o), "identical": same}
        if not same:
            i = next((k for k, (a, b) in enumerate(zip(e, o)) if a != b), min(len(e), len(o)))
            if i < len(ologs):
                t2 = ologs[i].float().topk(2).values
                row.update({"first_diff": i, "orig_margin_at_diff": float(t2[0] - t2[1]),
                            "engine_tok": e[i] if i < len(e) else None, "orig_tok": o[i]})
            else:
                row.update({"first_diff": i, "orig_margin_at_diff": None, "note": "길이만 다름(EOS 시점)"})
        rows.append(row)
    n_same = sum(r["identical"] for r in rows)
    near = all((r["identical"] or (r.get("orig_margin_at_diff") is not None and r["orig_margin_at_diff"] < 0.5)) for r in rows)
    res = {"ckpt": ckpt, "n": len(rows), "identical": n_same, "all_divergences_near_tie": near,
           "pass": n_same >= 6 and near, "rows": rows}
    print(f"[{ckpt}] V3 identical {n_same}/{len(rows)} near_tie_ok={near} pass={res['pass']}", flush=True)
    return res


@torch.no_grad()
def engine_teacher_forced(eng, prompt: list[int], forced: list[int]) -> list[torch.Tensor]:
    """원본 토큰열을 엔진에 강제로 넣으며 단계별 로짓을 얻는다 (단계 0 = prefill 마지막 위치)."""
    logs = [eng.prefill(prompt, 0)[-1]]
    # 엔진 generate와 같은 실행 형태 (batch_invariant면 max_batch 행·max_len 어텐션)
    Bx = eng.max_batch if eng.batch_invariant else 1
    Lb = eng.max_len if eng.batch_invariant else eng._bucket(len(prompt) + len(forced) + 1)
    pos = len(prompt)
    for t in forced[:-1]:
        tok = torch.zeros(Bx, dtype=torch.long, device="cuda")
        tok[0] = t
        p = torch.zeros(Bx, dtype=torch.long, device="cuda")
        p[0] = pos
        logs.append(eng._step(tok, p, Bx, Lb)[0].clone())
        pos += 1
    return logs


def run_v3bc(ckpt, m16, m32, eng, ids_long, n=256) -> dict:
    """V3b(교사 강제 결정 일치, 판정) + V3c(원본 BF16 vs FP32 greedy, 참고). tolerance_supplementary.json"""
    b_rows, c_rows = [], []
    for p in v3_prompts(ckpt, ids_long):
        o, ologs = original_greedy(m16, p, n, eos_ids=(2,), record_logits=True)
        elogs = engine_teacher_forced(eng, p, o)
        margins = [float(x.float().topk(2).values[0] - x.float().topk(2).values[1]) for x in ologs]
        agree = [int(e.argmax()) == t for e, t in zip(elogs, o)]
        bad = [i for i, (ok, mg) in enumerate(zip(agree, margins)) if not ok and mg >= 0.5]
        b_rows.append({"prompt_len": len(p), "steps": len(o), "disagree_any": agree.count(False),
                       "disagree_margin_ge_0.5": len(bad), "disagree_margins": [round(margins[i], 3) for i, ok in enumerate(agree) if not ok],
                       "max_logit_diff": max(float((e.float() - x.float()).abs().max()) for e, x in zip(elogs, ologs))})
        with fp32_reference_attention():
            o32, o32logs = original_greedy(m32, p, n, eos_ids=(2,), record_logits=True)
        same = o == o32
        row = {"prompt_len": len(p), "identical_bf16_vs_fp32": same}
        if not same:
            i = next((k for k, (a, b) in enumerate(zip(o, o32)) if a != b), min(len(o), len(o32)))
            row["first_diff"] = i
            if i < len(ologs):
                t2 = ologs[i].float().topk(2).values
                row["bf16_margin_at_diff"] = float(t2[0] - t2[1])
        c_rows.append(row)
    v3b_pass = all(r["disagree_margin_ge_0.5"] == 0 for r in b_rows)
    res = {"ckpt": ckpt,
           "V3b_teacher_forced": {"pass": v3b_pass, "rows": b_rows},
           "V3c_control_bf16_vs_fp32": {"identical": sum(r["identical_bf16_vs_fp32"] for r in c_rows), "n": len(c_rows), "rows": c_rows}}
    print(f"[{ckpt}] V3b pass={v3b_pass} disagree(any)={sum(r['disagree_any'] for r in b_rows)} "
          f"| V3c 원본 BF16=FP32 greedy {res['V3c_control_bf16_vs_fp32']['identical']}/{len(c_rows)}", flush=True)
    return res


def run_v4(ckpt, eng, ids_long, max_eB_v2: float, n=128) -> dict:
    lens = [5, 1000, 37, 256, 600, 12, 128, 777] * 4
    prompts = [ids_long[i * 97: i * 97 + L] for i, L in enumerate(lens)]
    ref = [eng.generate([p], n, eos_ids=(), record_logits=True)[0] for p in prompts]
    thr = 1.5 * max_eB_v2 + 0.05
    out = {"ckpt": ckpt, "threshold_logit_diff": thr, "batches": {}}
    for B in (8, 32):
        res = eng.generate(prompts[:B], n, eos_ids=(), record_logits=True)
        rows = []
        for b in range(B):
            r1, rb = ref[b], res[b]
            i = next((k for k, (a, c) in enumerate(zip(r1.tokens, rb.tokens)) if a != c), None)
            upto = len(r1.tokens) if i is None else i
            diffs = [float((rb.logits[k].float() - r1.logits[k].float()).abs().max()) for k in range(upto)]
            row = {"len": len(prompts[b]), "identical": i is None, "max_logit_diff_before_div": max(diffs) if diffs else 0.0}
            if i is not None:
                t2 = r1.logits[i].float().topk(2).values
                row.update({"first_diff": i, "batch1_margin_at_diff": float(t2[0] - t2[1])})
            rows.append(row)
        ok = all((r["identical"] or r["batch1_margin_at_diff"] < 0.5) and r["max_logit_diff_before_div"] <= thr for r in rows)
        bitwise = [all(torch.equal(x, y) for x, y in zip(rb.logits, r1.logits)) and len(rb.logits) == len(r1.logits)
                   for r1, rb in zip(ref[:B], res)]
        for r, bw in zip(rows, bitwise):
            r["logits_bitwise_equal"] = bw
        strict = all(r["identical"] for r in rows) and all(bitwise)
        out["batches"][B] = {"identical": sum(r["identical"] for r in rows), "n": B, "pass_v1_tolerance": ok,
                             "logits_bitwise_equal": sum(bitwise), "pass_strict": strict, "pass": strict,
                             "max_logit_diff": max(r["max_logit_diff_before_div"] for r in rows), "rows": rows}
        print(f"[{ckpt}] V4 batch {B}: identical {out['batches'][B]['identical']}/{B} bitwise {sum(bitwise)}/{B} "
              f"maxdiff={out['batches'][B]['max_logit_diff']:.4f} strict_pass={strict}", flush=True)
    # 참고: CUDA graph와 eager 경로의 일치
    g = eng.generate(prompts[:8], 32, eos_ids=(), record_logits=True, use_graph=True)
    e = eng.generate(prompts[:8], 32, eos_ids=(), record_logits=True, use_graph=False)
    out["graph_vs_eager"] = {"tokens_equal": all(a.tokens == b.tokens for a, b in zip(g, e)),
                             "max_logit_diff": max(float((x.float() - y.float()).abs().max())
                                                   for a, b in zip(g, e) for x, y in zip(a.logits, b.logits))}
    out["pass"] = all(v["pass"] for v in out["batches"].values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpts", nargs="+", default=["pretrain", "sft-v2", "slerp"])
    ap.add_argument("--only", choices=["V1V2", "V3", "V4", "V3bc"], default=None)
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tol = out / ("tolerance_supplementary.json" if a.only == "V3bc" else "tolerance_preregistered.json")
    assert tol.exists(), f"허용 오차 사전 등록 파일이 없다: {tol}"
    ids_long = long_text_ids(4100)
    verification = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tolerance_file": str(tol),
                    "tolerance_sha256": formats._sha256(tol), "ckpts": {}}
    path = out / ("verification.json" if a.only is None else f"verification_{a.only}.json")
    for ck in a.ckpts:
        t0 = time.time()
        m16, _ = load(ck, torch.bfloat16)
        eng = EvafrillEngine(m16, max_batch=32, max_len=4608)
        rec = {}
        if a.only in (None, "V1V2", "V4"):
            m32, _ = load(ck, torch.float32)
            rec["V1V2"] = run_v1v2(ck, m16, m32, eng, ids_long)
            del m32
            torch.cuda.empty_cache()
        if a.only in (None, "V3"):
            rec["V3"] = run_v3(ck, m16, eng, ids_long)
        if a.only == "V3bc":
            m32, _ = load(ck, torch.float32)
            rec["V3bc"] = run_v3bc(ck, m16, m32, eng, ids_long)
            del m32
            torch.cuda.empty_cache()
        if a.only in (None, "V4"):
            max_eB = max(r["V2_decode"]["max_eB"] for r in rec["V1V2"])
            rec["V4"] = run_v4(ck, eng, ids_long, max_eB)
        rec["seconds"] = round(time.time() - t0, 1)
        verification["ckpts"][ck] = rec
        path.write_text(json.dumps(verification, ensure_ascii=False, indent=1), encoding="utf-8")
        del eng, m16
        torch.cuda.empty_cache()
    verification["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(verification, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
