"""EVAFRILL 생성 성능: 원본(캐시 없음) vs 증분 디코딩 엔진 (CUDA graph on/off).

입력 128/1024/3968 × 출력 128/512 × 배치 1/8/32, 체크포인트 slerp(최종), BF16, H200 단독.
- 엔진: 설정마다 그 크기의 엔진을 새로 만들고, 같은 설정으로 한 번 예열(그래프 캡처 포함)한 뒤 두 번째 실행을 측정한다.
  EOS로 멈추지 않게 해 출력 길이를 고정한다.
- 원본: 매 단계 전체 시퀀스 forward(원본 generate 방식). 시퀀스 길이 3지점(L, L+out/2, L+out-1)에서
  forward 시간을 실측(예열 1 + 3회 중앙값)하고 구간별 선형 보간으로 Σ_{i=0}^{out-1} f(L+i)를 추정한다.
  작은 설정 2개는 전체 루프도 실제로 돌려 추정 오차를 보고한다.
TTFT = 첫 토큰까지(배치 전체의 prefill 포함), decode tok/s = B·(out-1)/decode 시간, 총 처리량 = B·out/전체 시간.

python -m kobench.standard.evafrill_perf --out ../runs/evafrill_engine_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from kobench.standard.evafrill_engine import EvafrillEngine
from kobench.standard.evafrill_verify import load, long_text_ids

INPUTS, OUTPUTS, BATCHES = (128, 1024, 3968), (128, 512), (1, 8, 32)
GIB = 1024 ** 3


def _sync():
    torch.cuda.synchronize()


@torch.no_grad()
def fwd_time(model, B: int, T: int, ids_long) -> tuple[float, float]:
    x = torch.tensor([ids_long[i * 13: i * 13 + T] for i in range(B)], device="cuda")
    model(x)
    _sync()
    ts = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        t = time.perf_counter()
        model(x)
        _sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), torch.cuda.max_memory_allocated() / GIB


def original_estimate(model, L, out, B, ids_long) -> dict:
    pts = sorted({L, L + out // 2, L + out - 1})
    meas, peak = {}, 0.0
    for T in pts:
        meas[T], p = fwd_time(model, B, T, ids_long)
        peak = max(peak, p)
    xs = sorted(meas)

    def f(t):
        for a, b in zip(xs, xs[1:]):
            if a <= t <= b:
                return meas[a] + (meas[b] - meas[a]) * (t - a) / (b - a)
        return meas[xs[-1]]

    ttft = meas[L]
    decode = sum(f(L + i) for i in range(1, out))
    return {"ttft_s": ttft, "decode_s": decode, "total_s": ttft + decode, "peak_alloc_gib": peak,
            "measured": "interpolated", "forward_samples_s": {str(k): v for k, v in meas.items()}}


@torch.no_grad()
def original_full(model, L, out, B, ids_long) -> dict:
    seq = torch.tensor([ids_long[i * 13: i * 13 + L] for i in range(B)], device="cuda")
    model(seq)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(out):
        logits, _ = model(seq)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        if i == 0:
            _sync()
            t_first = time.perf_counter()
        seq = torch.cat([seq, nxt], 1)
    _sync()
    t1 = time.perf_counter()
    return {"ttft_s": t_first - t0, "decode_s": t1 - t_first, "total_s": t1 - t0,
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / GIB, "measured": "full"}


def engine_run(model, L, out, B, ids_long, use_graph, capacity: tuple[int, int] | None = None) -> dict:
    """capacity=None: 설정 크기에 맞춘 엔진. capacity=(max_batch, max_len): 고정 용량 배치 불변 엔진(배포 기본)."""
    prompts = [ids_long[i * 13: i * 13 + L] for i in range(B)]
    if capacity is None:
        eng = EvafrillEngine(model, max_batch=B, max_len=EvafrillEngine._bucket_static(L + out))
    else:
        eng = EvafrillEngine(model, max_batch=capacity[0], max_len=capacity[1], batch_invariant=True)
    eng.generate(prompts, out, eos_ids=(), use_graph=use_graph)          # 예열 (+ 그래프 캡처)
    torch.cuda.reset_peak_memory_stats()
    tm = {}
    eng.generate(prompts, out, eos_ids=(), use_graph=use_graph, timings=tm)
    peak = torch.cuda.max_memory_allocated() / GIB
    del eng
    torch.cuda.empty_cache()
    return {"ttft_s": tm["ttft_s"], "decode_s": tm["decode_s"], "total_s": tm["total_s"], "peak_alloc_gib": peak,
            "measured": "full", "bucket_len": tm["bucket_len"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="slerp")
    ap.add_argument("--reuse-original", default=None, help="원본 측정 행을 이 perf.csv에서 재사용 (원본 코드 불변)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    torch.backends.cuda.matmul.allow_tf32 = False
    ids_long = long_text_ids(4100 + 13 * 32 + 600)
    model, _ = load(a.ckpt, torch.bfloat16)
    weights_gib = torch.cuda.memory_allocated() / GIB
    rows = []

    def add(mode, L, o, B, r, note=""):
        row = {"mode": mode, "input_len": L, "output_len": o, "batch": B, **{k: r.get(k) for k in
               ("ttft_s", "decode_s", "total_s", "peak_alloc_gib", "measured", "bucket_len")}, "note": note}
        row["decode_tok_s"] = B * (o - 1) / r["decode_s"] if r.get("decode_s") else None
        row["total_tok_s"] = B * o / r["total_s"] if r.get("total_s") else None
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, ensure_ascii=False), flush=True)
        with open(out / "perf.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    for L in INPUTS:
        for o in OUTPUTS:
            for B in BATCHES:
                for mode, g, cap in (("engine_graph", True, None), ("engine_eager", False, None),
                                     ("engine_graph_cap32", True, (32, 4608)), ("engine_eager_cap32", False, (32, 4608))):
                    try:
                        add(mode, L, o, B, engine_run(model, L, o, B, ids_long, g, cap))
                    except torch.cuda.OutOfMemoryError as e:
                        torch.cuda.empty_cache()
                        add(mode, L, o, B, {}, f"OOM: {str(e)[:80]}")
                if a.reuse_original:
                    continue
                try:
                    add("original", L, o, B, original_estimate(model, L, o, B, ids_long))
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    add("original", L, o, B, {}, f"OOM: {str(e)[:80]}")
    if a.reuse_original:
        prev = [r for r in csv.DictReader(open(a.reuse_original)) if r["mode"].startswith("original")]
        for r in prev:
            r["note"] = (r.get("note") or "") + f" [재사용: {a.reuse_original}]"
            rows.append({k: r.get(k) for k in rows[0]})
        with open(out / "perf.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    else:
        # 보간 추정의 오차 확인: 작은 설정은 전체 루프를 실제로 돌린다
        for L, o, B in ((128, 128, 1), (1024, 128, 1), (128, 128, 8)):
            add("original_full_check", L, o, B, original_full(model, L, o, B, ids_long))
    (out / "perf_meta.json").write_text(json.dumps({
        "ckpt": a.ckpt, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "weights_alloc_gib": weights_gib, "tf32": False, "dtype": "bfloat16",
        "notes": ["원본은 flash_attn 없이 _standard_attention(전체 T×T 점수 행렬)",
                  "엔진 VRAM은 설정 크기에 맞춘 상태 텐서 포함, peak_alloc은 torch 할당 기준(가중치 포함)"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
