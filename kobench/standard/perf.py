"""클라이언트 측 서빙 성능 측정 (엔진 독립): TTFT, TPOT, 처리량, 동시 요청, prefix caching, VRAM.

python -m kobench.standard.perf --models frankenstallm-bf16 qwen3-4b-2507 [--engine vllm] [--concurrency 1 4 16 32 64]
python -m kobench.standard.perf --models frankenstallm-gguf-q4_k_m --engine llamacpp

절차 (모델마다 서버 하나):
  1. 서버 기동: vLLM OpenAI 서버(BF16, max_model_len 4096) 또는 llama-server(GGUF, 슬롯 64). /health가 응답할 때까지의 시간 = 기동 시간
  2. 예열: 요청 8개(집계 제외)
  3. 워크로드 W1(일반): ko-arena-hard 질문 128개를 각 모델 chat 형식으로 토큰화(생성 평가와 같은 encode),
     출력 256 토큰 고정(ignore_eos) — 동시 요청 c ∈ {1,4,16,32,64}, 요청 수 = max(32, 4c)
  4. 워크로드 W2(공유 prefix): 약 2,000토큰 공통 문서 + 짧은 질문 64개 — prefix caching on/off 서버 각각에서 동시 16, 출력 64 토큰
  5. VRAM: 측정 중 nvidia-smi memory.used 최고치(엔진이 미리 잡는 KV 풀 포함) + 엔진 로그의 가중치·KV 용량
결과: results/perf/<model>/<engine>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from kobench.standard.registry import ModelSpec, load_models

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "perf"
SEED = 1234


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VramPoller:
    def __init__(self, interval: float = 0.5):
        self.peak, self.interval, self._stop = 0, interval, threading.Event()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


# ── 서버 ────────────────────────────────────────────────────────────────────

def vllm_cmd(spec: ModelSpec, port: int, prefix_caching: bool, max_len: int, gpu_mem: float) -> list[str]:
    py = os.environ.get("LT_VENV", "/workspace/llm-test/venv") + "/bin/vllm"
    cmd = [py, "serve", spec.path, "--port", str(port), "--host", "127.0.0.1", "--dtype", "bfloat16",
           "--max-model-len", str(max_len), "--gpu-memory-utilization", str(gpu_mem), "--seed", str(SEED),
           "--max-num-seqs", "128", "--disable-uvicorn-access-log", "--served-model-name", "m"]
    cmd += ["--enable-prefix-caching"] if prefix_caching else ["--no-enable-prefix-caching"]
    if not spec.is_local:
        cmd += ["--revision", spec.revision]
    if spec.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if spec.owner == "self":
        from kobench.standard import formats

        cmd += ["--tokenizer", str(formats.prepare_tokenizer_dir(spec.id))]
    if spec.extra.get("relax_head_dim_check"):
        cmd = [os.environ.get("LT_VENV", "/workspace/llm-test/venv") + "/bin/python", "-m", "kobench.standard.perf_vllm_entry"] + cmd[1:]
    return cmd


def llamacpp_cmd(spec: ModelSpec, port: int, prefix_caching: bool, max_len: int, slots: int = 64) -> list[str]:
    exe = str(Path(os.environ.get("LT_BIN", "/workspace/llm-test/bin")) / "llama-server")
    # 호스트 RAM 프롬프트 캐시(--cache-ram, 기본 켜짐)는 끈다: vLLM에 없는 기능이고, 슬롯 64개에서 요청마다 슬롯 상태(~220 MiB)를
    # RAM에 복사·축출해 TTFT가 1.6초로 늘었다(2026-11-21). prefix on/off는 슬롯 KV 재사용을 요청별 cache_prompt로 켜고 끈다
    # (서버 --no-cache-prompt만으로는 슬롯 KV 재사용이 남았다).
    cmd = [exe, "-m", spec.path, "--port", str(port), "--host", "127.0.0.1", "-ngl", "999", "-np", str(slots),
           "-c", str(max_len * slots), "-cb", "--seed", str(SEED), "--no-webui", "-fa", "on", "--cache-ram", "0"]
    if not prefix_caching:
        cmd += ["--no-cache-prompt"]
    return cmd


class Server:
    def __init__(self, cmd: list[str], port: int, log: Path, env: dict | None = None):
        self.cmd, self.port, self.log = cmd, port, log
        self.env = {**os.environ, **(env or {})}

    def __enter__(self):
        import urllib.request

        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.log, "w")
        t0 = time.time()
        self.p = subprocess.Popen(self.cmd, stdout=self.fh, stderr=subprocess.STDOUT, env=self.env, start_new_session=True,
                                  cwd=str(ROOT))
        while True:
            if self.p.poll() is not None:
                raise RuntimeError(f"서버 종료(code {self.p.returncode}), 로그 {self.log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 1800:
                raise TimeoutError("서버 기동 30분 초과")
            time.sleep(1)
        self.startup_s = time.time() - t0
        return self

    def __exit__(self, *a):
        try:
            os.killpg(self.p.pid, signal.SIGTERM)
            self.p.wait(timeout=60)
        except Exception:
            try:
                os.killpg(self.p.pid, signal.SIGKILL)
            except Exception:
                pass
        self.fh.close()
        time.sleep(5)


# ── 클라이언트 ──────────────────────────────────────────────────────────────

async def one_request(session, url: str, engine: str, prompt_ids: list[int], max_tokens: int, cache_prompt: bool = True) -> dict:
    if engine == "vllm":
        body = {"model": "m", "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True}}
        path = "/v1/completions"
    else:
        body = {"prompt": prompt_ids, "n_predict": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
                "cache_prompt": cache_prompt}   # 2026-11-21 이전엔 True 고정이라 llama.cpp prefix off가 실제로 꺼지지 않았다
        path = "/completion"
    t0 = time.perf_counter()
    first, last, n = None, None, 0
    async with session.post(url + path, json=body) as resp:
        if resp.status != 200:
            return {"error": f"HTTP {resp.status}: {(await resp.text())[:200]}"}
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            now = time.perf_counter()
            if engine == "vllm":
                ch = j.get("choices") or []
                if ch and ch[0].get("text") is not None and (ch[0]["text"] or ch[0].get("finish_reason") is None):
                    first = first or now
                    last = now
                if j.get("usage"):
                    n = j["usage"]["completion_tokens"]
            else:
                if j.get("content") is not None and not j.get("stop"):
                    first = first or now
                    last = now
                    n += 1
                if j.get("stop"):
                    n = j.get("tokens_predicted", n)
    end = time.perf_counter()
    if first is None:
        return {"error": "no tokens"}
    return {"ttft": first - t0, "e2e": end - t0, "n_out": n, "tpot": (last - first) / max(1, n - 1), "n_in": len(prompt_ids)}


async def run_load(url: str, engine: str, prompts: list[list[int]], concurrency: int, n_requests: int, max_tokens: int,
                   cache_prompt: bool = True) -> dict:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=3600)
    # llama-server는 keep-alive 연결을 응답 뒤 닫아 재사용 요청이 ServerDisconnectedError가 된다(2026-11-21, GGUF 4건 실패)
    # → llama.cpp는 요청마다 새 연결(localhost 연결 비용은 TTFT 대비 무시할 수준). vLLM 결과는 keep-alive 그대로.
    conn = aiohttp.TCPConnector(limit=concurrency, force_close=(engine == "llamacpp"))
    results = []
    q = asyncio.Queue()
    for i in range(n_requests):
        q.put_nowait(prompts[i % len(prompts)])
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        async def worker():
            while True:
                try:
                    p = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(await one_request(s, url, engine, p, max_tokens, cache_prompt))
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    ok = [r for r in results if "error" not in r]
    errs = [r["error"] for r in results if "error" in r]

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None

    tt = [r["ttft"] for r in ok]
    tp = [r["tpot"] for r in ok]
    out_tok = sum(r["n_out"] for r in ok)
    return {"concurrency": concurrency, "requests": n_requests, "ok": len(ok), "errors": errs[:3], "wall_s": round(wall, 3),
            "output_tok_per_s": round(out_tok / wall, 1), "req_per_s": round(len(ok) / wall, 3),
            "ttft_ms_p50": round(1000 * pct(tt, 50), 1) if tt else None, "ttft_ms_p90": round(1000 * pct(tt, 90), 1) if tt else None,
            "tpot_ms_p50": round(1000 * pct(tp, 50), 2) if tp else None, "tpot_ms_p90": round(1000 * pct(tp, 90), 2) if tp else None,
            "per_stream_tok_per_s_p50": round(1 / pct(tp, 50), 1) if tp else None,
            "mean_in_tokens": round(statistics.mean(r["n_in"] for r in ok), 1) if ok else None}


# ── 워크로드 ────────────────────────────────────────────────────────────────

def chat_encoder(spec: ModelSpec):
    """생성 평가와 같은 프롬프트 토큰화 (thinking off)."""
    from kobench.standard.gen_data import load_gen_config

    if spec.owner == "self":
        from kobench.standard import formats

        if spec.kind == "base":
            return lambda text: formats.encode_text(spec.id, text)
        return lambda text: formats.encode_prompt(spec.id, [{"role": "user", "content": text}])
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.path, revision=spec.revision, trust_remote_code=spec.trust_remote_code)
    th = load_gen_config()["thinking"].get(spec.id, {}).get("think_off", {})
    if spec.kind == "base" or not tok.chat_template:
        return lambda text: tok.encode(text)
    return lambda text: tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                           tokenize=False, **th), add_special_tokens=False)


def workloads(spec: ModelSpec, fmt_id: str | None = None) -> dict:
    from kobench.standard.gen_data import load_task

    sp = spec
    if fmt_id:                       # GGUF: 같은 학습 형식의 BF16 항목으로 토큰화
        sp = load_models()[fmt_id]
    enc = chat_encoder(sp)
    qs = [it["turns"][0] for it in load_task("ko_arena_hard")[:128]]
    w1 = [enc(q)[:3000] for q in qs]
    docs = [it["turns"][0] for it in load_task("ko_arena_hard")[200:260]]
    shared = "다음 문서를 참고해 질문에 답하세요.\n\n" + "\n\n".join(docs)
    base_ids = enc(shared + "\n\n질문: 첫 번째 문단을 한 문장으로 요약하세요.")
    # 공유 prefix 약 2,000토큰으로 맞춘 뒤 질문만 바꾼다
    head = shared
    while len(enc(head)) > 2000:
        head = head[: int(len(head) * 0.9)]
    w2 = [enc(head + f"\n\n질문 {i}: 위 문서의 {i % 7 + 1}번째 내용을 요약하세요.") for i in range(64)]
    return {"w1": w1, "w2": w2, "w2_prefix_tokens": len(enc(head)), "w1_mean_in": statistics.mean(map(len, w1)), "_unused": len(base_ids)}


def parse_engine_log(text: str) -> dict:
    d = {}
    m = re.search(r"Model loading took ([\d.]+) GiB", text)
    if m:
        d["weights_gib"] = float(m.group(1))
    m = re.search(r"Available KV cache memory: ([\d.]+) GiB", text)
    if m:
        d["kv_cache_gib"] = float(m.group(1))
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        d["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", text)
    if m:
        d["max_concurrency_at_max_len"] = float(m.group(1))
    m = re.search(r"model buffer size\s*=\s*([\d.]+) MiB", text)
    if m:
        d["weights_gib"] = round(float(m.group(1)) / 1024, 2)
    m = re.findall(r"KV buffer size\s*=\s*([\d.]+) MiB", text)
    if m:
        d["kv_cache_gib"] = round(sum(map(float, m)) / 1024, 2)
    return d


def bench_model(mid: str, engine: str, conc: list[int], gpu_mem: float) -> dict:
    from kobench.standard.run import _versions

    spec = load_models()[mid]
    if spec.extra.get("relax_head_dim_check"):    # 클라이언트 쪽 토크나이저·config 로드도 같은 완화가 필요 (kanana-1.5, 2026-11-21)
        from kobench.standard.run import relax_head_dim_check

        relax_head_dim_check()
    fmt_id = None
    if engine == "llamacpp":
        fmt_id = {"JANSORI": "jansori-bf16", "FRANKENSTALLM": "frankenstallm-bf16"}[spec.family]
    wl = workloads(spec, fmt_id)
    res = {"model": mid, "engine": engine, "params_b": spec.params_b, "active_b": spec.active_b, "family": spec.family,
           "kind": spec.kind, "revision": spec.revision, "max_model_len": 4096, "workload": {
               "w1": "ko-arena-hard 질문 128개(모델 chat 형식), 출력 256 토큰 고정(ignore_eos), greedy",
               "w1_mean_in_tokens": round(wl["w1_mean_in"], 1),
               "w2": f"공유 prefix {wl['w2_prefix_tokens']}토큰 + 질문 64개, 동시 16, 출력 64 토큰"},
           "versions": _versions(), "at": datetime.now(timezone.utc).isoformat(), "runs": {}}
    logd = OUT / mid
    for pc in (True, False):
        port = free_port()
        cmd = vllm_cmd(spec, port, pc, 4096, gpu_mem) if engine == "vllm" else llamacpp_cmd(spec, port, pc, 4096)
        log = logd / f"{engine}_server_{'pc' if pc else 'nopc'}.log"
        tag = "prefix_cache_on" if pc else "prefix_cache_off"
        with VramPoller() as vp, Server(cmd, port, log) as srv:
            url = f"http://127.0.0.1:{port}"
            asyncio.run(run_load(url, engine, wl["w1"][:8], 8, 8, 32))          # 예열 (집계 제외)
            r = {"startup_s": round(srv.startup_s, 1), "cmd": cmd}
            if pc:                                                              # W1은 기본 설정(prefix caching on)에서만
                r["w1"] = [asyncio.run(run_load(url, engine, wl["w1"], c, max(32, 4 * c), 256)) for c in conc]
                for x in r["w1"]:
                    print(f"[perf] {mid} c={x['concurrency']}: {x['output_tok_per_s']} tok/s, TTFT p50 {x['ttft_ms_p50']} ms, "
                          f"TPOT p50 {x['tpot_ms_p50']} ms", flush=True)
            asyncio.run(run_load(url, engine, wl["w2"][:4], 4, 4, 8, cache_prompt=pc))           # W2 prefix 예열 (on이면 캐시가 채워진다)
            r["w2"] = asyncio.run(run_load(url, engine, wl["w2"], 16, 64, 64, cache_prompt=pc))
            print(f"[perf] {mid} W2 {tag}: TTFT p50 {r['w2']['ttft_ms_p50']} ms", flush=True)
        r["vram_peak_mib"] = vp.peak
        r.update(parse_engine_log(log.read_text(errors="replace")))
        res["runs"][tag] = r
    OUT.joinpath(mid).mkdir(parents=True, exist_ok=True)
    (OUT / mid / f"{engine}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def bench_evafrill(mid: str, conc: list[int]) -> dict:
    """EVAFRILL: 서버가 없으므로 Goal 1 증분 디코딩 엔진으로 같은 W1 워크로드를 정적 배치로 잰다.
    동시 c = 배치 c(크기 맞춤 모드 batch_invariant=False, CUDA graph). 연속 배칭이 없어 배치 안 요청은 같이 시작·끝난다:
    TTFT = 배치의 순차 prefill이 끝나 첫 토큰이 나온 시각, TPOT = decode 시간/(출력−1). prefix caching 없음(W2 해당 없음)."""
    import torch

    from kobench.standard.evafrill_engine import EvafrillEngine
    from kobench.standard.evafrill_lm import EvafrillLM
    from kobench.standard.run import _versions

    spec = load_models()[mid]
    wl = workloads(spec)
    t0 = time.time()
    lm = EvafrillLM(pretrained=spec.path, revision=spec.revision, subfolder=spec.subfolder, max_length=4096)
    startup = time.time() - t0
    res = {"model": mid, "engine": "evafrill", "params_b": spec.params_b, "active_b": spec.active_b, "family": spec.family,
           "kind": spec.kind, "revision": spec.revision, "max_model_len": 4096,
           "workload": {"w1": "ko-arena-hard 질문 128개(학습 chat 형식), 출력 256 토큰 고정(EOS 무시), greedy, 정적 배치",
                        "w1_mean_in_tokens": round(wl["w1_mean_in"], 1), "w2": "해당 없음 (엔진에 prefix cache 없음)"},
           "versions": _versions(), "at": datetime.now(timezone.utc).isoformat(), "runs": {}}
    out_len = 256
    runs = []
    torch.cuda.reset_peak_memory_stats()
    with VramPoller() as vp:
        for c in conc:
            prompts = [wl["w1"][i % len(wl["w1"])] for i in range(max(32, 4 * c))]
            need = max(map(len, prompts)) + out_len + 1
            eng = EvafrillEngine(lm.model, max_batch=c, max_len=EvafrillEngine._bucket_static(need), batch_invariant=False)
            eng.generate(prompts[:c], 8, eos_ids=())                       # 예열 (그래프 캡처 포함, 집계 제외)
            ttft, tpot, toks = [], [], 0
            w0 = time.perf_counter()
            for s in range(0, len(prompts), c):
                tm = {}
                out = eng.generate(prompts[s: s + c], out_len, eos_ids=(), timings=tm)
                n = len(out)
                ttft += [tm["ttft_s"]] * n
                tpot += [tm["decode_s"] / (out_len - 1)] * n
                toks += sum(len(r.tokens) for r in out)
            wall = time.perf_counter() - w0
            pm = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]  # noqa: E731
            runs.append({"concurrency": c, "requests": len(prompts), "ok": len(prompts), "wall_s": round(wall, 3),
                         "output_tok_per_s": round(toks / wall, 1), "req_per_s": round(len(prompts) / wall, 3),
                         "ttft_ms_p50": round(1000 * pm(ttft, 50), 1), "ttft_ms_p90": round(1000 * pm(ttft, 90), 1),
                         "tpot_ms_p50": round(1000 * pm(tpot, 50), 2), "tpot_ms_p90": round(1000 * pm(tpot, 90), 2),
                         "per_stream_tok_per_s_p50": round(1 / pm(tpot, 50), 1)})
            print(f"[perf] {mid} c={c}: {runs[-1]['output_tok_per_s']} tok/s, TTFT p50 {runs[-1]['ttft_ms_p50']} ms", flush=True)
            del eng
            torch.cuda.empty_cache()
    res["runs"]["static_batch"] = {"startup_s": round(startup, 1), "w1": runs, "vram_peak_mib": vp.peak,
                                   "torch_max_allocated_gib": round(torch.cuda.max_memory_allocated() / (1 << 30), 2)}
    OUT.joinpath(mid).mkdir(parents=True, exist_ok=True)
    (OUT / mid / "evafrill.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "llamacpp", "evafrill"])
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16, 32, 64])
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for mid in a.models:
        if (OUT / mid / f"{a.engine}.json").exists() and not a.force:
            print(f"[perf] {mid}: 완료됨")
            continue
        try:
            if a.engine == "evafrill":
                bench_evafrill(mid, a.concurrency)
            else:
                bench_model(mid, a.engine, a.concurrency, a.gpu_mem)
        except Exception as e:
            print(f"[perf] {mid} 실패: {type(e).__name__}: {e}", flush=True)
            OUT.joinpath(mid).mkdir(parents=True, exist_ok=True)
            (OUT / mid / f"{a.engine}.error.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}",
                                                                          "at": datetime.now(timezone.utc).isoformat()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
