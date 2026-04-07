"""
EVAFRILL-Mo-3B PyTorch 직접 추론 러너

Ollama를 사용할 수 없는 커스텀 Mamba-2 + Transformer 하이브리드 모델을
PyTorch로 직접 로딩하여 generate() 인터페이스를 제공한다.
runner.py의 generate() 반환 형식과 호환되도록 래핑.
"""

import gc
import json
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# EVAFRILL 모델 소스 경로를 sys.path에 추가
_EVAFRILL_SRC = Path("/home/lanco/models/EVAFRILL-Mo")
if str(_EVAFRILL_SRC) not in sys.path:
    sys.path.insert(0, str(_EVAFRILL_SRC))

from model.config import LMConfig
from model.transformer import LLM
from tokenizers import Tokenizer
from safetensors.torch import load_file as load_safetensors

from . import config

# ── 경로 ────────────────────────────────────────────────────────────────────
EVAFRILL_CHECKPOINT = Path("/home/lanco/models/EVAFRILL-Mo-3B/slerp")
EVAFRILL_MODEL_NAME = "evafrill-mo-3b-slerp"

# ── 싱글톤 모델 캐시 ────────────────────────────────────────────────────────
_model: Optional[LLM] = None
_tokenizer: Optional[Tokenizer] = None
def _get_evafrill_device() -> str:
    """GPU 격리 전략에 따라 EVAFRILL 디바이스 결정"""
    if config.EVAFRILL_GPU_STRATEGY == "ollama_suspend":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cpu"

_device: str = _get_evafrill_device()


def _load_config(ckpt_dir: Path) -> LMConfig:
    """config.json (HF 형식) → LMConfig"""
    with open(ckpt_dir / "config.json", encoding="utf-8") as f:
        data = json.load(f)
    # HF 전용 필드 제거
    for key in ("model_type", "architectures", "_variant", "_description"):
        data.pop(key, None)
    return LMConfig(**data)


def gpu_is_healthy() -> bool:
    """nvidia-smi로 GPU가 정상 작동하는지 동적 확인"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False


def _cuda_cleanup() -> None:
    """CUDA 실패 후 프로세스 레벨 정리. 드라이버 레벨 오염은 복구 불가.

    4단계 정리 시퀀스:
      1) gc.collect() — Python 참조 순환 해제
      2) torch.cuda.synchronize() — 대기 중인 GPU 연산 완료
      3) torch.cuda.empty_cache() — GPU 캐시 메모리 해제
      4) torch.cuda.reset_peak_memory_stats() — peak 통계 리셋

    각 단계를 개별 try-except로 감싸는 이유: 드라이버가 이미 오염된 경우
    synchronize() 자체가 실패할 수 있으므로, 한 단계 실패 시에도 나머지를 시도한다.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  ⚠ cuda.synchronize() 실패: {e}")
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ⚠ cuda.empty_cache() 실패: {e}")
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"  ⚠ cuda.reset_peak_memory_stats() 실패: {e}")


def load_model() -> tuple[LLM, Tokenizer]:
    """모델과 토크나이저를 로딩 (최초 1회만)"""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print(f"  📦 EVAFRILL-Mo-3B SLERP 로딩 중 ({_device})...")
    start = time.time()

    # Config
    cfg = _load_config(EVAFRILL_CHECKPOINT)
    # Flash attention 비활성화 (추론 시 불필요, 호환성)
    cfg.use_flash_attn = False

    # 모델 생성 + safetensors 로딩
    model = None
    state_dict = None
    try:
        model = LLM(cfg)

        state_dict = load_safetensors(
            str(EVAFRILL_CHECKPOINT / "model.safetensors"),
            device="cpu",
        )
        model.load_state_dict(state_dict)
        model = model.to(device=_device, dtype=torch.bfloat16)
    except Exception as e:
        # CUDA 실패 또는 모델 생성 실패 시 정리
        print(f"  ⚠ EVAFRILL 모델 로딩 실패 — 정리 중: {e}")
        del model
        del state_dict
        _cuda_cleanup()
        if not gpu_is_healthy():
            print("  ❌ GPU 드라이버 오염 감지 (nvidia-smi 무응답)")
        else:
            print("  ℹ GPU 드라이버는 정상 (프로세스 레벨 CUDA 오류 가능성)")
        raise

    # state_dict CPU 복사본 해제 (모델은 이미 GPU에 있으므로 불필요)
    del state_dict
    gc.collect()

    model.eval()

    # 토크나이저
    tok_path = EVAFRILL_CHECKPOINT / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    elapsed = time.time() - start
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ EVAFRILL 로딩 완료: {n_params / 1e9:.2f}B params, {elapsed:.1f}s")

    _model = model
    _tokenizer = tokenizer
    return _model, _tokenizer


def unload_model() -> None:
    """VRAM 해제"""
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    _tokenizer = None
    _cuda_cleanup()


def _top_p_filtering(
    logits: torch.Tensor, top_p: float = 0.9, top_k: int = 50
) -> torch.Tensor:
    """Top-K + Top-P 필터링"""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    # 항상 2D [1, vocab] 유지 — 이후 cat 연산과 호환
    return logits


@torch.inference_mode()
def generate(
    prompt: str,
    system: str = "",
    options: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> dict:
    """
    EVAFRILL 추론 — runner.py generate()와 동일한 반환 형식

    Returns:
        dict with: response, eval_count, eval_duration_s, prompt_eval_count,
        prompt_eval_duration_s, total_duration_s, wall_time_s, tokens_per_sec, error
    """
    if options is None:
        options = dict(config.SAMPLING_PARAMS)

    temperature = options.get("temperature", 0.7)
    top_p = options.get("top_p", 0.9)
    top_k = options.get("top_k", 50)
    max_tokens = options.get("num_predict", 512)
    rep_penalty = options.get("repeat_penalty", 1.2)

    try:
        model, tokenizer = load_model()

        # 시스템 프롬프트 + 유저 프롬프트 결합
        full_prompt = f"{system}\n{prompt}" if system else prompt

        # 토크나이즈
        prompt_start = time.time()
        input_ids = torch.tensor(
            [tokenizer.encode(full_prompt).ids],
            dtype=torch.long,
            device=_device,
        )
        prompt_eval_time = time.time() - prompt_start
        prompt_len = input_ids.shape[1]

        eos_id = tokenizer.token_to_id("</s>")
        generated_ids = input_ids

        # 생성 루프
        gen_start = time.time()
        gen_count = 0

        for _ in range(max_tokens):
            logits_all, _ = model(generated_ids)
            logits = logits_all[:, -1, :]  # [1, vocab]

            # Repetition penalty
            if rep_penalty != 1.0:
                for token_id in generated_ids[0].tolist():
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= rep_penalty
                    else:
                        logits[0, token_id] *= rep_penalty

            # Temperature
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # Top-K/P filtering + sample
            logits = _top_p_filtering(logits, top_p=top_p, top_k=top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_id], dim=-1)
            gen_count += 1

            if eos_id is not None and next_id.item() == eos_id:
                break

        gen_duration = time.time() - gen_start
        total_duration = time.time() - prompt_start

        # 디코딩 — 프롬프트 이후 부분만
        new_ids = generated_ids[0, prompt_len:].tolist()
        response = tokenizer.decode(new_ids)

        return {
            "response": response,
            "eval_count": gen_count,
            "eval_duration_s": gen_duration,
            "prompt_eval_count": prompt_len,
            "prompt_eval_duration_s": prompt_eval_time,
            "total_duration_s": total_duration,
            "wall_time_s": total_duration,
            "tokens_per_sec": gen_count / gen_duration if gen_duration > 0 else 0,
            "error": None,
        }

    except Exception as e:
        return {
            "response": "",
            "eval_count": 0,
            "eval_duration_s": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration_s": 0,
            "total_duration_s": 0,
            "wall_time_s": 0,
            "tokens_per_sec": 0,
            "error": str(e),
        }


def is_evafrill(model_name: str) -> bool:
    """모델명이 EVAFRILL인지 확인"""
    return "evafrill" in model_name.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess 격리 계층 — CUDA 오류로부터 메인 프로세스 보호
#
# EVAFRILL의 model.to(cuda:0) → cudaErrorLaunchFailure가 GPU 드라이버를
# 오염시켜 Ollama까지 죽이는 문제를 방지한다.
# 모든 PyTorch/CUDA 연산은 별도 subprocess에서 실행되고, 오류 시 해당
# subprocess만 죽으며 드라이버는 OS 레벨에서 정상 회수된다.
# ═══════════════════════════════════════════════════════════════════════════════


class _WorkerBridge:
    """EVAFRILL subprocess 관리자

    통신: stdin/stdout JSON line protocol.
    Worker의 print() → stderr → 부모 콘솔에 표시.
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def _ensure_alive(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        project_root = str(Path(__file__).resolve().parent.parent)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "eval_framework.evafrill_runner"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # 부모 stderr 상속 → print 출력 공유
            text=True,
            bufsize=1,
            cwd=project_root,
        )
        return True

    def send(self, cmd: dict, timeout: int = 300) -> dict:
        if not self._ensure_alive():
            return {"ok": False, "error": "Worker 시작 실패"}
        try:
            self._proc.stdin.write(json.dumps(cmd) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._proc = None
            return {"ok": False, "error": f"Worker 파이프 끊김: {e}"}

        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            self._proc.kill()
            self._proc.wait()
            self._proc = None
            return {"ok": False, "error": f"Worker 타임아웃 ({timeout}s)"}

        line = self._proc.stdout.readline()
        if not line:
            rc = self._proc.wait()
            self._proc = None
            return {"ok": False, "error": f"Worker 비정상 종료 (exit={rc})"}

        return json.loads(line.strip())

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def shutdown(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
                self._proc.wait()
        self._proc = None


_bridge = _WorkerBridge()


def subprocess_load_model() -> bool:
    """Subprocess에서 EVAFRILL 모델 로딩"""
    result = _bridge.send({"action": "load"}, timeout=120)
    return result.get("ok", False)


def subprocess_generate(
    prompt: str,
    system: str = "",
    options: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> dict:
    """Subprocess에서 EVAFRILL 추론"""
    cmd = {
        "action": "generate",
        "prompt": prompt,
        "system": system,
        "options": options or dict(config.SAMPLING_PARAMS),
    }
    effective_timeout = (timeout or 120) + 30
    result = _bridge.send(cmd, timeout=effective_timeout)

    if "ok" in result and not result["ok"]:
        return {
            "response": "", "eval_count": 0, "eval_duration_s": 0,
            "prompt_eval_count": 0, "prompt_eval_duration_s": 0,
            "total_duration_s": 0, "wall_time_s": 0, "tokens_per_sec": 0,
            "error": result.get("error", "Worker error"),
        }
    return result


def subprocess_unload_model() -> None:
    """Worker subprocess 종료 — CUDA 컨텍스트 OS 레벨 정리"""
    _bridge.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP 원격 추론 (EVAFRILL_HTTP_URL 설정 시 활성화)
# ═══════════════════════════════════════════════════════════════════════════════

import os as _os
EVAFRILL_HTTP_URL = _os.getenv("EVAFRILL_HTTP_URL", "")  # e.g. "http://172.30.31.2:8000"

_ERROR_RESPONSE = {
    "response": "", "eval_count": 0, "eval_duration_s": 0,
    "prompt_eval_count": 0, "prompt_eval_duration_s": 0,
    "total_duration_s": 0, "wall_time_s": 0, "tokens_per_sec": 0,
}


def http_generate(
    prompt: str,
    system: str = "",
    options: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> dict:
    """원격 EVAFRILL HTTP 서버에 추론 요청"""
    import requests
    try:
        resp = requests.post(
            f"{EVAFRILL_HTTP_URL}/generate",
            json={
                "prompt": prompt,
                "system": system,
                "options": options or dict(config.SAMPLING_PARAMS),
            },
            timeout=timeout or 1200,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {**_ERROR_RESPONSE, "error": f"HTTP error: {e}"}


def http_load_model() -> bool:
    """원격 서버에 모델 로딩 요청 (보통 서버 시작 시 자동 로딩됨)"""
    import requests
    try:
        resp = requests.post(f"{EVAFRILL_HTTP_URL}/load", timeout=300)
        return resp.json().get("ok", False)
    except Exception:
        return False


def http_unload_model() -> None:
    """원격 서버에 모델 언로딩 요청 (보통 불필요)"""
    import requests
    try:
        requests.post(f"{EVAFRILL_HTTP_URL}/unload", timeout=30)
    except Exception:
        pass


def http_health() -> bool:
    """원격 서버 헬스체크"""
    import requests
    try:
        resp = requests.get(f"{EVAFRILL_HTTP_URL}/", timeout=5)
        return resp.json().get("model_loaded", False)
    except Exception:
        return False


def use_http() -> bool:
    """HTTP 원격 모드 활성화 여부"""
    return bool(EVAFRILL_HTTP_URL)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker 프로세스 메인 루프 (python -m eval_framework.evafrill_runner)
# ═══════════════════════════════════════════════════════════════════════════════


def _worker_loop():
    """Subprocess worker — stdin JSON 명령 → stdout JSON 응답"""
    _json_out = sys.stdout
    sys.stdout = sys.stderr  # print() → stderr (부모 콘솔에 표시)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = cmd.get("action")

        if action == "load":
            try:
                load_model()
                _json_out.write(json.dumps({"ok": True}) + "\n")
            except Exception as e:
                _json_out.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
            _json_out.flush()

        elif action == "generate":
            result = generate(
                prompt=cmd.get("prompt", ""),
                system=cmd.get("system", ""),
                options=cmd.get("options"),
                timeout=cmd.get("timeout"),
            )
            _json_out.write(json.dumps(result) + "\n")
            _json_out.flush()

        elif action == "unload":
            unload_model()
            _json_out.write(json.dumps({"ok": True}) + "\n")
            _json_out.flush()

        elif action == "quit":
            unload_model()
            break


if __name__ == "__main__":
    _worker_loop()
