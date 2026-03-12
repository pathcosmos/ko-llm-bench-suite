"""
안정적 Ollama API 호출 러너 — 재시도, 쿨다운, 점진적 저장
"""

import json
import time
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config


def ollama_health_check() -> bool:
    """Ollama 서버 상태 확인"""
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _restart_ollama() -> bool:
    """Ollama 서버 재시작 — graceful shutdown 후 재시작"""
    import os
    print("  🔄 Ollama 서버 재시작 중...")
    # Graceful shutdown 먼저 시도 (SIGTERM)
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True, timeout=10)
    time.sleep(5)
    # 아직 살아있으면 강제 종료
    subprocess.run(["pkill", "-9", "-f", "ollama serve"], capture_output=True, timeout=10)
    time.sleep(3)
    # 재시작 — GPU 장애 시 CPU-only 모드로 폴백
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/var/snap/ollama/common/models"
    if not config.GPU_AVAILABLE:
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama_serve.log", "a"),
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    # CPU 모드에서는 시작이 좀 더 걸릴 수 있음
    time.sleep(8)
    return ollama_health_check()


def wait_for_ollama(max_wait: int = 60, auto_restart: bool = True) -> bool:
    """Ollama 서버가 응답할 때까지 대기, 실패 시 자동 재시작"""
    start = time.time()
    while time.time() - start < max_wait:
        if ollama_health_check():
            return True
        print("  ⏳ Ollama 서버 대기 중...")
        time.sleep(3)

    # max_wait 초과 — 자동 재시작 시도
    if auto_restart:
        print(f"  ⚠ Ollama {max_wait}s 무응답 — 자동 재시작 시도")
        for attempt in range(3):
            if _restart_ollama():
                print("  ✅ Ollama 자동 재시작 성공")
                return True
            print(f"  ↻ 재시작 재시도 {attempt + 2}/3...")
            time.sleep(5)

    return False


def unload_model(model: str) -> None:
    """모델 언로드 — keep_alive=0으로 빈 요청"""
    try:
        requests.post(
            config.OLLAMA_API_GENERATE,
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass


def get_loaded_models() -> list[str]:
    """현재 로딩된 모델 목록"""
    try:
        r = requests.get(config.OLLAMA_API_PS, timeout=10)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def unload_all_models() -> None:
    """모든 로딩된 모델 언로드"""
    for model in get_loaded_models():
        unload_model(model)
    time.sleep(2)


def warmup_model(model: str) -> bool:
    """모델 로딩 (warm-up) — 짧은 프롬프트로 모델을 메모리에 올림"""
    try:
        r = requests.post(
            config.OLLAMA_API_GENERATE,
            json={
                "model": model,
                "prompt": "hello",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=config.WARMUP_TIMEOUT,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"  ⚠ Warmup 실패: {e}")
        return False


def generate(
    model: str,
    prompt: str,
    system: str = "",
    options: Optional[dict] = None,
    timeout: Optional[int] = None,
    stream: bool = False,
) -> dict:
    """
    Ollama API 호출 (재시도 포함)

    Returns:
        dict with keys: response, eval_count, eval_duration_s,
        prompt_eval_count, prompt_eval_duration_s, total_duration_s,
        wall_time_s, tokens_per_sec, error
    """
    if timeout is None:
        timeout = config.MODEL_TIMEOUTS.get(model, 120)
    if options is None:
        options = dict(config.SAMPLING_PARAMS)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options,
    }
    if system:
        payload["system"] = system

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        if attempt > 0:
            backoff = config.RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"    ↻ 재시도 {attempt + 1}/{config.MAX_RETRIES} ({backoff}s 대기)")
            time.sleep(backoff)
            if not wait_for_ollama():
                last_error = "Ollama 서버 무응답"
                continue

        try:
            start = time.time()
            resp = requests.post(
                config.OLLAMA_API_GENERATE,
                json=payload,
                timeout=timeout,
            )
            wall_time = time.time() - start
            data = resp.json()

            eval_dur = data.get("eval_duration", 0)
            eval_cnt = data.get("eval_count", 0)

            return {
                "response": data.get("response", ""),
                "eval_count": eval_cnt,
                "eval_duration_s": eval_dur / 1e9,
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "prompt_eval_duration_s": data.get("prompt_eval_duration", 0) / 1e9,
                "total_duration_s": data.get("total_duration", 0) / 1e9,
                "wall_time_s": wall_time,
                "tokens_per_sec": eval_cnt / (eval_dur / 1e9) if eval_dur > 0 else 0,
                "error": None,
            }
        except requests.exceptions.Timeout:
            last_error = f"타임아웃 ({timeout}s)"
        except requests.exceptions.ConnectionError:
            last_error = "Ollama 연결 실패"
        except Exception as e:
            last_error = str(e)

    return _error_result(last_error)


def chat(
    model: str,
    messages: list[dict],
    options: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> dict:
    """Ollama chat API 호출 (멀티턴용)"""
    if timeout is None:
        timeout = config.MODEL_TIMEOUTS.get(model, 120)
    if options is None:
        options = dict(config.SAMPLING_PARAMS)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        if attempt > 0:
            backoff = config.RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            time.sleep(backoff)
            if not wait_for_ollama():
                last_error = "Ollama 서버 무응답"
                continue

        try:
            start = time.time()
            resp = requests.post(config.OLLAMA_API_CHAT, json=payload, timeout=timeout)
            wall_time = time.time() - start
            data = resp.json()

            msg = data.get("message", {})
            eval_dur = data.get("eval_duration", 0)
            eval_cnt = data.get("eval_count", 0)

            return {
                "response": msg.get("content", ""),
                "eval_count": eval_cnt,
                "eval_duration_s": eval_dur / 1e9,
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "prompt_eval_duration_s": data.get("prompt_eval_duration", 0) / 1e9,
                "total_duration_s": data.get("total_duration", 0) / 1e9,
                "wall_time_s": wall_time,
                "tokens_per_sec": eval_cnt / (eval_dur / 1e9) if eval_dur > 0 else 0,
                "error": None,
            }
        except requests.exceptions.Timeout:
            last_error = f"타임아웃 ({timeout}s)"
        except requests.exceptions.ConnectionError:
            last_error = "Ollama 연결 실패"
        except Exception as e:
            last_error = str(e)

    return _error_result(last_error)


def switch_model(new_model: str, current_model: Optional[str] = None) -> bool:
    """
    모델 전환 — 이전 모델 언로드 + 새 모델 웜업
    VRAM 충돌 방지를 위해 이전 모델을 먼저 해제.
    Ollama 크래시 시 자동 재시작 후 재시도.
    """
    if current_model and current_model != new_model:
        print(f"  🔄 {current_model} → {new_model}")
        unload_model(current_model)
        time.sleep(config.COOLDOWN_BETWEEN_MODELS)
    else:
        print(f"  📦 모델 로딩: {new_model}")

    for attempt in range(3):
        if not ollama_health_check():
            print(f"  ⚠ Ollama 서버 다운 감지 (모델 전환 시도 {attempt + 1}/3)")
            if not wait_for_ollama(max_wait=30, auto_restart=True):
                continue

        if warmup_model(new_model):
            return True

        print(f"  ⚠ Warmup 실패 — Ollama 재시작 후 재시도 ({attempt + 1}/3)")
        _restart_ollama()
        time.sleep(5)

    print(f"  ❌ 모델 전환 실패: {new_model}")
    return False


def save_results_incremental(results: dict, track_name: str) -> Path:
    """결과를 점진적으로 저장 — 중간 크래시에도 데이터 보존"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.RESULTS_DIR / f"{track_name}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


def save_checkpoint(data: dict, track_name: str) -> Path:
    """체크포인트 저장 (덮어쓰기 방식)"""
    path = config.RESULTS_DIR / f"{track_name}_checkpoint.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_checkpoint(track_name: str) -> Optional[dict]:
    """이전 체크포인트 로드"""
    path = config.RESULTS_DIR / f"{track_name}_checkpoint.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_vram_usage() -> dict:
    """nvidia-smi로 현재 VRAM 사용량 조회"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "vram_used_mb": int(parts[0]),
                "vram_total_mb": int(parts[1]),
                "vram_free_mb": int(parts[2]),
                "gpu_util_pct": int(parts[3]),
            }
    except Exception:
        pass
    return {"vram_used_mb": 0, "vram_total_mb": 0, "vram_free_mb": 0, "gpu_util_pct": 0}


def _error_result(error_msg: str) -> dict:
    return {
        "response": "",
        "eval_count": 0,
        "eval_duration_s": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration_s": 0,
        "total_duration_s": 0,
        "wall_time_s": 0,
        "tokens_per_sec": 0,
        "error": error_msg,
    }
