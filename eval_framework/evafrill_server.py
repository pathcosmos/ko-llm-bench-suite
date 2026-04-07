"""EVAFRILL-Mo-3B HTTP Inference Server

원격 서버에서 모델을 GPU에 상주시키고 HTTP API로 추론 요청을 처리한다.
모델은 서버 시작 시 1회 로딩되어 메모리에 유지된다.

Usage:
    cd ~/cursor/temp_git/frankenstallm_test
    ~/ai-env/bin/python -m uvicorn eval_framework.evafrill_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import time
import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# evafrill_runner의 generate/load/unload를 직접 사용
# GPU 전략을 ollama_suspend로 설정하여 cuda:0 사용
os.environ.setdefault("EVAFRILL_GPU_STRATEGY", "ollama_suspend")

app = FastAPI(title="EVAFRILL-Mo-3B Inference Server")

_loaded = False


class GenerateRequest(BaseModel):
    prompt: str
    system: str = ""
    options: Optional[dict] = None
    timeout: Optional[int] = None


class GenerateResponse(BaseModel):
    response: str
    eval_count: int
    eval_duration_s: float
    prompt_eval_count: int
    prompt_eval_duration_s: float
    total_duration_s: float
    wall_time_s: float
    tokens_per_sec: float
    error: Optional[str] = None


@app.on_event("startup")
def startup_load_model():
    """서버 시작 시 모델을 GPU에 로딩"""
    global _loaded
    from eval_framework.evafrill_runner import load_model
    print("🔄 EVAFRILL 모델 로딩 중...")
    start = time.time()
    load_model()
    elapsed = time.time() - start
    print(f"✅ EVAFRILL 모델 로딩 완료 ({elapsed:.1f}s)")
    _loaded = True


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": _loaded}


@app.post("/generate", response_model=GenerateResponse)
async def http_generate(req: GenerateRequest):
    from eval_framework.evafrill_runner import generate
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generate(
            prompt=req.prompt,
            system=req.system,
            options=req.options,
            timeout=req.timeout,
        ),
    )
    return result


@app.post("/load")
def http_load():
    global _loaded
    from eval_framework.evafrill_runner import load_model
    load_model()
    _loaded = True
    return {"ok": True}


@app.post("/unload")
def http_unload():
    global _loaded
    from eval_framework.evafrill_runner import unload_model
    unload_model()
    _loaded = False
    return {"ok": True}
