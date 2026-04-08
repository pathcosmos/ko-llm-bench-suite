"""실시간 웹 대시보드 서버 — FastAPI + SSE."""
import json
import asyncio
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ── Dashboard State ──────────────────────────────────────────────────────────


class DashboardState:
    """Thread-safe dashboard state shared between executor and server."""

    def __init__(self):
        self.started_at: float = time.time()
        self.total_tracks: int = 0
        self.completed_tracks: int = 0
        self.current_track: Optional[int] = None
        self.current_track_name: str = ""
        self.current_model: str = ""
        self.current_item: str = ""
        self.items_done: int = 0
        self.items_total: int = 0
        self.tracks_status: dict = {}   # {track_num: "done"|"running"|"pending"|"error"}
        self.tracks_elapsed: dict = {}  # {track_num: seconds}
        self.gpu_name: str = ""
        self.gpu_vram_total: str = ""
        self.gpu_vram_free: str = ""
        self.errors: list = []  # [{track, model, message, time}]
        self.finished: bool = False
        self.models: list = []

    def to_dict(self) -> dict:
        elapsed = time.time() - self.started_at
        # Estimate remaining
        if self.completed_tracks > 0 and self.total_tracks > 0:
            avg_per_track = elapsed / self.completed_tracks
            remaining = avg_per_track * (self.total_tracks - self.completed_tracks)
        else:
            remaining = 0

        return {
            "elapsed_s": round(elapsed, 1),
            "est_remaining_s": round(remaining, 1),
            "total_tracks": self.total_tracks,
            "completed_tracks": self.completed_tracks,
            "current_track": self.current_track,
            "current_track_name": self.current_track_name,
            "current_model": self.current_model,
            "current_item": self.current_item,
            "items_done": self.items_done,
            "items_total": self.items_total,
            "tracks_status": self.tracks_status,
            "tracks_elapsed": self.tracks_elapsed,
            "gpu": {
                "name": self.gpu_name,
                "vram_total": self.gpu_vram_total,
                "vram_free": self.gpu_vram_free,
            },
            "errors": self.errors[-10:],  # Last 10
            "error_count": len(self.errors),
            "finished": self.finished,
            "models": self.models,
        }


# ── Global State ─────────────────────────────────────────────────────────────

_state = DashboardState()
_event_queue: Optional[Queue] = None
_server: Optional[uvicorn.Server] = None
_server_thread: Optional[threading.Thread] = None
_should_stop = threading.Event()

TRACK_NAMES = {
    1: "Korean Bench",
    2: "Ko-Bench",
    3: "Korean Deep",
    4: "Code & Math",
    5: "Consistency",
    6: "Performance",
    7: "Pairwise Elo",
}

def _drain_queue():
    """Process all pending events from the queue into state."""
    if _event_queue is None:
        return
    try:
        while True:
            event = _event_queue.get_nowait()
            _process_event(event)
    except Empty:
        pass


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="KoBench Dashboard")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    _drain_queue()
    return JSONResponse(content=_state.to_dict())


@app.get("/api/events")
async def api_events():
    """SSE endpoint -- streams events to browser."""

    async def event_generator():
        while not _should_stop.is_set():
            _drain_queue()

            # Send current state as heartbeat
            data = json.dumps(_state.to_dict(), ensure_ascii=False)
            yield f"data: {data}\n\n"

            await asyncio.sleep(1)  # 1-second update interval

        # Final state
        data = json.dumps(_state.to_dict(), ensure_ascii=False)
        yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _process_event(event: dict):
    """Update dashboard state from event."""
    t = event.get("type", "")

    if t == "init":
        _state.total_tracks = event.get("total_tracks", 0)
        _state.models = event.get("models", [])
        _state.started_at = time.time()
        for tn in event.get("track_nums", []):
            _state.tracks_status[tn] = "pending"

    elif t == "track_start":
        tn = event.get("track", 0)
        _state.current_track = tn
        _state.current_track_name = TRACK_NAMES.get(tn, f"Track {tn}")
        _state.tracks_status[tn] = "running"
        _state.items_done = 0
        _state.items_total = 0

    elif t == "track_done":
        tn = event.get("track", 0)
        _state.tracks_status[tn] = "done"
        _state.tracks_elapsed[tn] = round(event.get("elapsed", 0), 1)
        _state.completed_tracks += 1
        _state.current_model = ""

    elif t == "track_error":
        tn = event.get("track", 0)
        _state.tracks_status[tn] = "error"
        _state.errors.append({
            "track": tn,
            "model": event.get("model", ""),
            "message": event.get("message", ""),
            "time": time.time(),
        })

    elif t == "model_start":
        _state.current_model = event.get("model", "")
        _state.items_done = 0
        _state.items_total = event.get("total_items", 0)

    elif t == "model_done":
        _state.current_model = ""

    elif t == "progress":
        _state.items_done = event.get("done", 0)
        _state.items_total = event.get("total", 0)
        _state.current_item = event.get("item", "")

    elif t == "gpu":
        _state.gpu_name = event.get("name", "")
        _state.gpu_vram_total = event.get("vram_total", "")
        _state.gpu_vram_free = event.get("vram_free", "")

    elif t == "error":
        _state.errors.append({
            "track": event.get("track", 0),
            "model": event.get("model", ""),
            "message": event.get("message", ""),
            "time": time.time(),
        })

    elif t == "finished":
        _state.finished = True


# ── Server Lifecycle ─────────────────────────────────────────────────────────


def start_dashboard(event_queue: Queue, port: int = 8888) -> str:
    """백그라운드 스레드에서 대시보드 서버 시작. URL 반환.

    Raises:
        RuntimeError: 포트 바인딩 실패 시
    """
    global _event_queue, _server_thread, _server, _state

    # Queue에 maxsize 적용 (메모리 누수 방지 — 브라우저 미연결 시 오래된 이벤트 드랍)
    _event_queue = Queue(maxsize=500) if event_queue is None else event_queue
    # 호출자가 넘긴 큐도 참조 유지 (양쪽에서 사용 가능)
    if event_queue is not None:
        _event_queue = event_queue
    _should_stop.clear()

    # Reset state
    _state = DashboardState()

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    _server = uvicorn.Server(config)

    _server_thread = threading.Thread(target=_server.run, daemon=True)
    _server_thread.start()

    # Wait and verify server thread is still alive (bind failure kills thread)
    time.sleep(1.5)
    if not _server_thread.is_alive():
        raise RuntimeError(f"대시보드 서버 시작 실패 (port={port} 사용 중이거나 바인딩 오류)")
    # Double-check with HTTP
    try:
        import requests as _req
        r = _req.get(f"http://localhost:{port}/api/status", timeout=3)
        if r.status_code != 200:
            raise RuntimeError(f"대시보드 서버 응답 오류 (status={r.status_code})")
    except RuntimeError:
        raise
    except Exception as e:
        _should_stop.set()
        raise RuntimeError(f"대시보드 서버 시작 실패 (port={port}): {e}")

    url = f"http://localhost:{port}"
    return url


def stop_dashboard():
    """대시보드 서버 종료."""
    _should_stop.set()
    if _server is not None:
        _server.should_exit = True
