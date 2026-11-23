"""생성 코드 격리 실행 (HumanEval+/MBPP+ 채점).

이 컨테이너에서는 네임스페이스(unshare -n/-r)와 seccomp를 쓸 수 없다(unshare: Operation not permitted, 2026-11-20 확인).
그래서 다음을 겹쳐 쓴다.
  - 프로세스: 새 세션(프로세스 그룹 전체를 시간 초과 시 SIGKILL), 문항마다 새 임시 디렉터리를 cwd로
  - rlimit: CPU 시간, 주소 공간(메모리), 파일 크기, 열린 파일 수, 코어 덤프 0
  - 환경: python -I(환경변수·사용자 site 무시), 최소 env(HOME/TMPDIR = 임시 디렉터리, 프록시 없음)
  - 파이썬 audit hook(제거 불가): 소켓 생성·연결, 하위 프로세스·exec·fork, ctypes 로드, 임시 디렉터리 밖 파일 쓰기·삭제·이름 변경을
    막는다. 파이썬 수준 차단이므로 C 확장을 통한 우회까지 막지는 못한다 — 결과에 그 한계를 기록한다.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

GUARD = r'''
import sys as _s, os as _o
try:                      # 채점 코드가 쓰는 numpy는 훅 설치 전에 올린다 (import 시 ctypes.dlopen을 쓴다)
    import numpy as _np
except Exception:
    pass
_ROOT = _o.path.realpath(_o.getcwd())
def _inside(p):
    try:
        p = _o.fsdecode(p)
    except Exception:
        return False
    return _o.path.realpath(_o.path.join(_ROOT, p)).startswith(_ROOT + _o.sep)
_DENY = {"socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto",
         "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
         "os.kill", "os.killpg", "ctypes.dlopen", "ctypes.dlsym", "pty.spawn", "webbrowser.open"}
_FILEOPS = {"os.remove", "os.rmdir", "os.rename", "shutil.rmtree", "shutil.move", "os.chmod", "os.chown",
            "os.truncate", "os.symlink", "os.link", "os.mkdir", "os.unlink"}
def _hook(ev, args):
    if ev in _DENY:
        raise PermissionError("sandbox: " + ev)
    if ev in _FILEOPS:
        if not all(_inside(a) for a in args[:2] if isinstance(a, (str, bytes, _o.PathLike))):
            raise PermissionError("sandbox: " + ev)
    elif ev == "open":
        path, mode = args[0], args[1]
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (_o.O_WRONLY | _o.O_RDWR | _o.O_CREAT))
        if writing and isinstance(path, (str, bytes, _o.PathLike)) and not _inside(path):
            raise PermissionError("sandbox: write outside workdir")
_s.addaudithook(_hook)
del _hook
'''


@dataclass
class ExecResult:
    passed: bool
    status: str          # pass | fail | timeout | error
    detail: str
    seconds: float


def _limits(cpu_s: int, mem_bytes: int, fsize: int):
    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


def run_program(code: str, timeout: float = 30.0, mem_gb: float = 4.0, python: str | None = None) -> ExecResult:
    d = tempfile.mkdtemp(prefix="kobench_code_")
    try:
        with open(os.path.join(d, "prog.py"), "w", encoding="utf-8") as f:
            f.write(GUARD + "\n" + code)
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "TMPDIR": d, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        t0 = time.time()
        p = subprocess.Popen([python or sys.executable, "-I", "-B", "prog.py"], cwd=d, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             preexec_fn=_limits(int(timeout) + 1, int(mem_gb * (1 << 30)), 16 << 20))
        try:
            _, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.communicate()
            return ExecResult(False, "timeout", f"> {timeout}s", time.time() - t0)
        dt = time.time() - t0
        tail = err.decode("utf-8", "replace")[-400:]
        if p.returncode == 0:
            return ExecResult(True, "pass", "", dt)
        status = "timeout" if p.returncode in (-signal.SIGXCPU, -signal.SIGKILL) else ("fail" if "AssertionError" in tail else "error")
        return ExecResult(False, status, tail, dt)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_many(programs: list[str], workers: int = 8, **kw) -> list[ExecResult]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda c: run_program(c, **kw), programs))


# ── 프로그램 조립 ───────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """응답에서 코드 추출: python 태그 블록 → 태그 없는/다른 태그 블록 → 응답 전체 (각 단계에서 'def '가 있는 블록 우선).

    2026-11-21 수정: 이전엔 태그와 무관하게 첫 ``` 블록부터 짝지어, ```markdown 블록 안에 ```python 블록을 넣은 응답
    (llama-3.2-3b-instruct A HumanEval+ 다수)에서 바깥 블록의 설명문이나 엉뚱한 구간이 뽑혀 SyntaxError가 났다.
    """
    import re

    def pick(blocks):
        withdef = [b for b in blocks if "def " in b]
        return (withdef or blocks)[0] if blocks else None

    # 1) python 태그 블록 (닫는 ``` 까지; 닫히지 않았으면 끝까지)
    py = re.findall(r"```[ \t]*(?:[Pp]ython3?|py)[ \t]*\n(.*?)(?:```|\Z)", text, re.S)
    got = pick(py)
    if got is not None:
        return got
    # 2) 그 밖의 블록
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    if not blocks and "```" in text:                       # 닫히지 않은 블록 (길이 한도 등)
        m = re.search(r"```[^\n]*\n(.*)", text, re.S)
        blocks = [m.group(1)] if m else []
    got = pick(blocks)
    return got if got is not None else text


def build_program(gold: dict, output: str, base: bool) -> str:
    prompt, ep = gold["prompt"], gold.get("entry_point")
    if base:
        code = gold.get("raw", prompt) + output
    else:
        code = extract_code(output)
        if ep and f"def {ep}" not in code:               # 함수 본문만 쓴 경우: 시그니처·docstring을 앞에 둔다
            code = prompt + code
        else:                                            # 완성 함수를 쓴 경우: 프롬프트의 import 줄을 앞에 둔다 (예: typing.List)
            imp = [l for l in prompt.splitlines() if l.startswith(("import ", "from "))]
            if imp:
                code = "\n".join(imp) + "\n" + code
    imports = "\n".join(gold.get("imports") or [])
    prog = (imports + "\n" if imports else "") + code + "\n\n" + gold["test"] + "\n"
    if ep:
        prog += f"\ncheck({ep})\n"
    return prog
