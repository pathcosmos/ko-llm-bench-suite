# Interactive Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 4개 서브커맨드(setup, eval, config, status) 인터랙티브 위저드 구현

**Architecture:** kobench/wizard/ 패키지에 서브커맨드별 모듈 분리. Rich 라이브러리로 UI. 기존 kobench.py 배치 모드 100% 호환 유지.

**Tech Stack:** Python 3.12, rich (13.7.1, 이미 설치됨), 기존 kobench 패키지

**Spec:** `docs/superpowers/specs/2026-04-08-interactive-wizard-design.md`

---

## Task 1: UI 헬퍼 + 위저드 패키지 기반

**Files:**
- Create: `kobench/wizard/__init__.py`
- Create: `kobench/wizard/ui.py`

**ui.py — Rich 기반 공통 UI 컴포넌트:**

```python
"""Rich 기반 인터랙티브 UI 헬퍼."""
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

def banner():
    console.print(Panel.fit(
        "[bold cyan]KoBench Suite[/] — 한국어 LLM 벤치마크",
        subtitle="v0.1.0"
    ))

def step(num, total, title):
    console.print(f"\n[bold white]\\[{num}/{total}][/] [bold]{title}[/]")

def success(msg): console.print(f"  [green]✅ {msg}[/]")
def warn(msg):    console.print(f"  [yellow]⚠️  {msg}[/]")
def fail(msg):    console.print(f"  [red]❌ {msg}[/]")
def info(msg):    console.print(f"  [dim]{msg}[/]")

def ask(prompt, **kwargs):       return Prompt.ask(f"  {prompt}", **kwargs)
def confirm(prompt, **kwargs):   return Confirm.ask(f"  {prompt}", **kwargs)

def select_one(prompt, choices):
    for i, c in enumerate(choices, 1):
        console.print(f"  ({i}) {c}")
    idx = Prompt.ask(f"  {prompt}", choices=[str(i) for i in range(1, len(choices)+1)])
    return choices[int(idx) - 1]

def select_multi(prompt, items):
    selected = []
    for item in items:
        if Confirm.ask(f"  {item}?", default=True):
            selected.append(item)
    return selected

def show_table(title, columns, rows):
    table = Table(title=title, show_header=True)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(c) for c in row])
    console.print(table)

def progress_context():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    )
```

- [ ] Step 1: Create kobench/wizard/ 패키지 + ui.py
- [ ] Step 2: `python -c "from kobench.wizard.ui import banner; banner()"` 동작 확인
- [ ] Step 3: Commit

---

## Task 2: `status` 서브커맨드 (가장 단순)

**Files:**
- Create: `kobench/wizard/status_cmd.py`

```python
"""시스템 진단 표시."""
import subprocess, shutil
from kobench import config
from kobench.wizard import ui

def run():
    ui.banner()
    ui.step(1, 3, "시스템 정보")
    # Python, OS, GPU
    
    ui.step(2, 3, "Ollama 상태")
    # 연결 테스트 + 모델 목록
    
    ui.step(3, 3, "이전 평가 결과")
    # results/ 디렉토리 스캔
```

- [ ] Step 1: status_cmd.py 작성
- [ ] Step 2: 테스트 (수동 실행)
- [ ] Step 3: Commit

---

## Task 3: `setup` 서브커맨드

**Files:**
- Create: `kobench/wizard/setup_cmd.py`
- Create: `kobench/wizard/machine.py`

**setup_cmd.py 플로우:**
```
Step 1: Python 의존성 확인 (import 테스트)
Step 2: Ollama 확인 + 설치 안내
Step 3: 원격 서버 추가? → URL 입력 + 연결 테스트
Step 4: Judge 모델 확인 + 다운로드
Step 5: 요약 테이블 + 설정 저장 제안
```

**machine.py — 머신 연결 테스트:**
```python
import requests, socket

def test_connection(url, timeout=5):
    """Ollama 서버 연결 테스트. {connected, models, error} 반환."""

def test_port(host, port, timeout=3):
    """TCP 포트 연결 가능 여부."""
```

- [ ] Step 1: machine.py 작성
- [ ] Step 2: setup_cmd.py 작성
- [ ] Step 3: 연결 테스트 검증
- [ ] Step 4: Commit

---

## Task 4: `config` 서브커맨드

**Files:**
- Create: `kobench/wizard/config_cmd.py`

**플로우:**
```
Step 1: 프로젝트 이름/경로 입력
Step 2: 백엔드 URL/원격 설정
Step 3: 모델 선택 (Ollama에서 발견된 목록)
Step 4: 트랙 선택
Step 5: Judge 설정
Step 6: YAML 미리보기 + 저장
```

- [ ] Step 1: config_cmd.py 작성
- [ ] Step 2: 생성된 YAML이 validate_config() 통과하는지 테스트
- [ ] Step 3: Commit

---

## Task 5: `eval` 서브커맨드 (핵심)

**Files:**
- Create: `kobench/wizard/eval_cmd.py`
- Create: `kobench/wizard/executor.py`

**eval_cmd.py — 인터랙티브 평가 플로우:**
```
Step 1: 설정 파일 선택 (발견된 YAML 표시)
Step 2: 환경 확인 (Ollama, GPU, 모델)
Step 3: 모델 선택 (available 표시, missing pull 제안)
Step 4: 시나리오 선택 (빠른/전체/커스텀)
Step 5: 실행 (executor에 위임)
Step 6: 결과 표시 (Rich Table)
```

**executor.py — 실행 컨트롤러:**
```python
def run_evaluation_interactive(tracks, models, config_path=None):
    """인터랙티브 실행: 진행률 바 + 에러 복구 UI."""
    
    for track_num in tracks:
        # Rich Progress 바
        # try/except → 에러 시 ask("retry/skip/abort")
        # 체크포인트 저장
```

**에러 복구:**
```python
def handle_error(track_num, model, error):
    choice = ui.ask(
        f"Track {track_num} / {model} 오류: {error}",
        choices=["retry", "skip_model", "skip_track", "abort"]
    )
    return choice
```

- [ ] Step 1: executor.py 작성 (진행률 + 에러 복구)
- [ ] Step 2: eval_cmd.py 작성 (6단계 플로우)
- [ ] Step 3: 빠른 평가 시나리오 테스트 (T6, 모델 1개)
- [ ] Step 4: Commit

---

## Task 6: kobench.py 서브커맨드 통합

**Files:**
- Modify: `kobench.py`

기존 argparse에 서브커맨드 추가:

```python
subparsers = parser.add_subparsers(dest="command")

# 서브커맨드 등록
sub_setup = subparsers.add_parser("setup", help="인터랙티브 설치 위저드")
sub_eval = subparsers.add_parser("eval", help="인터랙티브 평가 실행")
sub_config = subparsers.add_parser("config", help="YAML 설정 빌더")
sub_status = subparsers.add_parser("status", help="시스템 진단")
```

main()에서:
```python
if args.command == "setup":
    from kobench.wizard.setup_cmd import run
    run(); return
elif args.command == "eval":
    from kobench.wizard.eval_cmd import run
    run(); return
elif args.command == "config":
    from kobench.wizard.config_cmd import run
    run(); return
elif args.command == "status":
    from kobench.wizard.status_cmd import run
    run(); return
# else: 기존 배치 모드 (하위호환)
```

- [ ] Step 1: 서브커맨드 등록 + 라우팅
- [ ] Step 2: `kobench.py setup` 동작 확인
- [ ] Step 3: `kobench.py eval` 동작 확인
- [ ] Step 4: `kobench.py --config X.yaml` 기존 배치 모드 하위호환 확인
- [ ] Step 5: 전체 테스트 `pytest tests/ -x`
- [ ] Step 6: Commit

---

## Task 7: requirements.txt + 문서 + 최종 push

**Files:**
- Modify: `requirements.txt` — `rich>=13.7` 추가
- Modify: `pyproject.toml` — rich 의존성 추가
- Modify: `README.md`, `README_EN.md` — 위저드 사용법 섹션 추가

- [ ] Step 1: 의존성 업데이트
- [ ] Step 2: README에 위저드 섹션 추가
- [ ] Step 3: 전체 테스트 통과 확인
- [ ] Step 4: Commit + push

---

## Verification Checklist

- [ ] `python kobench.py setup` — 인터랙티브 설치 완료
- [ ] `python kobench.py eval` — 모델 선택 → 트랙 실행 → 결과 표시
- [ ] `python kobench.py config` — YAML 생성 + validate 통과
- [ ] `python kobench.py status` — 시스템 정보 표시
- [ ] `python kobench.py --config X.yaml --tracks 1` — 배치 모드 하위호환
- [ ] `python kobench.py --help` — 서브커맨드 목록 표시
- [ ] 에러 시나리오: 없는 모델 → skip → 나머지 계속
- [ ] `pytest tests/ -x` — 전체 통과
