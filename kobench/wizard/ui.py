"""Rich 기반 인터랙티브 UI 헬퍼."""
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from pathlib import Path

console = Console()

def banner():
    """프로젝트 배너 표시."""
    console.print(Panel.fit(
        "[bold cyan]KoBench Suite[/] — 한국어 LLM 종합 벤치마크",
        subtitle="v0.1.0",
        border_style="cyan",
    ))

def step(num, total, title):
    """단계 헤더."""
    console.print(f"\n[bold white]\\[{num}/{total}][/] [bold]{title}[/]")

def success(msg):
    console.print(f"  [green]✅ {msg}[/]")

def warn(msg):
    console.print(f"  [yellow]⚠️  {msg}[/]")

def fail(msg):
    console.print(f"  [red]❌ {msg}[/]")

def info(msg):
    console.print(f"  [dim]{msg}[/]")

def ask(prompt, **kwargs):
    """텍스트 입력."""
    return Prompt.ask(f"  {prompt}", **kwargs)

def ask_int(prompt, **kwargs):
    """숫자 입력."""
    return IntPrompt.ask(f"  {prompt}", **kwargs)

def confirm(prompt, default=True):
    """예/아니오 확인."""
    return Confirm.ask(f"  {prompt}", default=default)

def select_one(prompt, choices, descriptions=None):
    """단일 선택 (번호 기반)."""
    for i, c in enumerate(choices, 1):
        desc = f" — {descriptions[i-1]}" if descriptions else ""
        console.print(f"  [cyan]({i})[/] {c}{desc}")
    idx = Prompt.ask(f"  {prompt}", choices=[str(i) for i in range(1, len(choices) + 1)])
    return choices[int(idx) - 1]

def select_multi(prompt, items):
    """다중 선택 (각 항목 confirm)."""
    console.print(f"  [bold]{prompt}[/]")
    selected = []
    for item in items:
        if Confirm.ask(f"    {item}", default=True):
            selected.append(item)
    return selected

def show_table(title, columns, rows):
    """테이블 표시."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(c) for c in row])
    console.print(table)

def show_yaml(yaml_text):
    """YAML 미리보기."""
    syntax = Syntax(yaml_text, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="YAML 설정 미리보기", border_style="green"))

def progress_context():
    """진행률 바 컨텍스트."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

def error_choice(track, model, error_msg):
    """에러 발생 시 사용자 선택."""
    fail(f"{track} / {model}: {error_msg}")
    return select_one("어떻게 진행하시겠습니까?",
        ["retry", "skip_model", "skip_track", "abort"],
        ["재시도", "이 모델 건너뛰기", "이 트랙 건너뛰기", "평가 중단"]
    )

def divider():
    console.print("─" * 60, style="dim")
