"""lm-eval 커스텀 태스크 (TaskManager include_path). TASKS_DIR을 include_path로 넘긴다."""
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent
