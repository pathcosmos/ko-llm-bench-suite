"""모델·스위트 레지스트리 로더 (configs/standard/*.yaml)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "standard"
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: str) -> str:
    """${VAR} / ${VAR:-default} 확장."""
    return _ENV.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    kind: str                 # base | instruct
    backend: str              # vllm | evafrill
    path: str
    revision: str | None
    params_b: float
    active_b: float
    owner: str = "public"
    subfolder: str = ""
    max_length: int = 4096
    gated: bool = False
    trust_remote_code: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return Path(self.path).exists()


@dataclass(frozen=True)
class TaskSpec:
    task: str
    num_fewshot: int
    source: str = ""
    metric: str = "acc"       # 보고용 대표 지표 (문항별 샘플 필드명과 같음: acc | acc_norm)


_MODEL_KEYS = {f for f in ModelSpec.__dataclass_fields__} - {"id", "extra"}


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    data = yaml.safe_load((path or CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
    out = {}
    for mid, d in data["models"].items():
        if d.get("kind") not in ("base", "instruct"):
            raise ValueError(f"{mid}: kind는 base|instruct")
        if d.get("backend") not in ("vllm", "evafrill"):
            raise ValueError(f"{mid}: backend는 vllm|evafrill")
        known = {k: v for k, v in d.items() if k in _MODEL_KEYS}
        known["path"] = _expand(str(known["path"]))
        out[mid] = ModelSpec(id=mid, extra={k: v for k, v in d.items() if k not in _MODEL_KEYS}, **known)
    return out


def load_suites(path: Path | None = None) -> dict[str, dict]:
    data = yaml.safe_load((path or CONFIG_DIR / "suites.yaml").read_text(encoding="utf-8"))
    out = {}
    for name, s in data["suites"].items():
        tasks = [TaskSpec(**t) for t in s["tasks"]]
        if len({t.task for t in tasks}) != len(tasks):
            raise ValueError(f"suite {name}: 태스크 중복")
        out[name] = {"description": s.get("description", ""), "tasks": tasks}
    return out
