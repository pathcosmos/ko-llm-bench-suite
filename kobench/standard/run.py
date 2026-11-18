"""표준 벤치 실행기 — lm-evaluation-harness를 모델당 한 번 로드해 스위트의 모든 태스크를 실행한다.

python -m kobench.standard.run --models jansori-bf16 frankenstallm-bf16 --suite ko_mc_v1
python -m kobench.standard.run --models evafrill-mo-3b --suite smoke --limit 50
python -m kobench.standard.run --list

출력: results/standard/<model>/<task>__<n>shot[__limitN].json   (집계·설정·메타)
      results/standard/<model>/<task>__<n>shot[__limitN].samples.jsonl.gz (문항별 로그우도·정답)
이미 있는 결과는 건너뛴다(--force로 재실행). 공개 모델은 revision이 고정돼 있어야 실행한다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kobench.standard.registry import ModelSpec, load_models, load_suites

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "standard"
SEED = 1234


def _git_commit() -> str | None:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(ROOT), "diff", "--quiet"]) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return None


def _versions() -> dict:
    import importlib.metadata as md

    v = {}
    for pkg in ("lm_eval", "vllm", "torch", "transformers"):
        try:
            v[pkg] = md.version(pkg.replace("_", "-"))
        except md.PackageNotFoundError:
            v[pkg] = None
    try:
        import torch

        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        v["cuda"] = torch.version.cuda
    except Exception:
        pass
    v["python"] = platform.python_version()
    return v


@dataclass(frozen=True)
class Variant:
    """평가 설정 변형. 기본값 = A 표(공통 프로토콜)."""
    doc_start: str = "none"        # none | eos  (사용자 모델만: 사전학습 문서 시작 조건 재현)
    chat: bool = False             # 지시 모델에 chat 템플릿 적용 (사용자 모델은 formats.yaml 학습 형식)
    shots: int | None = None       # 스위트 few-shot 수 대신 쓸 값

    @property
    def name(self) -> str:
        parts = []
        if self.doc_start != "none":
            parts.append(f"docstart_{self.doc_start}")
        if self.chat:
            parts.append("chat")
        if self.shots is not None:
            parts.append(f"shots{self.shots}")
        return "A" if not parts else "+".join(parts)

    @classmethod
    def parse(cls, text: str | None) -> "Variant":
        """예: 'docstart=eos,chat=1,shots=0'. 빈 값이면 A."""
        if not text or text == "A":
            return cls()
        kw = {}
        for item in text.split(","):
            k, _, v = item.partition("=")
            if k == "docstart":
                kw["doc_start"] = v
            elif k == "chat":
                kw["chat"] = v in ("1", "true", "yes")
            elif k == "shots":
                kw["shots"] = int(v)
            else:
                raise ValueError(f"알 수 없는 변형 항목: {k}")
        return cls(**kw)


def build_lm(spec: ModelSpec, gpu_mem: float, batch_size: str, variant: "Variant | None" = None):
    """모델당 한 번만 로드한다 (vLLM 로딩이 수십 초~분 단위)."""
    variant = variant or Variant()
    if variant.doc_start != "none" and spec.owner != "self":
        raise ValueError("문서 시작 토큰 변형은 사용자 모델 전용 (공개 모델은 자체 BOS 규칙을 따른다)")
    if spec.backend == "vllm":
        from lm_eval.models.vllm_causallms import VLLM

        fmt_kwargs = {}
        if spec.owner == "self":
            # 사용자 모델은 학습 형식(configs/standard/formats.yaml)의 토크나이저를 쓰고,
            # 빈 문맥(PPL 등)의 prefix는 BOS 대신 EOS로 둔다 (3종 모두 BOS로 학습하지 않음).
            from kobench.standard import formats

            fmt = formats.model_format(spec.id)
            fmt_kwargs["tokenizer"] = str(formats.prepare_tokenizer_dir(spec.id, doc_start=variant.doc_start))
            fmt_kwargs["prefix_token_id"] = fmt["stop_token_ids"][0]
        return VLLM(
            pretrained=spec.path,
            revision=None if spec.is_local else spec.revision,
            dtype="bfloat16",
            max_model_len=spec.max_length,
            gpu_memory_utilization=gpu_mem,
            batch_size=batch_size,
            trust_remote_code=spec.trust_remote_code,
            seed=SEED,
            **fmt_kwargs,
            **spec.extra.get("vllm_args", {}),
        )
    if spec.backend == "evafrill":
        from kobench.standard.evafrill_lm import EvafrillLM

        return EvafrillLM(
            pretrained=spec.path,
            revision=spec.revision,
            subfolder=spec.subfolder,
            max_length=spec.max_length,
            doc_prefix_eos=variant.doc_start == "eos",
            **spec.extra.get("evafrill_args", {}),
        )
    raise ValueError(spec.backend)


def prefetch_datasets(tasks, retries: int = 5, backoff: float = 10.0) -> None:
    """모델을 올리기 전에 태스크 데이터셋을 캐시에 받아 둔다.

    KMMLU는 과목별 CSV 90여 개를 받는데, 실행 중 HF가 일시적으로 500을 내 평가가 중단된 적이 있다(2026-11-18).
    """
    from lm_eval.tasks import TaskManager, get_task_dict

    tm = task_manager()
    for t in tasks:
        for attempt in range(1, retries + 1):
            try:
                get_task_dict([t.task], tm)
                print(f"[data] {t.task} 준비됨")
                break
            except Exception as e:  # 네트워크·HF 5xx
                if attempt == retries:
                    raise
                wait = backoff * attempt
                print(f"[data] {t.task} 실패 ({type(e).__name__}: {str(e)[:120]}) — {wait:.0f}초 후 재시도 {attempt}/{retries}")
                time.sleep(wait)


def _out_paths(model_id: str, task: str, nshot: int, limit, variant: "Variant | None" = None) -> tuple[Path, Path]:
    stem = f"{task}__{nshot}shot" + (f"__limit{limit}" if limit else "")
    v = variant or Variant()
    d = RESULTS / model_id if v.name == "A" else RESULTS / model_id / f"v_{v.name}"
    return d / f"{stem}.json", d / f"{stem}.samples.jsonl.gz"


def task_manager():
    """lm-eval 기본 태스크 + kobench 커스텀 태스크(dev 분할 등)."""
    from lm_eval.tasks import TaskManager

    from kobench.standard.tasks import TASKS_DIR

    return TaskManager(include_path=[str(p) for p in sorted(TASKS_DIR.iterdir()) if p.is_dir() and not p.name.startswith("_")])


def _jsonable(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)


def run_model(spec: ModelSpec, tasks, limit, force: bool, gpu_mem: float, batch_size: str,
              variant: "Variant | None" = None) -> list[dict]:
    from lm_eval import simple_evaluate

    variant = variant or Variant()
    if variant.chat and spec.kind != "instruct":
        raise ValueError(f"{spec.id}: base 모델에는 chat 변형을 쓰지 않는다")
    nshot_of = lambda t: t.num_fewshot if variant.shots is None else variant.shots  # noqa: E731
    todo = [t for t in tasks if force or not _out_paths(spec.id, t.task, nshot_of(t), limit, variant)[0].exists()]
    if not todo:
        print(f"[{spec.id}] 모든 태스크 완료됨 — 건너뜀")
        return []
    t0 = time.time()
    lm = build_lm(spec, gpu_mem, batch_size, variant)
    tm = task_manager()
    load_s = time.time() - t0
    meta_common = {
        "model": {k: _jsonable(v) for k, v in spec.__dict__.items()},
        "versions": _versions(),
        "suite_git": _git_commit(),
        "load_seconds": round(load_s, 1),
        "seed": SEED,
        "apply_chat_template": variant.chat,
        "variant": {"name": variant.name, "doc_start": variant.doc_start, "chat": variant.chat, "shots": variant.shots},
        "backend_info": getattr(lm, "load_info", None),
    }
    summaries = []
    for t in todo:
        nshot = nshot_of(t)
        out_json, out_samples = _out_paths(spec.id, t.task, nshot, limit, variant)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{spec.id}|{variant.name}] {t.task} ({nshot}-shot{', limit=' + str(limit) if limit else ''})")
        ts = time.time()
        res = simple_evaluate(
            model=lm,
            tasks=[t.task],
            num_fewshot=nshot,
            limit=limit,
            log_samples=True,
            bootstrap_iters=1000,
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
            apply_chat_template=variant.chat,
            task_manager=tm,
        )
        wall = time.time() - ts
        samples = res.pop("samples", {}) or {}
        tmp = out_samples.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for sub, rows in samples.items():
                for r in rows:
                    f.write(json.dumps({"subtask": sub, **r}, ensure_ascii=False, default=str) + "\n")
        tmp.replace(out_samples)
        doc = {
            "task": t.task,
            "num_fewshot": nshot,
            "source": t.source,
            "limit": limit,
            "wall_seconds": round(wall, 1),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "results": res.get("results"),
            "groups": res.get("groups"),
            "n-samples": res.get("n-samples"),
            "versions_task": res.get("versions"),
            "configs": res.get("configs"),
            **meta_common,
        }
        tmpj = out_json.with_suffix(".tmp")
        tmpj.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmpj.replace(out_json)  # 원자적 기록 = 완료 표시
        summaries.append({"model": spec.id, "variant": variant.name, "task": t.task, "wall_s": round(wall, 1), "results": res.get("results", {}).get(t.task)})
    del lm
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return summaries


def pin_models(model_ids: list[str], models: dict[str, ModelSpec], path: Path | None = None) -> dict[str, str]:
    """HF API의 현재 main commit SHA로 models.yaml의 `revision: null`을 채운다 (한 줄 정의만, 주석 보존)."""
    import re

    from huggingface_hub import HfApi

    from kobench.standard.registry import CONFIG_DIR

    path = path or CONFIG_DIR / "models.yaml"
    text = path.read_text(encoding="utf-8")
    api, pinned = HfApi(), {}
    for mid in model_ids:
        spec = models[mid]
        if spec.revision:
            print(f"{mid}: 이미 고정됨 {spec.revision}")
            continue
        sha = api.model_info(spec.path).sha
        pat = re.compile(rf"^(\s*{re.escape(mid)}:\s*\{{.*?revision:\s*)null(.*)$", re.M)
        text, n = pat.subn(rf"\g<1>{sha}\g<2>", text)
        if n != 1:
            raise RuntimeError(f"{mid}: 한 줄 정의에서 revision: null을 찾지 못함 (n={n})")
        pinned[mid] = sha
        print(f"{mid}: {spec.path} → {sha}")
    path.write_text(text, encoding="utf-8")
    return pinned


DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.tiktoken", "*.txt", "*.py", "*.jinja"]


def download_models(model_ids: list[str], models: dict[str, ModelSpec]) -> list[dict]:
    """고정된 revision으로 가중치·설정·토크나이저만 내려받는다 (원본 .pth·GGUF 등 제외). 실패해도 다음 모델로 진행."""
    from huggingface_hub import snapshot_download

    report = []
    for mid in model_ids:
        spec = models[mid]
        if spec.is_local:
            report.append({"model": mid, "status": "local"})
            continue
        if not spec.revision:
            report.append({"model": mid, "status": "unpinned"})
            continue
        t = time.time()
        try:
            p = Path(snapshot_download(spec.path, revision=spec.revision, allow_patterns=DOWNLOAD_PATTERNS))
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            report.append({"model": mid, "status": "ok", "gb": round(size / 1e9, 2), "seconds": round(time.time() - t)})
        except Exception as e:  # gated 미승인 등
            report.append({"model": mid, "status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--suite", default="ko_mc_v1")
    ap.add_argument("--tasks", nargs="+", help="스위트 중 일부 태스크만")
    ap.add_argument("--limit", type=int, default=None, help="태스크당 문항 수 제한 (스모크용, 결과 파일명에 표시)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--variant", default=None, help="설정 변형, 예: 'docstart=eos,chat=1,shots=0' (기본 A)")
    ap.add_argument("--allow-unpinned", action="store_true", help="revision 미고정 공개 모델 실행 허용")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pin", action="store_true", help="--models의 revision을 HF 현재 commit으로 고정하고 종료")
    ap.add_argument("--download", action="store_true", help="--models를 고정 revision으로 내려받고 종료")
    a = ap.parse_args(argv)

    models, suites = load_models(), load_suites()
    if a.pin or a.download:
        unknown = [m for m in a.models if m not in models]
        if unknown:
            ap.error(f"알 수 없는 모델: {unknown}")
        if a.pin:
            pin_models(a.models, models)
        else:
            download_models(a.models, models)
        return 0
    if a.list:
        for mid, m in models.items():
            pin = m.revision[:10] if m.revision else "UNPINNED"
            print(f"{mid:34s} {m.kind:8s} {m.backend:8s} {m.active_b:>5}/{m.params_b:<5}B {pin}  {m.path}")
        for s, d in suites.items():
            print(f"suite {s}: {[t.task for t in d['tasks']]}")
        return 0
    if a.suite not in suites:
        ap.error(f"알 수 없는 suite: {a.suite}")
    tasks = suites[a.suite]["tasks"]
    if a.tasks:
        unknown = set(a.tasks) - {t.task for t in tasks}
        if unknown:
            ap.error(f"suite {a.suite}에 없는 태스크: {sorted(unknown)}")
        tasks = [t for t in tasks if t.task in a.tasks]
    for mid in a.models:
        if mid not in models:
            ap.error(f"알 수 없는 모델: {mid}")
        m = models[mid]
        if not m.is_local and not m.revision and not a.allow_unpinned:
            ap.error(f"{mid}: revision 미고정 (models.yaml에 commit SHA를 적거나 --allow-unpinned)")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prefetch_datasets(tasks)
    all_summ = []
    for mid in a.models:
        all_summ += run_model(models[mid], tasks, a.limit, a.force, a.gpu_mem, a.batch_size, Variant.parse(a.variant))
    print(json.dumps(all_summ, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    code = main()
    # 결과는 태스크마다 원자적으로 기록됐다. vLLM 엔진 소멸자가 종료 시
    # "terminate called without an active exception"(SIGABRT, 134)을 내므로 정리 단계를 건너뛴다.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
