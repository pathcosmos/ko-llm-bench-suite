"""통일 비교 프로토콜 U (configs/standard/unified.yaml): chat 템플릿 없이 모든 모델에 같은 원문 프롬프트.

gen 변형 'U'에서 쓴다(gen.run_task가 uni_text/uni_stop/uni_turns로 원문을 만들고 encode_raw로 넣는다).
  - few-shot 이어쓰기: hrm8k(GSM8K-ko train 예시 5개), kosimpleqa(직접 쓴 예시 5개), 코드(base 이어쓰기 원문)
  - URIAL(문맥 내 정렬): ifeval_ko, komt, logickor, k2_generation, ko_arena_hard, scenario_self — 고정 머리말 + 예시 3개
"""

from __future__ import annotations

import re
from functools import lru_cache

import yaml

from kobench.standard.registry import CONFIG_DIR

CODE_TASKS = ("humaneval_plus", "mbpp_plus")
DEV_ALIAS = {"gsm8k_ko_dev": "hrm8k", "scenario_dev_self": "scenario_self"}   # 통일 프롬프트 동작 확인용 dev → 같은 프롬프트 규칙


@lru_cache(maxsize=1)
def load_uni() -> dict:
    return yaml.safe_load((CONFIG_DIR / "unified.yaml").read_text(encoding="utf-8"))


def is_uni(variant: str) -> bool:
    return variant.split("+")[0] == "U"


def is_raw(kind: str, variant: str) -> bool:
    """원문(이어쓰기) 경로를 쓰는가: base 모델 또는 통일 변형 U."""
    return kind == "base" or is_uni(variant)


@lru_cache(maxsize=1)
def hrm8k_prefix() -> str:
    """GSM8K-ko train 예시 5개를 테스트와 같은 lm-eval hrm8k 문구로 감싼 few-shot 머리."""
    import pandas as pd
    from lm_eval.tasks.hrm8k.default import utils as h

    from kobench.standard.gen_data import dataset_dir

    cfg = load_uni()["fewshot"]["hrm8k"]
    df = pd.read_parquet(dataset_dir("gsm8k_ko") / "data" / "train-00000-of-00001.parquet")
    shots = []
    for i in cfg["exemplar_train_index"]:
        r = df.iloc[int(i)]
        body, ans = str(r["answer"]).split("####")
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        ans = ans.strip().replace(",", "")
        shots.append(f"{h.doc_to_text({'question': r['question']})} {body} 최종 답: $\\boxed{{{ans}}}$")
    return cfg["joiner"].join(shots) + cfg["joiner"]


def kosimpleqa_prefix() -> str:
    c = load_uni()["fewshot"]["kosimpleqa"]
    return c["joiner"].join(c["format"].format(q=q, a=a) for q, a in c["exemplars"]) + c["joiner"]


def urial_prefix() -> str:
    u = load_uni()["urial"]
    parts = [u["preamble"]]
    for q, a in u["examples"]:
        parts.append(f"{u['user_tag']}\n{q}\n\n{u['assistant_tag']}\n{a}")
    return "\n\n".join(parts) + "\n\n"


def uni_turns(task: str, item: dict) -> int:
    task = DEV_ALIAS.get(task, task)
    return len(item["turns"]) if task in load_uni()["urial"]["tasks"] else 1


def uni_text(task: str, item: dict, turn: int = 0, prev: list[str] | None = None) -> str:
    """통일 프로토콜 원문 프롬프트. prev = 앞 턴 답(URIAL 다턴)."""
    task = DEV_ALIAS.get(task, task)
    if task in CODE_TASKS:
        return item["raw"]
    if task == "hrm8k":
        return hrm8k_prefix() + item["turns"][0]
    if task == "kosimpleqa":
        c = load_uni()["fewshot"]["kosimpleqa"]
        return kosimpleqa_prefix() + c["format"].format(q=item["turns"][0], a="").rstrip()
    u = load_uni()["urial"]
    if task not in u["tasks"]:
        raise KeyError(f"통일 프로토콜에 없는 태스크: {task}")
    s = urial_prefix()
    for q, a in zip(item["turns"][:turn], prev or []):
        s += f"{u['user_tag']}\n{q}\n\n{u['assistant_tag']}\n{a.strip()}\n\n"
    return s + f"{u['user_tag']}\n{item['turns'][turn]}\n\n{u['assistant_tag']}\n"


def uni_stop(task: str, item: dict) -> list[str]:
    task = DEV_ALIAS.get(task, task)
    if task in CODE_TASKS:
        return list(item["stop"])
    c = load_uni()
    if task in c["fewshot"]:
        return list(c["fewshot"][task]["stop"])
    return list(c["urial"]["stop"])


def uni_max_new(task: str, default: int) -> int:
    task = DEV_ALIAS.get(task, task)
    f = load_uni()["fewshot"]
    return int(f[task]["max_new"]) if task in f else int(f["code"]["max_new"]) if task in CODE_TASKS else default
