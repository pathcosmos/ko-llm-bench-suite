"""생성 벤치 문항 로더 (configs/standard/gen.yaml).

각 태스크는 문항 리스트를 돌려준다. 문항은 다음 키를 가진다.
  id      : 문항 식별자 (문자열, 결과 jsonl 재개 키)
  turns   : user 발화 리스트 (instruct, 1~2턴)
  raw     : base 모델 이어쓰기 프롬프트 (코드 태스크만)
  stop    : base 이어쓰기 정지 문자열 (코드 태스크만)
  gold    : 규칙 채점용 정답 (태스크마다 형식이 다름)
  meta    : 채점·집계용 부가 정보

선택지 순서: 정답 위치가 구조적으로 치우친 데이터(KoBBQ: '알 수 없음'이 항상 3번째, ko_truthful mc1: 정답이 항상 첫째,
K2-Eval knowledge: 정답 B가 46%)는 문항 id 해시로 결정적으로 회전한다. KorMedMCQA는 원 시험 순서를 유지한다.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

from kobench.standard.registry import CONFIG_DIR

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MC_INSTRUCTION = "답변의 마지막 줄에 '정답: {letters_hint}' 형식으로 알파벳 하나만 쓰세요."


@lru_cache(maxsize=1)
def load_gen_config(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "gen.yaml").read_text(encoding="utf-8"))


def dataset_dir(name: str) -> Path:
    from huggingface_hub import snapshot_download

    d = load_gen_config()["datasets"][name]
    return Path(snapshot_download(d["repo"], repo_type="dataset", revision=d["revision"]))


def _rot(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


def rotate(options: list[str], answer: int, key: str) -> tuple[list[str], int]:
    """결정적 회전: 선택지 순환 이동 (상대 순서 보존). 정답 위치가 문항마다 고르게 퍼진다."""
    k = _rot(key, len(options))
    opts = options[k:] + options[:k]
    return opts, (answer - k) % len(options)


def mc_prompt(body: str, options: list[str]) -> str:
    letters = LETTERS[: len(options)]
    lines = [body.strip(), ""] + [f"{l}. {o}" for l, o in zip(letters, options)]
    hint = "/".join(letters)
    return "\n".join(lines) + "\n\n" + MC_INSTRUCTION.format(letters_hint=hint)


# ── 태스크별 로더 ───────────────────────────────────────────────────────────

def _hrm8k() -> list[dict]:
    import pandas as pd
    from lm_eval.tasks.hrm8k.default import utils as h

    d = dataset_dir("hrm8k") / "HRM8K"
    out = []
    for f, sub in [("gsm8k_test.csv", "gsm8k"), ("ksm_test.csv", "ksm"), ("math_do_test.csv", "math"),
                   ("mmmlu_test.csv", "mmmlu"), ("omni-math_do_test.csv", "omni_math")]:
        df = pd.read_csv(d / f)
        for i, r in enumerate(df.to_dict("records")):
            text = h.doc_to_text_mmmlu(r) if sub == "mmmlu" else h.doc_to_text(r)
            out.append({"id": f"{sub}/{i}", "turns": [text], "gold": h.postprocess(r["answer"]), "meta": {"subset": sub}})
    return out


def _gsm8k_ko_dev(n: int = 300) -> list[dict]:
    """GSM8K-ko train(번역본)의 결정적 표본 n개. HRM8K의 GSM8K는 test 번역본이라 겹치지 않는다."""
    import pandas as pd
    from lm_eval.tasks.hrm8k.default import utils as h

    df = pd.read_parquet(dataset_dir("gsm8k_ko") / "data" / "train-00000-of-00001.parquet")
    idx = sorted(range(len(df)), key=lambda i: hashlib.sha256(f"gsm8k_ko_dev/{i}".encode()).hexdigest())[:n]
    out = []
    for i in sorted(idx):
        r = df.iloc[i]
        gold = str(r["answer"]).split("####")[-1].strip().replace(",", "")
        out.append({"id": f"train/{i}", "turns": [h.doc_to_text({"question": r["question"]})], "gold": h.postprocess(gold),
                    "meta": {"subset": "gsm8k_ko_train"}})
    return out


def _ifeval_ko() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(dataset_dir("ifeval_ko") / "data" / "train-00000-of-00001.parquet")
    out = []
    for r in df.to_dict("records"):
        # parquet가 정수 인자를 float(예: 3.0)로 바꿔 놓았다 → 원래 정수로 되돌린다 (IFEval 채점기는 int를 기대)
        # 목록 인자(keywords 등)는 numpy 배열로 와서 채점기의 `if v`가 ValueError → list로 되돌린다 (2026-11-20, 전체 실행에서 발견)
        kw = [{k: (int(v) if isinstance(v, float) and v.is_integer() else (list(v) if hasattr(v, "tolist") else v))
               for k, v in (x or {}).items() if v is not None} for x in r["kwargs"]]
        doc = {"key": int(r["key"]), "instruction_id_list": list(r["instruction_id_list"]), "prompt": r["prompt"], "kwargs": kw}
        out.append({"id": str(r["key"]), "turns": [r["prompt"]], "gold": doc, "meta": {}})
    return out


def _kormedmcqa(split: str) -> list[dict]:
    import pandas as pd

    d = dataset_dir("kormedmcqa")
    out = []
    for sub in ("doctor", "nurse", "pharm", "dentist"):
        df = pd.read_parquet(glob.glob(str(d / sub / f"{split}-*.parquet"))[0])
        for i, r in enumerate(df.to_dict("records")):
            opts = [str(r[c]) for c in "ABCDE"]
            out.append({"id": f"{sub}/{i}", "turns": [mc_prompt(r["question"], opts)], "gold": LETTERS[int(r["answer"]) - 1],
                        "meta": {"subset": sub, "n_options": 5}})
    return out


def _kobbq() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(dataset_dir("kobbq") / "data" / "kobbq-evaluation-test.parquet")
    out = []
    for r in df.to_dict("records"):
        choices = ast.literal_eval(r["choices"])
        ans = choices.index(r["answer"])
        opts, a = rotate(choices, ans, r["sample_id"])
        body = f"주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 고르세요.\n\n맥락: {r['context']}\n질문: {r['question']}"
        cond, pol = r["sample_id"].split("-")[-2:]               # amb|dis, bsd|cnt
        out.append({"id": r["sample_id"], "turns": [mc_prompt(body, opts)], "gold": LETTERS[a],
                    "meta": {"condition": cond, "polarity": pol, "category": r["bbq_category"],
                             "biased": LETTERS[opts.index(r["biased_answer"])] if r["biased_answer"] in opts else None,
                             "unknown": LETTERS[opts.index(choices[2])],      # 원 데이터의 3번째 = '알 수 없음' 류
                             "n_options": len(opts)}})
    return out


def _ko_truthful() -> list[dict]:
    rows = [json.loads(l) for l in open(dataset_dir("ko_truthful") / "ko_truthful_qa-multiple_choice-validation.jsonl", encoding="utf-8")]
    out = []
    for i, r in enumerate(rows):
        ch, lab = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        opts, a = rotate(list(ch), lab.index(1), f"ko_truthful/{i}")
        out.append({"id": str(i), "turns": [mc_prompt(f"질문: {r['question']}", opts)], "gold": LETTERS[a],
                    "meta": {"n_options": len(opts)}})
    return out


def _k2_knowledge() -> list[dict]:
    import pandas as pd

    df = pd.read_csv(dataset_dir("k2_eval") / "data" / "k2-eval-knowledge.csv")
    out = []
    for i, r in enumerate(df.to_dict("records")):
        opts, a = rotate([str(r[c]) for c in "abcd"], int(r["gold_answer"]), f"k2_knowledge/{i}")
        out.append({"id": str(i), "turns": [mc_prompt(r["question"], opts)], "gold": LETTERS[a], "meta": {"n_options": 4}})
    return out


def _komt() -> list[dict]:
    """KoMT-Bench 80문항 × 2턴. parquet에 참조 답(39문항)이 있다 (raw jsonl과 turns 동일 확인).
    MT-Bench 관례대로 참조 답은 math·reasoning·coding 범주 채점에만 쓴다."""
    import pandas as pd

    df = pd.read_parquet(glob.glob(str(dataset_dir("komt") / "data" / "*.parquet"))[0])
    out = []
    for r in df.to_dict("records"):
        ref = r.get("reference")
        ref = [str(x) for x in ref] if ref is not None and len(ref) else None
        out.append({"id": str(r["question_id"]), "turns": [str(x) for x in r["turns"]], "gold": None,
                    "meta": {"category": r["category"], "reference": ref}})
    return out


def _logickor() -> list[dict]:
    rows = [json.loads(l) for l in open(dataset_dir("logickor") / "questions.jsonl", encoding="utf-8")]
    return [{"id": str(r["id"]), "turns": list(r["questions"]), "gold": None,
             "meta": {"category": r["category"], "references": r.get("references")}} for r in rows]


def _k2_generation() -> list[dict]:
    import pandas as pd

    df = pd.read_csv(dataset_dir("k2_eval") / "data" / "k2-eval-generation.csv")
    return [{"id": str(i), "turns": [r["instruction"]], "gold": None, "meta": {"subject": r["subject"], "ability": r["ability"]}}
            for i, r in enumerate(df.to_dict("records"))]


def _ko_arena() -> list[dict]:
    rows = [json.loads(l) for l in open(dataset_dir("ko_arena") / "ko_question.jsonl", encoding="utf-8")]
    return [{"id": r["question_id"], "turns": [r["turns"][0]["ko_content"]], "gold": None,
             "meta": {"cluster": r.get("ko_cluster")}} for r in rows]


def _kosimpleqa() -> list[dict]:
    rows = [json.loads(l) for l in open(dataset_dir("kosimpleqa") / "kosimpleqa.jsonl", encoding="utf-8")]
    return [{"id": str(i), "turns": [r["problem"]], "gold": r["answer"], "meta": {"topic": r["topic"]}} for i, r in enumerate(rows)]


CODE_INSTRUCT = "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```\n{prompt}\n```"
CODE_BASE_STOP = ["\nif __name__", "\ndef main(", "\nprint(", '\n"""', "\nassert", "\n```"]   # evalplus base 모델 정지 문자열
# base 이어쓰기 프롬프트는 끝 공백·개행을 지운다(bigcode-evaluation-harness 관례). 끝의 단독 개행 토큰은 BBPE 토크나이저에서
# 학습에 드문 경계라(본문은 보통 '\n    ' 한 토큰) JANSORI가 EOS를 0.49 확률로 고른다 (2026-11-20 확인)


def _humanevalplus() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(glob.glob(str(dataset_dir("humanevalplus") / "data" / "*.parquet"))[0])
    out = []
    for r in df.to_dict("records"):
        out.append({"id": r["task_id"], "turns": [CODE_INSTRUCT.format(prompt=r["prompt"].strip())], "raw": r["prompt"].rstrip(),
                    "stop": CODE_BASE_STOP, "gold": {"test": r["test"], "entry_point": r["entry_point"], "prompt": r["prompt"],
                                                     "raw": r["prompt"].rstrip()},
                    "meta": {"bench": "humaneval"}})
    return out


def _mbppplus() -> list[dict]:
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(glob.glob(str(dataset_dir("mbppplus") / "data" / "*.parquet"))[0])
    out = []
    for r in df.to_dict("records"):
        tl = list(r["test_list"]) if isinstance(r["test_list"], (list, np.ndarray)) else ast.literal_eval(r["test_list"])
        p = f'"""\n{r["prompt"].strip()}\n{tl[0].strip()}\n"""\n'          # evalplus MBPP+ 프롬프트
        imports = list(r["test_imports"]) if isinstance(r["test_imports"], (list, np.ndarray)) else ast.literal_eval(r["test_imports"] or "[]")
        out.append({"id": f"Mbpp/{r['task_id']}", "turns": [CODE_INSTRUCT.format(prompt=p.strip())], "raw": p.rstrip(), "stop": CODE_BASE_STOP,
                    "gold": {"test": r["test"], "imports": imports, "entry_point": None, "prompt": p, "raw": p.rstrip()},
                    "meta": {"bench": "mbpp"}})
    return out


def _scenario(split: str, history: str):
    from kobench.standard import scenario

    return lambda: scenario.build(split, history)


LOADERS = {
    "scenario_self": _scenario("test", "self"),
    "scenario_oracle": _scenario("test", "oracle"),
    "scenario_dev_self": _scenario("dev", "self"),
    "scenario_dev_oracle": _scenario("dev", "oracle"),
    "hrm8k": _hrm8k,
    "gsm8k_ko_dev": _gsm8k_ko_dev,
    "ifeval_ko": _ifeval_ko,
    "kormedmcqa": lambda: _kormedmcqa("test"),
    "kormedmcqa_dev": lambda: _kormedmcqa("dev"),
    "kobbq": _kobbq,
    "ko_truthful_mc1": _ko_truthful,
    "k2_knowledge": _k2_knowledge,
    "komt": _komt,
    "logickor": _logickor,
    "k2_generation": _k2_generation,
    "ko_arena_hard": _ko_arena,
    "kosimpleqa": _kosimpleqa,
    "humaneval_plus": _humanevalplus,
    "mbpp_plus": _mbppplus,
}


def load_task(name: str) -> list[dict]:
    if name not in LOADERS:
        raise KeyError(f"알 수 없는 생성 태스크: {name}")
    items = LOADERS[name]()
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: 문항 id 중복")
    return items
