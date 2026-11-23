"""통일 프로토콜 U1: 생성으로 풀던 객관식 4종의 선택지 문장 로그우도(cloze) 버전 (configs/standard/unified.yaml mc_loglik).

데이터는 gen 태스크와 같은 고정 revision 스냅샷(kobench.standard.gen_data.dataset_dir)에서 읽는다. 원 선택지 순서, 0-shot.
문서 형식: {"query": 프롬프트, "choices": [선택지 문장], "gold": 정답 index, "n_options": 선택지 수}
"""
import ast
import glob
import json

import datasets
import yaml

from kobench.standard.gen_data import dataset_dir
from kobench.standard.registry import CONFIG_DIR

SUBJECT = {"doctor": "의사", "nurse": "간호사", "pharm": "약사", "dentist": "치과의사"}


def _fmt(task: str) -> str:
    return yaml.safe_load((CONFIG_DIR / "unified.yaml").read_text(encoding="utf-8"))["mc_loglik"][task]["format"]


def _dd(rows):
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def kormedmcqa(**kw):
    import pandas as pd

    d, f = dataset_dir("kormedmcqa"), _fmt("kormedmcqa_cloze")
    rows = []
    for sub in ("doctor", "nurse", "pharm", "dentist"):
        df = pd.read_parquet(glob.glob(str(d / sub / "test-*.parquet"))[0])
        for i, r in enumerate(df.to_dict("records")):
            ch = [str(r[c]).strip() for c in "ABCDE"]
            rows.append({"id": f"{sub}/{i}", "query": f.format(subject=SUBJECT[sub], question=str(r["question"]).strip()),
                         "choices": ch, "gold": int(r["answer"]) - 1, "n_options": 5})
    return _dd(rows)


def kobbq(**kw):
    import pandas as pd

    f = _fmt("kobbq_cloze")
    df = pd.read_parquet(dataset_dir("kobbq") / "data" / "kobbq-evaluation-test.parquet")
    rows = []
    for r in df.to_dict("records"):
        ch = [str(c).strip() for c in ast.literal_eval(r["choices"])]
        rows.append({"id": r["sample_id"], "query": f.format(context=r["context"].strip(), question=r["question"].strip()),
                     "choices": ch, "gold": ch.index(str(r["answer"]).strip()), "n_options": len(ch)})
    return _dd(rows)


def ko_truthful(**kw):
    f = _fmt("ko_truthful_cloze")
    src = dataset_dir("ko_truthful") / "ko_truthful_qa-multiple_choice-validation.jsonl"
    rows = []
    for i, line in enumerate(open(src, encoding="utf-8")):
        r = json.loads(line)
        ch, lab = [str(c).strip() for c in r["mc1_targets"]["choices"]], r["mc1_targets"]["labels"]
        rows.append({"id": str(i), "query": f.format(question=r["question"].strip()), "choices": ch, "gold": lab.index(1),
                     "n_options": len(ch)})
    return _dd(rows)


def k2_knowledge(**kw):
    import pandas as pd

    f = _fmt("k2_knowledge_cloze")
    df = pd.read_csv(dataset_dir("k2_eval") / "data" / "k2-eval-knowledge.csv")
    rows = [{"id": str(i), "query": f.format(question=str(r["question"]).strip()), "choices": [str(r[c]).strip() for c in "abcd"],
             "gold": int(r["gold_answer"]), "n_options": 4} for i, r in enumerate(df.to_dict("records"))]
    return _dd(rows)
