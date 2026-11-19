"""생성 태스크 채점 (Goal 2).

HRM8K: lm-eval 원본 채점(hrm8k/default/utils.py)은 \\boxed{}가 없으면 응답의 마지막 숫자를 답으로 본다.
그래서 우연한 정답이 생긴다(CLAUDE.md "수학 채점 대체 규칙"). 여기서는 세 값을 함께 기록한다.
  - exact_match_strict : \\boxed{...}가 있고 그 값이 정답과 같을 때만 1 (대표 지표)
  - exact_match        : lm-eval 원본 규칙 그대로 (기존 수치와 비교용)
  - format_fail        : \\boxed가 없어 형식을 지키지 않은 응답 (오답과 분리 집계)
IFEval-Ko: lm-eval IFEval 채점기를 그대로 쓴다(스키마 동일).
"""
from lm_eval.tasks.hrm8k.default import utils as _h
from lm_eval.tasks.ifeval import utils as _if

doc_to_text = _h.doc_to_text
doc_to_text_mmmlu = _h.doc_to_text_mmmlu
doc_to_target = _h.doc_to_target


def hrm8k_process_results(doc, results):
    cand = results[0]
    gold = _h.postprocess(doc["answer"])
    lenient = int(_h.is_equiv(cand, gold))
    has_boxed = "\\boxed" in cand or "\\fbox" in cand
    strict = int(has_boxed and lenient)
    return {"exact_match_strict": strict, "exact_match": lenient, "format_fail": int(not has_boxed)}


ifeval_process_results = _if.process_results
agg_inst_level_acc = _if.agg_inst_level_acc
