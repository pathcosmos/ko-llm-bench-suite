"""LLM judge 채점 (configs/standard/gen.yaml judge). judge 모델을 vLLM에 한 번 올리고, 아직 채점하지 않은 모든
(모델, 변형, 태스크) 요청을 한 번에 넣는다(offline 배치 = 높은 동시성). greedy, thinking off.

python -m kobench.standard.judge run [--models ...] [--variants A card think chosen] [--tasks komt ...]
python -m kobench.standard.judge score                 # judge 결과 → <task>.score.json

태스크별 방식 (원 벤치 방식을 따르되 프롬프트는 이 파일에 고정; 모든 판정 원문을 저장한다)
  komt          : MT-Bench single 채점(1~10, [[n]]). 2턴은 multi-turn 템플릿으로 둘째 답만 평가.
                  math·reasoning·coding은 데이터셋 참조 답을 쓰는 reference 템플릿 (MT-Bench 관례).
                  KoMT-Bench 관례대로 한국어로 답하지 않으면 1점을 주도록 지시를 덧붙였다(질문이 다른 언어를 요구하면 예외).
  logickor      : LogicKor 방식 단일 채점(1~10). 턴마다 따로, 참조 답이 있으면 제공. 한국어가 아니면 0점.
  k2_generation : 단일 채점(1~10): 정확성·한국 문화/지식 이해·유용성·한국어 유창성.
  ko_arena_hard : arena-hard v0.1 쌍대 비교. 기준 = gen.yaml arena_baseline 모델의 같은 변형 A 응답.
                  judge가 먼저 자기 답을 쓰고 비교, [[A>>B]]…[[B>>A]]. 위치를 바꿔 두 번(game 1: 기준이 A, game 2: 기준이 B).
  kosimpleqa    : SimpleQA 채점기 방식 — CORRECT / INCORRECT / NOT_ATTEMPTED.
judge와 같은 계열(gen.yaml same_family_as_judge)의 모델은 결과 표에 자기 선호 편향 가능성을 표시한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kobench.standard.gen import RESULTS, out_dir, read_jsonl
from kobench.standard.gen_data import load_gen_config, load_task
from kobench.standard.gen_score import apply_policy, policies_for
from kobench.standard.registry import load_models

JUDGE_TASKS = ("komt", "logickor", "k2_generation", "ko_arena_hard", "kosimpleqa")
KOREAN_RULE = ("\n\nAdditional rule: the assistant is expected to answer in Korean. If the assistant's answer is not written in "
               "Korean (except where the user explicitly asks for another language), give a rating of 1.")

MT_SINGLE = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the "
    "user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, "
    "creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as "
    "possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this "
    "format: \"[[rating]]\", for example: \"Rating: [[5]]\".{rule}\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n"
    "{answer}\n[The End of Assistant's Answer]")
MT_SINGLE_REF = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the "
    "user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference "
    "answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. "
    "Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response "
    "on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".{rule}\n\n"
    "[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref}\n[The End of Reference Answer]\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]")
MT_MULTI = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question "
    "displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and "
    "level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin "
    "your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must "
    "rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\"."
    "{rule}\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{q1}\n\n### Assistant A:\n{a1}\n\n"
    "### User:\n{q2}\n\n### Assistant A:\n{a2}\n\n<|The End of Assistant A's Conversation with User|>")
MT_MULTI_REF = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. "
    "Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. "
    "You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the "
    "assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing "
    "your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for "
    "example: \"Rating: [[5]]\".{rule}\n\n<|The Start of Reference Answer|>\n\n### User:\n{q1}\n\n### Reference answer:\n{r1}\n\n"
    "### User:\n{q2}\n\n### Reference answer:\n{r2}\n\n<|The End of Reference Answer|>\n\n\n"
    "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{q1}\n\n### Assistant A:\n{a1}\n\n### User:\n{q2}\n\n"
    "### Assistant A:\n{a2}\n\n<|The End of Assistant A's Conversation with User|>")
MT_REF_CATS = {"math", "reasoning", "coding"}

LOGICKOR = (
    "너는 질문에 대한 한국어 언어 모델의 답변을 매우 꼼꼼히 평가할 것이다. 공정한 평가를 위해 아래의 규칙을 준수한다.\n\n"
    "# 기본 규칙\n1. 질문의 요구사항을 충분히 반영하였는지 상세히 분석할 것.\n"
    "2. 답변 과정에서 누락되었거나 포함되지 못하여 아쉬운 부분에 대하여 상세히 분석할 것.\n"
    "3. 답변의 길이가 평가 결과에 영향을 미치지 않도록 할 것.\n4. 참고 답변(Reference)이 제공되면 평가 시 그 정보를 참고할 것.\n\n"
    "# 언어 요구사항\n- 모델은 반드시 한국어로 답변해야 하며, 다른 언어로 답한 경우 점수는 0점이다.\n"
    "- 예외: 질문이 다른 언어로 답할 것을 요구할 때만 그 언어를 허용한다.\n\n"
    "# 평가 출력 방식\n주어진 답변을 분석한 뒤 짧게 평가 이유를 쓰고, 마지막 줄에 점수를 0~10 사이 정수로 '[[점수]]' 형식으로 쓴다. "
    "예: [[7]]\n\n{context}# 질문\n{question}\n\n{reference}# 답변\n{answer}")
K2GEN = (
    "다음은 한국어 사용자의 지시와 AI 어시스턴트의 답변이다. 답변을 공정하게 평가하라.\n"
    "평가 기준: (1) 사실·추론의 정확성, (2) 한국 문화·역사·사회에 대한 이해와 맥락 적절성, (3) 지시에 대한 충실성과 유용성, "
    "(4) 자연스럽고 유창한 한국어. 답변 길이 자체로 가산하지 않는다. 한국어로 답하지 않았으면 1점.\n"
    "짧게 평가 이유를 쓰고, 마지막에 1~10 사이 정수 점수를 '[[점수]]' 형식으로 쓴다. 예: [[6]]\n\n"
    "[지시]\n{question}\n\n[답변 시작]\n{answer}\n[답변 끝]")
ARENA_SYS = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt "
    "displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's "
    "answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before "
    "judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must "
    "identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, "
    "relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user "
    "prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or "
    "more information from the user than providing an answer based on assumptions. Relevant means all parts of the response "
    "closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or "
    "excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing "
    "important information in the assistants' answers that would be beneficial to include when responding to the user prompt."
    "\n\nThe user prompt is in Korean; an answer that is not in Korean (unless requested) is less helpful.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: "
    "[[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\n"
    "Example output: \"My final verdict is tie: [[A=B]]\".")
ARENA_USER = ("<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{a}\n<|The End of Assistant A's Answer|>\n\n"
              "<|The Start of Assistant B's Answer|>\n{b}\n<|The End of Assistant B's Answer|>")
SIMPLEQA = (
    "Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either "
    "[\"CORRECT\", \"INCORRECT\", \"NOT_ATTEMPTED\"].\n\n"
    "Rules:\n- CORRECT: the predicted answer fully contains the important information of the gold target and does not contain any "
    "information that contradicts it. Hedging is fine as long as the gold target is fully included and nothing contradicts it. "
    "Minor spelling differences, different but equivalent transliterations (e.g. Korean vs. English names), and omitted "
    "words that are clearly implied by the question are allowed. For numbers, the prediction must be correct to the last "
    "significant figure in the gold target.\n- INCORRECT: any factual statement in the predicted answer contradicts the gold "
    "target, even if hedged (e.g. \"I think it is X but it may be Y\" where X is wrong). Listing several candidate answers counts "
    "as INCORRECT unless all are consistent with the gold target.\n- NOT_ATTEMPTED: the important information of the gold target "
    "is not included and nothing contradicts it (e.g. \"I don't know\", asking for more context, or answering a different "
    "question without giving the target).\n\nGrade only the final answer; do not penalize style. Do not try to answer the "
    "question yourself and do not apologize.\n\nQuestion: {question}\nGold target: {gold}\nPredicted answer: {answer}\n\n"
    "Grade the predicted answer as one of:\nA: CORRECT\nB: INCORRECT\nC: NOT_ATTEMPTED\n\nJust return the letters \"A\", \"B\", or "
    "\"C\", with no text around it.")


def unk_policy_for(model_id: str) -> str:
    """judge에 넘길 텍스트 정책: korean_sp 사용자 모델은 생성 dev에서 고른 정책(best_settings.yaml gen 항목), 없으면 unk_nl."""
    pols = policies_for(model_id)
    if pols == ["raw"]:
        return "raw"
    try:
        import yaml

        from kobench.standard.registry import CONFIG_DIR

        b = yaml.safe_load((CONFIG_DIR / "best_settings.yaml").read_text(encoding="utf-8")) or {}
        return b.get("gen", {}).get(model_id, {}).get("unk_policy", "unk_nl")
    except FileNotFoundError:
        return "unk_nl"


def build_requests(task: str, model_id: str, variant: str, rows: dict, items: list[dict], baseline_rows: dict | None) -> list[dict]:
    """요청: {key, messages, kind}. key는 (모델, 변형, 태스크, 문항, 턴/게임) 단위로 유일."""
    pol = unk_policy_for(model_id)
    reqs = []
    txt = lambda s: apply_policy(s or "", pol)  # noqa: E731
    for it in items:
        r = rows.get(it["id"])
        if r is None:
            continue
        outs = [txt(o) for o in r["outputs"]]
        base_key = f"{model_id}|{variant}|{task}|{it['id']}"
        if task == "komt":
            cat, ref = it["meta"]["category"], it["meta"]["reference"]
            use_ref = cat in MT_REF_CATS and ref
            q1, q2 = it["turns"][0], it["turns"][1]
            p1 = (MT_SINGLE_REF.format(rule=KOREAN_RULE, question=q1, ref=ref[0], answer=outs[0]) if use_ref
                  else MT_SINGLE.format(rule=KOREAN_RULE, question=q1, answer=outs[0]))
            reqs.append({"key": base_key + "|t1", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                               {"role": "user", "content": p1}]})
            if len(outs) > 1:
                p2 = (MT_MULTI_REF.format(rule=KOREAN_RULE, q1=q1, q2=q2, r1=ref[0], r2=ref[1] if len(ref) > 1 else "",
                                          a1=outs[0], a2=outs[1]) if use_ref
                      else MT_MULTI.format(rule=KOREAN_RULE, q1=q1, q2=q2, a1=outs[0], a2=outs[1]))
                reqs.append({"key": base_key + "|t2", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                   {"role": "user", "content": p2}]})
        elif task == "logickor":
            refs = it["meta"].get("references") or []
            for t, (q, a) in enumerate(zip(it["turns"], outs)):
                ctx = ""
                if t > 0:
                    ctx = "# 이전 대화\n" + "".join(f"사용자: {pq}\n어시스턴트: {pa}\n\n" for pq, pa in zip(it["turns"][:t], outs[:t]))
                ref = refs[t] if t < len(refs) and refs[t] else None
                p = LOGICKOR.format(context=ctx, question=q, answer=a,
                                    reference=f"# 참고 답변(Reference)\n{ref}\n\n" if ref else "")
                reqs.append({"key": f"{base_key}|t{t + 1}", "messages": [{"role": "user", "content": p}]})
        elif task == "k2_generation":
            reqs.append({"key": base_key + "|t1", "messages": [{"role": "user", "content": K2GEN.format(question=it["turns"][0], answer=outs[0])}]})
        elif task == "ko_arena_hard":
            b = baseline_rows.get(it["id"]) if baseline_rows else None
            if b is None:
                continue
            bo = b["outputs"][0]
            for g, (a_, b_) in enumerate([(bo, outs[0]), (outs[0], bo)], start=1):
                reqs.append({"key": f"{base_key}|g{g}", "messages": [
                    {"role": "system", "content": ARENA_SYS},
                    {"role": "user", "content": ARENA_USER.format(question=it["turns"][0], a=a_, b=b_)}]})
        elif task == "kosimpleqa":
            reqs.append({"key": base_key + "|t1", "messages": [{"role": "user", "content": SIMPLEQA.format(
                question=it["turns"][0], gold=it["gold"], answer=outs[0])}]})
    return reqs


def parse(task: str, text: str):
    ans = text.split("</think>")[-1]
    if task in ("komt", "logickor", "k2_generation"):
        m = re.findall(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", ans) or re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*\]", ans)
        if not m:
            return None
        v = float(m[-1])
        lo, hi = (0, 10) if task == "logickor" else (1, 10)
        return v if lo <= v <= hi else None
    if task == "ko_arena_hard":
        m = re.findall(r"\[\[([AB]>>[AB]|[AB]>[AB]|A=B|B=A)\]\]", ans)
        return m[-1] if m else None
    if task == "kosimpleqa":
        m = re.search(r"\b([ABC])\b", ans.strip())
        return {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[m.group(1)] if m else None
    raise ValueError(task)


def _stem(task: str, limit: int | None) -> str:
    return task + (f"__limit{limit}" if limit else "")


def judge_path(model_id: str, variant: str, task: str, limit: int | None = None) -> Path:
    return out_dir(model_id, variant) / f"{_stem(task, limit)}.judge.jsonl"


def variants_for(model_id: str, variants: list[str]) -> list[str]:
    """'chosen'을 그 모델의 생성 dev 선택 설정(규칙 v3, best_settings.yaml)으로 바꾼다. 선택 기록이 없는 모델은 건너뛴다.
    2026-11-21 이전 기본값(A card rep think)은 v2 시절 목록이라 v3 선택 설정(rp11, fp03+sys 등)이 judge에서 빠졌다."""
    out = []
    for v in variants:
        if v == "chosen":
            from kobench.standard.gen_report import chosen

            try:
                v = chosen(model_id)[0]
            except Exception:  # noqa: BLE001
                continue
        if v not in out:
            out.append(v)
    return out


def collect(models: list[str], variants: list[str], tasks: list[str], limit: int | None = None) -> list[dict]:
    cfg = load_gen_config()
    base_id = cfg["judge"]["arena_baseline"]
    baseline_rows = {r["id"]: r for r in read_jsonl(out_dir(base_id, "A") / f"{_stem('ko_arena_hard', limit)}.jsonl")}
    allreq = []
    for mid in models:
        for v in variants_for(mid, variants):
            for t in tasks:
                p = out_dir(mid, v) / f"{_stem(t, limit)}.jsonl"
                rows = {r["id"]: r for r in read_jsonl(p)}
                items = load_task(t)[:limit] if limit else load_task(t)
                if len(rows) < len(items):
                    if rows:
                        print(f"[judge] {mid}|{v}|{t}: 생성 미완료 ({len(rows)}/{len(items)}) — 건너뜀")
                    continue
                done = {r["key"] for r in read_jsonl(judge_path(mid, v, t, limit))}
                reqs = build_requests(t, mid, v, rows, items, baseline_rows if t == "ko_arena_hard" else None)
                if t == "ko_arena_hard" and mid == base_id and v == "A":
                    reqs = []      # 기준 모델 자신과의 비교는 하지 않는다 (승률 0.5로 정의)
                for r in reqs:
                    if r["key"] not in done:
                        r.update({"model": mid, "variant": v, "task": t, "limit": limit})
                        allreq.append(r)
    return allreq


def cmd_run(a) -> int:
    from vllm import LLM, SamplingParams

    from kobench.standard import run as R

    cfg = load_gen_config()
    jspec = load_models()[cfg["judge"]["model"]]
    models = a.models or sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith("_"))
    reqs = collect(models, a.variants, a.tasks, a.limit)
    print(f"[judge] 요청 {len(reqs)}개", flush=True)
    if not reqs:
        return 0
    t0 = time.time()
    llm = LLM(model=jspec.path, revision=jspec.revision, dtype="bfloat16", max_model_len=cfg["judge"]["max_model_len"],
              gpu_memory_utilization=a.gpu_mem, seed=R.SEED, trust_remote_code=jspec.trust_remote_code)
    tok = llm.get_tokenizer()
    load_s = time.time() - t0
    th = cfg["thinking"].get(jspec.id, {}).get("think_off", {})
    limit = cfg["judge"]["max_model_len"] - cfg["judge"]["max_new"]
    prompts, keep, too_long = [], [], []
    for r in reqs:
        ids = tok.encode(tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False, **th),
                         add_special_tokens=False)
        if len(ids) > limit:
            too_long.append(r)
            continue
        prompts.append(ids)
        keep.append(r)
    from vllm import TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=cfg["judge"]["max_new"], seed=R.SEED)
    t1 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=True)
    gen_s = time.time() - t1
    now = datetime.now(timezone.utc).isoformat()
    by_file: dict[Path, list[dict]] = {}
    for r, o in zip(keep, outs):
        text = o.outputs[0].text
        rec = {"key": r["key"], "verdict": parse(r["task"], text), "judge_output": text, "judge_prompt": r["messages"],
               "judge": jspec.id, "judge_revision": jspec.revision, "at": now}
        by_file.setdefault(judge_path(r["model"], r["variant"], r["task"], r["limit"]), []).append(rec)
    for r in too_long:
        rec = {"key": r["key"], "verdict": None, "judge_output": None, "error": "judge_prompt_too_long", "judge": jspec.id, "at": now}
        by_file.setdefault(judge_path(r["model"], r["variant"], r["task"], r["limit"]), []).append(rec)
    for p, recs in by_file.items():
        with open(p, "a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {"judge": jspec.id, "revision": jspec.revision, "requests": len(reqs), "too_long": len(too_long),
            "load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "at": now, "thinking": th, "temperature": 0.0}
    with open(RESULTS / "_judge_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


ARENA_SCORE = {"B>>A": 1.0, "B>A": 1.0, "A=B": 0.5, "B=A": 0.5, "A>B": 0.0, "A>>B": 0.0}   # game 1(기준=A) 관점: 평가 모델 = B
ARENA_WEIGHT = {"A>>B": 3, "B>>A": 3}                                                      # arena-hard: 강한 판정은 3배


def score_judge(model_id: str, variant: str, task: str, limit: int | None = None) -> dict | None:
    cfg = load_gen_config()
    recs = {r["key"]: r for r in read_jsonl(judge_path(model_id, variant, task, limit))}
    if not recs:
        return None
    items = load_task(task)[:limit] if limit else load_task(task)
    rows = {r["id"]: r for r in read_jsonl(out_dir(model_id, variant) / f"{_stem(task, limit)}.jsonl")}
    per = []
    for it in items:
        k = f"{model_id}|{variant}|{task}|{it['id']}"
        if task in ("komt", "logickor"):
            ts = [recs.get(f"{k}|t{t + 1}") for t in range(len(it["turns"]))]
            vals = [x["verdict"] if x else None for x in ts]
            ok = [v for v in vals if v is not None]
            per.append({"id": it["id"], "turns": vals, "score": sum(ok) / len(ok) if ok else None,
                        "category": it["meta"]["category"], "parse_fail": sum(v is None for v in vals)})
        elif task == "k2_generation":
            v = recs.get(k + "|t1", {}).get("verdict")
            per.append({"id": it["id"], "score": v, "parse_fail": int(v is None), "ability": it["meta"]["ability"]})
        elif task == "kosimpleqa":
            v = recs.get(k + "|t1", {}).get("verdict")
            per.append({"id": it["id"], "grade": v, "correct": int(v == "CORRECT"), "not_attempted": int(v == "NOT_ATTEMPTED"),
                        "parse_fail": int(v is None)})
        elif task == "ko_arena_hard":
            g1, g2 = recs.get(k + "|g1", {}).get("verdict"), recs.get(k + "|g2", {}).get("verdict")
            sc, w = [], []
            if g1:
                sc.append(ARENA_SCORE[g1]); w.append(ARENA_WEIGHT.get(g1, 1))
            if g2:            # game 2는 평가 모델이 A → 관점을 뒤집는다
                flip = g2.replace("A", "x").replace("B", "A").replace("x", "B")
                sc.append(ARENA_SCORE[flip]); w.append(ARENA_WEIGHT.get(g2, 1))
            per.append({"id": it["id"], "g1": g1, "g2": g2, "score": sum(s * x for s, x in zip(sc, w)) / sum(w) if w else None,
                        "parse_fail": int(g1 is None) + int(g2 is None)})
    vals = [p for p in per if p.get("score") is not None or task == "kosimpleqa"]
    m: dict = {"n": len(per), "parse_fail": sum(p["parse_fail"] for p in per)}
    if task == "kosimpleqa":
        c = sum(p["correct"] for p in per) / len(per)
        na = sum(p["not_attempted"] for p in per) / len(per)
        att = 1 - na
        cga = c / att if att > 0 else 0.0
        m.update({"correct": c, "not_attempted": na, "incorrect": 1 - c - na, "correct_given_attempted": cga,
                  "f_score": 2 * c * cga / (c + cga) if (c + cga) > 0 else 0.0})
    else:
        s = [p["score"] for p in vals]
        m["score" if task != "ko_arena_hard" else "winrate"] = sum(s) / len(s) if s else None
        if task in ("komt", "logickor"):
            m["by_category"] = {c: (lambda xs: sum(xs) / len(xs) if xs else None)([p["score"] for p in vals if p["category"] == c])
                                for c in sorted({p["category"] for p in per})}
            for t in (0, 1):
                xs = [p["turns"][t] for p in per if len(p["turns"]) > t and p["turns"][t] is not None]
                m[f"turn{t + 1}"] = sum(xs) / len(xs) if xs else None
        if task == "ko_arena_hard":
            m["baseline"] = cfg["judge"]["arena_baseline"]
    spec = load_models()[model_id]
    m["same_family_as_judge"] = spec.family in cfg["judge"]["same_family_as_judge"]
    m["truncated"] = sum(1 for r in rows.values() if any(f in ("length", "prompt_too_long") for f in r["finish"])) / max(1, len(rows))
    out = {"model": model_id, "variant": variant, "task": task, "judge": cfg["judge"]["model"],
           "unk_policy": unk_policy_for(model_id), "metrics": {"judge": m}, "items": {"judge": per}}
    (out_dir(model_id, variant) / f"{_stem(task, limit)}.score.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[judge-score] {model_id}|{variant}|{task}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)))
    return out


def cmd_score(a) -> int:
    models = a.models or sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for mid in models:
        for v in variants_for(mid, a.variants):
            for t in a.tasks:
                if judge_path(mid, v, t, a.limit).exists():
                    score_judge(mid, v, t, a.limit)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+")
        p.add_argument("--variants", nargs="+", default=["A", "card", "rep", "think", "chosen"])
        p.add_argument("--tasks", nargs="+", default=list(JUDGE_TASKS))
        p.add_argument("--gpu-mem", type=float, default=0.88)
        p.add_argument("--limit", type=int, help="스모크: 생성 __limitN 결과만")
    a = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
