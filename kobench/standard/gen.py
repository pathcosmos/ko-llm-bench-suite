"""생성 벤치 실행기 (configs/standard/gen.yaml). 모델을 한 번 올리고 스위트의 생성 태스크를 모두 돌린다.

python -m kobench.standard.gen run --models frankenstallm-bf16 --suite gen_v1 --variant A
python -m kobench.standard.gen run --models qwen3-4b-2507 --suite gen_dev_v1 --variant card
python -m kobench.standard.gen run --models smollm3-3b --suite gen_v1 --variant think
python -m kobench.standard.gen score --models ... --variant A          # 규칙·코드 채점 (GPU 불필요)

변형: A = greedy·thinking off / card = 모델 권장 샘플링 / rep = greedy + 카드 권장 repetition_penalty /
      think = thinking on (card 샘플링, 문맥 16384)
출력: results/gen/<model>/<variant>/<task>.jsonl (문항별, 재개 단위 = chunk), <task>.meta.json (설정·버전·시간)
     문항 레코드: id, outputs[턴], reasoning[턴], gen_tokens[턴], prompt_tokens[턴], finish[턴]
사용자 모델은 formats.yaml 학습 형식(encode_prompt 토큰 ID)을 쓰고, korean_sp 모델의 <unk>(3)는 '<unk>' 문자열로 남긴다
(채점 시 정책 unk_policy로 처리: 개행 복원 또는 삭제 — dev로 선택).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kobench.standard.gen_data import load_gen_config, load_task
from kobench.standard.registry import ModelSpec, load_models

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "gen"
VARIANTS = None   # 자유 형식: 기본[+수정자] (settings_for 참고)
UNK = "<unk>"


def is_korean_sp(model_id: str) -> bool:
    from kobench.standard import formats
    from kobench.standard.registry import load_models

    return load_models()[model_id].owner == "self" and formats.model_format(model_id)["tokenizer"] == "korean_sp_unigram64k"


def unk_stop_variants(stop: list[str] | None) -> list[str] | None:
    """korean_sp 모델의 정지 문자열: 개행은 <unk>로 생성되고, 줄머리 단어가 '▁'로 시작하면 '<unk> X'로 디코딩된다
    → '\n'이 든 정지 문자열마다 '<unk>X'와 '<unk> X' 형태를 더한다. 2026-11-22 수정: 이전에는 '\n' 정지 문자열이
    korean_sp 출력에 한 번도 맞지 않았다(통일 프로토콜 dev 확인에서 EVAFRILL pretrain 출력 300/300이 길이 한도까지 생성)."""
    if not stop:
        return stop
    out = list(stop)
    for s in stop:
        if "\n" in s:
            for v in (s.replace("\n", UNK), s.replace("\n", UNK + " ")):
                if v not in out:
                    out.append(v)
    return out


def out_dir(model_id: str, variant: str) -> Path:
    return RESULTS / model_id / variant


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:      # 중단 중 잘린 마지막 줄은 버린다 (재개 시 다시 생성)
                    pass
    return rows


# 모델별 thinking 구분 표시를 <think>/</think>로 통일 (Gemma 4: '<|channel>thought\n … <channel|>', 2026-11-22 확인 — 이 표시는
# special 토큰이라 예전 디코더가 지워 버려 생각과 답의 경계가 사라졌다)
THINK_MARKERS = (("<|channel>thought\n", "<think>"), ("<|channel>thought", "<think>"), ("<|channel>", "<think>"), ("<channel|>", "</think>"))


def normalize_think(text: str) -> str:
    for a, b in THINK_MARKERS:
        text = text.replace(a, b)
    return text


def parse_think(text: str, finish: str, prefilled: bool) -> tuple[str, str]:
    """thinking on 출력 → (답, 생각). '</think>'가 있으면 그 뒤가 답.
    없으면: 템플릿이 '<think>'를 미리 넣었거나(prefilled) 출력에 '<think>'가 열려 있으면 생각이 잘린 것 → 답 없음.
    템플릿이 미리 넣지 않았고 출력에 표시가 전혀 없이 정상 종료(stop)했으면 생각 없이 바로 답한 것 → 전체가 답
    (2026-11-22: SmolLM3는 2턴째부터 생각 없이 답해 이 경우가 빈 답으로 채점되던 버그)."""
    if "</think>" in text:
        return split_reasoning(text)
    if prefilled or "<think>" in text or finish != "stop":
        return "", text.replace("<think>", "").strip()
    return text.strip(), ""


def split_reasoning(text: str) -> tuple[str, str]:
    """thinking 출력에서 최종 답을 분리한다. '</think>'가 있으면 그 뒤가 답, 없고 '<think>'만 있으면(잘림) 답은 빈 문자열."""
    if "</think>" in text:
        r, a = text.rsplit("</think>", 1)
        return a.strip(), r.replace("<think>", "").strip()
    if "<think>" in text:
        return "", text.replace("<think>", "").strip()
    return text, ""


# ── 샘플링 설정 ─────────────────────────────────────────────────────────────

def _snapshot_dir(spec: ModelSpec) -> Path | None:
    if spec.is_local:
        return Path(spec.path)
    d = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / \
        f"models--{spec.path.replace('/', '--')}" / "snapshots" / str(spec.revision)
    return (d / spec.subfolder) if spec.subfolder else d


def card_sampling(spec: ModelSpec) -> dict | None:
    """모델 권장 샘플링. 사용자 모델은 gen.yaml user_card, 공개 모델은 저장소 generation_config.json. 없으면 None."""
    cfg = load_gen_config()
    if spec.owner == "self":
        key = "evafrill" if spec.backend == "evafrill" else spec.id
        return dict(cfg["user_card"][key]) if key in cfg["user_card"] else None
    p = _snapshot_dir(spec) / "generation_config.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text(encoding="utf-8"))
    if not g.get("do_sample", False) and "temperature" not in g:
        return None
    out = {k: g[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "min_p") if k in g}
    return out or None


KO_SYSTEM = ("당신은 도움이 되는 한국어 어시스턴트입니다. 사용자의 질문에 한국어로 정확하고 간결하게 답하세요. "
             "같은 문장이나 내용을 되풀이하지 말고, 답을 마치면 바로 끝내세요.")



def decode_keep_unk(tok, ids: list[int]) -> str:
    """korean_sp 출력 디코딩: <unk>(3, 학습 개행)은 문자열 '<unk>'로 남기고 나머지 특수 토큰(pad/bos/eos)은 뺀다.
    전체를 한 번에 디코딩한다. 2026-11-20 수정: 이전에는 <unk>마다 끊어 조각별로 디코딩해 Metaspace 디코더가 조각 첫
    '▁'(공백)을 지웠다 → 개행 뒤 줄머리 공백이 하나씩 빠짐(코드 들여쓰기 4→3). 이전 출력은 `gen_score.repair_unk_space`로 근사 복원."""
    special = {i for i, a in tok.get_added_tokens_decoder().items() if a.special} - {3}
    return tok.decode([i for i in ids if i not in special], skip_special_tokens=False)

@dataclass
class GenSettings:
    variant: str
    sampling: dict            # temperature 0 = greedy
    think_kwargs: dict        # chat 템플릿 인자
    max_model_len: int
    system: str | None = None     # +sys: 한국어 시스템 지시 (사용자 모델은 첫 user 앞에 붙임 — 학습 데이터에 system 역할이 없음)
    template: str | None = None   # +orpo: frankenstallm ORPO 학습 형식(orpo_raw_bos, 단일 턴만; 다턴은 기본 형식)


# 변형 = 디코딩 기본[+수정자...]. 기본: A(greedy) card(권장 샘플링) rep(greedy+카드 반복 패널티) rp11/rp12(greedy+반복 패널티 1.1/1.2)
# fp03(greedy+frequency_penalty 0.3, vLLM만) think(thinking on). 수정자: sys(한국어 시스템 지시), orpo(frankenstallm ORPO 형식)
BASES = ("A", "card", "rep", "rp11", "rp12", "fp03", "think", "thinkrd", "U")   # thinkrd = thinking on + README 권장 샘플링(gen.yaml think_readme_sampling)
THINK_BASES = ("think", "thinkrd")


def is_think(variant: str) -> bool:
    return variant.split("+")[0] in THINK_BASES

# U = 통일 프로토콜(configs/standard/unified.yaml): 템플릿 없는 원문, greedy
MODS = ("sys", "orpo")


def settings_for(spec: ModelSpec, variant: str) -> GenSettings | None:
    cfg = load_gen_config()
    th = cfg["thinking"].get(spec.id)
    base, *mods = variant.split("+")
    if base not in BASES or any(m not in MODS for m in mods):
        raise ValueError(variant)
    off = dict(th["think_off"]) if th else {}
    L = cfg["common"]["max_model_len"]
    card = card_sampling(spec)
    if base in ("A", "U"):
        st = GenSettings(variant, {"temperature": 0.0}, off, L)
    elif base == "card":
        if card is None:
            return None
        st = GenSettings(variant, card, off, L)
    elif base == "rep":
        # greedy + 모델 카드 권장 repetition_penalty (결정적이면서 반복 억제). 카드에 반복 패널티가 없으면(또는 1.0) 해당 없음
        rp = (card or {}).get("repetition_penalty")
        if not rp or float(rp) == 1.0:
            return None
        st = GenSettings(variant, {"temperature": 0.0, "repetition_penalty": float(rp)}, off, L)
    elif base in ("rp11", "rp12"):
        st = GenSettings(variant, {"temperature": 0.0, "repetition_penalty": 1.1 if base == "rp11" else 1.2}, off, L)
    elif base == "fp03":
        if spec.backend != "vllm":
            return None                    # EVAFRILL 엔진은 frequency_penalty 미지원
        st = GenSettings(variant, {"temperature": 0.0, "frequency_penalty": 0.3}, off, L)
    elif base == "thinkrd":
        # thinking on + 모델 README의 thinking 권장 샘플링 (저장소에 generation_config.json이 없어 think가 greedy 대체값으로 돈 모델용,
        # 2026-11-23 사용자 결정: 기존 think 결과는 그대로 두고 별도 열로 보고)
        rd = (cfg.get("think_readme_sampling") or {}).get(spec.id)
        if not th or not rd:
            return None
        st = GenSettings(variant, dict(rd["sampling"]), dict(th["think_on"]), cfg["common"]["think_max_model_len"])
    else:                                  # think
        if not th:
            return None
        st = GenSettings(variant, card or {"temperature": 0.0}, dict(th["think_on"]), cfg["common"]["think_max_model_len"])
    if "sys" in mods:
        st.system = KO_SYSTEM
    if "orpo" in mods:
        if spec.id != "frankenstallm-bf16":
            return None
        st.template = "orpo_raw_bos"
    return st


# ── 생성 백엔드 ─────────────────────────────────────────────────────────────

def user_encode(model_id: str, messages: list[dict], template: str | None) -> list[int]:
    """사용자 모델 학습 형식 인코딩. system 메시지는 첫 user 앞에 붙인다(학습 데이터에 system 역할 없음).
    orpo_raw_bos는 단일 턴만 지원 → 다턴(assistant 이력 있음)은 기본 형식."""
    from kobench.standard import formats

    has_sys = any(m["role"] == "system" for m in messages)
    if template and not any(m["role"] == "assistant" for m in messages):
        return formats.encode_prompt(model_id, messages, template=template)
    return formats.encode_prompt(model_id, messages, system="prepend_to_first_user" if has_sys else None)

class VllmGen:
    """vLLM offline. 프롬프트는 토큰 ID로 넣는다 (공개 모델: 공식 chat 템플릿 → encode, 사용자 모델: formats.encode_prompt)."""

    def __init__(self, spec: ModelSpec, max_model_len: int, gpu_mem: float = 0.85):
        from vllm import LLM

        from kobench.standard import run as R

        if spec.extra.get("relax_head_dim_check"):
            R.relax_head_dim_check()
        kw = {}
        self.spec = spec
        self.user = spec.owner == "self"
        if self.user:
            from kobench.standard import formats

            kw["tokenizer"] = str(formats.prepare_tokenizer_dir(spec.id))
        self.llm = LLM(model=spec.path, revision=None if spec.is_local else spec.revision, dtype="bfloat16",
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_mem, seed=R.SEED,
                       trust_remote_code=spec.trust_remote_code, **kw, **spec.extra.get("vllm_args", {}))
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len

    def encode_chat(self, messages: list[dict], think_kwargs: dict, template: str | None = None) -> list[int]:
        if self.user:
            return user_encode(self.spec.id, messages, template)
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **think_kwargs)
        return self.tok.encode(text, add_special_tokens=False)

    def encode_raw(self, text: str) -> list[int]:
        if self.user:
            from kobench.standard import formats

            return formats.encode_text(self.spec.id, text)
        ids = self.tok.encode(text, add_special_tokens=True)       # base 공개 모델: 토크나이저 기본 BOS 규칙
        return ids

    def decode(self, ids: list[int]) -> str:
        if self.user:
            from kobench.standard import formats

            t = formats.training_tokenizer(self.spec.id)
            fmt = formats.model_format(self.spec.id)
            stop = set(fmt["stop_token_ids"])
            ids = [i for i in ids if i not in stop]
            if fmt["tokenizer"] == "korean_sp_unigram64k":
                return decode_keep_unk(t, ids)
            return t.decode(ids, skip_special_tokens=True)
        # 공개 모델: 특수 토큰을 남긴 채 디코딩하고, <think>/</think>만 남긴 뒤 나머지 특수 토큰 문자열을 지운다
        # (일부 모델은 <think>가 special이라 skip_special_tokens=True면 생각과 답의 경계가 사라진다)
        text = self.tok.decode(ids, skip_special_tokens=False)
        for s in self._special_strings():
            text = text.replace(s, "")
        return normalize_think(text)

    def _special_strings(self) -> list[str]:
        if not hasattr(self, "_specials"):
            sp = set(getattr(self.tok, "all_special_tokens", []) or [])
            for t in getattr(self.tok, "added_tokens_decoder", {}).values():
                if getattr(t, "special", False):
                    sp.add(str(t))
            keep = {"<think>", "</think>", "<|channel>", "<channel|>"}          # thinking 구분 표시는 남긴다 (normalize_think)
            self._specials = sorted((s for s in sp if s and s not in keep), key=len, reverse=True)
        return self._specials

    def generate(self, prompts: list[list[int]], max_new: list[int], sampling: dict, stop: list[str] | None,
                 seed: int) -> list[dict]:
        from vllm import SamplingParams, TokensPrompt

        stop_ids = None
        ksp = {}
        if self.user:
            from kobench.standard import formats

            stop_ids = list(formats.model_format(self.spec.id)["stop_token_ids"])
            if is_korean_sp(self.spec.id):          # <unk>(개행)를 vLLM 정지 검사 문자열에 남겨 '<unk>…' 정지 문자열이 맞게 한다
                stop = unk_stop_variants(stop)
                ksp = {"skip_special_tokens": False, "spaces_between_special_tokens": False}
        params = []
        for m in max_new:
            sp = dict(sampling)
            params.append(SamplingParams(max_tokens=m, seed=seed, stop=stop or None, stop_token_ids=stop_ids,
                                         include_stop_str_in_output=False, **ksp, **sp))
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=True)
        res = []
        for o in outs:
            c = o.outputs[0]
            # 토큰 ID를 다시 디코딩하면 정지 문자열을 완성한 토큰까지 남는다 → 같은 규칙으로 자른다 (2026-11-21 수정)
            from kobench.standard.gen_score import cut_at_stop

            full = self.decode(list(c.token_ids))
            text = cut_at_stop(full, stop)
            res.append({"text": text, "n": len(c.token_ids), "finish": "stop" if text != full else c.finish_reason})
        return res


class EvafrillGen:
    """EVAFRILL 증분 디코딩 엔진 (배치 불변, 배치 32). 프롬프트는 formats.encode_prompt/encode_text."""

    def __init__(self, spec: ModelSpec, max_model_len: int):
        import torch

        from kobench.standard.evafrill_lm import EvafrillLM

        self.spec = spec
        # 생성 배치 64 (배치 불변 모드: decode를 항상 64행으로 실행 → 결과는 배치 구성과 무관하게 같다). 처리량 때문에 lm-eval 기본 32보다 키움
        kw = {"gen_batch_size": 64, **spec.extra.get("evafrill_args", {})}
        self.lm = EvafrillLM(pretrained=spec.path, revision=spec.revision, subfolder=spec.subfolder,
                             max_length=max_model_len, **kw)
        self.max_model_len = max_model_len
        self.torch = torch

    def encode_chat(self, messages, think_kwargs, template=None):
        return user_encode(self.spec.id, messages, template)

    def encode_raw(self, text):
        from kobench.standard import formats

        return formats.encode_text(self.spec.id, text)

    def decode(self, ids):
        from kobench.standard import formats

        tok = formats.training_tokenizer(self.spec.id)
        return decode_keep_unk(tok, [i for i in ids if i != 2])

    def generate(self, prompts, max_new, sampling, stop, seed):
        from kobench.standard import formats

        from kobench.standard.gen_score import cut_at_stop

        tok = formats.training_tokenizer(self.spec.id)
        stop = unk_stop_variants(stop)          # korean_sp: '\n' 정지 문자열을 <unk> 형태로도 (unk_stop_variants)
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        res: list[dict | None] = [None] * len(prompts)
        bs = self.lm.gen_batch_size
        temp = float(sampling.get("temperature", 0.0))
        for s in range(0, len(order), bs):
            idx = order[s: s + bs]
            # 배치는 하나의 생성 길이를 쓴다: 각 행의 한도(문맥 4096 − 프롬프트)를 넘지 않게 배치 최대 프롬프트 기준으로 자르고,
            # 행별 결과는 다시 자기 max_new로 자른다 (길이순 정렬이라 같은 배치의 프롬프트 길이는 비슷하다)
            mx = min(max(max_new[i] for i in idx), self.max_model_len - max(len(prompts[i]) for i in idx))
            need = max(len(prompts[i]) for i in idx) + mx + 1
            if mx < 1 or need > 4608:
                raise ValueError(f"EVAFRILL 생성 길이 {need} > 4608")
            eng = self.lm._get_engine(4608)       # 엔진 크기 고정 (배치 불변 모드는 decode 어텐션 길이 = max_len)
            out = eng.generate([prompts[i] for i in idx], mx, eos_ids=(2,), stop_strings=stop or None,
                               decode_fn=(lambda ids: decode_keep_unk(tok, ids)) if stop else None,
                               temperature=temp, top_p=float(sampling.get("top_p", 1.0)) if temp > 0 else 1.0,
                               seed=seed if temp > 0 else None,
                               repetition_penalty=float(sampling.get("repetition_penalty", 1.0)))
            for i, r in zip(idx, out):
                toks = r.tokens[: max_new[i]]
                full = self.decode(toks)
                text = cut_at_stop(full, stop)
                fin = "stop" if text != full else (r.finish_reason if len(r.tokens) <= max_new[i] else "length")
                res[i] = {"text": text, "n": len(toks), "finish": fin}
            print(f"[evafrill] {min(s + bs, len(order))}/{len(order)}", flush=True)
        return res  # type: ignore[return-value]


def build_backend(spec: ModelSpec, max_model_len: int, gpu_mem: float):
    if spec.backend == "vllm":
        return VllmGen(spec, max_model_len, gpu_mem)
    if spec.backend == "evafrill":
        return EvafrillGen(spec, max_model_len)
    raise ValueError(spec.backend)


# ── 실행 ────────────────────────────────────────────────────────────────────

USER_STOP = ["<|user|>"]      # 사용자 모델(user_assistant_v1)이 다음 턴을 스스로 이어 쓰는 경우 정지


def run_task(be, spec: ModelSpec, st: GenSettings, task: str, limit: int | None) -> dict:
    cfg = load_gen_config()
    tdef = cfg["tasks"][task]
    items = load_task(task)
    if limit:
        items = items[:limit]
    od = out_dir(spec.id, st.variant)
    od.mkdir(parents=True, exist_ok=True)
    stem = task + (f"__limit{limit}" if limit else "")
    path = od / f"{stem}.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        print(f"[{spec.id}|{st.variant}] {task}: 완료됨 ({len(done)})")
        return {"task": task, "skipped": True}
    from kobench.standard import unified as U

    uni = U.is_uni(st.variant)
    base = spec.kind == "base" and not uni
    max_new = U.uni_max_new(task, int(tdef["max_new"])) if uni else int(tdef["max_new"])
    chunk = int(cfg["common"]["chunk"])
    seed = int(cfg["common"]["seed"])
    t0 = time.time()
    n_gen = 0
    for c0 in range(0, len(todo), chunk):
        part = todo[c0: c0 + chunk]
        nturn = 1 if base else max((U.uni_turns(task, it) if uni else len(it["turns"])) for it in part)
        hist = {it["id"]: [] for it in part}
        recs = {it["id"]: {"id": it["id"], "outputs": [], "reasoning": [], "gen_tokens": [], "prompt_tokens": [], "finish": []}
                for it in part}
        for turn in range(nturn):
            active = [it for it in part if (base and turn == 0) or (uni and turn < U.uni_turns(task, it))
                      or (not base and not uni and turn < len(it["turns"]))]
            prompts, mx, keep = [], [], []
            for it in active:
                if base:
                    ids = be.encode_raw(it["raw"])
                elif uni:
                    prev_ans = it["oracle"][:turn] if it.get("oracle") else hist[it["id"]]
                    ids = be.encode_raw(U.uni_text(task, it, turn, prev_ans))
                else:
                    msgs = []
                    # 정답 이력 시나리오: 앞 턴 assistant 내용 = 참조 답 (모델 자신의 답 대신)
                    prev_ans = it["oracle"][:turn] if it.get("oracle") else hist[it["id"]]
                    for u, a in zip(it["turns"][:turn], prev_ans):
                        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                    msgs.append({"role": "user", "content": it["turns"][turn]})
                    if st.system:
                        msgs = [{"role": "system", "content": st.system}] + msgs
                    ids = be.encode_chat(msgs, st.think_kwargs, st.template)
                room = st.max_model_len - len(ids)
                m = room if is_think(st.variant) else min(max_new, room)
                if m < 16:
                    r = recs[it["id"]]
                    r["outputs"].append("")
                    r["reasoning"].append("")
                    r["gen_tokens"].append(0)
                    r["prompt_tokens"].append(len(ids))
                    r["finish"].append("prompt_too_long")
                    hist[it["id"]].append("")
                    continue
                prompts.append(ids)
                mx.append(m)
                keep.append(it)
            if not prompts:
                continue
            if base:
                stop = keep[0]["stop"]
            elif uni:
                stop = U.uni_stop(task, keep[0]) + (USER_STOP if spec.owner == "self" else [])
            elif spec.owner == "self":
                stop = USER_STOP
            else:
                stop = None
            outs = be.generate(prompts, mx, st.sampling, stop, seed)
            for it, ids, o in zip(keep, prompts, outs):
                if is_think(st.variant):
                    ans, rea = parse_think(o["text"], o["finish"], think_prefilled(be, ids))
                elif "<think>" in o["text"] or "</think>" in o["text"]:
                    ans, rea = split_reasoning(o["text"])
                else:
                    ans, rea = o["text"], ""
                r = recs[it["id"]]
                r["outputs"].append(ans)
                r["reasoning"].append(rea)
                r["gen_tokens"].append(o["n"])
                r["prompt_tokens"].append(len(ids))
                r["finish"].append(o["finish"])
                hist[it["id"]].append(ans)
                n_gen += o["n"]
        with open(path, "a", encoding="utf-8") as f:
            for it in part:
                f.write(json.dumps(recs[it["id"]], ensure_ascii=False) + "\n")
        print(f"[{spec.id}|{st.variant}] {task}: {min(c0 + chunk, len(todo))}/{len(todo)} 기록", flush=True)
    wall = time.time() - t0
    meta = {
        "task": task, "task_def": tdef, "dataset": cfg["datasets"].get(tdef["dataset"]) if tdef.get("dataset") else {"generator": tdef.get("generator")}, "limit": limit,
        "model": {k: (v if isinstance(v, (str, int, float, bool, type(None), dict, list)) else str(v)) for k, v in spec.__dict__.items()},
        "variant": st.variant, "sampling": st.sampling, "think_kwargs": st.think_kwargs, "max_model_len": st.max_model_len,
        "system": st.system, "template": st.template,
        "n_items": len(items), "generated_tokens_this_run": n_gen, "wall_seconds": round(wall, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
    }
    from kobench.standard.run import _git_commit, _versions

    meta["versions"] = _versions()
    meta["suite_git"] = _git_commit()
    (od / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"task": task, "wall_s": round(wall, 1), "gen_tokens": n_gen}


def think_prefilled(be, ids: list[int]) -> bool:
    """생성 프롬프트 끝에 템플릿이 '<think>'를 미리 넣었는가 (Qwen3.5/3.6·Nemotron은 넣고, SmolLM3·Gemma 4는 안 넣는다)."""
    tok = getattr(be, "tok", None)
    return bool(tok) and "<think>" in tok.decode(ids[-6:], skip_special_tokens=False)


def tasks_for(spec: ModelSpec, suite: str, only: list[str] | None) -> list[str]:
    cfg = load_gen_config()
    names = cfg["suites"][suite]
    if only:
        bad = set(only) - set(names)
        if bad:
            raise SystemExit(f"suite {suite}에 없는 태스크: {sorted(bad)}")
        names = [t for t in names if t in only]
    if suite in ("uni_v1", "uni_dev_v1"):
        return names                       # 통일 프로토콜: 모든 모델에 같은 태스크
    return [t for t in names if spec.kind in cfg["tasks"][t]["kinds"]]


def cmd_run(a) -> int:
    models = load_models()
    summ = []
    for mid in a.models:
        spec = models[mid]
        st = settings_for(spec, a.variant)
        if st is None:
            print(f"[{mid}] 변형 {a.variant} 없음 (권장 샘플링 또는 thinking 미지원) — 건너뜀")
            continue
        if spec.kind == "base" and a.variant not in ("A", "U"):
            print(f"[{mid}] base 모델은 A·U만")
            continue
        names = tasks_for(spec, a.suite, a.tasks)
        stem = lambda t: t + (f"__limit{a.limit}" if a.limit else "")  # noqa: E731
        pending = []
        for t in names:
            p = out_dir(mid, st.variant) / f"{stem(t)}.jsonl"
            n_items = len(load_task(t)[: a.limit] if a.limit else load_task(t))
            if len(read_jsonl(p)) < n_items:
                pending.append(t)
        if not pending:
            print(f"[{mid}|{st.variant}] 모든 태스크 완료됨")
            continue
        t0 = time.time()
        be = build_backend(spec, st.max_model_len, a.gpu_mem)
        load_s = time.time() - t0
        print(f"[{mid}|{st.variant}] 로드 {load_s:.0f}s, 태스크 {pending}", flush=True)
        for t in pending:
            r = run_task(be, spec, st, t, a.limit)
            r.update({"model": mid, "variant": st.variant, "load_s": round(load_s, 1)})
            summ.append(r)
        del be
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


def cmd_score(a) -> int:
    from kobench.standard import gen_score

    models = load_models()
    for mid in a.models:
        spec = models[mid]
        for t in tasks_for(spec, a.suite, a.tasks):
            gen_score.score_task(mid, a.variant, t, a.limit, force=a.force)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+", required=True)
        p.add_argument("--suite", default="gen_v1")
        p.add_argument("--tasks", nargs="+")
        p.add_argument("--variant", default="A", help="기본[+수정자]: A card rep rp11 rp12 fp03 think [+sys] [+orpo]")
        p.add_argument("--limit", type=int)
        p.add_argument("--gpu-mem", type=float, default=0.85)
        p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)       # vLLM 소멸자 SIGABRT 회피 (run.py와 같은 이유)
