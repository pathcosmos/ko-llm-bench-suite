"""사용자 모델의 학습 형식(configs/standard/formats.yaml)을 평가 경로에 적용한다.

- build_prompt(): 학습 코드(_build_conversation_turns / _build_alpaca_turns)와 같은 프롬프트 문자열
- encode_prompt(): 학습과 같은 방식(프롬프트 문자열 전체를 한 번 encode)으로 토큰 ID
- prepare_tokenizer_dir(): vLLM/transformers용 로컬 토크나이저 디렉터리 (SHA 검증한 tokenizer.json + chat 템플릿)
  HF 원본 스냅샷은 수정하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

import yaml

from kobench.standard.registry import CONFIG_DIR, _expand

FORMATS_PATH = CONFIG_DIR / "formats.yaml"

# 학습 코드의 user/assistant 형식을 그대로 옮긴 Jinja 템플릿 (system 역할은 학습 데이터에서 무시됨 → drop)
# pending 판정은 학습 코드처럼 truthiness(빈 문자열 = 없음)로 한다.
CHAT_TEMPLATE_USER_ASSISTANT_V1 = (
    "{%- set ns = namespace(history='', pending='') -%}"
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'user' -%}"
    "{%- if ns.pending -%}{%- set ns.pending = ns.pending ~ '\\n' ~ m['content'] -%}"
    "{%- else -%}{%- set ns.pending = m['content'] -%}{%- endif -%}"
    "{%- elif m['role'] == 'assistant' -%}"
    "{%- if ns.pending -%}"
    "{%- set ns.history = ns.history ~ '<|user|>\\n' ~ ns.pending ~ '\\n<|assistant|>\\n' ~ m['content'] ~ '\\n' -%}"
    "{%- set ns.pending = '' -%}"
    "{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{{- ns.history -}}"
    "{%- if ns.pending -%}{{- '<|user|>\\n' ~ ns.pending ~ '\\n<|assistant|>\\n' -}}{%- endif -%}"
)


@lru_cache(maxsize=1)
def load_formats(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or FORMATS_PATH).read_text(encoding="utf-8"))


def model_format(model_id: str) -> dict:
    f = load_formats()["models"]
    if model_id not in f:
        raise KeyError(f"formats.yaml에 {model_id} 형식이 없다")
    entry = f[model_id]
    if "alias_of" in entry:                     # 같은 형식을 쓰는 다른 체크포인트
        return {**model_format(entry["alias_of"]), **{k: v for k, v in entry.items() if k != "alias_of"}}
    return entry


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def tokenizer_file(name: str) -> Path:
    """formats.yaml의 sources 중 SHA가 일치하는 첫 파일. 없으면 실패한다."""
    spec = load_formats()["tokenizers"][name]
    if spec.get("use") is False:
        raise ValueError(f"{name}: 사용 금지 토크나이저 ({spec.get('reason', '')[:80]})")
    tried = []
    for src in spec["sources"]:
        p = Path(_expand(src.replace("${HF_HOME}", os.environ.get("HF_HOME", "${HF_HOME}"))))
        if p.exists():
            got = _sha256(p)
            if got == spec["sha256"]:
                return p
            tried.append(f"{p} (sha {got[:12]} 불일치)")
        else:
            tried.append(f"{p} (없음)")
    raise FileNotFoundError(f"{name}: SHA {spec['sha256'][:12]} 파일을 찾지 못함: {tried}")


@lru_cache(maxsize=None)
def training_tokenizer(model_id: str):
    """학습 코드가 쓴 것과 같은 tokenizers.Tokenizer 객체."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_file(model_format(model_id)["tokenizer"])))


def build_prompt(model_id: str, messages: list[dict], template: str | None = None, system: str | None = None) -> str:
    """messages([{role, content}]) → 학습 형식 프롬프트 문자열 (마지막은 assistant 태그로 끝난다)."""
    fmt = model_format(model_id)
    if not fmt.get("chat"):
        raise ValueError(f"{model_id}: chat 형식 없음 (base 모델은 원문 이어쓰기)")
    tname = template or fmt["chat"]["default"]
    t = load_formats()["templates"][tname]
    system_mode = system or t.get("system", "drop")
    msgs = list(messages)
    if tname == "orpo_raw_bos":
        sys_txt = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        users = [m["content"] for m in msgs if m["role"] == "user"]
        if len(users) != 1 or any(m["role"] == "assistant" for m in msgs):
            raise ValueError("orpo_raw_bos는 단일 턴만 지원")
        return (sys_txt + "\n" + users[0]).strip() if sys_txt else users[0]
    if system_mode == "prepend_to_first_user":
        sys_txt = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        msgs = [m for m in msgs if m["role"] != "system"]
        if sys_txt:
            for i, m in enumerate(msgs):
                if m["role"] == "user":
                    msgs[i] = {**m, "content": f"{sys_txt}\n{m['content']}"}
                    break
    elif system_mode != "drop":
        raise ValueError(f"알 수 없는 system 처리: {system_mode}")
    # 학습 코드 _build_conversation_turns와 같은 규칙
    ut, at = t["user_tag"], t["assistant_tag"]
    history, pending = "", ""
    for m in msgs:
        role, content = m.get("role", "").lower(), m.get("content", "")
        if role == "user":
            pending = f"{pending}\n{content}" if pending else content
        elif role == "assistant":
            if not pending:
                continue
            history = f"{history}{ut}{pending}\n{at}{content}\n"
            pending = ""
    if not pending:
        raise ValueError("마지막에 답할 user 메시지가 없다")
    return f"{history}{ut}{pending}\n{at}"


def encode_prompt(model_id: str, messages: list[dict], template: str | None = None, system: str | None = None) -> list[int]:
    """학습과 같은 토큰 ID: 프롬프트 문자열 전체를 한 번 encode (+ 템플릿이 요구하면 BOS)."""
    fmt = model_format(model_id)
    tname = template or fmt["chat"]["default"]
    t = load_formats()["templates"][tname]
    tok = training_tokenizer(model_id)
    ids = tok.encode(build_prompt(model_id, messages, tname, system)).ids
    if t.get("bos"):
        bos = load_formats()["tokenizers"][fmt["tokenizer"]]["special"]["bos"]
        ids = [bos] + ids
    return ids


def encode_text(model_id: str, text: str) -> list[int]:
    """사전학습과 같은 원문 인코딩 (특수 토큰 자동 추가 없음)."""
    return training_tokenizer(model_id).encode(text, add_special_tokens=False).ids


def prepare_tokenizer_dir(model_id: str, root: str | Path | None = None, doc_start: str = "none") -> Path:
    """vLLM/transformers가 읽을 로컬 토크나이저 디렉터리를 만든다 (멱등, 내용이 같으면 재사용).

    doc_start="eos": 모든 인코딩 앞에 EOS를 붙이는 post_processor를 넣는다 (사전학습 문서 시작 조건 재현).
    lm-eval은 문맥과 전체를 각각 인코딩해 앞부분을 잘라내므로 연속 토큰에는 영향이 없다.
    """
    if doc_start not in ("none", "eos"):
        raise ValueError(doc_start)
    fmt = model_format(model_id)
    tok_name = fmt["tokenizer"]
    spec = load_formats()["tokenizers"][tok_name]
    src = tokenizer_file(tok_name)
    root = Path(root or Path(os.environ.get("LT_LOCAL", "/tmp")) / "tokenizer_overrides")
    out = root / (model_id if doc_start == "none" else f"{model_id}__docstart_{doc_start}")
    out.mkdir(parents=True, exist_ok=True)
    sp = spec["special"]
    if tok_name == "jansori_bbpe65k":
        # 배포본 디렉터리의 설정을 그대로 쓴다 (chat 템플릿 없음)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            s = src.parent / name
            if s.exists():
                shutil.copyfile(s, out / name)
    else:
        shutil.copyfile(src, out / "tokenizer.json")
        cfg = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>",
            "add_bos_token": False, "add_eos_token": False,
            "clean_up_tokenization_spaces": False,
            "model_max_length": 4096,
        }
        if fmt.get("chat"):
            cfg["chat_template"] = CHAT_TEMPLATE_USER_ASSISTANT_V1
        (out / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "special_tokens_map.json").write_text(json.dumps(
            {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}), encoding="utf-8")
        assert sp == {"pad": 0, "bos": 1, "eos": 2, "unk": 3}
    if doc_start == "eos":
        eos_id = spec["special"]["eos"]
        tj = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
        eos_tok = next(a["content"] for a in tj["added_tokens"] if a["id"] == eos_id)
        tj["post_processor"] = {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": eos_tok, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": eos_tok, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {eos_tok: {"id": eos_tok, "ids": [eos_id], "tokens": [eos_tok]}},
        }
        (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    (out / "FORMAT_SOURCE.json").write_text(json.dumps(
        {"model": model_id, "tokenizer": tok_name, "sha256": spec["sha256"], "source": str(src),
         "doc_start": doc_start}, indent=1), encoding="utf-8")
    return out


# korean_sp의 added special token (HF가 텍스트를 이 문자열로 먼저 조각낸다). 긴 것부터 매칭.
_KSP_SPECIALS = ("</s>", "<pad>", "<unk>", "<s>")


def llamacpp_text(text: str, specials: tuple[str, ...] = _KSP_SPECIALS) -> str:
    """학습 토크나이저 GGUF(add_space_prefix=False)에 문자열을 넣을 때의 클라이언트 규칙.

    HF(korean_sp)는 텍스트를 특수 토큰으로 조각낸 뒤, 특수 토큰이 아닌 각 조각에 Metaspace(prepend_scheme=always)를
    적용한다: 조각이 공백으로 시작하지 않으면 앞에 ▁을 붙인다. llama.cpp UGM도 특수 토큰으로 조각내지만
    add_space_prefix=True면 공백으로 시작하는 조각에도 ▁을 더 붙인다. 그래서 GGUF는 접두를 끄고, 여기서
    각 조각에 HF와 같은 규칙으로 공백 하나를 넣는다.
    """
    import re

    pat = re.compile("(" + "|".join(re.escape(t) for t in specials) + ")")
    out = []
    for i, part in enumerate(pat.split(text)):
        if i % 2 == 1:          # 특수 토큰
            out.append(part)
        elif part:
            out.append(part if part.startswith(" ") else " " + part)
    return "".join(out)
