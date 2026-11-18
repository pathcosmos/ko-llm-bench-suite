"""학습 형식 ↔ 평가 경로 토큰 일치 검증 (Goal 1-A).

학습 기준값은 원본 학습 코드의 함수를 그대로 import해서 만든다:
  - frankenstallm: HF pathcosmos/frankenstallm data/sft_dataset.py (_build_conversation_turns) + korean_sp tokenizer.encode
  - EVAFRILL:      EVAFRILL-Mo data/sft_dataset.py (_build_conversation_turns) + korean_sp tokenizer.encode
  - JANSORI:       llm-traditional data/tokenize_shards.py:116과 같은 encode_batch(add_special_tokens=False)
평가 경로: formats.py, vLLM 토크나이저(chat 템플릿), lm-eval VLLM.tok_encode, llama.cpp /tokenize, Ollama(prompt_eval_count)

python -m kobench.standard.conformance --out ../runs/format_conformance_v1 [--llamacpp] [--ollama]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from kobench.standard import formats

# ── 입력 사례 (한/영/공백/개행/특수문자/OOV/코드/특수토큰 문자열) ─────────────────────
TEXTS = [
    "한국의 수도는 어디인가요?",
    "What is the capital of South Korea?",
    "한국어와 English가 섞인 문장입니다.",
    "  앞에 공백 두 칸",
    "뒤에 공백 ",
    "중간에  공백  두 칸씩",
    "첫 줄\n둘째 줄\n\n넷째 줄",
    "\t탭으로 시작\t중간 탭",
    "줄 끝 개행\n",
    "숫자 1234567890과 3.14159, 1/3, 8,622,698명",
    "km² · ≈ · ½ · ①②③ · ★☆♥",
    "윷놀이와 廉宗(염종) — 사전에 없는 문자",
    "이모지 😀👍🇰🇷 포함",
    "ｆｕｌｌ　ｗｉｄｔｈ 전각 문자",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "SELECT * FROM users WHERE age >= 19;",
    "<|user|>\n태그 문자열이 본문에 있는 경우\n<|assistant|>\n",
    "본문 중간에 </s> 특수 토큰 문자열",
    "<s> 로 시작하는 본문",
    "가\u0301 결합 문자, x\u200by 폭 없는 공백",
    "C:\\Users\\path\\to\\file.txt",
    "“따옴표”와 ‘작은따옴표’, 그리고 「괄호」",
    "a" * 3 + " " + "가" * 5,
    "Ⅰ. 서론\nⅡ. 본론\nⅢ. 결론",
]

CHATS = [[{"role": "user", "content": t}] for t in TEXTS] + [
    [{"role": "system", "content": "당신은 친절한 비서입니다."}, {"role": "user", "content": "안녕하세요"}],
    [{"role": "user", "content": "첫 질문"}, {"role": "assistant", "content": "첫 답변"}, {"role": "user", "content": "두 번째 질문"}],
    [{"role": "user", "content": "연속"}, {"role": "user", "content": "사용자 메시지"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b\n여러 줄\n답변"}, {"role": "user", "content": "c"},
     {"role": "assistant", "content": "d"}, {"role": "user", "content": "e  "}],
    [{"role": "assistant", "content": "먼저 나온 assistant는 학습에서 무시"}, {"role": "user", "content": "질문"}],
]

TRAINING_SFT_SOURCES = {
    "frankenstallm-bf16": "${HF_HOME}/hub/models--pathcosmos--frankenstallm/snapshots/b9f7777f01479b18e5a3a55f9cad4582df122f7d/data/sft_dataset.py",
    "evafrill-mo-3b": "${LT_SRC}/EVAFRILL-Mo/data/sft_dataset.py",
}
JANSORI_GGUF = "/home/work/llm-dense/llm-traditional/runs/3b_47k_hub_release_v1/nfs-release-v1/gguf/dense-3b-47k-Q8_0.gguf"
FRANKEN_GGUF = "${HF_HOME}/hub/models--pathcosmos--frankenstallm/snapshots/b9f7777f01479b18e5a3a55f9cad4582df122f7d/gguf/frankenstallm-3b-v2-Q8_0.gguf"


def _p(s: str) -> Path:
    return Path(os.path.expandvars(s))


def load_training_module(model_id: str):
    """원본 학습 코드 모듈 (data/sft_dataset.py)을 파일 경로로 import."""
    path = _p(TRAINING_SFT_SOURCES[model_id])
    spec = importlib.util.spec_from_file_location(f"_train_sft_{model_id.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def training_prompt(model_id: str, messages: list[dict]) -> str:
    """학습 코드 _build_conversation_turns로 '다음 assistant 응답'의 prompt 문자열을 만든다."""
    mod = load_training_module(model_id)
    conv = [m for m in messages] + [{"role": "assistant", "content": "__NEXT__"}]
    pairs = mod._build_conversation_turns(conv)
    if not pairs:
        raise ValueError("학습 코드가 학습 쌍을 만들지 않음")
    return pairs[-1][0]


def training_ids_chat(model_id: str, messages: list[dict]) -> list[int]:
    return formats.training_tokenizer(model_id).encode(training_prompt(model_id, messages)).ids


def training_ids_text(model_id: str, text: str) -> list[int]:
    tok = formats.training_tokenizer(model_id)
    return tok.encode_batch([text], add_special_tokens=False)[0].ids


# ── 평가 경로 ───────────────────────────────────────────────────────────────
def vllm_tokenizer(model_id: str):
    from vllm.tokenizers.registry import get_tokenizer

    return get_tokenizer(str(formats.prepare_tokenizer_dir(model_id)))


def lmeval_stub(model_id: str):
    """lm-eval VLLM.tok_encode를 GPU 없이 실행하기 위한 최소 인스턴스 (실제 메서드 사용)."""
    from lm_eval.models.vllm_causallms import VLLM

    obj = VLLM.__new__(VLLM)
    obj.tokenizer = vllm_tokenizer(model_id)
    obj.add_bos_token = None
    obj.custom_prefix_token_id = None
    return obj


class LlamaServer:
    def __init__(self, gguf: Path, port: int):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        self.port = port
        self.proc = subprocess.Popen(
            [str(Path(os.environ["LT_BIN"]) / "llama-server"), "-m", str(gguf), "-ngl", "0", "--port", str(port),
             "--host", "127.0.0.1", "-c", "512", "-t", "4"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        import requests

        for _ in range(240):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        self.close()
        raise RuntimeError("llama-server 기동 실패")

    def tokenize(self, text: str, add_special: bool) -> list[int]:
        import requests

        r = requests.post(f"http://127.0.0.1:{self.port}/tokenize",
                          json={"content": text, "add_special": add_special, "parse_special": True}, timeout=30)
        r.raise_for_status()
        return r.json()["tokens"]

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _cmp(ref: list[int], got: list[int]) -> dict:
    if ref == got:
        return {"equal": True}
    i = next((k for k, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
    return {"equal": False, "first_diff": i, "ref": ref[max(0, i - 3): i + 6], "got": got[max(0, i - 3): i + 6],
            "len_ref": len(ref), "len_got": len(got)}


def _summ(rows: list[dict]) -> dict:
    return {"n": len(rows), "equal": sum(r["equal"] for r in rows),
            "mismatches": [r for r in rows if not r["equal"]][:10]}


def check_chat_model(model_id: str, llama: LlamaServer | None) -> dict:
    tok_v = vllm_tokenizer(model_id)
    lm = lmeval_stub(model_id)
    paths = {k: [] for k in ("formats", "vllm_chat_template_str", "vllm_chat_no_special", "vllm_chat_add_special",
                             "lmeval_tok_encode", "llamacpp_no_special", "llamacpp_add_special")}
    for msgs in CHATS:
        ref_prompt = training_prompt(model_id, msgs)
        ref = formats.training_tokenizer(model_id).encode(ref_prompt).ids
        rendered = tok_v.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        case = {"messages": msgs}
        paths["formats"].append({**case, **_cmp(ref, formats.encode_prompt(model_id, msgs))})
        paths["vllm_chat_template_str"].append({**case, "equal": rendered == ref_prompt,
                                               **({} if rendered == ref_prompt else {"ref": ref_prompt, "got": rendered})})
        paths["vllm_chat_no_special"].append({**case, **_cmp(ref, tok_v(rendered, add_special_tokens=False).input_ids)})
        paths["vllm_chat_add_special"].append({**case, **_cmp(ref, tok_v(rendered, add_special_tokens=True).input_ids)})
        paths["lmeval_tok_encode"].append({**case, **_cmp(ref, lm.tok_encode(rendered))})
        if llama is not None:
            paths["llamacpp_no_special"].append({**case, **_cmp(ref, llama.tokenize(rendered, False))})
            paths["llamacpp_add_special"].append({**case, **_cmp(ref, llama.tokenize(rendered, True))})
    return {k: _summ(v) for k, v in paths.items() if v}


def check_text_model(model_id: str, llama: LlamaServer | None) -> dict:
    tok_v = vllm_tokenizer(model_id)
    lm = lmeval_stub(model_id)
    paths = {k: [] for k in ("formats_encode_text", "vllm_add_special", "vllm_no_special", "lmeval_tok_encode",
                             "llamacpp_no_special", "llamacpp_add_special")}
    for t in TEXTS:
        ref = training_ids_text(model_id, t)
        case = {"text": t}
        paths["formats_encode_text"].append({**case, **_cmp(ref, formats.encode_text(model_id, t))})
        paths["vllm_add_special"].append({**case, **_cmp(ref, tok_v(t, add_special_tokens=True).input_ids)})
        paths["vllm_no_special"].append({**case, **_cmp(ref, tok_v(t, add_special_tokens=False).input_ids)})
        paths["lmeval_tok_encode"].append({**case, **_cmp(ref, lm.tok_encode(t))})
        if llama is not None:
            paths["llamacpp_no_special"].append({**case, **_cmp(ref, llama.tokenize(t, False))})
            paths["llamacpp_add_special"].append({**case, **_cmp(ref, llama.tokenize(t, True))})
    return {k: _summ(v) for k, v in paths.items() if v}


TRAIN_TOK_GGUF = "${LT_LOCAL}/gguf/frankenstallm-3b-v2-Q8_0.train-tokenizer.gguf"
SPECIAL_ADJACENT = ["</s>abc 특수 토큰 뒤 공백 없음", "앞</s>뒤", "<s>\n개행으로 시작", "x<s>y<unk>z", "<pad><pad> 연속 특수",
                    "<unk>", "끝에 특수</s>", "</s>"]


def check_llamacpp_train_tokenizer(server: "LlamaServer") -> dict:
    """학습 토크나이저 GGUF + formats.llamacpp_text() 문자열 경로가 학습 토큰과 같은지."""
    out = {"texts": [], "chats": [], "special_adjacent": []}
    for t in TEXTS:
        out["texts"].append({"text": t, **_cmp(training_ids_text("frankenstallm-bf16", t),
                                                server.tokenize(formats.llamacpp_text(t), True))})
    for m in CHATS:
        out["chats"].append({"messages": m, **_cmp(training_ids_chat("frankenstallm-bf16", m),
                             server.tokenize(formats.llamacpp_text(formats.build_prompt("frankenstallm-bf16", m)), True))})
    for t in SPECIAL_ADJACENT:
        out["special_adjacent"].append({"text": t, **_cmp(training_ids_text("frankenstallm-bf16", t),
                                                           server.tokenize(formats.llamacpp_text(t), True))})
    return {k: _summ(v) for k, v in out.items()}


def check_token_id_path(server: "LlamaServer", model_id: str, chats: bool) -> dict:
    """llama-server에 토큰 ID 배열을 직접 넣을 때 그대로 평가되는지 (tokens_evaluated == 길이, BOS 미추가)."""
    import requests

    bos = formats.load_formats()["tokenizers"][formats.model_format(model_id)["tokenizer"]]["special"]["bos"]

    def comp(ids):
        r = requests.post(f"http://127.0.0.1:{server.port}/completion",
                          json={"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": 5, "cache_prompt": False},
                          timeout=120)
        r.raise_for_status()
        return r.json()

    cases = [training_ids_text(model_id, t) for t in TEXTS if t.strip()]
    if chats:
        cases += [training_ids_chat(model_id, m) for m in CHATS]
    rows = []
    for ids in cases:
        a, b = comp(ids), comp([bos] + ids)
        pa = json.dumps(a.get("completion_probabilities"), sort_keys=True)
        pb = json.dumps(b.get("completion_probabilities"), sort_keys=True)
        rows.append({"n_ids": len(ids), "equal": a.get("tokens_evaluated") == len(ids) and pa != pb})
    return _summ(rows)

def _ollama_internal_port(serve_log: Path) -> int:
    import re

    ports = re.findall(r"--port (\d+)", serve_log.read_text(errors="replace"))
    if not ports:
        raise RuntimeError("Ollama 내부 llama-server 포트를 로그에서 찾지 못함")
    return int(ports[-1])


def check_ollama(host: str, serve_log: Path, fs_model: str, jansori_model: str) -> dict:
    """Ollama 경로의 실제 토큰 ID 검증.

    1) /api/chat·/api/generate의 `_debug_render_only`로 Ollama가 렌더링한 프롬프트 문자열을 받는다.
    2) 그 문자열을 Ollama가 띄운 내부 llama-server(번들 llama.cpp)의 /tokenize로 토큰화한다.
    3) 실제 실행의 prompt_eval_count와도 대조한다.
    frankenstallm은 학습 토크나이저 GGUF(gguf_retokenize, add_space_prefix=False)와 앞 공백 템플릿을 쓴다.
    """
    import requests

    O = {"num_predict": 1, "num_ctx": 4096, "temperature": 0}

    def post(path, body):
        r = requests.post(host + path, json={**body, "stream": False}, timeout=900)
        r.raise_for_status()
        return r.json()

    def internal_tokenize(text):
        port = _ollama_internal_port(serve_log)
        r = requests.post(f"http://127.0.0.1:{port}/tokenize",
                          json={"content": text, "add_special": True, "parse_special": True}, timeout=60)
        r.raise_for_status()
        return r.json()["tokens"]

    out = {"frankenstallm_chat": [], "jansori_text": []}
    for msgs in CHATS:
        ref = training_ids_chat("frankenstallm-bf16", msgs)
        n = post("/api/chat", {"model": fs_model, "messages": msgs, "options": O})["prompt_eval_count"]  # 모델 적재 + 실제 실행
        rendered = post("/api/chat", {"model": fs_model, "messages": msgs, "_debug_render_only": True})["_debug_info"]["rendered_template"]
        ids = internal_tokenize(rendered)
        out["frankenstallm_chat"].append({"messages": msgs, "rendered": rendered, "prompt_eval_count": n,
                                          "count_equal": n == len(ref), **_cmp(ref, ids)})
    for t in TEXTS:
        ref = training_ids_text("jansori-bf16", t)
        n = post("/api/generate", {"model": jansori_model, "prompt": t, "options": O})["prompt_eval_count"]
        rendered = post("/api/generate", {"model": jansori_model, "prompt": t, "_debug_render_only": True})["_debug_info"]["rendered_template"]
        ids = internal_tokenize(rendered)
        out["jansori_text"].append({"text": t, "rendered_equal_input": rendered == t, "prompt_eval_count": n,
                                    "count_equal": n == len(ref), **_cmp(ref, ids)})
    for k, rows in out.items():
        print(f"ollama {k}: token IDs equal {sum(r['equal'] for r in rows)}/{len(rows)}, "
              f"prompt_eval_count equal {sum(r['count_equal'] for r in rows)}/{len(rows)}")
    return out


def negative_control_release_tokenizer() -> dict:
    """검증의 민감도 확인: frankenstallm HF 배포 tokenizer.json(byte-fallback)은 OOV 사례에서 달라야 한다."""
    from tokenizers import Tokenizer

    rel = Tokenizer.from_file(str(_p("${HF_HOME}/hub/models--pathcosmos--frankenstallm/snapshots/"
                                     "b9f7777f01479b18e5a3a55f9cad4582df122f7d/tokenizer.json")))
    rows = []
    for msgs in CHATS:
        ref_prompt = training_prompt("frankenstallm-bf16", msgs)
        ref = formats.training_tokenizer("frankenstallm-bf16").encode(ref_prompt).ids
        rows.append({"messages": msgs, **_cmp(ref, rel.encode(ref_prompt).ids)})
    return _summ(rows)


def old_suite_examples() -> list[dict]:
    """기존 수트 경로와 학습 형식의 실제 토큰 차이 (사례 3개)."""
    out = []
    track1_system = "당신은 한국어 시험 문제에 정확히 답하는 AI입니다. 정답 기호만 출력하세요."
    q = "한국의 수도는 어디인가요?"
    for mid in ("evafrill-mo-3b", "frankenstallm-bf16"):
        tok = formats.training_tokenizer(mid)
        new_ids = formats.encode_prompt(mid, [{"role": "system", "content": track1_system}, {"role": "user", "content": q}])
        if mid == "evafrill-mo-3b":
            old_text = f"{track1_system}\n{q}"  # kobench/evafrill_runner.py:217
            old_ids = tok.encode(old_text).ids
            path = "kobench/evafrill_runner.py:217 f'{system}\\n{prompt}' → tokenizer.encode"
        else:
            old_text = q  # Modelfile TEMPLATE {{ .Prompt }} → system 미포함, llama.cpp가 BOS 추가
            old_ids = [1] + tok.encode(old_text).ids
            path = "modelfiles/Modelfile.v2-*: TEMPLATE {{ .Prompt }} (SYSTEM 무시) + GGUF 기본 BOS"
        out.append({
            "model": mid, "old_path": path,
            "old_text": old_text, "old_ids": old_ids, "old_pieces": [tok.id_to_token(i) for i in old_ids],
            "train_text": formats.build_prompt(mid, [{"role": "user", "content": q}]),
            "train_ids": new_ids, "train_pieces": [tok.id_to_token(i) for i in new_ids],
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--llamacpp", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"n_texts": len(TEXTS), "n_chats": len(CHATS), "models": {}}
    llama_j = LlamaServer(Path(JANSORI_GGUF), 18611) if a.llamacpp else None
    try:
        res["models"]["jansori-bf16"] = check_text_model("jansori-bf16", llama_j)
    finally:
        if llama_j:
            llama_j.close()
    llama_f = LlamaServer(_p(FRANKEN_GGUF), 18612) if a.llamacpp else None
    try:
        res["models"]["frankenstallm-bf16"] = check_chat_model("frankenstallm-bf16", llama_f)
        res["models"]["frankenstallm-bf16__pretrain_text"] = check_text_model("frankenstallm-bf16", llama_f)
    finally:
        if llama_f:
            llama_f.close()
    if a.llamacpp:
        tg = _p(TRAIN_TOK_GGUF)
        if tg.exists():
            s = LlamaServer(tg, 18613)
            try:
                res["models"]["frankenstallm-bf16__train_tokenizer_gguf"] = check_llamacpp_train_tokenizer(s)
            finally:
                s.close()
        for mid, g, port, chats in (("jansori-bf16", Path(JANSORI_GGUF), 18614, False),
                                    ("frankenstallm-bf16", _p(FRANKEN_GGUF), 18615, True)):
            s = LlamaServer(g, port)
            try:
                res["models"][f"{mid}__token_id_input"] = {"llamacpp_token_id": check_token_id_path(s, mid, chats)}
            finally:
                s.close()
    res["models"]["evafrill-mo-3b"] = check_chat_model("evafrill-mo-3b", None)
    res["models"]["evafrill-mo-3b__pretrain_text"] = check_text_model("evafrill-mo-3b", None)
    # frankenstallm의 llamacpp_* 키는 배포 GGUF(학습 후 만든 토크나이저) 결과다 — 평가에 쓰지 않는 대조 기록임을 이름에 드러낸다
    for key in ("frankenstallm-bf16", "frankenstallm-bf16__pretrain_text"):
        for pth in [k for k in list(res["models"].get(key, {})) if k.startswith("llamacpp_")]:
            res["models"][key][f"RELEASE_GGUF_not_used__{pth}"] = res["models"][key].pop(pth)
    res["negative_control_frankenstallm_release_tokenizer"] = negative_control_release_tokenizer()
    res["old_suite_examples"] = old_suite_examples()
    (out / "token_conformance.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for m, paths in res["models"].items():
        print(m)
        for p, sm in paths.items():
            print(f"   {p:26s} {sm['equal']}/{sm['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
