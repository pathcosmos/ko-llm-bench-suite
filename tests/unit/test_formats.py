"""학습 형식 ↔ 평가 경로 토큰 일치 회귀 테스트 (configs/standard/formats.yaml).

기준값은 원본 학습 코드(data/sft_dataset.py의 _build_conversation_turns + korean_sp tokenizer.encode,
llm-traditional tokenize_shards의 encode_batch(add_special_tokens=False))로 만든다.
모델/소스 파일이 없는 환경에서는 건너뛴다. llama.cpp 검사는 KOBENCH_SLOW=1일 때만.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("lm_eval")

from kobench.standard import conformance as C  # noqa: E402
from kobench.standard import formats  # noqa: E402

CHAT_MODELS = ["frankenstallm-bf16", "evafrill-mo-3b"]


def _have(model_id: str) -> bool:
    try:
        formats.tokenizer_file(formats.model_format(model_id)["tokenizer"])
    except (FileNotFoundError, KeyError):
        return False
    if model_id in C.TRAINING_SFT_SOURCES:
        return C._p(C.TRAINING_SFT_SOURCES[model_id]).exists()
    return True


def _skip_unless(model_id):
    if not _have(model_id):
        pytest.skip(f"{model_id} 토크나이저/학습 코드 없음")


@pytest.mark.parametrize("model_id", CHAT_MODELS)
def test_prompt_string_and_ids_match_training_code(model_id):
    _skip_unless(model_id)
    for msgs in C.CHATS:
        assert formats.build_prompt(model_id, msgs) == C.training_prompt(model_id, msgs)
        assert formats.encode_prompt(model_id, msgs) == C.training_ids_chat(model_id, msgs)


@pytest.mark.parametrize("model_id", CHAT_MODELS)
def test_vllm_chat_template_and_lmeval_encoding_match(model_id, tmp_path):
    _skip_unless(model_id)
    pytest.importorskip("vllm")
    tok = C.vllm_tokenizer(model_id)
    lm = C.lmeval_stub(model_id)
    for msgs in C.CHATS:
        ref_prompt = C.training_prompt(model_id, msgs)
        ref = C.training_ids_chat(model_id, msgs)
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert rendered == ref_prompt
        assert tok(rendered, add_special_tokens=True).input_ids == ref
        assert lm.tok_encode(rendered) == ref


@pytest.mark.parametrize("model_id", ["jansori-bf16"] + CHAT_MODELS)
def test_raw_text_encoding_matches_pretraining(model_id):
    _skip_unless(model_id)
    pytest.importorskip("vllm")
    tok = C.vllm_tokenizer(model_id)
    lm = C.lmeval_stub(model_id)
    for t in C.TEXTS:
        ref = C.training_ids_text(model_id, t)
        assert formats.encode_text(model_id, t) == ref
        assert tok(t, add_special_tokens=True).input_ids == ref  # BOS 자동 추가 없음
        assert lm.tok_encode(t) == ref


def test_newline_and_tab_are_unk_in_korean_sp():
    _skip_unless("evafrill-mo-3b")
    tok = formats.training_tokenizer("evafrill-mo-3b")
    assert tok.encode("a\nb").ids == tok.encode("a\tb").ids == tok.encode("a\n\nb").ids
    assert 3 in tok.encode("a\nb").ids


def test_release_tokenizer_is_rejected_and_differs():
    _skip_unless("frankenstallm-bf16")
    with pytest.raises(ValueError):
        formats.tokenizer_file("frankenstallm_release_bytefallback")
    nc = C.negative_control_release_tokenizer()
    assert nc["equal"] < nc["n"]  # 검증이 실제 차이를 잡아내는지 (개행이 byte 토큰으로 바뀜)


def test_sha_guard_rejects_modified_file(tmp_path, monkeypatch):
    bad = tmp_path / "tokenizer.json"
    bad.write_text("{}")
    data = formats.load_formats()
    spec = dict(data["tokenizers"]["korean_sp_unigram64k"], sources=[str(bad)])
    monkeypatch.setitem(data["tokenizers"], "korean_sp_unigram64k", spec)
    formats.tokenizer_file.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            formats.tokenizer_file("korean_sp_unigram64k")
    finally:
        formats.load_formats.cache_clear()
        formats.tokenizer_file.cache_clear()


def test_orpo_raw_template_matches_trl_layout():
    _skip_unless("frankenstallm-bf16")
    msgs = [{"role": "system", "content": "시스템"}, {"role": "user", "content": "질문"}]
    assert formats.build_prompt("frankenstallm-bf16", msgs, template="orpo_raw_bos") == "시스템\n질문"
    ids = formats.encode_prompt("frankenstallm-bf16", msgs, template="orpo_raw_bos")
    assert ids[0] == 1 and ids[1:] == formats.training_tokenizer("frankenstallm-bf16").encode("시스템\n질문").ids


@pytest.mark.skipif(os.environ.get("KOBENCH_SLOW") != "1", reason="llama.cpp 검사는 KOBENCH_SLOW=1일 때만")
def test_llamacpp_jansori_matches_and_frankenstallm_does_not():
    if not Path(C.JANSORI_GGUF).exists() or not os.environ.get("LT_BIN"):
        pytest.skip("GGUF 또는 llama-server 없음")
    s = C.LlamaServer(Path(C.JANSORI_GGUF), 18631)
    try:
        for t in C.TEXTS:
            assert s.tokenize(t, True) == C.training_ids_text("jansori-bf16", t)
    finally:
        s.close()
    s = C.LlamaServer(C._p(C.FRANKEN_GGUF), 18632)
    try:
        ref = C.training_ids_chat("frankenstallm-bf16", C.CHATS[0])
        rendered = formats.build_prompt("frankenstallm-bf16", C.CHATS[0])
        assert s.tokenize(rendered, False) != ref  # GGUF 토크나이저 불일치 → 토큰 ID 직접 입력 경로가 필요
    finally:
        s.close()


def test_llamacpp_text_rule_matches_hf_segment_prefix():
    f = formats.llamacpp_text
    assert f("abc") == " abc"
    assert f("  x") == "  x"                       # 이미 공백으로 시작하면 그대로 (HF Metaspace 규칙)
    assert f("</s>abc") == "</s> abc"              # 특수 토큰으로 나눈 각 조각에 같은 규칙
    assert f("<s> 로") == "<s> 로"
    assert f("앞</s>뒤") == " 앞</s> 뒤"
    assert f("<pad><pad>") == "<pad><pad>"
    assert f("") == ""


@pytest.mark.skipif(os.environ.get("KOBENCH_SLOW") != "1", reason="llama.cpp 검사는 KOBENCH_SLOW=1일 때만")
def test_llamacpp_train_tokenizer_gguf_and_token_id_path():
    tg = C._p(C.TRAIN_TOK_GGUF)
    if not tg.exists() or not os.environ.get("LT_BIN"):
        pytest.skip("학습 토크나이저 GGUF(kobench.standard.gguf_retokenize) 또는 llama-server 없음")
    s = C.LlamaServer(tg, 18633)
    try:
        r = C.check_llamacpp_train_tokenizer(s)
        assert all(v["equal"] == v["n"] for v in r.values()), r
        t = C.check_token_id_path(s, "frankenstallm-bf16", chats=False)
        assert t["equal"] == t["n"]
    finally:
        s.close()


@pytest.mark.skipif(os.environ.get("KOBENCH_SLOW") != "1", reason="Ollama 검사는 KOBENCH_SLOW=1일 때만 (CPU Ollama를 직접 띄운다)")
def test_ollama_paths_match_training_ids(tmp_path):
    """수트의 Modelfile 그대로 Ollama(CPU)를 띄워 학습 토큰 ID와 대조한다 (준비·정리를 테스트가 직접 한다)."""
    import shutil
    import socket
    import subprocess
    import time

    import requests

    ollama = shutil.which("ollama")
    fs_gguf, j_gguf = C._p(C.TRAIN_TOK_GGUF), Path(C.JANSORI_GGUF)
    if not ollama or not fs_gguf.exists() or not j_gguf.exists():
        pytest.skip("ollama 바이너리 또는 GGUF(학습 토크나이저 GGUF, JANSORI) 없음")
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    host = f"127.0.0.1:{port}"
    repo = Path(__file__).resolve().parents[2]
    mf_fs = tmp_path / "Modelfile.fs"
    mf_fs.write_text((repo / "modelfiles/Modelfile.frankenstallm-3b-v2-train-tokenizer").read_text()
                     .replace("<PATH_TO_TRAIN_TOKENIZER_GGUF>", str(fs_gguf)))
    mf_j = tmp_path / "Modelfile.j"
    mf_j.write_text((repo / "modelfiles/Modelfile.jansori-3b").read_text().replace("<PATH_TO_GGUF>", str(j_gguf)))
    env = {**os.environ, "OLLAMA_HOST": host, "OLLAMA_MODELS": str(tmp_path / "models"), "CUDA_VISIBLE_DEVICES": "",
           "OLLAMA_DEBUG": "1", "OLLAMA_NO_CLOUD": "true"}
    log = tmp_path / "serve.log"
    proc = subprocess.Popen([ollama, "serve"], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                if requests.get(f"http://{host}/api/version", timeout=2).ok:
                    break
            except Exception:
                pass
            time.sleep(1)
        for name, mf in (("kb-fs", mf_fs), ("kb-jansori", mf_j)):
            subprocess.run([ollama, "create", name, "-f", str(mf)], env=env, check=True, capture_output=True)
        r = C.check_ollama(f"http://{host}", log, "kb-fs", "kb-jansori")
        for rows in r.values():
            assert all(x["equal"] and x["count_equal"] for x in rows)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
