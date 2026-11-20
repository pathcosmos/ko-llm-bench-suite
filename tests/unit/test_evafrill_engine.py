"""EVAFRILL 생성 경로 테스트.

- CPU: EvafrillLM.generate_until의 요청 묶음·정지 문자열·EOS 처리·순서 보존 (가짜 엔진)
- GPU (KOBENCH_GPU=1 + 모델 파일): 엔진 prefill이 원본 forward와 비트 단위로 같은지, CUDA graph = eager,
  배치 결과 = 배치 1, generate_until = 엔진 단독 실행
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from kobench.standard.evafrill_lm import EOS_ID, EvafrillLM  # noqa: E402


class _FakeEngine:
    """프롬프트 토큰 합으로 정해지는 결정적 출력."""

    max_len = 10_000

    def __init__(self):
        self.calls = []

    def generate(self, prompts, max_new_tokens, eos_ids, stop_strings, decode_fn, temperature, top_p, seed):
        self.calls.append({"n": len(prompts), "max_new": max_new_tokens, "until": stop_strings, "temp": temperature})
        out = []
        for p in prompts:
            k = sum(p) % 3
            toks = [10 + k, 11, 12, EOS_ID, 99] if k == 0 else [20 + k, 21, 22, 23][:max_new_tokens]
            out.append(SimpleNamespace(tokens=toks))
        return out


def _fake_lm():
    lm = EvafrillLM.__new__(EvafrillLM)
    lm._max_length = 64
    lm._gen_batch_size = 2
    lm.doc_prefix_eos = False
    lm._engine = _FakeEngine()
    lm._get_engine = lambda max_len: lm._engine
    vocab = {10: "가", 11: "나", 12: "다", 20: "A", 21: "B", 22: "<STOP>", 23: "C", 22 + 1: "C"}
    lm.tok_decode = lambda toks: "".join(vocab.get(t, "?") for t in toks)
    lm.tok_encode = lambda s, add_special_tokens=None: [ord(c) % 7 for c in s]
    return lm


def _req(ctx, **kw):
    return SimpleNamespace(args=(ctx, kw))


def test_generate_until_groups_orders_and_truncates():
    lm = _fake_lm()
    reqs = [_req("aaa", until=["<STOP>"], max_gen_toks=8), _req("b", until=["<STOP>"], max_gen_toks=8),
            _req("cc", until=[], max_gen_toks=3, do_sample=True, temperature=0.7), _req("dddd", until=["<STOP>"], max_gen_toks=8)]
    out = lm.generate_until(reqs, disable_tqdm=True)
    assert len(out) == 4
    for r, text in zip(reqs, out):
        ctx, kw = r.args
        k = sum(lm.tok_encode(ctx)) % 3
        toks = [10 + k, 11, 12, EOS_ID, 99] if k == 0 else [20 + k, 21, 22, 23][: kw["max_gen_toks"]]
        expected = lm.postprocess_generation(toks, kw["until"])
        assert text == expected
        assert "?" not in text  # EOS 이후(99)는 버려진다
        if "<STOP>" in kw["until"]:
            assert "<STOP>" not in text
    # 같은 gen_kwargs끼리 묶이고, 배치 크기 2로 나뉜다; 샘플링 요청만 temperature>0
    temps = sorted(c["temp"] for c in lm._engine.calls)
    assert temps[-1] == 0.7 and temps.count(0.0) == len(temps) - 1
    assert all(c["n"] <= 2 for c in lm._engine.calls)


def test_postprocess_stops_at_eos_then_until():
    lm = _fake_lm()
    assert lm.postprocess_generation([10, 11, EOS_ID, 12], []) == "가나"
    assert lm.postprocess_generation([20, 22, 23], ["<STOP>"]) == "A"


# ── GPU ──────────────────────────────────────────────────────────────────────
gpu = pytest.mark.skipif(
    os.environ.get("KOBENCH_GPU") != "1" or not torch.cuda.is_available(),
    reason="GPU 테스트는 KOBENCH_GPU=1일 때만 (모델 3B 로드)")


@pytest.fixture(scope="module")
def slerp():
    """실제 EVAFRILL 소스로 로드한다. tests/conftest.py가 'model.*' 모듈을 MagicMock으로 바꿔 두므로
    로드하는 동안 치웠다가 모듈 종료 시 되돌린다. 모델 파일이 없을 때만 건너뛴다 (로드 오류는 숨기지 않는다)."""
    import sys
    from unittest.mock import MagicMock

    snap = Path(os.path.expandvars("${HF_HOME}/hub/models--pathcosmos--EVAFRILL-Mo-3B/snapshots/"
                                   "e95edece955af9df9a6215210db594a9984fddf5/slerp/model.safetensors"))
    if not snap.exists():
        pytest.skip("EVAFRILL slerp 체크포인트 없음")
    mocked = {k: sys.modules.pop(k) for k in list(sys.modules)
              if (k == "model" or k.startswith("model.")) and isinstance(sys.modules[k], MagicMock)}
    from kobench.standard.evafrill_verify import load

    m, _ = load("slerp", torch.bfloat16)
    yield m
    for k in [k for k in sys.modules if k == "model" or k.startswith("model.")]:
        sys.modules.pop(k)
    sys.modules.update(mocked)


@gpu
def test_engine_prefill_bitwise_and_graph_and_batch(slerp):
    from kobench.standard import formats
    from kobench.standard.evafrill_engine import EvafrillEngine

    eng = EvafrillEngine(slerp, max_batch=4, max_len=1024)
    p = formats.encode_prompt("evafrill-mo-3b", [{"role": "user", "content": "한국의 수도는 어디인가요?"}])
    E = eng.prefill(p, 0, all_logits=True)
    B, _ = slerp(torch.tensor([p], device="cuda"))
    assert torch.equal(E, B[0])
    g = eng.generate([p], 24, eos_ids=(), use_graph=True)[0].tokens
    e = eng.generate([p], 24, eos_ids=(), use_graph=False)[0].tokens
    assert g == e
    q = formats.encode_prompt("evafrill-mo-3b", [{"role": "user", "content": "Explain photosynthesis briefly."}])
    singles = [eng.generate([x], 12, eos_ids=())[0].tokens for x in (p, q, p[:5])]
    batch = [r.tokens for r in eng.generate([p, q, p[:5]], 12, eos_ids=())]
    assert batch == singles


@gpu
def test_generate_until_matches_engine(slerp):
    from kobench.standard import formats
    from kobench.standard.evafrill_engine import EvafrillEngine

    lm = EvafrillLM.__new__(EvafrillLM)
    lm.model, lm._max_length, lm._gen_batch_size, lm.doc_prefix_eos = slerp, 4096, 4, False
    lm.tokenizer_obj = formats.training_tokenizer("evafrill-mo-3b")
    lm._engine = EvafrillEngine(slerp, max_batch=4, max_len=1024)
    prompt = lm.apply_chat_template([{"role": "user", "content": "좋은 수면 습관을 알려 주세요."}])
    assert prompt == formats.build_prompt("evafrill-mo-3b", [{"role": "user", "content": "좋은 수면 습관을 알려 주세요."}])
    text = lm.generate_until([_req(prompt, until=["\n\n"], max_gen_toks=40)], disable_tqdm=True)[0]
    ids = lm.tok_encode(prompt, add_special_tokens=False)
    assert ids == formats.encode_prompt("evafrill-mo-3b", [{"role": "user", "content": "좋은 수면 습관을 알려 주세요."}])
    r = lm._engine.generate([ids], 40, eos_ids=(EOS_ID,), stop_strings=["\n\n"], decode_fn=lm.tok_decode)[0]
    assert text == lm.postprocess_generation(r.tokens, ["\n\n"])
