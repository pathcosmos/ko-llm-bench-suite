"""kobench.standard: 레지스트리 검증 + EVAFRILL 로그우도 배치 계산 정확성."""

import math
import textwrap

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from kobench.standard import registry  # noqa: E402
from kobench.standard.evafrill_lm import EOS_ID, EvafrillLM  # noqa: E402


# ── 레지스트리 ────────────────────────────────────────────────────────────────

def test_repo_registry_loads_and_has_own_models():
    models = registry.load_models()
    own = {m for m, s in models.items() if s.owner == "self"}
    assert {"jansori-bf16", "frankenstallm-bf16", "evafrill-mo-3b"} <= own
    assert all(s.kind in ("base", "instruct") for s in models.values())
    suites = registry.load_suites()
    assert {"ko_mc_v1", "en_mc_v1", "smoke"} <= set(suites)


def test_registry_env_expansion_and_validation(tmp_path, monkeypatch):
    p = tmp_path / "m.yaml"
    p.write_text(textwrap.dedent("""
        models:
          a: {family: X, kind: base, backend: vllm, path: "${KB_TEST_PATH:-/fallback}", revision: abc, params_b: 1, active_b: 1, vllm_args: {enforce_eager: true}}
    """))
    assert registry.load_models(p)["a"].path == "/fallback"
    monkeypatch.setenv("KB_TEST_PATH", "/real")
    m = registry.load_models(p)["a"]
    assert m.path == "/real" and m.extra == {"vllm_args": {"enforce_eager": True}}
    p.write_text("models:\n  b: {family: X, kind: chat, backend: vllm, path: x, revision: null, params_b: 1, active_b: 1}\n")
    with pytest.raises(ValueError):
        registry.load_models(p)


def test_suite_rejects_duplicate_tasks(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("suites:\n  s: {tasks: [{task: a, num_fewshot: 0}, {task: a, num_fewshot: 5}]}\n")
    with pytest.raises(ValueError):
        registry.load_suites(p)


def test_pin_models_rewrites_only_target_line(tmp_path, monkeypatch):
    from kobench.standard import run

    p = tmp_path / "m.yaml"
    p.write_text(textwrap.dedent("""
        # 주석 보존
        models:
          a: {family: X, kind: base, backend: vllm, path: org/a, revision: null, params_b: 1, active_b: 1}
          b: {family: X, kind: base, backend: vllm, path: org/b, revision: null, params_b: 1, active_b: 1}
    """))

    class _Api:
        def model_info(self, repo):
            return type("I", (), {"sha": "f" * 40 if repo == "org/a" else "0" * 40})()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    models = registry.load_models(p)
    assert run.pin_models(["a"], models, p) == {"a": "f" * 40}
    again = registry.load_models(p)
    assert again["a"].revision == "f" * 40 and again["b"].revision is None
    assert "# 주석 보존" in p.read_text()


# ── EVAFRILL 로그우도 ─────────────────────────────────────────────────────────

class _ToyLM(torch.nn.Module):
    """다음 토큰 로짓이 (현재 토큰, 위치)의 결정적 함수인 인과 모델 — 패딩 영향이 있으면 결과가 달라진다."""

    def __init__(self, vocab=11):
        super().__init__()
        self.vocab = vocab

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T).expand(B, T)
        cums = ids.cumsum(1)  # 인과: 앞 토큰에만 의존
        v = torch.arange(self.vocab)
        logits = torch.sin(cums[..., None] * 0.37 + pos[..., None] * 0.11 + v * 0.53) * 3
        return logits.float(), None


def _make_lm(max_length=8, max_batch_tokens=16):
    lm = EvafrillLM.__new__(EvafrillLM)
    torch.nn.Module.__init__  # noqa: B018 (EvafrillLM은 Module이 아님)
    lm.model = _ToyLM()
    lm._device = torch.device("cpu")
    lm._max_length = max_length
    lm._batch_size = 4
    lm.max_batch_tokens = max_batch_tokens
    lm.doc_prefix_eos = True
    return lm


def _reference(model, ctx, cont, max_length):
    whole = (ctx + cont)[-(max_length + 1):]
    logits, _ = model(torch.tensor([whole[:-1]]))
    lp = torch.log_softmax(logits[0], -1)
    total, greedy = 0.0, True
    for i, t in enumerate(cont):
        pos = len(whole) - 1 - len(cont) + i
        total += lp[pos, t].item()
        greedy &= lp[pos].argmax().item() == t
    return total, greedy


def test_loglikelihood_matches_unbatched_reference():
    lm = _make_lm()
    reqs = [
        (None, [EOS_ID, 5, 3], [4]),
        (None, [EOS_ID, 1], [7, 8, 2]),
        (None, [EOS_ID, 9, 9, 9, 9, 9, 9, 9, 9], [1, 2]),  # max_length 초과 → 왼쪽 절단
        (None, [EOS_ID], [6]),
        (None, [EOS_ID, 4, 4, 4, 4], [3]),
    ]
    got = lm._loglikelihood_tokens(reqs, disable_tqdm=True)
    for (_, ctx, cont), (lp, g) in zip(reqs, got):
        ref_lp, ref_g = _reference(lm.model, ctx, cont, lm.max_length)
        assert math.isclose(lp, ref_lp, rel_tol=1e-5, abs_tol=1e-5)
        assert g == ref_g


def test_tok_encode_prefix_consistency():
    lm = _make_lm()

    class _Tok:
        def encode(self, s):
            class E:  # noqa: D401
                ids = [ord(c) % 7 + 3 for c in s]
            return E()

    lm.tokenizer_obj = _Tok()
    assert lm.tok_encode("ab")[0] == EOS_ID
    assert lm.tok_encode("ab", add_special_tokens=False)[0] != EOS_ID or lm.tok_encode("ab", add_special_tokens=False) == []
    ctx, cont = lm._encode_pair("가나", "다")
    assert ctx[0] == EOS_ID and EOS_ID not in cont


# ── 집계 ─────────────────────────────────────────────────────────────────────

def test_bootstrap_and_paired_diff():
    import numpy as np

    from kobench.standard.report import bootstrap_ci, paired_diff

    x = np.array([1.0] * 70 + [0.0] * 30)
    lo, hi = bootstrap_ci(x)
    assert lo < 0.7 < hi and 0.55 < lo and hi < 0.85
    a = {("t", i): 1.0 for i in range(10)}
    b = {("t", i): (1.0 if i < 5 else 0.0) for i in range(10)}
    b[("t", 99)] = 1.0  # 짝이 없는 문항은 제외돼야 한다
    d, lo, hi, n = paired_diff(a, b)
    assert n == 10 and abs(d - 0.5) < 1e-9 and lo <= 0.5 <= hi
    assert paired_diff(a, a)[:3] == (0.0, 0.0, 0.0)


def test_bpb_ratio_micro_ci_and_paired():
    import math

    from kobench.standard.report import _item_scores, bootstrap_ci, micro, paired_diff

    # lm-eval bits_per_byte = −Σloglik / ln2 / Σbytes (문서 평균이 아니라 코퍼스 비율)
    items = {("w", 0): {"bits_per_byte": [-100.0, 200]}, ("w", 1): {"bits_per_byte": [-10.0, 10]}}
    sc = _item_scores(items, "bits_per_byte")
    want = 110.0 / math.log(2) / 210
    assert abs(micro(sc) - want) < 1e-12
    lo, hi = bootstrap_ci(list(sc.values()))
    assert lo <= want <= hi
    worse = {k: (n * 1.1, d) for k, (n, d) in sc.items()}
    d, lo, hi, n = paired_diff(worse, sc)
    assert n == 2 and abs(d - want * 0.1) < 1e-12 and lo <= d <= hi


def test_variant_from_name_and_chosen(tmp_path):
    from kobench.standard.report import chosen_variants
    from kobench.standard.run import Variant

    for text in ["A", "docstart=eos", "chat=1", "docstart=eos,chat=1,shots=0", "shots=3"]:
        v = Variant.parse(text)
        assert Variant.from_name(v.name) == v
    f = tmp_path / "best.yaml"
    f.write_text("models:\n  m1: {chosen: chat}\n  m2: {chosen: A}\n", encoding="utf-8")
    ch = chosen_variants(f)
    assert ch["m1"] == Variant(chat=True) and ch["m2"] == Variant()


# ── 설정 변형 ────────────────────────────────────────────────────────────────

def test_variant_parse_name_and_paths():
    from kobench.standard.run import Variant, _out_paths

    assert Variant.parse(None).name == "A" and Variant.parse("A") == Variant()
    v = Variant.parse("docstart=eos,chat=1,shots=0")
    assert (v.doc_start, v.chat, v.shots) == ("eos", True, 0)
    assert v.name == "docstart_eos+chat+shots0"
    a_json, _ = _out_paths("m", "kobest", 5, None)
    v_json, _ = _out_paths("m", "kobest", 0, None, v)
    assert a_json.parent.name == "m" and v_json.parent.name == "v_docstart_eos+chat+shots0"
    with pytest.raises(ValueError):
        Variant.parse("unknown=1")


def test_dev_tasks_registered():
    pytest.importorskip("lm_eval")
    from kobench.standard.run import task_manager

    names = set(task_manager().all_tasks)
    assert {"kobest_dev", "kmmlu_dev", "global_mmlu_ko_dev", "kobest_boolq_dev"} <= names


def test_doc_start_eos_tokenizer_variant():
    from kobench.standard import formats

    try:
        d = formats.prepare_tokenizer_dir("frankenstallm-bf16", doc_start="eos")
    except FileNotFoundError:
        pytest.skip("학습 토크나이저 없음")
    from tokenizers import Tokenizer

    t = Tokenizer.from_file(str(d / "tokenizer.json"))
    plain = formats.encode_text("frankenstallm-bf16", "안녕 세상")
    assert t.encode("안녕 세상").ids == [2] + plain
    assert formats.model_format("evafrill-mo-3b-orpo")["tokenizer"] == "korean_sp_unigram64k"


def test_custom_mc_task_helpers():
    import importlib.util

    from kobench.standard.tasks import TASKS_DIR

    spec = importlib.util.spec_from_file_location("kb_custom_utils", TASKS_DIR / "custom" / "utils.py")
    u = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u)
    doc = {"question": " 질문? ", "options": "['가', '나', '다', '라', '마']", "solution": "5"}
    assert u.kmmlu_x_text(doc) == "질문?\nA. 가\nB. 나\nC. 다\nD. 라\nE. 마\n정답："
    assert u.kmmlu_x_choices(doc) == list("ABCDE") and u.kmmlu_x_target(doc) == 4
    assert u.kobalt_target({"Answer": "J"}) == 9


def test_hrm8k_strict_scoring_separates_format_failures():
    pytest.importorskip("lm_eval")
    import importlib.util

    from kobench.standard.tasks import TASKS_DIR

    spec = importlib.util.spec_from_file_location("kb_gen_utils", TASKS_DIR / "gen" / "utils.py")
    u = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u)
    doc = {"answer": "18"}
    assert u.hrm8k_process_results(doc, ["따라서 $\\boxed{18}$"]) == {"exact_match_strict": 1, "exact_match": 1, "format_fail": 0}
    # 원본 규칙은 마지막 숫자를 답으로 봐서 맞다고 하지만, 엄격 채점은 형식 실패로 분리한다
    assert u.hrm8k_process_results(doc, ["답은 18"]) == {"exact_match_strict": 0, "exact_match": 1, "format_fail": 1}
    assert u.hrm8k_process_results(doc, ["\\boxed{7}"]) == {"exact_match_strict": 0, "exact_match": 0, "format_fail": 0}


def test_relax_head_dim_check_only_for_explicit_head_dim():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig

    from kobench.standard.run import relax_head_dim_check

    relax_head_dim_check()
    assert LlamaConfig(hidden_size=1792, num_attention_heads=24, num_key_value_heads=8, head_dim=128).head_dim == 128
    with pytest.raises(Exception):
        LlamaConfig(hidden_size=1792, num_attention_heads=24)   # head_dim 미지정이면 여전히 거부
