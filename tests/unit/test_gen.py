"""생성 벤치(kobench.standard.gen*), 코드 격리 실행, judge 파싱, GGUF 로그우도 묶음, 엔진 repetition_penalty."""

import json
import math

import pytest

pytest.importorskip("lm_eval")
np = pytest.importorskip("numpy")

from kobench.standard import code_exec, gen, gen_data, gen_report, gen_score, judge  # noqa: E402


# ── 데이터 도우미 ─────────────────────────────────────────────────────────────

def test_rotate_is_deterministic_and_keeps_answer():
    opts = ["a", "b", "c", "알 수 없음"]
    o1, a1 = gen_data.rotate(opts, 3, "k1")
    o2, a2 = gen_data.rotate(opts, 3, "k1")
    assert (o1, a1) == (o2, a2) and o1[a1] == "알 수 없음" and sorted(o1) == sorted(opts)
    positions = {gen_data.rotate(opts, 3, f"k{i}")[1] for i in range(64)}
    assert positions == {0, 1, 2, 3}                       # 해시 회전으로 정답 위치가 퍼진다


def test_mc_prompt_letters():
    p = gen_data.mc_prompt("질문", ["x", "y", "z"])
    assert "A. x" in p and "C. z" in p and "정답: A/B/C" in p


def test_gen_config_tasks_have_loaders_and_datasets():
    cfg = gen_data.load_gen_config()
    for name, t in cfg["tasks"].items():
        assert name in gen_data.LOADERS
        if t["scorer"] == "scenario":                      # 절차 생성 (데이터셋 없음)
            assert t["dataset"] is None and t["generator"] == "kobench.standard.scenario"
            continue
        assert t["dataset"] in cfg["datasets"] and len(cfg["datasets"][t["dataset"]]["revision"]) == 40
        assert t["scorer"] in ("rule", "judge", "code")
    for s in cfg["suites"].values():
        assert set(s) <= set(cfg["tasks"])


# ── 정답 추출·채점 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,n,exp", [
    ("정답: B", 4, "B"), ("따라서 정답은 C입니다.", 4, "C"), ("**정답**: (D)", 4, "D"), ("정답은 **B**", 4, "B"),
    ("A", 4, "A"), ("(B)", 4, "B"), ("C. 보건소장에게 신고", 5, "C"), ("\\boxed{A}", 4, "A"), ("The answer is C", 4, "C"),
    ("정답: A\n다시 보니 정답: B", 4, "B"),
    ("A. 첫째<unk>B. 둘째<unk>C. 셋째", 5, None),           # 선택지 나열은 답이 아니다
    ("정답을 고르기 어렵다", 4, None), ("정답: E", 4, None), ("Answer: Banana", 4, None),
])
def test_extract_choice(text, n, exp):
    assert gen_score.extract_choice(text, n) == exp


def test_apply_policy():
    assert gen_score.apply_policy("a<unk>b", "unk_nl") == "a\nb"
    assert gen_score.apply_policy("a<unk>b", "unk_drop") == "ab"
    assert gen_score.apply_policy("a<unk>b", "raw") == "a<unk>b"


def test_math_and_mc_scoring():
    it = {"gold": "18", "meta": {}}
    assert gen_score._score_math(it, "답은 $\\boxed{18}$") == {"exact_match_strict": 1, "exact_match": 1, "format_fail": 0}
    r = gen_score._score_math(it, "답은 18")             # 원본 규칙(마지막 숫자)은 정답, 엄격 규칙은 형식 실패
    assert r["exact_match"] == 1 and r["exact_match_strict"] == 0 and r["format_fail"] == 1
    mc = {"gold": "B", "meta": {"n_options": 3, "condition": "amb", "polarity": "bsd", "biased": "A", "unknown": "B"}}
    r = gen_score._score_mc(mc, "정답: A")
    assert r["acc"] == 0 and r["is_biased"] == 1 and r["is_unknown"] == 0


def test_kobbq_aggregate_bias_scores():
    rows = [
        {"acc": 0, "format_fail": 0, "pred": "A", "condition": "amb", "polarity": "bsd", "is_biased": 1, "is_unknown": 0},
        {"acc": 1, "format_fail": 0, "pred": "C", "condition": "amb", "polarity": "cnt", "is_biased": 0, "is_unknown": 1},
        {"acc": 1, "format_fail": 0, "pred": "A", "condition": "dis", "polarity": "bsd", "is_biased": 1, "is_unknown": 0},
        {"acc": 0, "format_fail": 0, "pred": "B", "condition": "dis", "polarity": "cnt", "is_biased": 0, "is_unknown": 0},
    ]
    m = gen_score.aggregate("kobbq", rows)
    assert m["acc_amb"] == 0.5 and m["acc_dis"] == 0.5
    assert m["diff_bias_amb"] == 0.5          # (편향 1 − 반편향 0) / 모호 2
    assert m["diff_bias_dis"] == 1.0          # bsd 정확도 1 − cnt 정확도 0


def test_split_reasoning():
    assert gen.split_reasoning("<think>생각</think>\n답") == ("답", "생각")
    assert gen.split_reasoning("생각만</think>답") == ("답", "생각만")
    assert gen.split_reasoning("<think>잘린 생각") == ("", "잘린 생각")
    assert gen.split_reasoning("그냥 답") == ("그냥 답", "")


def test_read_jsonl_drops_truncated_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"id": "1"}) + "\n" + '{"id": "2", "outp')
    assert [r["id"] for r in gen.read_jsonl(p)] == ["1"]


# ── 코드 격리 실행 ───────────────────────────────────────────────────────────

def test_extract_code_prefers_python_block_with_def():
    t = "설명\n```\nprint(1)\n```\n```python\ndef f():\n    return 1\n```"
    assert code_exec.extract_code(t).strip().startswith("def f")
    assert "def g" in code_exec.extract_code("```python\ndef g():\n    pass")      # 닫히지 않은 블록


def test_build_program_humaneval_styles():
    gold = {"prompt": "from typing import List\n\ndef f(x: List[int]) -> int:\n    \"\"\"doc\"\"\"\n", "raw": "from typing import List\n\ndef f(x: List[int]) -> int:\n    \"\"\"doc\"\"\"",
            "test": "def check(c):\n    assert c([1, 2]) == 3\n", "entry_point": "f"}
    full = code_exec.build_program(gold, "```python\ndef f(x: List[int]) -> int:\n    return sum(x)\n```", base=False)
    assert full.startswith("from typing import List") and full.rstrip().endswith("check(f)")
    body = code_exec.build_program(gold, "```python\n    return sum(x)\n```", base=False)
    assert "def f" in body
    base = code_exec.build_program(gold, "\n    return sum(x)\n", base=True)
    assert base.startswith(gold["raw"])
    for prog in (full, body, base):
        assert code_exec.run_program(prog, timeout=20).passed


@pytest.mark.parametrize("code,status", [
    ("import socket; socket.socket()", "error"),
    ("open('/tmp/kobench_sandbox_probe', 'w')", "error"),
    ("import subprocess; subprocess.run(['true'])", "error"),
    ("import os; os.system('true')", "error"),
    ("while True: pass", "timeout"),
    ("open('inside.txt', 'w').write('ok')", "pass"),
])
def test_sandbox_blocks(code, status):
    r = code_exec.run_program(code, timeout=3)
    assert r.status == status, r.detail


# ── judge ───────────────────────────────────────────────────────────────────

def test_judge_parse():
    assert judge.parse("komt", "설명... Rating: [[7]]") == 7.0
    assert judge.parse("komt", "Rating: [[11]]") is None
    assert judge.parse("logickor", "평가 [[0]]") == 0.0
    assert judge.parse("ko_arena_hard", "My final verdict is: [[B>A]]") == "B>A"
    assert judge.parse("kosimpleqa", "A") == "CORRECT"
    assert judge.parse("kosimpleqa", "<think>x</think>C") == "NOT_ATTEMPTED"


def test_arena_requests_swap_positions():
    items = [{"id": "q1", "turns": ["질문"], "meta": {}}]
    rows = {"q1": {"outputs": ["모델 답"]}}
    base = {"q1": {"outputs": ["기준 답"]}}
    reqs = judge.build_requests("ko_arena_hard", "qwen3-4b-2507", "A", rows, items, base)
    assert [r["key"].split("|")[-1] for r in reqs] == ["g1", "g2"]
    u1, u2 = reqs[0]["messages"][1]["content"], reqs[1]["messages"][1]["content"]
    assert u1.index("기준 답") < u1.index("모델 답") and u2.index("모델 답") < u2.index("기준 답")


def test_komt_uses_reference_only_for_ref_categories():
    it = {"id": "1", "turns": ["q1", "q2"], "meta": {"category": "math", "reference": ["r1", "r2"]}}
    rows = {"1": {"outputs": ["a1", "a2"]}}
    reqs = judge.build_requests("komt", "qwen3-4b-2507", "A", rows, [it], None)
    assert "Reference" in reqs[0]["messages"][1]["content"] and "r2" in reqs[1]["messages"][1]["content"]
    it["meta"]["category"] = "writing"
    reqs = judge.build_requests("komt", "qwen3-4b-2507", "A", rows, [it], None)
    assert "Reference" not in reqs[0]["messages"][1]["content"]


# ── 집계 ────────────────────────────────────────────────────────────────────

def test_paired_bootstrap_sign():
    a = {str(i): 1.0 for i in range(50)}
    b = {str(i): 0.0 for i in range(50)}
    d, lo, hi = gen_report.paired(a, b)
    assert d == 1.0 and lo == 1.0 and hi == 1.0
    lo, hi = gen_report.boot_ci(np.array([0.0, 1.0] * 50))
    assert lo < 0.5 < hi


# ── GGUF 로그우도 묶음 ───────────────────────────────────────────────────────

def test_llamacpp_groups_shared_context(monkeypatch):
    from kobench.standard import llamacpp_lm

    lm = llamacpp_lm.LlamaCppLM.__new__(llamacpp_lm.LlamaCppLM)
    lm._max_length = 8
    lm.n_ctx = 65536
    captured = {}

    def fake_run(groups):
        captured["groups"] = groups
        return [{"ll": [-float(len(c)) for c in conts], "greedy": [1] * len(conts)} for _, conts in groups]

    lm._run = fake_run
    reqs = [(None, [1, 2, 3], [4]), (None, [1, 2, 3], [5, 6]), (None, [9], [7]), (None, [1, 2, 3, 4, 5, 6, 7], [8, 8])]
    out = lm._loglikelihood_tokens(reqs)
    g = captured["groups"]
    assert g[0] == ([1, 2, 3], [[4], [5, 6]]) and g[1] == ([9], [[7]])
    assert g[2][0] == [3, 4, 5, 6, 7] and g[2][1] == [[8, 8]]          # (문맥+연속)[-(max_length-1):] 자르기
    assert out == [(-1.0, True), (-2.0, True), (-1.0, True), (-2.0, True)]
    # 같은 문맥이 많아도 묶음당 연속 32개까지
    lm.n_ctx = 1 << 20
    lm._loglikelihood_tokens([(None, [1], [i]) for i in range(70)])
    assert [len(c) for _, c in captured["groups"]] == [32, 32, 6]


# ── EVAFRILL 엔진 repetition_penalty ─────────────────────────────────────────

def test_engine_penalize_matches_hf_rule():
    torch = pytest.importorskip("torch")
    from kobench.standard.evafrill_engine import EvafrillEngine

    logits = torch.tensor([[2.0, -2.0, 1.0, 0.5]])
    seen = torch.tensor([[True, True, False, False]])
    out = EvafrillEngine._penalize(logits, seen, 2.0)
    assert torch.allclose(out, torch.tensor([[1.0, -4.0, 1.0, 0.5]]))
    assert math.isclose(EvafrillEngine._penalize(logits, seen, 1.0)[0, 0].item(), 2.0)


# ── 성능 측정 도우미 ─────────────────────────────────────────────────────────

def test_perf_parse_engine_log():
    from kobench.standard import perf

    vllm_log = ("Model loading took 5.62 GiB memory and 3.1 seconds\nAvailable KV cache memory: 112.22 GiB\n"
                "GPU KV cache size: 1,050,608 tokens\nMaximum concurrency for 4,096 tokens per request: 256.50x\n")
    d = perf.parse_engine_log(vllm_log)
    assert d == {"weights_gib": 5.62, "kv_cache_gib": 112.22, "kv_cache_tokens": 1050608, "max_concurrency_at_max_len": 256.5}
    ll = "load_tensors:   CUDA0 model buffer size =  3264.00 MiB\nllama_kv_cache:      CUDA0 KV buffer size =  1024.00 MiB\n"
    d = perf.parse_engine_log(ll)
    assert d["weights_gib"] == round(3264 / 1024, 2) and d["kv_cache_gib"] == 1.0


def test_perf_server_commands():
    from kobench.standard import perf
    from kobench.standard.registry import load_models

    m = load_models()
    cmd = perf.vllm_cmd(m["qwen3-4b-2507"], 8000, False, 4096, 0.85)
    assert "--no-enable-prefix-caching" in cmd and "--served-model-name" in cmd and "--revision" in cmd
    cmd = perf.vllm_cmd(m["kanana-1.5-2.1b"], 8000, True, 4096, 0.85)
    assert "kobench.standard.perf_vllm_entry" in cmd and cmd[cmd.index("kobench.standard.perf_vllm_entry") + 1] == "serve"
    cmd = perf.llamacpp_cmd(m["jansori-gguf-q8_0"], 8000, False, 4096)
    assert "--no-cache-prompt" in cmd and cmd[cmd.index("-c") + 1] == str(4096 * 64)
    # 2026-11-21: 호스트 RAM 프롬프트 캐시는 on/off 모두 끈다(vLLM과 같은 조건)
    assert cmd[cmd.index("--cache-ram") + 1] == "0"
    on = perf.llamacpp_cmd(m["jansori-gguf-q8_0"], 8000, True, 4096)
    assert "--no-cache-prompt" not in on and on[on.index("--cache-ram") + 1] == "0"


def test_perf_llamacpp_request_sends_cache_prompt_flag():
    # 2026-11-21: cache_prompt가 True로 고정되어 prefix off가 실제로 꺼지지 않았다
    import asyncio

    from kobench.standard import perf

    sent = []

    class Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @property
        def content(self):
            async def gen():
                yield b'data: {"content": "x", "stop": true}'
            return gen()

    class Sess:
        def post(self, url, json):
            sent.append(json)
            return Resp()

    for flag in (True, False):
        asyncio.run(perf.one_request(Sess(), "http://h", "llamacpp", [1, 2], 4, cache_prompt=flag))
    assert [b["cache_prompt"] for b in sent] == [True, False]


def test_perf_vllm_entry_guards_main():
    # spawn 하위 프로세스가 이 모듈을 다시 import해도 서버를 띄우지 않아야 한다 (완화만 적용)
    import importlib

    import kobench.standard.perf_vllm_entry as e

    importlib.reload(e)
    src = open(e.__file__, encoding="utf-8").read()
    assert 'if __name__ == "__main__":' in src and src.index("relax_head_dim_check()") < src.index('if __name__ == "__main__":')


# ── 반복 지표·선택 규칙 v2 ───────────────────────────────────────────────────

def test_loop_and_rep3():
    assert gen_score.is_loop("시작 " + "같은 문장이 계속 반복되고 있습니다 정말로. " * 5) == 1
    assert gen_score.is_loop("서로 다른 내용의 문장들로 이루어진 정상적인 답변입니다. 끝.") == 0
    assert gen_score.rep3("a b c " * 10) > 0.8 and gen_score.rep3("짧은 답") is None


def test_choose_rule_v2():
    c = {"A": {"mean": 0.30, "loop": 0.09}, "card": {"mean": 0.28, "loop": 0.01}, "rep": {"mean": 0.29, "loop": 0.02}}
    ch, gain, ex = gen_report.choose(c, "A")
    assert ch == "rep" and ex == ["A"] and abs(gain + 0.01) < 1e-9          # A는 루프 기준으로 제외
    c = {"A": {"mean": 0.300, "loop": 0.0}, "card": {"mean": 0.3005, "loop": 0.0}}
    assert gen_report.choose(c, "A")[0] == "A"                               # 차이 ≤ 0.001 → A
    c = {"A": {"mean": 0.3, "loop": 0.2}, "card": {"mean": 0.1, "loop": 0.1}}
    assert gen_report.choose(c, "A")[0] == "card"                            # 모두 넘으면 루프율 최저


def test_rep_variant_settings():
    from kobench.standard.registry import load_models

    m = load_models()
    st = gen.settings_for(m["frankenstallm-bf16"], "rep")
    assert st.sampling == {"temperature": 0.0, "repetition_penalty": 1.2}
    assert gen.settings_for(m["qwen2.5-3b-instruct"], "rep").sampling["repetition_penalty"] == 1.05
    assert gen.settings_for(m["qwen3-4b-2507"], "rep") is None                # 카드에 반복 패널티 없음


# ── 다턴 시나리오 ────────────────────────────────────────────────────────────

def test_scenario_build_deterministic_and_disjoint():
    from kobench.standard import scenario as S

    a, b = S.build("dev", "self"), S.build("dev", "self")
    assert [x["turns"] for x in a] == [x["turns"] for x in b]                 # 같은 시드 = 같은 문항
    t = S.build("test", "self")
    assert len(a) == 100 and len(t) == 300 and not ({x["id"] for x in a} & {x["id"] for x in t})
    assert {x["meta"]["category"] for x in t} == set(S.GENERATORS)
    o = S.build("dev", "oracle")
    assert o[0]["oracle"] == o[0]["gold"]["oracle"] and a[0]["oracle"] is None


def test_scenario_oracle_passes_and_controls_fail():
    from kobench.standard import scenario as S

    items = S.build("dev", "self")
    assert all(S.score_item(it, it["gold"]["oracle"])["pass"] for it in items)          # 참조 답은 모두 통과
    assert all(S.score_item(it, [""] * len(it["turns"]))["turn_pass"] == 0 for it in items)  # 빈 답은 모두 실패
    # 첫 턴 답을 모든 턴에 복사하면 어떤 범주도 시나리오 통과가 없다
    assert not any(S.score_item(it, [it["gold"]["oracle"][0]] * len(it["turns"]))["pass"] for it in items)


def test_scenario_checks():
    from kobench.standard import scenario as S

    assert S.check({"type": "num", "value": 76}, "계산하면 76입니다.\n답: 76개", {})
    assert not S.check({"type": "num", "value": 76}, "76입니다.", {})                     # '답:' 줄 필수
    assert S.check({"type": "answer_seq", "values": ["가", "나", "다"]}, "답: 가, 나, 다", {})
    assert not S.check({"type": "answer_seq", "values": ["가", "나", "다"]}, "답: 나, 가, 다", {})
    assert S.check({"type": "polite", "polite": True}, "좋습니다. 감사합니다.", {})
    assert S.check({"type": "polite", "polite": False}, "좋아. 고마워.", {})
    assert S.check({"type": "sentences", "min": 2, "max": 2}, "첫 문장입니다. 둘째 문장입니다. [끝]", {})
    assert S.check({"type": "json_records", "records": [{"name": "김민준", "age": 30, "city": "서울"}]},
                   '```json\n[{"name": "김민준", "age": 30, "city": "서울"}]\n```', {})
    assert S.check({"type": "overlap_prev", "max": 0.3}, "전혀 새로운 내용으로 이어지는 문장입니다 정말 다릅니다", {"prev": ["앞 내용은 이것과 다릅니다 확실히 그렇습니다"]})
    assert S.check({"type": "uncertain"}, "그런 박물관에 대한 정보를 찾을 수 없습니다.", {})


def test_scenario_quoted_polite_and_no_new_year():
    from kobench.standard import scenario as S

    wrapped = '물론이죠! 반말로 바꿔 보면 이렇게요:\n"그 말 듣고 어이없었어. 드디어 끝났어. 진짜 희한했어."\n자연스럽죠?'
    assert S.check({"type": "polite", "polite": False, "quoted": True}, wrapped, {})
    assert not S.check({"type": "polite", "polite": False}, wrapped, {})               # 인용 판정이 없으면 설명 문장 때문에 실패
    assert S.check({"type": "no_new_year", "allowed": ["2013"]}, "2013년 수상 기록은 없고 2010년대 자료도 없습니다.", {})
    assert not S.check({"type": "no_new_year", "allowed": []}, "이 박물관은 2005년에 개관했습니다.", {})


# ── 규칙 v3: 운용 설정 ───────────────────────────────────────────────────────

def test_variant_grammar_v3():
    from kobench.standard.registry import load_models

    m = load_models()
    st = gen.settings_for(m["frankenstallm-bf16"], "rp11+sys")
    assert st.sampling == {"temperature": 0.0, "repetition_penalty": 1.1} and st.system == gen.KO_SYSTEM
    assert gen.settings_for(m["frankenstallm-bf16"], "A+orpo").template == "orpo_raw_bos"
    assert gen.settings_for(m["qwen3-4b-2507"], "A+orpo") is None          # ORPO 형식은 frankenstallm 전용
    assert gen.settings_for(m["evafrill-mo-3b"], "fp03") is None            # 엔진은 frequency_penalty 미지원
    with pytest.raises(ValueError):
        gen.settings_for(m["qwen3-4b-2507"], "A+bogus")


def test_user_encode_system_and_orpo_fallback():
    from kobench.standard import formats

    sysmsg = [{"role": "system", "content": "지시"}, {"role": "user", "content": "질문"}]
    a = gen.user_encode("frankenstallm-bf16", sysmsg, None)
    b = formats.encode_prompt("frankenstallm-bf16", sysmsg, system="prepend_to_first_user")
    assert a == b
    multi = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
    assert gen.user_encode("frankenstallm-bf16", multi, "orpo_raw_bos") == formats.encode_prompt("frankenstallm-bf16", multi)


def test_malfunction_detectors():
    m = gen_score.malfunction
    assert m("정상적인 한국어 답변입니다.", "stop") == {"loop": 0, "lang": 0, "leak": 0, "empty": 0, "trunc": 0}
    assert m("관광, giải trí, sử dụng 등입니다", "stop")["lang"] == 1
    assert m("```python\nprint('hello world english only code')\n```", "stop")["lang"] == 0
    assert m("답입니다.\n<|user|>\n다음", "length") == {"loop": 0, "lang": 0, "leak": 1, "empty": 0, "trunc": 1}
    assert m("", "stop")["empty"] == 1


def test_choose_v3():
    ok = {"loop": 0.0, "lang": 0.0, "leak": 0.0, "empty": 0.0, "trunc": 0.0}
    bad = dict(ok, loop=0.10)
    c = {"A|raw": {"mean": 0.50, "malfunction": bad}, "rp11|raw": {"mean": 0.48, "malfunction": ok},
         "card|raw": {"mean": 0.47, "malfunction": ok}}
    assert gen_report.choose_v3(c, "A|raw")[0] == "rp11|raw"                # A는 루프 제약 위반
    c = {"A|raw": {"mean": 0.30, "malfunction": dict(ok, loop=0.30)}, "rp12|raw": {"mean": 0.10, "malfunction": dict(ok, loop=0.05)}}
    assert gen_report.choose_v3(c, "A|raw")[0] == "rp12|raw"                # 모두 위반 → 위반 최소
    c = {"A|raw": {"mean": 0.5000, "malfunction": ok}, "card|raw": {"mean": 0.5005, "malfunction": ok}}
    assert gen_report.choose_v3(c, "A|raw")[0] == "A|raw"                   # 차이 ≤ 0.001 → 기준


def test_tune_rows_summarizes_A_and_chosen():
    from kobench.standard.gen_report import tune_rows
    d = lambda q, loop: {"mean": q, "tasks": {"t": q}, "malfunction": {"loop": loop, "lang": 0, "leak": 0, "empty": 0, "trunc": 0}}
    best = {"gen": {"m": {"chosen": "rp11", "unk_policy": "raw",
                          "stage1": {"dev": {"A|raw": d(0.5, 0.1), "rp11|raw": d(0.52, 0.01)}, "feasible": ["rp11|raw"]},
                          "stage2": {"dev": {"rp11|raw": d(0.52, 0.01), "rp11+sys|raw": d(0.4, 0.0)}, "feasible": ["rp11|raw"]}},
                    "x": {"chosen": "A"}}}
    (r,) = tune_rows(best)
    assert r["model"] == "m" and r["feasible"] and r["n_candidates"] == 3
    assert r["A_quality"] == 0.5 and r["chosen_quality"] == 0.52 and r["A_loop"] == 0.1 and r["chosen_loop"] == 0.01


def test_evafrill_ckpt_tie_keeps_slerp_same_policy(tmp_path, monkeypatch):
    """동점이면 기준점 slerp(최종). 비교는 1위와 같은 unk 정책끼리 (정책 차이를 체크포인트 이득으로 보지 않는다)."""
    import shutil
    import yaml as _y
    for f in ("dev_selection.yaml", "best_settings.yaml"):
        shutil.copy(gen_report.CONFIG_DIR / f, tmp_path / f)
    monkeypatch.setattr(gen_report, "CONFIG_DIR", tmp_path)
    vals = {("evafrill-mo-3b-sft-v2", "unk_drop"): 0.0496, ("evafrill-mo-3b", "unk_drop"): 0.0496,
            ("evafrill-mo-3b", "unk_nl"): 0.0359}
    monkeypatch.setattr(gen_report, "dev_score", lambda ck, v, p: {"mean": vals.get((ck, p), 0.01)})
    assert gen_report.cmd_evafrill_ckpt(None) == 0
    e = _y.safe_load((tmp_path / "best_settings.yaml").read_text(encoding="utf-8"))["gen"]["evafrill"]
    assert e["checkpoint"] == "evafrill-mo-3b" and e["checkpoint_gain_over_slerp"] == 0


def test_korean_sp_decode_keeps_line_leading_spaces():
    from kobench.standard import formats
    t = formats.training_tokenizer("frankenstallm-bf16")
    src = "def f(x):\n    if x:\n        return 1\n    return 2\n안녕 하세요"
    ids = [1] + t.encode(src, add_special_tokens=False).ids + [2]
    out = gen.decode_keep_unk(t, ids)
    assert out == src.replace("\n", "<unk>")
    old = "def f(x):<unk>   if x:<unk>       return 1<unk>   return 2"      # 이전 조각별 디코딩 결과
    assert gen_score.repair_unk_space(old) == "def f(x):<unk>    if x:<unk>        return 1<unk>    return 2"


def test_cut_at_stop_removes_stop_residue_earliest_first():
    # 2026-11-21: VllmGen이 토큰 ID를 다시 디코딩해 '\nprint(d' 같은 정지 문자열 잔여가 base 코드 출력에 남았다
    from kobench.standard.gen_data import CODE_BASE_STOP
    from kobench.standard.gen_score import cut_at_stop

    out = "\ndef f(a):\n    return a\n\nprint(d"
    assert cut_at_stop(out, CODE_BASE_STOP) == "\ndef f(a):\n    return a\n"
    assert cut_at_stop(cut_at_stop(out, CODE_BASE_STOP), CODE_BASE_STOP) == "\ndef f(a):\n    return a\n"   # 멱등
    assert cut_at_stop('x\nassert f(1)\n"""', CODE_BASE_STOP) == "x"                                        # 가장 앞 정지 문자열
    assert cut_at_stop("abc", None) == "abc"


def test_judge_variants_for_resolves_chosen(monkeypatch):
    # 2026-11-21: judge 기본 변형이 v2 목록이라 v3 선택 설정이 빠졌다 → 'chosen'을 모델별 선택 설정으로 바꾼다
    from kobench.standard import gen_report, judge

    def fake(mid):
        if mid == "none":
            raise KeyError(mid)
        return {"m1": ("rp11", "raw"), "m2": ("A", "raw")}[mid]

    monkeypatch.setattr(gen_report, "chosen", fake)
    assert judge.variants_for("m1", ["A", "think", "chosen"]) == ["A", "think", "rp11"]
    assert judge.variants_for("m2", ["A", "chosen"]) == ["A"]          # 중복 제거
    assert judge.variants_for("none", ["A", "chosen"]) == ["A"]        # 선택 기록 없음 → 건너뜀


def test_extract_code_prefers_python_block_inside_markdown_wrapper():
    # 2026-11-21: ```markdown 블록 안의 ```python 블록을 제대로 뽑지 못해 SyntaxError (llama-3.2-3b-instruct HumanEval+)
    from kobench.standard.code_exec import extract_code

    t = "```markdown\n# 제목\n설명\n\n```python\ndef f(x):\n    return x\n```\n끝\n```"
    assert extract_code(t) == "def f(x):\n    return x\n"
    assert extract_code("설명\n```\ndef g():\n    pass\n```") == "def g():\n    pass\n"           # 태그 없는 블록
    assert extract_code("```python\ndef h():\n    return 1") == "def h():\n    return 1"          # 닫히지 않은 블록
    assert extract_code("```py\nimport os\n```\n```python\ndef k():\n    pass\n```") == "def k():\n    pass\n"  # def 우선
    assert extract_code("def plain():\n    pass") == "def plain():\n    pass"                     # 블록 없음


def test_parse_think_handles_unprefilled_direct_answers_and_gemma_markers():
    from kobench.standard.gen import normalize_think, parse_think

    assert parse_think("<think>생각</think>답", "stop", False) == ("답", "생각")
    assert parse_think("생각만 하다 잘림", "length", True) == ("", "생각만 하다 잘림")       # 템플릿이 <think>를 넣은 모델의 잘림
    assert parse_think("답: 36", "stop", False) == ("답: 36", "")                          # SmolLM3 2턴째: 생각 없이 바로 답
    assert parse_think("답: 36", "stop", True) == ("", "답: 36")                           # prefilled면 닫히지 않은 생각
    assert parse_think("<think>열린 생각", "stop", False) == ("", "열린 생각")
    g = normalize_think("<|channel>thought\n따져 보면<channel|>최종 답")                   # Gemma 4 표시
    assert g == "<think>따져 보면</think>최종 답" and parse_think(g, "stop", False) == ("최종 답", "따져 보면")


def test_unk_stop_variants_match_korean_sp_newline_forms():
    from kobench.standard import gen
    from kobench.standard.gen_score import cut_at_stop

    v = gen.unk_stop_variants(["\n# 사용자:", "<|user|>"])
    assert v[:2] == ["\n# 사용자:", "<|user|>"] and "<unk># 사용자:" in v and "<unk> # 사용자:" in v
    assert gen.unk_stop_variants(None) is None
    assert cut_at_stop("답입니다.<unk> # 사용자:<unk>다음", v) == "답입니다."
    assert cut_at_stop("답입니다.<unk># 사용자:", v) == "답입니다."
    assert gen.is_korean_sp("evafrill-mo-3b") and gen.is_korean_sp("frankenstallm-bf16")
    assert not gen.is_korean_sp("jansori-bf16") and not gen.is_korean_sp("qwen3-4b-base")


def test_thinkrd_uses_readme_sampling_only_where_registered():
    from kobench.standard import gen
    from kobench.standard.registry import load_models

    m = load_models()
    st = gen.settings_for(m["qwen3.5-4b"], "thinkrd")
    assert st.sampling["temperature"] == 1.0 and st.sampling["presence_penalty"] == 1.5 and st.sampling["top_k"] == 20
    assert st.think_kwargs == {"enable_thinking": True} and st.max_model_len == 16384
    assert gen.settings_for(m["qwen3.5-35b-a3b"], "thinkrd") is None      # 등록 안 된 모델은 해당 없음
    assert gen.settings_for(m["frankenstallm-bf16"], "thinkrd") is None
    assert gen.is_think("thinkrd") and gen.is_think("think") and not gen.is_think("A+sys")
