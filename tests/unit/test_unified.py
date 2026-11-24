"""통일 비교 프로토콜 U (configs/standard/unified.yaml, kobench/standard/unified.py)."""

import pytest

from kobench.standard import unified as U


def test_is_raw_covers_base_and_uni_variant():
    assert U.is_raw("base", "A") and U.is_raw("instruct", "U") and U.is_raw("instruct", "U+sys")
    assert not U.is_raw("instruct", "A") and not U.is_raw("instruct", "think")


def test_urial_prompt_multi_turn_appends_own_answers_and_ends_at_assistant_tag():
    item = {"turns": ["첫 질문", "둘째 질문"]}
    t0 = U.uni_text("komt", item, 0, [])
    t1 = U.uni_text("komt", item, 1, ["첫 답  "])
    pre = U.urial_prefix()
    assert t0 == pre + "# 사용자:\n첫 질문\n\n# 도우미:\n"
    assert t1 == pre + "# 사용자:\n첫 질문\n\n# 도우미:\n첫 답\n\n# 사용자:\n둘째 질문\n\n# 도우미:\n"
    assert U.uni_turns("komt", item) == 2 and U.uni_turns("hrm8k", {"turns": ["q"]}) == 1
    assert U.uni_stop("komt", item) == ["\n# 사용자:", "\n# 안내"]
    assert pre.count("# 사용자:") == 3                     # 예시 3개


def test_fewshot_and_code_prompts():
    qa = U.uni_text("kosimpleqa", {"turns": ["한국의 국화는?"]})
    assert qa.endswith("질문: 한국의 국화는?\n답:") and qa.count("질문:") == 6
    assert U.uni_stop("kosimpleqa", {}) == ["\n"] and U.uni_max_new("kosimpleqa", 512) == 128
    code = {"raw": "def f():", "stop": ["\nassert"]}
    assert U.uni_text("mbpp_plus", code) == "def f():" and U.uni_stop("mbpp_plus", code) == ["\nassert"]
    assert U.uni_max_new("komt", 2048) == 2048                # URIAL 태스크는 원래 max_new
    assert U.uni_text("scenario_dev_self", {"turns": ["a"]}) == U.uni_text("scenario_self", {"turns": ["a"]})   # dev 별칭


@pytest.mark.skipif(not __import__("os").environ.get("KOBENCH_SLOW"), reason="데이터 스냅샷 필요 (KOBENCH_SLOW=1)")
def test_hrm8k_fewshot_prefix_uses_registered_train_items_without_calc_annotations():
    p = U.hrm8k_prefix()
    assert p.count("최종 답: $\\boxed{") == 5 and "<<" not in p
    assert "$\\boxed{28}$" in p                             # train 796 (와플·스테이크 28분)


def test_gen_settings_and_tasks_for_uni_suite():
    from kobench.standard import gen
    from kobench.standard.registry import load_models

    models = load_models()
    st = gen.settings_for(models["jansori-bf16"], "U")
    assert st.sampling == {"temperature": 0.0} and st.system is None
    names = gen.tasks_for(models["jansori-bf16"], "uni_v1", None)
    assert "komt" in names and "ifeval_ko" in names          # base도 통일 스위트 전부
    assert gen.tasks_for(models["jansori-bf16"], "gen_v1", None) == ["humaneval_plus", "mbpp_plus"]
