"""
Track 6 (Performance Profiling) 단위 테스트

_make_filler_prompt, _get_quant_groups, _make_result_entry,
_test_prefill_speed, _test_decode_speed, _test_ttft, _test_vram,
_test_quant_comparison, _test_max_context, _test_concurrent,
_build_summary, run
"""

import pytest
from unittest.mock import patch, MagicMock, call

from eval_framework.tracks import track6_performance as t6


# ── _make_filler_prompt ─────────────────────────────────────────────────────


class TestMakeFillerPrompt:
    """한국어 필러 텍스트 생성"""

    def test_returns_string(self):
        """반환값이 문자열"""
        result = t6._make_filler_prompt(100)
        assert isinstance(result, str)

    def test_length_proportional_to_tokens(self):
        """더 많은 토큰을 요청하면 더 긴 텍스트"""
        short = t6._make_filler_prompt(100)
        long = t6._make_filler_prompt(1000)
        assert len(long) > len(short)

    def test_exact_char_length(self):
        """문자 수는 target_tokens * 2.0"""
        result = t6._make_filler_prompt(500)
        assert len(result) == int(500 * 2.0)

    def test_small_value(self):
        """작은 토큰 수 요청"""
        result = t6._make_filler_prompt(1)
        assert len(result) == 2

    def test_zero_tokens(self):
        """0 토큰 요청 시 빈 문자열이 아닌 결과 (repeats=1, trim to 0)"""
        result = t6._make_filler_prompt(0)
        assert len(result) == 0

    def test_contains_korean(self):
        """한국어 텍스트 포함"""
        result = t6._make_filler_prompt(100)
        assert "대한민국" in result


# ── _get_quant_groups ───────────────────────────────────────────────────────


class TestGetQuantGroups:
    """양자화 그룹 검출"""

    def test_two_variants_grouped(self):
        """동일 베이스에 2개 이상 양자화 변형이 있으면 그룹"""
        models = ["mymodel-f16", "mymodel-Q8_0"]
        groups = t6._get_quant_groups(models)
        assert "mymodel" in groups
        assert groups["mymodel"] == {"f16": "mymodel-f16", "Q8_0": "mymodel-Q8_0"}

    def test_three_variants(self):
        """3개 변형 모두 그룹"""
        models = ["m-f16", "m-Q8_0", "m-Q4_K_M"]
        groups = t6._get_quant_groups(models)
        assert len(groups["m"]) == 3

    def test_single_variant_excluded(self):
        """1개 변형만 있으면 제외"""
        models = ["m-f16", "other-model"]
        groups = t6._get_quant_groups(models)
        assert "m" not in groups

    def test_no_quant_tags(self):
        """양자화 태그가 없으면 빈 딕셔너리"""
        models = ["model-a", "model-b"]
        groups = t6._get_quant_groups(models)
        assert groups == {}

    def test_mixed_bases(self):
        """다른 베이스는 별도 그룹"""
        models = ["a-f16", "a-Q8_0", "b-f16", "b-Q4_K_M"]
        groups = t6._get_quant_groups(models)
        assert "a" in groups
        assert "b" in groups

    def test_empty_list(self):
        """빈 모델 목록"""
        assert t6._get_quant_groups([]) == {}


# ── _make_result_entry ──────────────────────────────────────────────────────


class TestMakeResultEntry:
    """결과 엔트리 생성"""

    def test_basic_fields(self):
        """기본 필드 포함"""
        entry = t6._make_result_entry("m1", "prefill_speed")
        assert entry["model"] == "m1"
        assert entry["test_type"] == "prefill_speed"
        assert entry["error"] is None

    def test_default_values(self):
        """기본값 확인"""
        entry = t6._make_result_entry("m1", "test")
        assert entry["input_length"] == 0
        assert entry["output_length"] == 0
        assert entry["tokens_per_sec"] == 0.0
        assert entry["prefill_tok_s"] == 0.0
        assert entry["ttft_s"] == 0.0
        assert entry["vram_used_mb"] == 0
        assert entry["wall_time_s"] == 0.0

    def test_rounding(self):
        """소수점 반올림 확인"""
        entry = t6._make_result_entry(
            "m1", "test",
            tokens_per_sec=12.345678,
            prefill_tok_s=99.999,
            ttft_s=0.12345678,
            wall_time_s=1.23456789,
        )
        assert entry["tokens_per_sec"] == 12.35
        assert entry["prefill_tok_s"] == 100.0
        assert entry["ttft_s"] == 0.1235
        assert entry["wall_time_s"] == 1.2346

    def test_extra_fields_merged(self):
        """extra 딕셔너리 병합"""
        entry = t6._make_result_entry(
            "m1", "test",
            extra={"custom_key": "custom_value", "num": 42},
        )
        assert entry["custom_key"] == "custom_value"
        assert entry["num"] == 42

    def test_extra_none(self):
        """extra=None이면 추가 필드 없음"""
        entry = t6._make_result_entry("m1", "test", extra=None)
        assert "extra" not in entry

    def test_error_string(self):
        """에러 메시지 설정"""
        entry = t6._make_result_entry("m1", "test", error="timeout")
        assert entry["error"] == "timeout"


# ── _test_prefill_speed ─────────────────────────────────────────────────────


class TestPrefillSpeed:
    """프리필 속도 측정"""

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_returns_results_per_length(self, mock_time, mock_runner, mock_config):
        """각 입력 길이마다 결과 하나씩"""
        mock_config.TRACK6_INPUT_LENGTHS = [100, 500]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 100,
            "prompt_eval_duration_s": 0.5,
            "eval_count": 1,
            "tokens_per_sec": 200.0,
            "wall_time_s": 0.6,
        }

        results = t6._test_prefill_speed("m1")

        assert len(results) == 2
        assert all(r["test_type"] == "prefill_speed" for r in results)
        assert all(r["model"] == "m1" for r in results)

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_prefill_tps_calculation(self, mock_time, mock_runner, mock_config):
        """프리필 tok/s 계산: prompt_eval_count / prompt_eval_duration_s"""
        mock_config.TRACK6_INPUT_LENGTHS = [100]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 200,
            "prompt_eval_duration_s": 0.4,
            "eval_count": 1,
            "tokens_per_sec": 50.0,
            "wall_time_s": 1.0,
        }

        results = t6._test_prefill_speed("m1")
        assert results[0]["prefill_tok_s"] == 500.0  # 200 / 0.4

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_zero_duration(self, mock_time, mock_runner, mock_config):
        """prompt_eval_duration_s=0이면 prefill_tps=0"""
        mock_config.TRACK6_INPUT_LENGTHS = [100]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 100,
            "prompt_eval_duration_s": 0,
            "eval_count": 1,
            "tokens_per_sec": 0,
            "wall_time_s": 0.1,
        }

        results = t6._test_prefill_speed("m1")
        assert results[0]["prefill_tok_s"] == 0.0

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_error_propagated(self, mock_time, mock_runner, mock_config):
        """에러가 결과에 전파"""
        mock_config.TRACK6_INPUT_LENGTHS = [100]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 0,
            "prompt_eval_duration_s": 0,
            "eval_count": 0,
            "tokens_per_sec": 0,
            "wall_time_s": 0,
            "error": "model_error",
        }

        results = t6._test_prefill_speed("m1")
        assert results[0]["error"] == "model_error"


# ── _test_decode_speed ──────────────────────────────────────────────────────


class TestDecodeSpeed:
    """디코드 속도 측정"""

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_four_output_lengths(self, mock_time, mock_runner, mock_config):
        """출력 길이 4가지 (50, 100, 256, 512)"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 20,
            "prompt_eval_duration_s": 0.1,
            "eval_count": 50,
            "tokens_per_sec": 30.0,
            "wall_time_s": 2.0,
        }

        results = t6._test_decode_speed("m1")

        assert len(results) == 4
        assert all(r["test_type"] == "decode_speed" for r in results)

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_requested_output_length_extra(self, mock_time, mock_runner, mock_config):
        """extra에 requested_output_length 포함"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 20,
            "prompt_eval_duration_s": 0.1,
            "eval_count": 50,
            "tokens_per_sec": 30.0,
            "wall_time_s": 2.0,
        }

        results = t6._test_decode_speed("m1")
        requested = [r["requested_output_length"] for r in results]
        assert requested == [50, 100, 256, 512]

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_zero_prefill_duration(self, mock_time, mock_runner, mock_config):
        """prompt_eval_duration_s=0이면 prefill_tok_s=0"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 20,
            "prompt_eval_duration_s": 0,
            "eval_count": 50,
            "tokens_per_sec": 30.0,
            "wall_time_s": 2.0,
        }

        results = t6._test_decode_speed("m1")
        assert results[0]["prefill_tok_s"] == 0.0


# ── _test_ttft ──────────────────────────────────────────────────────────────


class TestTTFT:
    """Time To First Token 측정"""

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.requests")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_three_prompt_lengths(self, mock_time, mock_requests, mock_config):
        """short, medium, long 3개 프롬프트"""
        mock_config.MODEL_TIMEOUTS = {}
        mock_config.OLLAMA_API_GENERATE = "http://localhost:11434/api/generate"
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_time.time.side_effect = [
            # short
            1.0,       # wall_start
            1.05,      # ttft (first token)
            1.1,       # wall_time
            # medium
            2.0,
            2.1,
            2.2,
            # long
            3.0,
            3.2,
            3.3,
        ]

        mock_resp = MagicMock()
        import json
        mock_resp.iter_lines.return_value = [
            json.dumps({"response": "hello"}).encode(),
            json.dumps({"response": " world", "done": True}).encode(),
        ]
        mock_requests.post.return_value = mock_resp

        results = t6._test_ttft("m1")

        assert len(results) == 3
        labels = [r["prompt_label"] for r in results]
        assert labels == ["short", "medium", "long"]
        assert all(r["test_type"] == "ttft" for r in results)

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.requests")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_no_token_received(self, mock_time, mock_requests, mock_config):
        """토큰 수신 실패 시 error='no_token_received'"""
        mock_config.MODEL_TIMEOUTS = {}
        mock_config.OLLAMA_API_GENERATE = "http://localhost:11434/api/generate"
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_time.time.side_effect = [1.0, 1.5, 1.5] * 3

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'{"response": "", "done": true}',
        ]
        mock_requests.post.return_value = mock_resp

        results = t6._test_ttft("m1")
        assert results[0]["error"] == "no_token_received"

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.requests")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_request_exception(self, mock_time, mock_requests, mock_config):
        """요청 예외 시 에러 메시지 저장"""
        mock_config.MODEL_TIMEOUTS = {}
        mock_config.OLLAMA_API_GENERATE = "http://localhost:11434/api/generate"
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_time.time.side_effect = [1.0, 1.5, 1.5] * 3

        mock_requests.post.side_effect = ConnectionError("connection refused")

        results = t6._test_ttft("m1")
        assert "connection refused" in results[0]["error"]


# ── _test_vram ──────────────────────────────────────────────────────────────


class TestVRAM:
    """VRAM 사용량 측정"""

    @patch("eval_framework.tracks.track6_performance.runner")
    def test_returns_single_result(self, mock_runner):
        """결과 1건 반환"""
        mock_runner.get_vram_usage.return_value = {
            "vram_used_mb": 4096,
            "vram_total_mb": 16384,
            "vram_free_mb": 12288,
            "gpu_util_pct": 25,
        }

        results = t6._test_vram("m1")

        assert len(results) == 1
        assert results[0]["test_type"] == "vram_usage"
        assert results[0]["vram_used_mb"] == 4096
        assert results[0]["vram_total_mb"] == 16384
        assert results[0]["vram_free_mb"] == 12288
        assert results[0]["gpu_util_pct"] == 25

    @patch("eval_framework.tracks.track6_performance.runner")
    def test_empty_vram_info(self, mock_runner):
        """VRAM 정보 없을 때 기본값"""
        mock_runner.get_vram_usage.return_value = {}

        results = t6._test_vram("m1")
        assert results[0]["vram_used_mb"] == 0


# ── _test_quant_comparison ──────────────────────────────────────────────────


class TestQuantComparison:
    """양자화 변형 비교"""

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_results_per_variant_and_prompt(self, mock_time, mock_runner, mock_config):
        """각 변형 x 각 프롬프트 = 결과 수"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.switch_model.return_value = True
        mock_runner.get_vram_usage.return_value = {"vram_used_mb": 2048}
        mock_runner.generate.return_value = {
            "prompt_eval_count": 30,
            "prompt_eval_duration_s": 0.2,
            "eval_count": 128,
            "tokens_per_sec": 40.0,
            "wall_time_s": 3.0,
        }

        quant_groups = {"base": {"f16": "base-f16", "Q8_0": "base-Q8_0"}}
        results, last_model = t6._test_quant_comparison(quant_groups)

        # 2 variants x 5 prompts = 10 results
        assert len(results) == 10
        assert all(r["test_type"] == "quant_comparison" for r in results)

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_extra_fields(self, mock_time, mock_runner, mock_config):
        """extra에 base_model, quant_tag, prompt_index 포함"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.switch_model.return_value = True
        mock_runner.get_vram_usage.return_value = {"vram_used_mb": 2048}
        mock_runner.generate.return_value = {
            "prompt_eval_count": 30,
            "prompt_eval_duration_s": 0.2,
            "eval_count": 128,
            "tokens_per_sec": 40.0,
            "wall_time_s": 3.0,
        }

        quant_groups = {"mybase": {"f16": "mybase-f16"}}
        # Single variant won't normally be here, but tests field presence
        # Need at least 2 to pass _get_quant_groups, but _test_quant_comparison
        # doesn't re-validate
        quant_groups = {"mybase": {"f16": "mybase-f16", "Q8_0": "mybase-Q8_0"}}
        results, _ = t6._test_quant_comparison(quant_groups)

        first = results[0]
        assert "base_model" in first
        assert "quant_tag" in first
        assert "prompt_index" in first

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_returns_last_model(self, mock_time, mock_runner, mock_config):
        """마지막으로 로드된 모델 반환"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.switch_model.return_value = True
        mock_runner.get_vram_usage.return_value = {}
        mock_runner.generate.return_value = {
            "prompt_eval_count": 10,
            "prompt_eval_duration_s": 0.1,
            "eval_count": 50,
            "tokens_per_sec": 20.0,
            "wall_time_s": 1.0,
        }

        quant_groups = {"b": {"Q4_K_M": "b-Q4_K_M", "f16": "b-f16"}}
        _, last_model = t6._test_quant_comparison(quant_groups)
        # sorted tags: Q4_K_M, f16 -> last is f16
        assert last_model == "b-f16"


# ── _test_max_context ───────────────────────────────────────────────────────


class TestMaxContext:
    """최대 컨텍스트 테스트"""

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_five_context_lengths(self, mock_time, mock_runner, mock_config):
        """5가지 컨텍스트 길이 테스트"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 512,
            "prompt_eval_duration_s": 0.5,
            "eval_count": 32,
            "tokens_per_sec": 64.0,
            "wall_time_s": 1.0,
        }

        results = t6._test_max_context("m1")

        assert len(results) == 5
        assert all(r["test_type"] == "max_context" for r in results)
        requested = [r["requested_context"] for r in results]
        assert requested == [512, 1024, 2048, 3072, 4096]

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_uses_prompt_eval_count_as_input(self, mock_time, mock_runner, mock_config):
        """prompt_eval_count가 있으면 input_length로 사용"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 450,
            "prompt_eval_duration_s": 0.5,
            "eval_count": 32,
            "tokens_per_sec": 64.0,
            "wall_time_s": 1.0,
        }

        results = t6._test_max_context("m1")
        assert results[0]["input_length"] == 450

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_fallback_to_ctx_len(self, mock_time, mock_runner, mock_config):
        """prompt_eval_count=0이면 ctx_len 사용"""
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_runner.generate.return_value = {
            "prompt_eval_count": 0,
            "prompt_eval_duration_s": 0,
            "eval_count": 0,
            "tokens_per_sec": 0,
            "wall_time_s": 0.1,
        }

        results = t6._test_max_context("m1")
        assert results[0]["input_length"] == 512


# ── _test_concurrent ────────────────────────────────────────────────────────


class TestConcurrent:
    """동시 요청 테스트"""

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_results_per_concurrency_level(self, mock_time, mock_runner, mock_config):
        """각 동시성 수준마다 결과 하나"""
        mock_config.TRACK6_CONCURRENT_LEVELS = [1, 2]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_time.time.side_effect = [
            # level=1: batch_start, single_request start, single_request end, batch_end
            0.0, 0.0, 1.0, 1.0,
            # level=2: batch_start, req0 start, req0 end, req1 start, req1 end, batch_end
            2.0, 2.0, 3.0, 2.0, 3.0, 3.0,
        ]
        mock_runner.generate.return_value = {
            "tokens_per_sec": 30.0,
            "eval_count": 64,
        }

        results = t6._test_concurrent("m1")

        assert len(results) == 2
        assert all(r["test_type"] == "concurrent" for r in results)

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_concurrency_level_in_extra(self, mock_time, mock_runner, mock_config):
        """extra에 concurrency_level 포함"""
        mock_config.TRACK6_CONCURRENT_LEVELS = [4]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_time.time.side_effect = [0.0] + [0.0, 1.0] * 4 + [4.0]
        mock_runner.generate.return_value = {
            "tokens_per_sec": 20.0,
            "eval_count": 64,
        }

        results = t6._test_concurrent("m1")
        assert results[0]["concurrency_level"] == 4

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_aggregate_tok_s(self, mock_time, mock_runner, mock_config):
        """집계 tok/s 계산"""
        mock_config.TRACK6_CONCURRENT_LEVELS = [1]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        # batch_start=0, req_start=0, req_end=1, batch_end=2
        mock_time.time.side_effect = [0.0, 0.0, 1.0, 2.0]
        mock_runner.generate.return_value = {
            "tokens_per_sec": 30.0,
            "eval_count": 60,
        }

        results = t6._test_concurrent("m1")
        # aggregate = 60 tokens / 2.0 seconds = 30.0
        assert results[0]["aggregate_tok_s"] == 30.0


# ── _build_summary ──────────────────────────────────────────────────────────


class TestBuildSummary:
    """결과 요약 생성"""

    def test_empty_results(self):
        """빈 결과 목록 → 빈 요약"""
        assert t6._build_summary([]) == {}

    def test_prefill_average(self):
        """프리필 속도 평균"""
        results = [
            {"model": "m1", "test_type": "prefill_speed", "prefill_tok_s": 100.0,
             "tokens_per_sec": 0, "ttft_s": 0, "vram_used_mb": 0, "input_length": 0},
            {"model": "m1", "test_type": "prefill_speed", "prefill_tok_s": 200.0,
             "tokens_per_sec": 0, "ttft_s": 0, "vram_used_mb": 0, "input_length": 0},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["avg_prefill_tok_s"] == 150.0

    def test_decode_average(self):
        """디코드 속도 평균"""
        results = [
            {"model": "m1", "test_type": "decode_speed", "tokens_per_sec": 40.0,
             "prefill_tok_s": 0, "ttft_s": 0, "vram_used_mb": 0, "input_length": 0},
            {"model": "m1", "test_type": "decode_speed", "tokens_per_sec": 60.0,
             "prefill_tok_s": 0, "ttft_s": 0, "vram_used_mb": 0, "input_length": 0},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["avg_decode_tok_s"] == 50.0

    def test_ttft_average(self):
        """TTFT 평균"""
        results = [
            {"model": "m1", "test_type": "ttft", "ttft_s": 0.1,
             "tokens_per_sec": 0, "prefill_tok_s": 0, "vram_used_mb": 0, "input_length": 0},
            {"model": "m1", "test_type": "ttft", "ttft_s": 0.3,
             "tokens_per_sec": 0, "prefill_tok_s": 0, "vram_used_mb": 0, "input_length": 0},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["avg_ttft_s"] == 0.2

    def test_vram_usage(self):
        """VRAM 사용량"""
        results = [
            {"model": "m1", "test_type": "vram_usage", "vram_used_mb": 8192,
             "tokens_per_sec": 0, "prefill_tok_s": 0, "ttft_s": 0, "input_length": 0},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["vram_used_mb"] == 8192

    def test_max_context_reached(self):
        """최대 컨텍스트 도달값"""
        results = [
            {"model": "m1", "test_type": "max_context", "input_length": 1024,
             "tokens_per_sec": 0, "prefill_tok_s": 0, "ttft_s": 0, "vram_used_mb": 0},
            {"model": "m1", "test_type": "max_context", "input_length": 4096,
             "tokens_per_sec": 0, "prefill_tok_s": 0, "ttft_s": 0, "vram_used_mb": 0},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["max_context_reached"] == 4096

    def test_max_context_with_error_ignored(self):
        """에러가 있는 max_context 결과는 무시"""
        results = [
            {"model": "m1", "test_type": "max_context", "input_length": 4096,
             "error": "oom", "tokens_per_sec": 0, "prefill_tok_s": 0,
             "ttft_s": 0, "vram_used_mb": 0},
            {"model": "m1", "test_type": "max_context", "input_length": 2048,
             "tokens_per_sec": 0, "prefill_tok_s": 0, "ttft_s": 0, "vram_used_mb": 0},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["max_context_reached"] == 2048

    def test_concurrent_aggregate(self):
        """동시 요청 집계 토큰/초"""
        results = [
            {"model": "m1", "test_type": "concurrent",
             "tokens_per_sec": 0, "prefill_tok_s": 0, "ttft_s": 0,
             "vram_used_mb": 0, "input_length": 0,
             "extra": {"concurrency_level": 2, "aggregate_tok_s": 55.5}},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["concurrent_aggregate_tok_s"]["2"] == 55.5

    def test_multi_model(self):
        """복수 모델 요약"""
        results = [
            {"model": "m1", "test_type": "prefill_speed", "prefill_tok_s": 100.0,
             "tokens_per_sec": 0, "ttft_s": 0, "vram_used_mb": 0, "input_length": 0},
            {"model": "m2", "test_type": "prefill_speed", "prefill_tok_s": 200.0,
             "tokens_per_sec": 0, "ttft_s": 0, "vram_used_mb": 0, "input_length": 0},
        ]
        summary = t6._build_summary(results)
        assert "m1" in summary
        assert "m2" in summary

    def test_zero_values_produce_zero_averages(self):
        """값이 0이면 평균도 0"""
        results = [
            {"model": "m1", "test_type": "prefill_speed", "prefill_tok_s": 0,
             "tokens_per_sec": 0, "ttft_s": 0, "vram_used_mb": 0, "input_length": 0},
        ]
        summary = t6._build_summary(results)
        assert summary["m1"]["avg_prefill_tok_s"] == 0

    def test_summary_fields_complete(self):
        """요약에 필수 필드가 모두 포함"""
        results = [
            {"model": "m1", "test_type": "vram_usage", "vram_used_mb": 1024,
             "tokens_per_sec": 0, "prefill_tok_s": 0, "ttft_s": 0, "input_length": 0},
        ]
        summary = t6._build_summary(results)
        required = {"avg_prefill_tok_s", "avg_decode_tok_s", "avg_ttft_s",
                     "vram_used_mb", "max_context_reached", "concurrent_aggregate_tok_s"}
        assert required == set(summary["m1"].keys())


# ── run ─────────────────────────────────────────────────────────────────────


class TestRun:
    """Track 6 메인 실행 흐름"""

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_full_flow(self, mock_time, mock_runner, mock_config):
        """전체 흐름: 모델 로드 → 6개 테스트 → 요약"""
        mock_config.TRACK6_INPUT_LENGTHS = [100]
        mock_config.TRACK6_CONCURRENT_LEVELS = [1]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_config.MODEL_TIMEOUTS = {}
        mock_config.OLLAMA_API_GENERATE = "http://localhost:11434/api/generate"

        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = True
        mock_runner.switch_model.return_value = True
        mock_runner.get_vram_usage.return_value = {"vram_used_mb": 4096}
        mock_runner.generate.return_value = {
            "prompt_eval_count": 100,
            "prompt_eval_duration_s": 0.5,
            "eval_count": 50,
            "tokens_per_sec": 30.0,
            "wall_time_s": 2.0,
        }
        mock_time.time.side_effect = lambda: 1.0

        result = t6.run(["m1"])

        assert result["track"] == "track6_performance"
        assert "error" not in result
        assert len(result["results"]) > 0
        assert "m1" in result["summary"]
        mock_runner.save_results_incremental.assert_called_once()

    @patch("eval_framework.tracks.track6_performance.runner")
    def test_ollama_unavailable(self, mock_runner):
        """Ollama 연결 실패 시 에러 반환"""
        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = False

        result = t6.run(["m1"])
        assert "error" in result
        assert result["track"] == "track6_performance"

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_model_load_failure_skips(self, mock_time, mock_runner, mock_config):
        """모델 로딩 실패 시 해당 모델 스킵"""
        mock_config.TRACK6_INPUT_LENGTHS = [100]
        mock_config.TRACK6_CONCURRENT_LEVELS = [1]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0

        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = True
        mock_runner.switch_model.return_value = False

        result = t6.run(["bad_model"])

        errors = [r for r in result["results"] if r.get("error") == "warmup_failed"]
        assert len(errors) == 1

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_checkpoint_resume(self, mock_time, mock_runner, mock_config):
        """체크포인트에서 재개"""
        mock_config.TRACK6_INPUT_LENGTHS = [100]
        mock_config.TRACK6_CONCURRENT_LEVELS = [1]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_config.MODEL_TIMEOUTS = {}
        mock_config.OLLAMA_API_GENERATE = "http://localhost:11434/api/generate"

        existing_result = t6._make_result_entry("m1", "prefill_speed", prefill_tok_s=100.0)
        mock_runner.load_checkpoint.return_value = {
            "results": [existing_result],
            "completed_keys": ["model_loaded:m1", "prefill:m1", "decode:m1",
                               "ttft:m1", "vram:m1", "max_context:m1", "concurrent:m1"],
        }
        mock_runner.wait_for_ollama.return_value = True
        mock_runner.switch_model.return_value = True

        result = t6.run(["m1"])

        # Should not call generate since all keys are completed
        mock_runner.generate.assert_not_called()
        # The existing result should still be in the output
        assert len(result["results"]) >= 1

    @patch("eval_framework.tracks.track6_performance.runner")
    def test_defaults_to_config_models(self, mock_runner):
        """models=None이면 config.ALL_MODELS 사용"""
        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = False

        with patch("eval_framework.tracks.track6_performance.config") as mock_config:
            mock_config.ALL_MODELS = ["default_m1", "default_m2"]
            result = t6.run(None)

        assert "error" in result

    @patch("eval_framework.tracks.track6_performance.config")
    @patch("eval_framework.tracks.track6_performance.runner")
    @patch("eval_framework.tracks.track6_performance.time")
    def test_quant_comparison_runs(self, mock_time, mock_runner, mock_config):
        """양자화 그룹이 있으면 quant_comparison 실행"""
        mock_config.TRACK6_INPUT_LENGTHS = [100]
        mock_config.TRACK6_CONCURRENT_LEVELS = [1]
        mock_config.COOLDOWN_BETWEEN_TESTS = 0
        mock_config.MODEL_TIMEOUTS = {}
        mock_config.OLLAMA_API_GENERATE = "http://localhost:11434/api/generate"

        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = True
        mock_runner.switch_model.return_value = True
        mock_runner.get_vram_usage.return_value = {"vram_used_mb": 2048}
        mock_runner.generate.return_value = {
            "prompt_eval_count": 50,
            "prompt_eval_duration_s": 0.3,
            "eval_count": 50,
            "tokens_per_sec": 30.0,
            "wall_time_s": 2.0,
        }
        mock_time.time.side_effect = lambda: 1.0

        result = t6.run(["base-f16", "base-Q8_0"])

        quant_results = [r for r in result["results"] if r["test_type"] == "quant_comparison"]
        assert len(quant_results) > 0
