"""데이터 외부화 테스트 — Track 2/7 JSON 로딩 및 스키마 검증"""

import json
import pytest
from pathlib import Path

from kobench import config


class TestTrack2QuestionsJson:
    """data/ko_bench/questions.json 검증"""

    def test_file_exists(self):
        path = config.DATA_DIR / "ko_bench" / "questions.json"
        assert path.exists(), f"Track 2 questions.json not found at {path}"

    def test_schema_valid(self):
        path = config.DATA_DIR / "ko_bench" / "questions.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        # 8개 카테고리
        expected_categories = {
            "writing", "roleplay", "reasoning", "math",
            "coding", "extraction", "stem", "humanities",
        }
        assert set(data.keys()) == expected_categories
        # 각 카테고리에 10개 질문
        for cat, questions in data.items():
            assert isinstance(questions, list), f"{cat}: not a list"
            assert len(questions) == 10, f"{cat}: expected 10, got {len(questions)}"
            for q in questions:
                assert "turn1" in q, f"{cat}: missing turn1"
                assert "turn2" in q, f"{cat}: missing turn2"

    def test_load_function(self):
        from kobench.tracks.ko_bench import _load_questions
        result = _load_questions()
        assert result is not None
        assert len(result) == 8


class TestTrack7PromptsJson:
    """data/track7_prompts.json 검증"""

    def test_file_exists(self):
        path = config.DATA_DIR / "track7_prompts.json"
        assert path.exists(), f"Track 7 prompts.json not found at {path}"

    def test_schema_valid(self):
        path = config.DATA_DIR / "track7_prompts.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 20
        for item in data:
            assert "id" in item
            assert "category" in item
            assert "prompt" in item

    def test_load_function(self):
        from kobench.tracks.pairwise import _load_prompts
        result = _load_prompts()
        assert result is not None
        assert len(result) == 20

    def test_categories_coverage(self):
        path = config.DATA_DIR / "track7_prompts.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        categories = {item["category"] for item in data}
        expected = {
            "korean_knowledge", "reasoning", "creative_writing",
            "code", "explanation", "practical", "korean_language",
        }
        assert categories == expected


class TestFallbackWhenJsonMissing:
    """JSON 없을 때 inline fallback 동작 검증"""

    def test_track2_fallback(self, tmp_path):
        """DATA_DIR에 파일이 없으면 None 반환"""
        from unittest.mock import patch
        with patch("kobench.config.DATA_DIR", tmp_path):
            from kobench.tracks.ko_bench import _load_questions
            result = _load_questions()
            assert result is None

    def test_track7_fallback(self, tmp_path):
        from unittest.mock import patch
        with patch("kobench.config.DATA_DIR", tmp_path):
            from kobench.tracks.pairwise import _load_prompts
            result = _load_prompts()
            assert result is None
