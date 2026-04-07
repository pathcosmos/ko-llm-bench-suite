"""
Track 5: 일관성 & 강건성 — 반복 일관성, 패러프레이즈, 길이 민감도,
언어 혼용, 지시 준수, 환각 탐지
"""

import json
import re
import time
import unicodedata
from typing import Optional

import numpy as np

from kobench import config
from kobench import runner

TRACK_NAME = "consistency"

# ═══════════════════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════════


def jaccard_similarity(set1: set, set2: set) -> float:
    """두 집합 간 Jaccard 유사도 (교집합/합집합)."""
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 1.0


def edit_distance_ratio(str1: str, str2: str) -> float:
    """문자 수준 편집 거리 비율 (0=동일, 1=완전히 다름).

    Levenshtein distance / max(len(str1), len(str2))
    DP 구현 — O(n*m) 이지만 응답 길이가 수백 자 수준이라 충분.
    """
    n, m = len(str1), len(str2)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    # 메모리 최적화: 두 행만 유지
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # 삭제
                curr[j - 1] + 1,   # 삽입
                prev[j - 1] + cost  # 치환
            )
        prev, curr = curr, prev

    distance = prev[m]
    return distance / max(n, m)


def detect_korean_ratio(text: str) -> float:
    """텍스트에서 한국어 문자 비율 (공백/구두점 제외)."""
    if not text:
        return 0.0
    alpha_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        # L* = Letter, N* = Number
        if cat.startswith("L") or cat.startswith("N"):
            alpha_chars.append(ch)
    if not alpha_chars:
        return 0.0
    korean_count = sum(
        1 for ch in alpha_chars
        if ("\uAC00" <= ch <= "\uD7A3")       # 완성형 한글
        or ("\u3131" <= ch <= "\u3163")         # 자모
        or ("\u1100" <= ch <= "\u11FF")         # 한글 자모
    )
    return korean_count / len(alpha_chars)


def check_instruction_compliance(
    response: str,
    instruction_type: str,
    constraint_value,
) -> dict:
    """지시 준수 여부를 규칙 기반으로 검사.

    Returns:
        {"compliant": bool, "detail": str}
    """
    response = response.strip()

    if instruction_type == "count_items":
        # "N개만 나열하세요" — 번호(1. 2. 3.) 또는 글머리(- •) 개수 확인
        numbered = re.findall(r"^\s*\d+[\.\)]\s", response, re.MULTILINE)
        bulleted = re.findall(r"^\s*[-•*]\s", response, re.MULTILINE)
        item_count = max(len(numbered), len(bulleted))
        # 번호/글머리 없으면 줄 단위로 세기
        if item_count == 0:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            item_count = len(lines)
        expected = int(constraint_value)
        compliant = item_count == expected
        return {
            "compliant": compliant,
            "detail": f"항목 수: {item_count} (기대: {expected})",
        }

    elif instruction_type == "max_chars":
        length = len(response)
        limit = int(constraint_value)
        compliant = length <= limit
        return {
            "compliant": compliant,
            "detail": f"글자 수: {length} (제한: {limit})",
        }

    elif instruction_type == "json_format":
        try:
            parsed = json.loads(response)
            return {"compliant": True, "detail": "유효한 JSON"}
        except json.JSONDecodeError:
            # JSON이 코드블록 안에 있을 수 있음
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1).strip())
                    return {"compliant": True, "detail": "코드블록 내 유효한 JSON"}
                except json.JSONDecodeError:
                    pass
            return {"compliant": False, "detail": "유효하지 않은 JSON"}

    elif instruction_type == "numbered_list":
        numbered = re.findall(r"^\s*\d+[\.\)]\s", response, re.MULTILINE)
        compliant = len(numbered) >= 2  # 최소 2개 이상의 번호 항목
        return {
            "compliant": compliant,
            "detail": f"번호 항목 수: {len(numbered)}",
        }

    elif instruction_type == "table_format":
        # Markdown 표: | 로 구분된 행이 2개 이상, 구분선(---|---) 포함
        pipe_lines = [l for l in response.split("\n") if "|" in l]
        separator = any(re.match(r"^\s*\|?[\s\-:]+\|", l) for l in pipe_lines)
        compliant = len(pipe_lines) >= 3 and separator
        return {
            "compliant": compliant,
            "detail": f"표 행: {len(pipe_lines)}, 구분선: {separator}",
        }

    else:
        return {"compliant": False, "detail": f"알 수 없는 지시 유형: {instruction_type}"}


# ═══════════════════════════════════════════════════════════════════════════════
# 테스트 데이터 (모두 인라인)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. 동일 프롬프트 반복 ────────────────────────────────────────────────────
REPETITION_PROMPTS = [
    # 사실 질문
    "대한민국의 수도는 어디인가요?",
    "물의 화학식은 무엇인가요?",
    "태양계에서 가장 큰 행성은 무엇인가요?",
    # 창작
    "가을을 주제로 짧은 시를 한 편 써주세요.",
    "외로운 로봇이 주인공인 짧은 이야기를 써주세요.",
    # 추론
    "모든 고양이는 동물이다. 모든 동물은 생물이다. 따라서 모든 고양이는?",
    "A는 B보다 크고, B는 C보다 큽니다. A와 C 중 어느 것이 더 큰가요?",
    # 코드
    "Python으로 피보나치 수열을 출력하는 함수를 작성해주세요.",
    "Python으로 리스트에서 중복을 제거하는 코드를 작성해주세요.",
    # 일반 지식
    "인공지능의 정의를 한 문장으로 설명해주세요.",
]
REPETITION_COUNT = 5

# ── 2. 패러프레이즈 강건성 ───────────────────────────────────────────────────
# 각 항목: (정답 키워드 목록, [패러프레이즈 5개])
PARAPHRASE_QUESTIONS = [
    {
        "answer_keywords": ["서울"],
        "variants": [
            "대한민국의 수도는?",
            "한국의 수도 도시는 어디인가요?",
            "Korea의 capital city는?",
            "대한민국에서 수도로 지정된 도시의 이름은 무엇입니까?",
            "한국의 수도가 어디인지 알려주세요.",
        ],
    },
    {
        "answer_keywords": ["H2O", "H₂O"],
        "variants": [
            "물의 화학식은?",
            "물을 화학 기호로 표현하면?",
            "Water의 chemical formula는 무엇인가요?",
            "H와 O로 이루어진 물의 분자식을 알려주세요.",
            "물 분자의 화학식을 써주세요.",
        ],
    },
    {
        "answer_keywords": ["목성", "Jupiter"],
        "variants": [
            "태양계에서 가장 큰 행성은?",
            "Solar system에서 제일 큰 planet은?",
            "우리 태양계의 행성 중 크기가 가장 큰 것은 무엇인가요?",
            "태양 주위를 도는 행성 중 가장 거대한 행성의 이름은?",
            "가장 큰 태양계 행성을 알려주세요.",
        ],
    },
    {
        "answer_keywords": ["도쿄", "東京", "Tokyo"],
        "variants": [
            "일본의 수도는 어디인가요?",
            "Japan의 capital은?",
            "일본에서 수도 역할을 하는 도시는?",
            "일본의 수도 도시 이름을 알려주세요.",
            "일본의 수도가 어디인지 답해주세요.",
        ],
    },
    {
        "answer_keywords": ["지구", "Earth"],
        "variants": [
            "태양에서 세 번째로 가까운 행성은?",
            "태양계에서 세 번째 행성은 무엇인가요?",
            "Sun에서 세 번째로 위치한 planet은?",
            "수금지화목토천해 중 세 번째는?",
            "태양에서 세 번째 궤도를 도는 행성의 이름은?",
        ],
    },
    {
        "answer_keywords": ["중국", "China"],
        "variants": [
            "세계에서 인구가 가장 많은 나라는?",
            "전 세계에서 인구수가 제일 많은 국가는 어디인가요?",
            "World에서 population이 가장 많은 country는?",
            "인구 기준 세계 1위 국가를 알려주세요.",
            "지구상에서 사람이 가장 많이 사는 나라는?",
        ],
    },
    {
        "answer_keywords": ["에베레스트", "Everest"],
        "variants": [
            "세계에서 가장 높은 산은?",
            "지구에서 제일 높은 mountain은 무엇인가요?",
            "세계 최고봉의 이름은?",
            "가장 높은 산의 이름을 알려주세요.",
            "해발 고도가 가장 높은 산은 어디에 있나요?",
        ],
    },
    {
        "answer_keywords": ["태평양", "Pacific"],
        "variants": [
            "세계에서 가장 큰 바다는?",
            "지구에서 면적이 가장 넓은 대양은?",
            "가장 큰 ocean의 이름은 무엇인가요?",
            "면적 기준 세계 최대 대양을 알려주세요.",
            "Earth에서 가장 넓은 바다는 어디인가요?",
        ],
    },
    {
        "answer_keywords": ["아인슈타인", "Einstein"],
        "variants": [
            "상대성이론을 발표한 과학자는 누구인가요?",
            "Theory of Relativity를 만든 사람은?",
            "E=mc²를 발견한 물리학자는?",
            "상대성이론으로 유명한 과학자의 이름은?",
            "특수상대성이론과 일반상대성이론을 발표한 사람은 누구입니까?",
        ],
    },
    {
        "answer_keywords": ["산소", "O2", "O₂", "Oxygen"],
        "variants": [
            "사람이 호흡할 때 필요한 기체는?",
            "인간이 숨 쉴 때 들이마시는 주요 기체는 무엇인가요?",
            "Breathing에 필수적인 gas는?",
            "폐에서 혈액으로 흡수되는 기체의 이름은?",
            "생존을 위해 사람이 반드시 마셔야 하는 공기 속 기체는?",
        ],
    },
]

# ── 3. 길이 민감도 ──────────────────────────────────────────────────────────
LENGTH_SENSITIVITY_DATA = [
    {
        "answer_keywords": ["서울"],
        "short": "한국 수도는?",
        "medium": "한국의 수도가 어디인지 알려주세요.",
        "long": (
            "대한민국은 동아시아에 위치한 국가로, 한반도의 남쪽 절반을 차지하고 있습니다. "
            "삼국시대부터 현대에 이르기까지 오랜 역사를 가지고 있으며, 다양한 도시들이 "
            "정치, 경제, 문화의 중심지 역할을 해왔습니다. "
            "위 내용을 참고하여 한국의 수도가 어디인지 자세히 설명해주세요."
        ),
    },
    {
        "answer_keywords": ["태양", "Sun"],
        "short": "태양계 중심 항성은?",
        "medium": "태양계의 중심에 위치한 항성은 무엇인지 설명해주세요.",
        "long": (
            "태양계는 약 46억 년 전에 형성된 것으로 추정되며, 중심에 하나의 항성이 있고 "
            "그 주위를 8개의 행성이 공전하고 있습니다. 이 항성은 태양계 전체 질량의 "
            "99.86%를 차지합니다. "
            "위 정보를 바탕으로 태양계 중심의 항성이 무엇인지 자세히 답변해주세요."
        ),
    },
    {
        "answer_keywords": ["DNA"],
        "short": "유전 물질 약자는?",
        "medium": "생물의 유전 정보를 담고 있는 물질의 약자를 알려주세요.",
        "long": (
            "생물학에서 유전 정보는 세포핵 안에 존재하는 특정 분자에 의해 저장되고 "
            "전달됩니다. 이 분자는 이중 나선 구조를 가지며 네 가지 염기의 서열로 "
            "유전 암호를 구성합니다. 1953년 왓슨과 크릭이 그 구조를 밝혀냈습니다. "
            "위 설명을 참고하여 이 유전 물질의 약자가 무엇인지 설명해주세요."
        ),
    },
    {
        "answer_keywords": ["광합성"],
        "short": "식물이 빛으로 양분 만드는 과정은?",
        "medium": "식물이 빛에너지를 이용하여 양분을 만드는 과정을 무엇이라 하나요?",
        "long": (
            "식물은 엽록체에서 빛에너지를 흡수하여 이산화탄소와 물을 원료로 삼아 "
            "포도당과 산소를 생성합니다. 이 과정은 지구상의 대부분의 생태계에서 "
            "에너지 흐름의 출발점이 됩니다. "
            "위 내용을 바탕으로 이 과정의 이름이 무엇인지 자세히 설명해주세요."
        ),
    },
    {
        "answer_keywords": ["뉴턴", "Newton"],
        "short": "만유인력 발견자는?",
        "medium": "만유인력의 법칙을 발견한 과학자는 누구인지 알려주세요.",
        "long": (
            "17세기 영국에서 한 과학자가 사과가 떨어지는 것을 보고 모든 물체 사이에 "
            "작용하는 인력의 존재를 깨달았다는 일화가 전해집니다. 이 과학자는 운동 법칙 "
            "3가지와 만유인력의 법칙을 정리하여 고전역학의 토대를 마련했습니다. "
            "위 내용을 참고하여 만유인력을 발견한 과학자가 누구인지 자세히 답해주세요."
        ),
    },
]

# ── 4. 언어 혼용 ────────────────────────────────────────────────────────────
LANGUAGE_MIXING_PROMPTS = [
    "Python에서 list comprehension의 사용법을 explain해주세요.",
    "Machine learning에서 overfitting을 방지하는 방법을 한국어로 설명해주세요.",
    "HTTP status code 404는 무엇을 의미하는지 describe해주세요.",
    "Docker container와 virtual machine의 차이점을 설명해주세요.",
    "Big O notation에서 O(n log n)의 의미를 한국어로 explain해주세요.",
    "REST API에서 GET과 POST method의 차이를 설명해주세요.",
    "Database에서 primary key와 foreign key의 역할을 알려주세요.",
    "CSS에서 flexbox layout의 주요 property를 한국어로 설명해주세요.",
    "Git에서 branch를 merge하는 방법을 step by step으로 설명해주세요.",
    "Object-oriented programming의 4가지 핵심 concept을 한국어로 설명해주세요.",
]

# ── 5. 지시 준수 ────────────────────────────────────────────────────────────
# (prompt, instruction_type, constraint_value)
INSTRUCTION_FOLLOWING_DATA = [
    # count_items: 정확한 항목 수
    ("한국의 대표적인 음식 3개만 나열하세요.", "count_items", 3),
    ("프로그래밍 언어 5개만 나열하세요.", "count_items", 5),
    ("세계 4대 문명을 나열하세요.", "count_items", 4),
    # max_chars: 글자 수 제한
    ("인공지능을 50자 이내로 정의하세요.", "max_chars", 50),
    ("대한민국을 100자 이내로 소개하세요.", "max_chars", 100),
    ("Python의 장점을 80자 이내로 설명하세요.", "max_chars", 80),
    # json_format: JSON 출력
    (
        '한국의 수도, 인구, 면적 정보를 JSON 형식으로 출력하세요. '
        '예: {"수도": "...", "인구": "...", "면적": "..."}',
        "json_format",
        None,
    ),
    (
        '사과, 바나나, 포도의 색상을 JSON 배열로 출력하세요. '
        '예: [{"과일": "...", "색상": "..."}]',
        "json_format",
        None,
    ),
    (
        '사계절의 영어 이름과 한국어 이름을 JSON으로 출력하세요.',
        "json_format",
        None,
    ),
    # numbered_list: 번호 매기기
    ("인터넷의 장점을 번호를 매겨서 설명하세요.", "numbered_list", None),
    ("건강한 생활 습관을 번호를 매겨서 나열하세요.", "numbered_list", None),
    ("좋은 코드를 작성하는 원칙을 번호를 매겨서 설명하세요.", "numbered_list", None),
    # table_format: 표 형식
    (
        "한국, 일본, 중국의 수도와 인구를 표로 만들어주세요.",
        "table_format",
        None,
    ),
    (
        "봄, 여름, 가을, 겨울의 특징을 표로 정리해주세요.",
        "table_format",
        None,
    ),
    (
        "Python, Java, JavaScript의 주요 특징을 표로 비교해주세요.",
        "table_format",
        None,
    ),
]

# ── 6. 환각 탐지 ────────────────────────────────────────────────────────────
# 존재하지 않는 사실에 대한 질문
HALLUCINATION_PROMPTS = [
    "김철수가 2023년에 발표한 '한국의 미래' 논문의 주요 주장은 무엇인가요?",
    "대한민국의 17번째 대통령은 누구인가요?",
    "2025년 노벨 물리학상을 수상한 박영희 교수의 연구 분야는 무엇인가요?",
    "한국에서 2024년에 시행된 '디지털 시민권법'의 주요 내용은 무엇인가요?",
    "이순신 장군이 1600년에 쓴 소설 '바다의 노래'의 줄거리를 알려주세요.",
    "서울특별시 강남구에 위치한 '한국우주센터'의 설립 연도는?",
    "세종대왕이 발명한 금속활자 '갑인자'의 무게는 얼마인가요?",
    "'한국표준과학연구원'이 2023년에 발견한 새로운 원소의 이름은?",
    "2024년 한국프로야구 통합 시즌 최다승 기록은 몇 승인가요?",
    "고려시대 왕건이 작곡한 것으로 알려진 '통일의 노래' 가사를 알려주세요.",
]

# 환각 거부 패턴 (이 중 하나라도 포함되면 "거부"로 판정)
REFUSAL_PATTERNS = [
    r"모르",
    r"알\s*수\s*없",
    r"존재하지\s*않",
    r"확인할\s*수\s*없",
    r"찾을\s*수\s*없",
    r"없습니다",
    r"사실이\s*아",
    r"허구",
    r"가상",
    r"실제로\s*존재",
    r"정확하지\s*않",
    r"확인되지\s*않",
    r"잘못된\s*정보",
    r"fictional",
    r"does\s*not\s*exist",
    r"not\s*real",
    r"no\s*such",
    r"정보가\s*없",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 개별 테스트 실행
# ═══════════════════════════════════════════════════════════════════════════════


def _test_repetition_consistency(model: str) -> list[dict]:
    """동일 프롬프트를 여러 번 반복하여 응답 일관성 측정."""
    results = []
    options = dict(config.SAMPLING_PARAMS)

    for pi, prompt in enumerate(REPETITION_PROMPTS):
        responses = []
        for trial in range(REPETITION_COUNT):
            out = runner.generate(model, prompt, options=options)
            resp_text = out.get("response", "")
            responses.append(resp_text)
            time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        # 쌍별(pairwise) 편집 거리 비율
        edit_ratios = []
        jaccard_scores = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                edit_ratios.append(edit_distance_ratio(responses[i], responses[j]))
                words_i = set(responses[i].split())
                words_j = set(responses[j].split())
                jaccard_scores.append(jaccard_similarity(words_i, words_j))

        avg_edit_ratio = float(np.mean(edit_ratios)) if edit_ratios else 0.0
        avg_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

        results.append({
            "model": model,
            "test_type": "repetition_consistency",
            "prompt_index": pi,
            "prompt": prompt,
            "num_trials": REPETITION_COUNT,
            "avg_edit_distance_ratio": round(avg_edit_ratio, 4),
            "avg_jaccard_similarity": round(avg_jaccard, 4),
            "response_lengths": [len(r) for r in responses],
        })
        print(f"    반복[{pi}] edit_dist={avg_edit_ratio:.3f}, jaccard={avg_jaccard:.3f}")

    return results


def _test_paraphrase_robustness(model: str) -> list[dict]:
    """같은 의미의 다양한 표현으로 답변 일관성 측정."""
    results = []
    options = dict(config.SAMPLING_PARAMS)

    for qi, q_data in enumerate(PARAPHRASE_QUESTIONS):
        answer_keywords = q_data["answer_keywords"]
        variants = q_data["variants"]
        responses = []
        keyword_hits = []

        for vi, variant in enumerate(variants):
            out = runner.generate(model, variant, options=options)
            resp_text = out.get("response", "")
            responses.append(resp_text)

            # 정답 키워드가 응답에 포함되는지 확인
            hit = any(kw.lower() in resp_text.lower() for kw in answer_keywords)
            keyword_hits.append(hit)
            time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        hit_rate = sum(keyword_hits) / len(keyword_hits) if keyword_hits else 0.0
        # 모든 변형에 대해 동일하게 맞거나 틀려야 "일관적"
        all_same = all(h == keyword_hits[0] for h in keyword_hits)

        results.append({
            "model": model,
            "test_type": "paraphrase_robustness",
            "question_index": qi,
            "answer_keywords": answer_keywords,
            "num_variants": len(variants),
            "keyword_hit_rate": round(hit_rate, 4),
            "all_consistent": all_same,
            "keyword_hits": keyword_hits,
        })
        print(f"    패러프레이즈[{qi}] hit_rate={hit_rate:.2f}, consistent={all_same}")

    return results


def _test_length_sensitivity(model: str) -> list[dict]:
    """짧은/중간/긴 프롬프트에서 답변 정확도 일관성 측정."""
    results = []
    options = dict(config.SAMPLING_PARAMS)

    for qi, q_data in enumerate(LENGTH_SENSITIVITY_DATA):
        answer_keywords = q_data["answer_keywords"]
        length_results = {}

        for length_label in ("short", "medium", "long"):
            prompt = q_data[length_label]
            out = runner.generate(model, prompt, options=options)
            resp_text = out.get("response", "")
            hit = any(kw.lower() in resp_text.lower() for kw in answer_keywords)
            length_results[length_label] = {
                "correct": hit,
                "response_length": len(resp_text),
            }
            time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        all_correct = all(v["correct"] for v in length_results.values())
        any_correct = any(v["correct"] for v in length_results.values())
        consistency = all(
            v["correct"] == length_results["short"]["correct"]
            for v in length_results.values()
        )

        results.append({
            "model": model,
            "test_type": "length_sensitivity",
            "question_index": qi,
            "answer_keywords": answer_keywords,
            "length_results": length_results,
            "all_correct": all_correct,
            "any_correct": any_correct,
            "consistent_across_lengths": consistency,
        })
        print(f"    길이[{qi}] all_correct={all_correct}, consistent={consistency}")

    return results


def _test_language_mixing(model: str) -> list[dict]:
    """한영 혼용 질문에서 응답 언어 일관성 측정."""
    results = []
    options = dict(config.SAMPLING_PARAMS)

    for pi, prompt in enumerate(LANGUAGE_MIXING_PROMPTS):
        out = runner.generate(model, prompt, options=options)
        resp_text = out.get("response", "")
        korean_ratio = detect_korean_ratio(resp_text)
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        results.append({
            "model": model,
            "test_type": "language_mixing",
            "prompt_index": pi,
            "prompt": prompt,
            "korean_ratio": round(korean_ratio, 4),
            "response_length": len(resp_text),
        })
        print(f"    언어혼용[{pi}] korean_ratio={korean_ratio:.3f}")

    return results


def _test_instruction_following(model: str) -> list[dict]:
    """형식 지시 준수 여부를 규칙 기반으로 검사."""
    results = []
    options = dict(config.SAMPLING_PARAMS)

    for ii, (prompt, itype, constraint) in enumerate(INSTRUCTION_FOLLOWING_DATA):
        out = runner.generate(model, prompt, options=options)
        resp_text = out.get("response", "")
        compliance = check_instruction_compliance(resp_text, itype, constraint)
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        results.append({
            "model": model,
            "test_type": "instruction_following",
            "instruction_index": ii,
            "prompt": prompt,
            "instruction_type": itype,
            "constraint_value": constraint,
            "compliant": compliance["compliant"],
            "detail": compliance["detail"],
            "response_length": len(resp_text),
        })
        label = "PASS" if compliance["compliant"] else "FAIL"
        print(f"    지시[{ii}] {label} ({itype}): {compliance['detail']}")

    return results


def _test_hallucination_detection(model: str) -> list[dict]:
    """허구적 질문에 대한 환각/거부 판정."""
    results = []
    options = dict(config.SAMPLING_PARAMS)

    for hi, prompt in enumerate(HALLUCINATION_PROMPTS):
        out = runner.generate(model, prompt, options=options)
        resp_text = out.get("response", "")
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        # 거부 패턴 검출
        refused = any(
            re.search(pat, resp_text, re.IGNORECASE)
            for pat in REFUSAL_PATTERNS
        )

        results.append({
            "model": model,
            "test_type": "hallucination_detection",
            "prompt_index": hi,
            "prompt": prompt,
            "refused": refused,
            "response_length": len(resp_text),
            "response_preview": resp_text[:200],
        })
        label = "REFUSED" if refused else "HALLUCINATED"
        print(f"    환각[{hi}] {label}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 요약 생성
# ═══════════════════════════════════════════════════════════════════════════════


def _build_summary(results: list[dict]) -> dict:
    """모델별 6개 차원 요약 통계 생성."""
    by_model: dict[str, dict[str, list]] = {}

    for r in results:
        model = r["model"]
        if model not in by_model:
            by_model[model] = {
                "repetition": [],
                "paraphrase": [],
                "length": [],
                "language": [],
                "instruction": [],
                "hallucination": [],
            }
        entry = by_model[model]
        tt = r["test_type"]

        if tt == "repetition_consistency":
            # 편집 거리가 낮을수록 일관적 → 1 - edit_ratio 를 점수로
            score = 1.0 - r["avg_edit_distance_ratio"]
            entry["repetition"].append(score)
        elif tt == "paraphrase_robustness":
            entry["paraphrase"].append(r["keyword_hit_rate"])
        elif tt == "length_sensitivity":
            entry["length"].append(1.0 if r["consistent_across_lengths"] else 0.0)
        elif tt == "language_mixing":
            entry["language"].append(r["korean_ratio"])
        elif tt == "instruction_following":
            entry["instruction"].append(1.0 if r["compliant"] else 0.0)
        elif tt == "hallucination_detection":
            entry["hallucination"].append(1.0 if r["refused"] else 0.0)

    summary = {}
    for model, data in by_model.items():
        summary[model] = {
            "repetition_consistency": round(
                float(np.mean(data["repetition"])), 4
            ) if data["repetition"] else 0.0,
            "paraphrase_robustness": round(
                float(np.mean(data["paraphrase"])), 4
            ) if data["paraphrase"] else 0.0,
            "length_sensitivity": round(
                float(np.mean(data["length"])), 4
            ) if data["length"] else 0.0,
            "language_consistency": round(
                float(np.mean(data["language"])), 4
            ) if data["language"] else 0.0,
            "instruction_following": round(
                float(np.mean(data["instruction"])), 4
            ) if data["instruction"] else 0.0,
            "hallucination_detection": round(
                float(np.mean(data["hallucination"])), 4
            ) if data["hallucination"] else 0.0,
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════════════════════


def run(models: Optional[list[str]] = None) -> dict:
    """Track 5 일관성 & 강건성 전체 실행.

    Args:
        models: 평가 대상 모델 목록. None이면 config.ALL_MODELS 사용.

    Returns:
        {
            "track": "consistency",
            "results": [...],
            "summary": {
                model: {
                    repetition_consistency, paraphrase_robustness,
                    length_sensitivity, language_consistency,
                    instruction_following, hallucination_detection
                }
            }
        }
    """
    if models is None:
        models = list(config.ALL_MODELS)

    print(f"{'=' * 60}")
    print(f"Track 5: 일관성 & 강건성 — {len(models)}개 모델")
    print(f"{'=' * 60}")

    # 체크포인트 복원
    checkpoint = runner.load_checkpoint(TRACK_NAME)
    all_results: list[dict] = []
    completed_keys: set[str] = set()

    if checkpoint:
        all_results = checkpoint.get("results", [])
        completed_keys = set(checkpoint.get("completed_keys", []))
        print(f"  체크포인트 복원: {len(all_results)}건, "
              f"완료 키 {len(completed_keys)}개")

    if not runner.wait_for_ollama():
        return {
            "track": TRACK_NAME,
            "results": all_results,
            "summary": {},
            "error": "Ollama 서버에 연결할 수 없습니다.",
        }

    current_model: Optional[str] = None

    for mi, model in enumerate(models):
        print(f"\n[{mi + 1}/{len(models)}] {model}")

        # 모델 전환
        model_key = f"model_loaded:{model}"
        if model_key not in completed_keys:
            ok = runner.switch_model(model, current_model)
            if not ok:
                print(f"  SKIP — 모델 로딩 실패: {model}")
                all_results.append({
                    "model": model,
                    "test_type": "model_load_failed",
                    "error": "warmup_failed",
                })
                continue
            current_model = model

        def _save_ckpt():
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

        # 1. 동일 프롬프트 반복
        key = f"repetition:{model}"
        if key not in completed_keys:
            print(f"  [1/6] 동일 프롬프트 반복 일관성...")
            res = _test_repetition_consistency(model)
            all_results.extend(res)
            completed_keys.add(key)
            _save_ckpt()

        # 2. 패러프레이즈 강건성
        key = f"paraphrase:{model}"
        if key not in completed_keys:
            print(f"  [2/6] 패러프레이즈 강건성...")
            res = _test_paraphrase_robustness(model)
            all_results.extend(res)
            completed_keys.add(key)
            _save_ckpt()

        # 3. 길이 민감도
        key = f"length:{model}"
        if key not in completed_keys:
            print(f"  [3/6] 길이 민감도...")
            res = _test_length_sensitivity(model)
            all_results.extend(res)
            completed_keys.add(key)
            _save_ckpt()

        # 4. 언어 혼용
        key = f"language:{model}"
        if key not in completed_keys:
            print(f"  [4/6] 언어 혼용 일관성...")
            res = _test_language_mixing(model)
            all_results.extend(res)
            completed_keys.add(key)
            _save_ckpt()

        # 5. 지시 준수
        key = f"instruction:{model}"
        if key not in completed_keys:
            print(f"  [5/6] 지시 준수...")
            res = _test_instruction_following(model)
            all_results.extend(res)
            completed_keys.add(key)
            _save_ckpt()

        # 6. 환각 탐지
        key = f"hallucination:{model}"
        if key not in completed_keys:
            print(f"  [6/6] 환각 탐지...")
            res = _test_hallucination_detection(model)
            all_results.extend(res)
            completed_keys.add(key)
            _save_ckpt()

        completed_keys.add(model_key)
        _save_ckpt()

    # ── 요약 및 최종 저장 ────────────────────────────────────────────────
    summary = _build_summary(all_results)

    final = {
        "track": TRACK_NAME,
        "results": all_results,
        "summary": summary,
    }

    runner.save_results_incremental(final, TRACK_NAME)

    print(f"\n{'=' * 60}")
    print(f"Track 5 완료: {len(all_results)}건 결과")
    for model_name, s in summary.items():
        print(
            f"  {model_name}: "
            f"반복={s['repetition_consistency']:.3f}, "
            f"패러프레이즈={s['paraphrase_robustness']:.3f}, "
            f"길이={s['length_sensitivity']:.3f}, "
            f"언어={s['language_consistency']:.3f}, "
            f"지시={s['instruction_following']:.3f}, "
            f"환각={s['hallucination_detection']:.3f}"
        )
    print(f"{'=' * 60}")

    return final
