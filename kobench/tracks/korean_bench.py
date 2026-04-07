"""
Track 1 — 한국어 표준 벤치마크 (KoBEST + KMMLU subset)

lm-evaluation-harness를 우선 사용하고, 설치되어 있지 않으면
내장 standalone 문항으로 평가한다.
"""

import json
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

from kobench import config
from kobench import runner

TRACK_NAME = "korean_bench"

# ─────────────────────────────────────────────────────────────────────────────
# lm-evaluation-harness wrapper
# ─────────────────────────────────────────────────────────────────────────────

LM_EVAL_TASKS = ",".join(config.TRACK1_TASKS)


def _lm_eval_available() -> bool:
    """lm_eval CLI가 PATH에 존재하는지 확인"""
    return shutil.which("lm_eval") is not None


def _run_lm_eval(model: str) -> Optional[dict]:
    """
    lm-evaluation-harness를 subprocess로 실행.
    Returns parsed results dict or None on failure.
    """
    base_url = f"{config.OLLAMA_BASE_URL}/v1/completions"
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--tasks", LM_EVAL_TASKS,
        "--model_args", f"model={model},base_url={base_url},num_concurrent=1,max_retries=3,tokenized_requests=False",
        "--num_fewshot", "0",
        "--batch_size", "1",
        "--output_path", str(config.RESULTS_DIR / f"lm_eval_{model}"),
        "--log_samples",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30분 제한
        )
        if result.returncode != 0:
            print(f"    lm_eval 실패 (exit {result.returncode}): {result.stderr[:500]}")
            return None

        # lm_eval은 results 폴더에 JSON을 저장한다
        output_dir = config.RESULTS_DIR / f"lm_eval_{model}"
        results_files = list(output_dir.rglob("results*.json"))
        if not results_files:
            print("    lm_eval 결과 파일을 찾을 수 없습니다")
            return None

        with open(results_files[0], encoding="utf-8") as f:
            raw = json.load(f)

        parsed = {}
        for task_key, metrics in raw.get("results", {}).items():
            acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or 0.0
            parsed[task_key] = round(acc, 4)
        return parsed

    except subprocess.TimeoutExpired:
        print("    lm_eval 타임아웃 (30분)")
        return None
    except Exception as e:
        print(f"    lm_eval 오류: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone 벤치마크 문항
# ─────────────────────────────────────────────────────────────────────────────

def _build_kobest_boolq() -> list[dict]:
    """KoBEST-BoolQ: 예/아니오 독해 문항 20개"""
    items = [
        {"passage": "대한민국의 수도는 서울특별시이며, 인구는 약 950만 명이다.", "question": "대한민국의 수도는 부산인가?", "answer": 1},
        {"passage": "한글은 1443년 세종대왕이 창제하여 1446년에 반포하였다.", "question": "한글은 세종대왕이 만들었는가?", "answer": 0},
        {"passage": "김치는 배추, 고춧가루, 마늘 등을 넣어 발효시킨 한국 전통 음식이다.", "question": "김치에는 고춧가루가 들어가는가?", "answer": 0},
        {"passage": "제주도는 대한민국에서 가장 큰 섬으로, 화산 활동으로 형성되었다.", "question": "제주도는 화산섬인가?", "answer": 0},
        {"passage": "한국의 법정 성인 나이는 만 19세이다.", "question": "한국에서 만 18세는 성인인가?", "answer": 1},
        {"passage": "경복궁은 조선 시대에 지어진 궁궐로, 서울 종로구에 위치해 있다.", "question": "경복궁은 조선 시대에 지어졌는가?", "answer": 0},
        {"passage": "한국전쟁은 1950년 6월 25일에 발발하여 1953년 7월 27일 정전 협정으로 멈추었다.", "question": "한국전쟁은 1945년에 시작되었는가?", "answer": 1},
        {"passage": "태극기는 흰색 바탕에 태극 무늬와 네 모서리의 괘로 이루어져 있다.", "question": "태극기에는 태극 무늬가 있는가?", "answer": 0},
        {"passage": "한강은 서울을 관통하며 서해로 흘러들어 간다.", "question": "한강은 동해로 흘러 들어가는가?", "answer": 1},
        {"passage": "한국의 전통 의복인 한복은 저고리와 치마(여성) 또는 바지(남성)로 구성된다.", "question": "한복은 한국의 전통 의복인가?", "answer": 0},
        {"passage": "비빔밥은 밥 위에 나물, 고추장, 고기 등을 얹어 비벼 먹는 음식이다.", "question": "비빔밥은 일본 전통 음식인가?", "answer": 1},
        {"passage": "독도는 대한민국 경상북도 울릉군에 속하는 섬이다.", "question": "독도는 대한민국 영토인가?", "answer": 0},
        {"passage": "한국어에서 존댓말은 상대의 나이와 사회적 지위에 따라 사용된다.", "question": "한국어에는 존댓말이 존재하는가?", "answer": 0},
        {"passage": "2002년 한일 월드컵에서 대한민국은 4강까지 진출하였다.", "question": "대한민국은 2002년 월드컵에서 결승에 진출했는가?", "answer": 1},
        {"passage": "서울의 지하철 노선은 총 9개 이상이며, 세계적으로 이용객이 많은 대중교통 시스템이다.", "question": "서울에는 지하철이 있는가?", "answer": 0},
        {"passage": "떡볶이는 떡, 어묵, 고추장 소스를 사용하는 한국 길거리 음식이다.", "question": "떡볶이에는 어묵이 사용되는가?", "answer": 0},
        {"passage": "판문점은 남북한의 군사분계선 위에 위치한 공동경비구역이다.", "question": "판문점은 남한 내부에만 위치하는가?", "answer": 1},
        {"passage": "한국의 교육 체계는 초등학교 6년, 중학교 3년, 고등학교 3년이다.", "question": "한국의 중학교는 3년제인가?", "answer": 0},
        {"passage": "불국사는 경주에 위치한 유네스코 세계문화유산이다.", "question": "불국사는 서울에 위치해 있는가?", "answer": 1},
        {"passage": "삼성전자는 반도체, 스마트폰 등을 생산하는 한국의 대표적인 기업이다.", "question": "삼성전자는 한국 기업인가?", "answer": 0},
    ]
    # answer: 0=예, 1=아니오
    result = []
    for i, item in enumerate(items):
        result.append({
            "id": f"boolq_{i:03d}",
            "benchmark": "kobest_boolq",
            "question": f"다음 지문을 읽고 질문에 '예' 또는 '아니오'로만 답하세요.\n\n지문: {item['passage']}\n질문: {item['question']}",
            "choices": ["예", "아니오"],
            "answer": item["answer"],
            "subject": "",
        })
    return result


def _build_kobest_copa() -> list[dict]:
    """KoBEST-COPA: 2지선다 인과관계 추론 20문항"""
    items = [
        {"premise": "비가 많이 내렸다.", "question": "결과는?", "choices": ["거리가 침수되었다.", "사막이 확장되었다."], "answer": 0},
        {"premise": "아이가 시험에서 만점을 받았다.", "question": "원인은?", "choices": ["공부를 열심히 했다.", "아침을 거르었다."], "answer": 0},
        {"premise": "냉장고가 고장났다.", "question": "결과는?", "choices": ["음식이 상했다.", "전기세가 줄었다."], "answer": 0},
        {"premise": "교통사고가 발생했다.", "question": "결과는?", "choices": ["도로가 정체되었다.", "주가가 상승했다."], "answer": 0},
        {"premise": "그녀는 매일 운동을 했다.", "question": "결과는?", "choices": ["건강이 좋아졌다.", "감기에 걸렸다."], "answer": 0},
        {"premise": "전쟁이 발발했다.", "question": "결과는?", "choices": ["경제가 성장했다.", "많은 사람이 피난을 갔다."], "answer": 1},
        {"premise": "학생이 도서관에서 책을 빌렸다.", "question": "원인은?", "choices": ["레포트를 써야 했다.", "도서관 문이 잠겨 있었다."], "answer": 0},
        {"premise": "감독이 선수를 교체했다.", "question": "원인은?", "choices": ["해당 선수의 경기력이 좋지 않았다.", "경기에서 이기고 있었다."], "answer": 0},
        {"premise": "회사가 대규모 채용을 진행했다.", "question": "원인은?", "choices": ["사업이 확장되었다.", "직원들이 파업했다."], "answer": 0},
        {"premise": "눈이 많이 왔다.", "question": "결과는?", "choices": ["학교가 휴교했다.", "벚꽃이 피었다."], "answer": 0},
        {"premise": "그는 매운 음식을 먹었다.", "question": "결과는?", "choices": ["배가 아팠다.", "살이 빠졌다."], "answer": 0},
        {"premise": "컴퓨터 바이러스에 감염되었다.", "question": "결과는?", "choices": ["파일이 손상되었다.", "인터넷 속도가 빨라졌다."], "answer": 0},
        {"premise": "가뭄이 계속되었다.", "question": "결과는?", "choices": ["농작물이 말랐다.", "홍수가 발생했다."], "answer": 0},
        {"premise": "아기가 울음을 멈추었다.", "question": "원인은?", "choices": ["엄마가 안아주었다.", "번개가 쳤다."], "answer": 0},
        {"premise": "그는 면접에서 떨어졌다.", "question": "원인은?", "choices": ["준비가 부족했다.", "날씨가 좋았다."], "answer": 0},
        {"premise": "알람이 울리지 않았다.", "question": "결과는?", "choices": ["지각을 했다.", "일찍 도착했다."], "answer": 0},
        {"premise": "공장에서 폐수를 방류했다.", "question": "결과는?", "choices": ["강물이 오염되었다.", "물고기가 많아졌다."], "answer": 0},
        {"premise": "새 법률이 시행되었다.", "question": "원인은?", "choices": ["국회에서 법안을 통과시켰다.", "대통령이 해외에 갔다."], "answer": 0},
        {"premise": "환자가 회복되었다.", "question": "원인은?", "choices": ["적절한 치료를 받았다.", "병원 문을 닫았다."], "answer": 0},
        {"premise": "겨울에 난방비가 올랐다.", "question": "원인은?", "choices": ["에너지 가격이 상승했다.", "여름이 길었다."], "answer": 0},
    ]
    result = []
    for i, item in enumerate(items):
        q = f"전제: {item['premise']}\n{item['question']}\n\nA. {item['choices'][0]}\nB. {item['choices'][1]}\n\n정답을 A 또는 B로만 답하세요."
        result.append({
            "id": f"copa_{i:03d}",
            "benchmark": "kobest_copa",
            "question": q,
            "choices": item["choices"],
            "answer": item["answer"],
            "subject": "",
        })
    return result


def _build_kobest_sentineg() -> list[dict]:
    """KoBEST-SentiNeg: 부정어 포함 감성 분석 20문항"""
    items = [
        {"text": "이 영화는 지루하지 않았다.", "answer": 0},   # 긍정
        {"text": "서비스가 좋지 않았다.", "answer": 1},         # 부정
        {"text": "음식이 맛없다고 할 수 없다.", "answer": 0},   # 긍정
        {"text": "배우의 연기는 나쁘지 않았다.", "answer": 0},   # 긍정
        {"text": "가격이 합리적이지 못하다.", "answer": 1},       # 부정
        {"text": "이 제품은 기대에 못 미치지 않는다.", "answer": 0},
        {"text": "직원들이 불친절하다.", "answer": 1},
        {"text": "여행이 재미없지는 않았다.", "answer": 0},
        {"text": "숙소가 깨끗하지 않았다.", "answer": 1},
        {"text": "경치가 아름답지 않다고 할 수 없다.", "answer": 0},
        {"text": "배송이 느려서 불만족스럽다.", "answer": 1},
        {"text": "강의가 이해하기 어렵지 않았다.", "answer": 0},
        {"text": "이 가게의 음식은 맛이 없다.", "answer": 1},
        {"text": "공연은 실망스럽지 않았다.", "answer": 0},
        {"text": "대기 시간이 짧지 않았다.", "answer": 1},
        {"text": "품질이 나쁘다고는 볼 수 없다.", "answer": 0},
        {"text": "소음이 심해서 불편하다.", "answer": 1},
        {"text": "결과가 만족스럽지 않다.", "answer": 1},
        {"text": "디자인이 별로라고 할 수 없다.", "answer": 0},
        {"text": "이 책은 읽을 가치가 없지 않다.", "answer": 0},
    ]
    result = []
    for i, item in enumerate(items):
        q = f"다음 문장의 감성을 판단하세요.\n\n\"{item['text']}\"\n\nA. 긍정\nB. 부정\n\n정답을 A 또는 B로만 답하세요."
        result.append({
            "id": f"sentineg_{i:03d}",
            "benchmark": "kobest_sentineg",
            "question": q,
            "choices": ["긍정", "부정"],
            "answer": item["answer"],
            "subject": "",
        })
    return result


def _build_kobest_hellaswag() -> list[dict]:
    """KoBEST-HellaSwag: 4지선다 상식 추론 20문항"""
    items = [
        {
            "context": "요리사가 프라이팬에 기름을 두르고 불을 켰다.",
            "choices": ["달걀을 깨서 팬에 넣었다.", "프라이팬을 냉장고에 넣었다.", "기름을 세면대에 버렸다.", "불을 끄고 집을 나갔다."],
            "answer": 0,
        },
        {
            "context": "학생이 도서관에 들어가 빈 자리를 찾았다.",
            "choices": ["가방을 놓고 책을 펼쳤다.", "수영복으로 갈아입었다.", "축구공을 꺼냈다.", "텐트를 설치했다."],
            "answer": 0,
        },
        {
            "context": "의사가 환자의 혈압을 측정한 뒤 고개를 끄덕였다.",
            "choices": ["혈압이 정상 범위라고 말했다.", "환자에게 점프를 시켰다.", "병원 문을 잠갔다.", "환자를 무시하고 퇴근했다."],
            "answer": 0,
        },
        {
            "context": "기차가 역에 도착하자 문이 열렸다.",
            "choices": ["승객들이 내리기 시작했다.", "기차가 하늘로 날아올랐다.", "역이 사라졌다.", "기관사가 수영을 시작했다."],
            "answer": 0,
        },
        {
            "context": "아이가 공원에서 그네를 타다가 갑자기 멈추었다.",
            "choices": ["엄마가 부르는 소리를 들었다.", "그네가 우주로 발사되었다.", "공원이 바다가 되었다.", "아이가 투명해졌다."],
            "answer": 0,
        },
        {
            "context": "바리스타가 에스프레소 머신의 전원을 켰다.",
            "choices": ["원두를 그라인더에 넣었다.", "머신을 창밖으로 던졌다.", "에스프레소 머신에 빨래를 넣었다.", "콘센트를 뽑고 나갔다."],
            "answer": 0,
        },
        {
            "context": "비행기가 활주로를 달리며 속도를 높이고 있었다.",
            "choices": ["이륙하여 하늘로 올라갔다.", "뒤로 후진하기 시작했다.", "활주로에서 주차를 했다.", "날개를 접고 지하로 들어갔다."],
            "answer": 0,
        },
        {
            "context": "그녀는 빈 캔버스 앞에 팔레트와 붓을 준비했다.",
            "choices": ["물감을 섞어 첫 붓질을 시작했다.", "캔버스를 먹었다.", "붓으로 머리를 빗었다.", "팔레트를 베개로 사용했다."],
            "answer": 0,
        },
        {
            "context": "소방관들이 불이 난 건물에 도착하여 호스를 연결했다.",
            "choices": ["물을 뿌려 진화를 시작했다.", "호스로 줄넘기를 했다.", "건물 옆에서 낮잠을 잤다.", "불에 장작을 더 넣었다."],
            "answer": 0,
        },
        {
            "context": "축구 선수가 페널티킥 위치에 공을 올려놓았다.",
            "choices": ["골대를 향해 공을 찼다.", "공을 들고 관중석으로 갔다.", "공 위에 앉았다.", "심판에게 공을 선물했다."],
            "answer": 0,
        },
        {
            "context": "농부가 아침에 일어나 밭으로 나갔다.",
            "choices": ["작물에 물을 주기 시작했다.", "밭에 수영장을 팠다.", "밭 위에서 스키를 탔다.", "작물을 모두 뽑아 버렸다."],
            "answer": 0,
        },
        {
            "context": "어부가 그물을 바다에 던졌다.",
            "choices": ["물고기가 잡히기를 기다렸다.", "그물을 타고 서핑을 했다.", "바다에 연을 날렸다.", "그물로 옷을 만들었다."],
            "answer": 0,
        },
        {
            "context": "프로그래머가 코드를 작성하다가 에러 메시지를 보았다.",
            "choices": ["디버깅을 시작했다.", "컴퓨터를 창밖으로 던졌다.", "에러 메시지에 답장을 보냈다.", "모니터를 냉장고에 넣었다."],
            "answer": 0,
        },
        {
            "context": "택배기사가 아파트 현관에 도착하여 벨을 눌렀다.",
            "choices": ["주민이 문을 열고 택배를 받았다.", "택배기사가 벽을 통과했다.", "현관문이 노래를 불렀다.", "아파트가 이사를 갔다."],
            "answer": 0,
        },
        {
            "context": "피아니스트가 무대에 올라 피아노 앞에 앉았다.",
            "choices": ["건반 위에 손을 올려놓고 연주를 시작했다.", "피아노를 들어서 옮겼다.", "피아노 위에 누워 잠들었다.", "객석으로 피아노를 밀었다."],
            "answer": 0,
        },
        {
            "context": "등산객이 산 정상에 도착하여 배낭을 내려놓았다.",
            "choices": ["주변 경치를 감상하며 물을 마셨다.", "정상에서 스키를 탔다.", "배낭을 산 아래로 던졌다.", "정상에 집을 지었다."],
            "answer": 0,
        },
        {
            "context": "간호사가 주사기를 준비하고 환자의 팔을 소독했다.",
            "choices": ["조심스럽게 주사를 놓았다.", "주사기로 그림을 그렸다.", "환자의 팔에 스티커를 붙였다.", "소독약을 마셨다."],
            "answer": 0,
        },
        {
            "context": "사진작가가 해질녘에 카메라를 삼각대 위에 올려놓았다.",
            "choices": ["노을을 배경으로 셔터를 눌렀다.", "카메라를 바다에 던졌다.", "삼각대로 낚시를 했다.", "카메라에 밥을 담았다."],
            "answer": 0,
        },
        {
            "context": "우체부가 편지 묶음을 들고 첫 번째 집 앞에 섰다.",
            "choices": ["우편함에 편지를 넣었다.", "편지로 종이비행기를 접었다.", "편지를 땅에 묻었다.", "집 지붕 위로 편지를 던졌다."],
            "answer": 0,
        },
        {
            "context": "심판이 호루라기를 불며 손을 들어올렸다.",
            "choices": ["선수에게 반칙을 선언했다.", "호루라기를 먹었다.", "하늘로 날아올랐다.", "운동장에 꽃을 심었다."],
            "answer": 0,
        },
    ]
    result = []
    for i, item in enumerate(items):
        choices_str = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(item["choices"]))
        q = f"다음 상황에 이어질 가장 자연스러운 문장을 고르세요.\n\n상황: {item['context']}\n\n{choices_str}\n\n정답을 A, B, C, D 중 하나로만 답하세요."
        result.append({
            "id": f"hellaswag_{i:03d}",
            "benchmark": "kobest_hellaswag",
            "question": q,
            "choices": item["choices"],
            "answer": item["answer"],
            "subject": "",
        })
    return result


def _build_kmmlu() -> list[dict]:
    """KMMLU subset: 10 과목 x 5문항 = 50문항"""
    subjects = {
        "한국사": [
            {"q": "고조선의 건국 시기로 알려진 해는?", "c": ["기원전 2333년", "기원전 108년", "기원전 57년", "기원전 37년"], "a": 0},
            {"q": "임진왜란이 발발한 해는?", "c": ["1592년", "1636년", "1894년", "1446년"], "a": 0},
            {"q": "대한민국 임시정부가 수립된 도시는?", "c": ["상하이", "도쿄", "베이징", "워싱턴"], "a": 0},
            {"q": "한글 창제를 주도한 왕은?", "c": ["세종대왕", "태조", "영조", "정조"], "a": 0},
            {"q": "6.25 전쟁의 정전 협정이 체결된 해는?", "c": ["1950년", "1951년", "1953년", "1955년"], "a": 2},
        ],
        "한국지리": [
            {"q": "대한민국에서 가장 높은 산은?", "c": ["한라산", "지리산", "설악산", "북한산"], "a": 0},
            {"q": "한강이 흘러드는 바다는?", "c": ["서해(황해)", "동해", "남해", "북해"], "a": 0},
            {"q": "대한민국에서 가장 큰 섬은?", "c": ["제주도", "거제도", "강화도", "울릉도"], "a": 0},
            {"q": "영남 지방에 해당하는 도가 아닌 것은?", "c": ["충청남도", "경상남도", "경상북도", "부산광역시"], "a": 0},
            {"q": "다음 중 광역시가 아닌 것은?", "c": ["수원시", "부산광역시", "대구광역시", "인천광역시"], "a": 0},
        ],
        "경제": [
            {"q": "GDP의 풀네임은?", "c": ["Gross Domestic Product", "General Demand Price", "Global Data Protocol", "Gross Direct Profit"], "a": 0},
            {"q": "인플레이션의 의미는?", "c": ["물가의 지속적 상승", "물가의 지속적 하락", "환율의 상승", "금리의 하락"], "a": 0},
            {"q": "중앙은행이 기준금리를 올리면 일반적으로 나타나는 현상은?", "c": ["시중 대출 금리 상승", "통화량 증가", "물가 상승", "소비 증가"], "a": 0},
            {"q": "수요-공급 법칙에서 가격이 상승하면 공급량은?", "c": ["증가한다", "감소한다", "변하지 않는다", "불확실하다"], "a": 0},
            {"q": "한국은행의 역할이 아닌 것은?", "c": ["세금 징수", "통화 정책 수행", "화폐 발행", "금융 안정 유지"], "a": 0},
        ],
        "법학": [
            {"q": "대한민국 헌법에서 보장하는 기본권이 아닌 것은?", "c": ["특권", "평등권", "자유권", "사회권"], "a": 0},
            {"q": "민법에서 성년의 나이는?", "c": ["만 19세", "만 18세", "만 20세", "만 21세"], "a": 0},
            {"q": "형법에서 '고의'란?", "c": ["범죄 사실을 인식하고 의도함", "우연히 발생한 것", "법률의 부재", "무죄 추정"], "a": 0},
            {"q": "대한민국의 최고 법규범은?", "c": ["헌법", "민법", "형법", "상법"], "a": 0},
            {"q": "삼권분립에 해당하지 않는 것은?", "c": ["언론권", "입법권", "행정권", "사법권"], "a": 0},
        ],
        "물리": [
            {"q": "뉴턴의 제2법칙 F = ma에서 a는?", "c": ["가속도", "면적", "진폭", "각도"], "a": 0},
            {"q": "빛의 속도는 약 몇 m/s인가?", "c": ["3 x 10^8", "3 x 10^6", "3 x 10^10", "3 x 10^4"], "a": 0},
            {"q": "전류의 단위는?", "c": ["암페어(A)", "볼트(V)", "와트(W)", "옴(Ω)"], "a": 0},
            {"q": "에너지 보존 법칙에 따르면 에너지는?", "c": ["형태는 변하지만 총량은 보존된다", "계속 증가한다", "계속 감소한다", "생성될 수 있다"], "a": 0},
            {"q": "파동에서 진동수의 단위는?", "c": ["Hz", "m", "N", "J"], "a": 0},
        ],
        "화학": [
            {"q": "물(H₂O)에서 수소와 산소의 원자 비율은?", "c": ["2:1", "1:2", "1:1", "3:1"], "a": 0},
            {"q": "pH 7은 어떤 성질을 나타내는가?", "c": ["중성", "산성", "염기성", "알칼리성"], "a": 0},
            {"q": "주기율표에서 원소를 구분하는 기준은?", "c": ["원자번호", "질량수", "전자수", "밀도"], "a": 0},
            {"q": "산과 염기가 반응하면 생성되는 것은?", "c": ["물과 염", "기체와 금속", "산화물과 수소", "알코올과 에스터"], "a": 0},
            {"q": "다이아몬드와 흑연의 관계는?", "c": ["동소체", "이성질체", "동위원소", "화합물"], "a": 0},
        ],
        "생물": [
            {"q": "세포에서 유전 정보를 담고 있는 물질은?", "c": ["DNA", "단백질", "탄수화물", "지방"], "a": 0},
            {"q": "광합성이 일어나는 세포 소기관은?", "c": ["엽록체", "미토콘드리아", "리보솜", "골지체"], "a": 0},
            {"q": "인체에서 산소를 운반하는 혈액 성분은?", "c": ["적혈구", "백혈구", "혈소판", "혈장"], "a": 0},
            {"q": "멘델의 유전 법칙 중 '우열의 법칙'이란?", "c": ["우성 형질이 표현형에 나타남", "유전자가 독립적으로 분리됨", "형질이 섞임", "돌연변이 발생"], "a": 0},
            {"q": "다음 중 원핵생물은?", "c": ["대장균", "아메바", "효모", "버섯"], "a": 0},
        ],
        "컴퓨터공학": [
            {"q": "2진수 1010을 10진수로 변환하면?", "c": ["10", "8", "12", "5"], "a": 0},
            {"q": "운영체제(OS)의 역할이 아닌 것은?", "c": ["웹 디자인", "프로세스 관리", "메모리 관리", "파일 시스템 관리"], "a": 0},
            {"q": "TCP/IP에서 TCP의 역할은?", "c": ["신뢰성 있는 데이터 전송", "IP 주소 할당", "도메인 이름 변환", "암호화"], "a": 0},
            {"q": "시간 복잡도 O(n log n)에 해당하는 정렬 알고리즘은?", "c": ["병합 정렬", "버블 정렬", "선택 정렬", "삽입 정렬"], "a": 0},
            {"q": "데이터베이스에서 기본키(Primary Key)의 특징은?", "c": ["유일하고 NULL 불가", "중복 가능", "NULL 허용", "자동 삭제"], "a": 0},
        ],
        "의학": [
            {"q": "인체에서 가장 큰 장기는?", "c": ["간", "심장", "위", "폐"], "a": 0},
            {"q": "혈당을 조절하는 호르몬은?", "c": ["인슐린", "아드레날린", "갑상선 호르몬", "성장 호르몬"], "a": 0},
            {"q": "항생제가 효과적인 대상은?", "c": ["세균 감염", "바이러스 감염", "알레르기", "유전 질환"], "a": 0},
            {"q": "심폐소생술(CPR)에서 가장 먼저 해야 할 것은?", "c": ["의식 확인", "인공호흡", "가슴 압박", "AED 사용"], "a": 0},
            {"q": "백신의 원리는?", "c": ["면역 반응을 미리 유도", "바이러스를 직접 제거", "항생제를 투여", "체온을 낮춤"], "a": 0},
        ],
        "윤리": [
            {"q": "칸트 윤리학의 핵심 개념은?", "c": ["정언명령", "공리주의", "사회계약", "덕 윤리"], "a": 0},
            {"q": "공리주의의 기본 원칙은?", "c": ["최대 다수의 최대 행복", "개인의 자유", "의무의 준수", "자연 상태 보존"], "a": 0},
            {"q": "생명윤리에서 '자율성 존중 원칙'이란?", "c": ["환자의 결정을 존중", "의사의 판단이 절대적", "가족의 동의만 필요", "국가가 결정"], "a": 0},
            {"q": "환경윤리에서 '지속가능한 발전'이란?", "c": ["미래 세대의 필요를 충족하면서 현재 세대의 필요도 충족", "경제 성장 우선", "자연 보호만 추구", "과거로의 회귀"], "a": 0},
            {"q": "정의론으로 유명한 철학자는?", "c": ["롤스", "마르크스", "니체", "공자"], "a": 0},
        ],
    }

    result = []
    idx = 0
    for subject, questions in subjects.items():
        for qi, item in enumerate(questions):
            choices_str = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(item["c"]))
            q = f"[{subject}] 다음 문제의 정답을 A, B, C, D 중 하나로만 답하세요.\n\n{item['q']}\n\n{choices_str}"
            result.append({
                "id": f"kmmlu_{idx:03d}",
                "benchmark": "kmmlu",
                "question": q,
                "choices": item["c"],
                "answer": item["a"],
                "subject": subject,
            })
            idx += 1
    return result


def _build_all_questions() -> list[dict]:
    """전체 standalone 문항 생성"""
    questions = []
    questions.extend(_build_kobest_boolq())
    questions.extend(_build_kobest_copa())
    questions.extend(_build_kobest_sentineg())
    questions.extend(_build_kobest_hellaswag())
    questions.extend(_build_kmmlu())
    return questions


# ─────────────────────────────────────────────────────────────────────────────
# 응답 파싱
# ─────────────────────────────────────────────────────────────────────────────

_CHOICE_RE = re.compile(r"^[^A-Da-d예아]*([A-Da-d]|예|아니오)", re.MULTILINE)


def _parse_answer(response: str, benchmark: str, num_choices: int) -> Optional[int]:
    """
    모델 응답에서 정답 인덱스(0-indexed) 추출.
    BoolQ: 예->0, 아니오->1
    COPA/SentiNeg: A->0, B->1
    HellaSwag/KMMLU: A->0, B->1, C->2, D->3
    """
    text = response.strip()

    # BoolQ: 예/아니오 매칭 우선
    if benchmark == "kobest_boolq":
        if "아니오" in text[:20]:
            return 1
        if "예" in text[:10]:
            return 0

    # A-D 매칭
    m = _CHOICE_RE.search(text)
    if m:
        choice = m.group(1).upper()
        if choice in "ABCD":
            idx = ord(choice) - ord("A")
            if idx < num_choices:
                return idx
        if choice == "예":
            return 0
        if choice == "아니오":
            return 1

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone 평가 실행
# ─────────────────────────────────────────────────────────────────────────────

def _run_standalone(model: str, questions: list[dict]) -> dict:
    """
    Standalone 모드로 단일 모델 평가.
    Returns: {"model": str, "scores": {benchmark: accuracy}, "details": [...]}
    """
    details = []
    correct_by_bench = defaultdict(int)
    total_by_bench = defaultdict(int)

    for qi, q in enumerate(questions):
        result = runner.generate(
            model=model,
            prompt=q["question"],
            system="당신은 한국어 시험 문제에 정확히 답하는 AI입니다. 정답 기호만 출력하세요.",
            options=dict(config.BENCHMARK_SAMPLING),
        )

        if result["error"]:
            parsed = None
            is_correct = False
        else:
            parsed = _parse_answer(result["response"], q["benchmark"], len(q["choices"]))
            is_correct = parsed == q["answer"]

        if is_correct:
            correct_by_bench[q["benchmark"]] += 1
        total_by_bench[q["benchmark"]] += 1

        details.append({
            "id": q["id"],
            "benchmark": q["benchmark"],
            "subject": q.get("subject", ""),
            "expected": q["answer"],
            "parsed": parsed,
            "correct": is_correct,
            "raw_response": result["response"][:200],
            "error": result["error"],
        })

        if (qi + 1) % 10 == 0:
            print(f"    진행: {qi + 1}/{len(questions)}")

        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

    scores = {}
    for bench in sorted(total_by_bench.keys()):
        total = total_by_bench[bench]
        correct = correct_by_bench[bench]
        scores[bench] = round(correct / total, 4) if total > 0 else 0.0

    return {
        "model": model,
        "scores": scores,
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행 함수
# ─────────────────────────────────────────────────────────────────────────────

def run(models: Optional[list[str]] = None) -> dict:
    """
    Track 1 실행 — 한국어 표준 벤치마크.

    Args:
        models: 평가할 모델 목록 (기본: config.ALL_MODELS)

    Returns:
        {
            "track": "korean_bench",
            "results": [{"model": str, "scores": {bench: acc}, "details": [...]}, ...],
            "summary": {model: {bench: acc, ...}, ...},
        }
    """
    if models is None:
        models = list(config.ALL_MODELS)

    print(f"═══ Track 1: 한국어 표준 벤치마크 ({len(models)}개 모델) ═══")

    # Ollama 서버 확인
    if not runner.wait_for_ollama():
        raise RuntimeError("Ollama 서버에 연결할 수 없습니다")

    use_lm_eval = _lm_eval_available()
    if use_lm_eval:
        print("  lm-evaluation-harness 감지 — lm_eval 모드로 실행")
    else:
        print("  lm-evaluation-harness 미설치 — standalone 모드로 실행")

    # standalone 문항 미리 빌드
    questions = _build_all_questions() if not use_lm_eval else []

    # 체크포인트 로드
    checkpoint = runner.load_checkpoint(TRACK_NAME)
    completed_models = set()
    all_results = []
    if checkpoint:
        all_results = checkpoint.get("results", [])
        completed_models = {r["model"] for r in all_results}
        print(f"  체크포인트에서 {len(completed_models)}개 모델 결과 로드됨")

    current_model = None

    for mi, model in enumerate(models):
        if model in completed_models:
            print(f"\n  [{mi + 1}/{len(models)}] {model} — 이미 완료, 건너뜀")
            continue

        print(f"\n  [{mi + 1}/{len(models)}] {model}")

        # 모델 전환
        if not runner.switch_model(model, current_model):
            print(f"    모델 로딩 실패: {model}")
            all_results.append({
                "model": model,
                "scores": {},
                "details": [],
                "error": "모델 로딩 실패",
            })
            runner.save_checkpoint(
                {"track": TRACK_NAME, "results": all_results},
                TRACK_NAME,
            )
            continue
        current_model = model

        # lm_eval 모드
        if use_lm_eval:
            lm_scores = _run_lm_eval(model)
            if lm_scores is not None:
                model_result = {
                    "model": model,
                    "scores": lm_scores,
                    "details": [],
                    "mode": "lm_eval",
                }
                all_results.append(model_result)
                runner.save_checkpoint(
                    {"track": TRACK_NAME, "results": all_results},
                    TRACK_NAME,
                )
                print(f"    완료 (lm_eval): {lm_scores}")
                continue
            else:
                print("    lm_eval 실패 — standalone으로 fallback")
                if not questions:
                    questions = _build_all_questions()

        # standalone 모드
        model_result = _run_standalone(model, questions)
        model_result["mode"] = "standalone"
        all_results.append(model_result)

        # 체크포인트 저장
        runner.save_checkpoint(
            {"track": TRACK_NAME, "results": all_results},
            TRACK_NAME,
        )
        print(f"    완료: {model_result['scores']}")

    # 최종 결과 정리
    summary = {}
    for r in all_results:
        summary[r["model"]] = r.get("scores", {})

    final = {
        "track": TRACK_NAME,
        "timestamp": datetime.now().isoformat(),
        "num_models": len(all_results),
        "mode": "lm_eval" if use_lm_eval else "standalone",
        "results": all_results,
        "summary": summary,
    }

    # 최종 결과 저장
    runner.save_results_incremental(final, TRACK_NAME)

    print(f"\n═══ Track 1 완료: {len(all_results)}개 모델 평가됨 ═══")
    return final
