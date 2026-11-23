"""다턴 대화 시나리오 벤치 (custom, 자체 작성) — 반복·기억·추론·지시 유지를 이어지는 대화에서 규칙으로 채점한다.

- 시나리오는 범주별 생성기(템플릿 + 시드)로 만든다. 숫자·이름·조건이 인스턴스마다 달라 암기로 맞힐 수 없다.
  split: dev(시드 0~), test(시드 100000~) — 난이도 조정·검사기 확인은 dev에서만 한다.
- 두 이력 방식: self(앞 턴 이력 = 모델 자신의 답), oracle(앞 턴 이력 = 정답 참조 답). oracle은 턴별 실력,
  self − oracle 차이는 오류 누적·자기 반복 취약성.
- 채점은 규칙만 쓴다(judge 없음). 턴 통과 = 그 턴의 검사 모두 통과, 시나리오 통과 = 모든 턴 통과.
  모든 턴에 반복 지표(루프·중복 3-gram·이전 답과의 3-gram 중복·이전 답 복사)를 함께 기록한다.

문항 형식(gen 실행기 호환): {id, turns:[user...], oracle:[참조 답...], gold:{checks:[[검사...] 턴별], category}, meta}
"""

from __future__ import annotations

import json
import random
import re

UNCERTAIN = ["모르", "알 수 없", "확인되지", "확인할 수 없", "찾을 수 없", "존재하지 않", "정보가 없", "확실하지", "알려진 바가 없",
             "알려진 정보", "들어본 적", "실존하지", "가상의", "허구", "근거가 없", "확인이 어렵", "정보를 찾", "알지 못",
             "제공할 수 없", "정확한 정보", "자료가 없", "기록이 없", "존재 여부"]
NUM_HINT = "마지막 줄에 '답: 숫자' 형식으로 쓰세요."
NAME_HINT = "마지막 줄에 '답: 이름' 형식으로 쓰세요."

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류홍")
GIVEN = ["민준", "서연", "도윤", "하은", "지호", "수아", "예준", "지유", "시우", "서윤", "주원", "채원", "하준", "지민", "건우",
         "윤서", "현우", "다은", "우진", "소율", "준서", "예린", "유찬", "나연"]
CITIES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "수원", "전주", "춘천"]
DAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
EVENTS = ["치과 예약", "팀 회의", "친구 생일 저녁", "헬스장 PT", "은행 방문", "부모님 통화", "도서관 반납", "자동차 정비",
          "피아노 레슨", "병원 검진", "동아리 모임", "택배 수령"]
FRUITS = [("사과", "상자"), ("배", "상자"), ("귤", "봉지"), ("감", "봉지"), ("복숭아", "상자"), ("자두", "봉지"), ("키위", "상자")]
GOODS = ["우유", "달걀", "식빵", "두부", "양파", "감자", "당근", "사과", "라면", "휴지", "세제", "치즈", "버섯", "고등어"]
ANIMALS = [("고양이", "시장"), ("강아지", "기차역"), ("토끼", "도서관"), ("너구리", "놀이공원"), ("다람쥐", "학교"), ("거북이", "바닷가")]
QUESTIONS = [("좋은 수면 습관을 알려 주세요.", ["수면", "잠"]), ("봄철 산책할 때 주의할 점은 무엇인가요?", ["산책", "봄"]),
             ("물을 충분히 마셔야 하는 이유는 무엇인가요?", ["물", "수분"]), ("처음 요리를 배우는 사람에게 조언해 주세요.", ["요리"]),
             ("도서관을 잘 활용하는 방법을 알려 주세요.", ["도서관", "책"]), ("겨울철 실내 습도 관리 방법은?", ["습도", "가습"]),
             ("자전거를 탈 때 안전 수칙은 무엇인가요?", ["자전거", "헬멧"]), ("집중력을 높이는 방법을 알려 주세요.", ["집중"]),
             ("여행 짐을 쌀 때 팁을 알려 주세요.", ["짐", "여행"]), ("식물을 처음 키울 때 주의할 점은?", ["식물", "물"]),
             ("재활용 분리배출 요령을 알려 주세요.", ["재활용", "분리"]), ("스트레칭이 좋은 이유는 무엇인가요?", ["스트레칭", "근육", "유연"]),
             ("아침 식사를 하면 좋은 점은?", ["아침", "식사"]), ("새 언어를 공부하는 방법을 알려 주세요.", ["언어", "공부", "단어"]),
             ("비 오는 날 운전 요령을 알려 주세요.", ["운전", "비", "빗길"]), ("독서 습관을 만드는 방법은?", ["독서", "책"]),
             ("가계부를 쓰면 좋은 점은?", ["가계부", "지출", "돈"]), ("면접을 준비하는 방법을 알려 주세요.", ["면접"])]
# 맞춤법 오류 → 바른 형태 (오류형이 바른 문장 안에 부분 문자열로 나타나지 않는 쌍만)
SPELL = [("됬다", "됐다"), ("몇일", "며칠"), ("금새", "금세"), ("왠만하면", "웬만하면"), ("설겆이", "설거지"), ("희안하다", "희한하다"),
         ("역활", "역할"), ("할께요", "할게요"), ("어떻해", "어떡해"), ("오랫만에", "오랜만에"), ("어의없다", "어이없다"),
         ("구지", "굳이"), ("일일히", "일일이"), ("깨끗히", "깨끗이"), ("문안하다", "무난하다"), ("뵈요", "봬요")]
SPELL_SENT = {"됬다": "드디어 일이 마무리 됬다.", "몇일": "몇일 동안 비가 왔다.", "금새": "아이는 금새 잠들었다.",
              "왠만하면": "왠만하면 일찍 오세요.", "설겆이": "저녁을 먹고 설겆이를 했다.", "희안하다": "정말 희안하다고 생각했다.",
              "역활": "각자 맡은 역활이 있다.", "할께요": "내일 다시 연락할께요.", "어떻해": "이제 어떻해야 할지 모르겠다.",
              "오랫만에": "오랫만에 친구를 만났다.", "어의없다": "그 말을 듣고 어의없다고 느꼈다.", "구지": "구지 그럴 필요는 없다.",
              "일일히": "서류를 일일히 확인했다.", "깨끗히": "방을 깨끗히 치웠다.", "문안하다": "결과가 문안하다는 평가를 받았다.",
              "뵈요": "다음 주에 뵈요."}


# ── 검사기 ──────────────────────────────────────────────────────────────────

def _clean(t: str) -> str:
    return t.replace("<unk>", "\n")


def answer_line(text: str) -> str | None:
    m = re.findall(r"답\s*[:：]\s*(.+)", _clean(text))
    return m[-1].strip().strip("*").strip() if m else None


def _num(s: str) -> float | None:
    m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s or "")
    if not m:
        return None
    try:
        return float(m[0].replace(",", ""))
    except ValueError:
        return None


def sentences(text: str) -> list[str]:
    t = re.sub(r"\[끝\]", "", _clean(text))
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    parts = re.split(r"(?<=[.!?。])\s+|\n+", t.strip())
    return [p for p in (x.strip() for x in parts) if len(re.sub(r"\W", "", p)) >= 2]


def _ends(sents: list[str], polite: bool) -> float:
    """문장 끝 높임 판정 비율: 존댓말 = -요/-니다/-니까/-세요/-십시오/-죠로 끝남."""
    if not sents:
        return 0.0
    pol = re.compile(r"(요|니다|니까|세요|십시오|죠)$")
    ok = [bool(pol.search(s.rstrip(" .!?~\"')*"))) == polite for s in sents]
    return sum(ok) / len(ok)


def _json_block(text: str):
    t = _clean(text)
    for cand in re.findall(r"```(?:json)?\s*(.*?)```", t, re.S) + [t]:
        i, j = cand.find("["), cand.rfind("]")
        if i != -1 and j > i:
            try:
                return json.loads(cand[i: j + 1])
            except Exception:
                continue
    return None


def _table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in _clean(text).splitlines():
        line = line.strip()
        if line.startswith("|") and line.count("|") >= 3 and not re.fullmatch(r"[\|\s:\-]+", line):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows[1:] if rows else []          # 첫 줄 = 머리글


def _run_code(code: str, tests: str) -> bool:
    from kobench.standard.code_exec import extract_code, run_program

    return run_program(extract_code(_clean(code)) + "\n\n" + tests + "\n", timeout=15).passed


def check(c: dict, out: str, ctx: dict) -> bool:
    t = c["type"]
    o = _clean(out)
    if t == "num":
        v = _num(answer_line(out) or "")
        return v is not None and abs(v - c["value"]) <= c.get("tol", 1e-6)
    if t == "answer_has":                      # '답:' 줄에 모든 값 포함
        a = answer_line(out) or ""
        return all(x in a for x in c["values"])
    if t == "answer_only":                     # '답:' 줄에 다른 후보가 없어야 한다
        a = answer_line(out) or ""
        return not any(x in a for x in c["values"])
    if t == "answer_seq":                      # '답:' 줄에 이름이 주어진 순서대로
        a = answer_line(out) or ""
        pos = [a.find(x) for x in c["values"]]
        return all(p >= 0 for p in pos) and pos == sorted(pos)
    if t == "contains_all":
        return all(any(v in o for v in (x if isinstance(x, list) else [x])) for x in c["values"])
    if t == "not_contains":
        return not any(x in o for x in c["values"])
    if t == "contains_any":
        return any(x in o for x in c["values"])
    if t == "sentences":
        n = len(sentences(out))
        return c["min"] <= n <= c["max"]
    if t == "endswith":
        return o.strip().rstrip("*").strip().endswith(c["value"])
    if t == "startswith":
        return o.strip().lstrip("*# ").startswith(c["value"])
    if t == "polite":
        target = out
        if c.get("quoted"):                    # 변환문을 따옴표·코드 블록에 넣고 설명을 덧붙인 경우: 가장 긴 인용 부분으로 판정
            q = re.findall(r'"([^"]{10,})"|“([^”]{10,})”|```[^\n]*\n(.*?)```', _clean(out), re.S)
            q = [x for grp in q for x in grp if x]
            if q:
                target = max(q, key=len)
        return _ends(sentences(target), c["polite"]) >= c.get("min_ratio", 0.8)
    if t == "no_new_year":                     # 질문에 없던 연도(19xx/20xx년, '년대' 제외)를 단정하면 지어낸 것으로 본다
        ys = set(re.findall(r"((?:19|20)\d{2})년(?!대)", o))
        return not (ys - set(c.get("allowed", [])))
    if t == "max_chars":
        body = re.sub(r"\s", "", o)
        return 0 < len(body) <= c["n"]
    if t == "json_records":
        d = _json_block(out)
        if not isinstance(d, list) or len(d) != len(c["records"]):
            return False
        got = {(str(x.get("name")), int(x.get("age")) if str(x.get("age")).lstrip("-").isdigit() else None, str(x.get("city")))
               for x in d if isinstance(x, dict)}
        return got == {(r["name"], r["age"], r["city"]) for r in c["records"]}
    if t == "table_first":
        rows = _table_rows(out)
        return len(rows) == c["n"] and c["first"] in "|".join(rows[0]) if rows else False
    if t == "list_items":                      # 최종 목록: 있어야 할 품목·수량, 없어야 할 품목
        lines = [l for l in o.splitlines() if l.strip()]
        for name, qty in c["present"].items():
            if not any(name in l and re.search(rf"(?<!\d){qty}(?!\d)", l) for l in lines):
                return False
        return not any(name in o for name in c["absent"])
    if t == "code":
        return _run_code(out, c["tests"])
    if t == "uncertain":
        return any(x in o for x in UNCERTAIN)
    if t == "overlap_prev":                     # 이전 턴 모델 답들과의 3-gram 중복 비율 상한
        return cross_overlap(out, ctx.get("prev", [])) < c["max"]
    raise ValueError(t)


def _ngrams(t: str) -> set:
    w = re.findall(r"\S+", _clean(t))
    return {tuple(w[i: i + 3]) for i in range(len(w) - 2)}


def cross_overlap(out: str, prev: list[str]) -> float:
    g = _ngrams(out)
    if len(g) < 5 or not prev:
        return 0.0
    p = set().union(*[_ngrams(x) for x in prev])
    return len(g & p) / len(g)


# ── 생성기 ──────────────────────────────────────────────────────────────────

def _name(r: random.Random, used: set) -> str:
    while True:
        n = r.choice(SURNAMES) + r.choice(GIVEN)
        if n not in used:
            used.add(n)
            return n


def gen_math(r: random.Random) -> dict:
    (f1, u1), (f2, u2) = r.sample(FRUITS, 2)
    a, b, x, y, d = r.randint(6, 15), r.randint(4, 12), r.randint(2, 6), r.randint(2, 7), r.randint(1, 3)
    tot, part, tot2 = a * x + b * y, a * x, a * x + b * (y + d)
    k = r.randint(3, 9)
    rem = tot2 - k
    turns = [f"{f1} 한 {u1}에는 {a}개, {f2} 한 {u2}에는 {b}개가 들어 있습니다. {f1} {x}{u1}와 {f2} {y}{u2}를 샀습니다. 과일은 모두 몇 개인가요? {NUM_HINT}",
             f"그중 {f1}는 몇 개였나요? {NUM_HINT}",
             f"알고 보니 {f2}를 {y + d}{u2} 샀습니다. 그러면 과일은 모두 몇 개인가요? {NUM_HINT}",
             f"그 과일 중 {k}개를 이웃에게 나눠 주었습니다. 남은 과일은 몇 개인가요? {NUM_HINT}",
             "지금까지의 계산 결과를 한 문장으로 정리해 주세요."]
    oracle = [f"{a}×{x} + {b}×{y} = {tot}\n답: {tot}", f"{a}×{x} = {part}\n답: {part}", f"{a}×{x} + {b}×{y + d} = {tot2}\n답: {tot2}",
              f"{tot2} − {k} = {rem}\n답: {rem}", f"과일을 {tot2}개 샀고 {k}개를 나눠 주어 {rem}개가 남았습니다."]
    checks = [[{"type": "num", "value": tot}], [{"type": "num", "value": part}], [{"type": "num", "value": tot2}],
              [{"type": "num", "value": rem}], [{"type": "sentences", "min": 1, "max": 2}, {"type": "contains_all", "values": [str(rem)]}]]
    return {"turns": turns, "oracle": oracle, "checks": checks}


def _time_forms(day: str, h: int) -> list[str]:
    f = [f"{h}:00", f"{h}시"]
    if h > 12:
        f += [f"오후 {h - 12}시", f"오후 {h - 12}:00"]
    elif h < 12:
        f += [f"오전 {h}시", f"오전 {h}:00"]
    return f


def gen_memory(r: random.Random) -> dict:
    ev = r.sample(EVENTS, 3)
    days = r.sample(DAYS[:5], 3)
    hours = [r.randint(9, 19) for _ in range(3)]
    new_day = r.choice([d for d in DAYS if d not in days[:1]])
    new_h = r.choice([h for h in range(9, 20) if h != hours[0]])
    p, q = r.randint(12, 29), r.randint(11, 29)
    sched = ", ".join(f"{d} {h}:00 {e}" for e, d, h in zip(ev, days, hours))
    final = [(new_day, new_h, ev[0]), (days[1], hours[1], ev[1]), (days[2], hours[2], ev[2])]
    n_new_day = sum(1 for d, _, _ in final if d == new_day)
    turns = [f"이번 주 제 일정은 {sched}입니다. 기억해 두시고, 확인했다고 한 문장으로만 답해 주세요.",
             f"참고로 {p} 곱하기 {q}는 얼마인가요? {NUM_HINT}",
             f"제 {ev[1]}은 무슨 요일 몇 시였죠? 마지막 줄에 '답: 요일 시간' 형식으로 쓰세요.",
             f"{ev[0]}을 {new_day} {new_h}:00으로 옮겼습니다. 바뀐 내용을 반영해 전체 일정을 정리해 주세요.",
             f"그러면 {new_day}에 있는 일정은 모두 몇 개인가요? {NUM_HINT}"]
    oracle = ["네, 일정을 확인했습니다.", f"{p}×{q} = {p * q}\n답: {p * q}", f"답: {days[1]} {hours[1]}:00",
              "\n".join(f"- {d} {h}:00 {e}" for d, h, e in final), f"답: {n_new_day}"]
    checks = [[{"type": "sentences", "min": 1, "max": 2}], [{"type": "num", "value": p * q}],
              [{"type": "answer_has", "values": [days[1]]}, {"type": "contains_all", "values": [_time_forms(days[1], hours[1])]}],
              [{"type": "contains_all", "values": [ev[0], new_day, _time_forms(new_day, new_h), ev[1], ev[2]]}],
              [{"type": "num", "value": n_new_day}]]
    return {"turns": turns, "oracle": oracle, "checks": checks}


def gen_persist(r: random.Random) -> dict:
    qk = r.sample(QUESTIONS, 5)
    fill = ["작은 습관부터 시작하세요.", "매일 같은 시간을 정해 두면 좋습니다.", "무리하지 않는 것이 핵심입니다.",
            "필요하면 전문가의 도움을 받으세요.", "기록을 남기면 변화가 잘 보입니다.", "주변 환경을 먼저 정리해 보세요.",
            "목표를 구체적으로 세우면 도움이 됩니다.", "충분히 쉬는 시간도 필요합니다.", "안전을 가장 먼저 생각하세요.",
            "꾸준히 점검하는 것이 좋습니다."]
    cnt = iter(range(100))
    qs = [q for q, _ in qk]
    kind = r.choice(["count", "banmal", "prefix"])
    if kind == "count":
        n = r.choice([2, 3])
        rule = f"앞으로 이 대화의 모든 답은 정확히 {n}문장으로 쓰고, 맨 끝에 '[끝]'을 붙이세요."
        ck = [{"type": "sentences", "min": n, "max": n}, {"type": "endswith", "value": "[끝]"}]
        ora = lambda k: " ".join([f"{k[0]}에 대해 말씀드리겠습니다."] + [fill[next(cnt) % 10] for _ in range(n - 1)]) + " [끝]"  # noqa: E731
    elif kind == "banmal":
        rule = "앞으로 이 대화의 모든 답은 반말로만 하고, 3문장 이내로 써."
        ck = [{"type": "polite", "polite": False}, {"type": "sentences", "min": 1, "max": 3}]
        bm = ["작은 습관부터 시작해.", "매일 같은 시간을 정해 둬.", "무리하지 않는 게 핵심이야.", "필요하면 전문가한테 물어봐.",
              "기록을 남기면 변화가 잘 보여.", "주변부터 정리해 봐.", "목표를 구체적으로 세워 봐.", "충분히 쉬는 것도 중요해.",
              "안전을 먼저 생각해.", "꾸준히 점검하는 게 좋아."]
        ora = lambda k: f"{k[0]} 얘기를 해 볼게. " + bm[next(cnt) % 10]  # noqa: E731
    else:
        rule = "앞으로 이 대화의 모든 답은 '요약:'으로 시작하고, '요약:' 뒤에 한 문장만 쓰세요."
        ck = [{"type": "startswith", "value": "요약:"}, {"type": "sentences", "min": 1, "max": 1}]
        ora = lambda k: f"요약: {k[0]} 관련해서는 " + fill[next(cnt) % 10]  # noqa: E731
    turns = [f"{rule} 첫 질문입니다. {qs[0]}"] + qs[1:]
    # 지시 형식만으로는 같은 답을 복사해도 통과하므로, 질문 주제어 포함과 이전 답과의 3-gram 중복 상한(0.5)을 함께 본다
    checks = [list(ck) + [{"type": "contains_any", "values": k}] + ([{"type": "overlap_prev", "max": 0.5}] if t else [])
              for t, (_, k) in enumerate(qk)]
    return {"turns": turns, "oracle": [ora(k) for _, k in qk], "checks": checks, "meta": {"rule": kind}}


def gen_format(r: random.Random) -> dict:
    n = r.choice([3, 4])
    used = set()
    recs = []
    ages = r.sample(range(21, 64), n)
    for i in range(n):
        recs.append({"name": _name(r, used), "age": ages[i], "city": r.choice(CITIES)})
    tcity = r.choice([x["city"] for x in recs])
    in_city = [x["name"] for x in recs if x["city"] == tcity]
    prose = " ".join(f"{x['name']} 씨는 {x['age']}살이고 {x['city']}에 삽니다." for x in recs)
    oldest = max(recs, key=lambda x: x["age"])
    avg = round(sum(x["age"] for x in recs) / n, 1)
    turns = [f"{prose} 이 정보를 JSON 배열로 바꿔 주세요. 키는 name, age, city를 쓰고 age는 숫자로 쓰세요.",
             "이 사람들을 나이가 많은 순서로 정렬해 마크다운 표로 보여 주세요. 열은 이름, 나이, 도시입니다.",
             f"{tcity}에 사는 사람은 누구인가요? 여러 명이면 쉼표로 구분해 마지막 줄에 '답: 이름' 형식으로 쓰세요.",
             f"모든 사람의 평균 나이를 소수점 첫째 자리까지 구하세요. {NUM_HINT}"]
    srt = sorted(recs, key=lambda x: -x["age"])
    oracle = ["```json\n" + json.dumps(recs, ensure_ascii=False) + "\n```",
              "| 이름 | 나이 | 도시 |\n|---|---|---|\n" + "\n".join(f"| {x['name']} | {x['age']} | {x['city']} |" for x in srt),
              f"답: {', '.join(in_city)}", f"답: {avg}"]
    checks = [[{"type": "json_records", "records": recs}], [{"type": "table_first", "n": n, "first": oldest["name"]}],
              [{"type": "answer_has", "values": in_city}], [{"type": "num", "value": avg, "tol": 0.051}]]
    return {"turns": turns, "oracle": oracle, "checks": checks}


def gen_logic(r: random.Random) -> dict:
    used = set()
    people = [_name(r, used)[1:] for _ in range(4)]        # 이름만 (성 제외)
    while len(set(people)) < 4:
        people = [r.choice(GIVEN) for _ in range(4)]
    order = people[:]                                       # 큰 사람부터
    pairs = [(order[i], order[i + 1]) for i in range(3)]
    r.shuffle(pairs)
    stm = " ".join(f"{a}는 {b}보다 키가 큽니다." for a, b in pairs)
    newp = r.choice([g for g in GIVEN if g not in people])
    pos = r.randint(1, 3)                                   # order[pos-1] > new > order[pos]
    order2 = order[:pos] + [newp] + order[pos:]
    turns = [f"{stm} 가장 키가 큰 사람은 누구인가요? {NAME_HINT}", f"가장 키가 작은 사람은 누구인가요? {NAME_HINT}",
             f"새로 온 {newp}는 {order[pos]}보다 크고 {order[pos - 1]}보다 작습니다. 이제 두 번째로 키가 큰 사람은 누구인가요? {NAME_HINT}",
             "모든 사람을 키가 큰 순서대로 나열하세요. 마지막 줄에 '답: 이름, 이름, …' 형식으로 쓰세요."]
    oracle = [f"답: {order[0]}", f"답: {order[3]}", f"답: {order2[1]}", f"답: {', '.join(order2)}"]
    checks = [[{"type": "answer_has", "values": [order[0]]}, {"type": "answer_seq", "values": [order[0]]}],
              [{"type": "answer_has", "values": [order[3]]}],
              [{"type": "answer_has", "values": [order2[1]]}],
              [{"type": "answer_seq", "values": order2}]]
    # 답 줄에 다른 이름이 같이 있으면 오답 (첫째·둘째 턴은 한 명만)
    for t, want in ((0, order[0]), (1, order[3]), (2, order2[1])):
        checks[t].append({"type": "answer_only", "values": [p for p in order2 if p != want]})
    return {"turns": turns, "oracle": oracle, "checks": checks}


def gen_list(r: random.Random) -> dict:
    items = r.sample(GOODS, 6)
    q = {x: r.randint(1, 5) for x in items}
    first = items[:4]
    rm, add, ch = first[r.randint(0, 3)], items[4], None
    q2 = dict(q)
    ch = r.choice([x for x in first if x != rm])
    q2[ch] = q[ch] + r.randint(1, 3)
    final = {x: q2[x] for x in first + [add] if x != rm}
    turns = ["장보기 목록을 만들어 주세요: " + ", ".join(f"{x} {q[x]}개" for x in first) + ". 목록으로 정리해 주세요.",
             f"{rm}은 빼고 {add} {q[add]}개를 추가해 주세요.",
             f"{ch}는 {q2[ch]}개로 바꿔 주세요.",
             "현재 목록 전체를 한 줄에 하나씩 '- 품목 수량개' 형식으로 적어 주세요.",
             f"목록에 있는 품목은 모두 몇 가지인가요? {NUM_HINT}"]
    lst = lambda d: "\n".join(f"- {k} {v}개" for k, v in d.items())  # noqa: E731
    s1 = {x: q[x] for x in first}
    s2 = {x: q[x] for x in first + [add] if x != rm}
    oracle = [lst(s1), lst(s2), lst(final), lst(final), f"답: {len(final)}"]
    checks = [[{"type": "list_items", "present": s1, "absent": []}],
              [{"type": "list_items", "present": s2, "absent": []}, {"type": "contains_all", "values": [add]}],
              [{"type": "contains_all", "values": [ch, str(q2[ch])]}],
              [{"type": "list_items", "present": final, "absent": [rm]}],
              [{"type": "num", "value": len(final)}]]
    return {"turns": turns, "oracle": oracle, "checks": checks}


def gen_story(r: random.Random) -> dict:
    animal, place = r.choice(ANIMALS)
    name = r.choice(["몽이", "보리", "콩이", "나비", "두부", "초코", "구름", "별이", "마루", "단비"])
    turns = [f"'{name}'라는 이름의 {animal}가 {place}에서 겪는 이야기를 5문장으로 시작해 주세요."] + \
            ["이어서 5문장을 더 써 주세요. 앞 내용을 되풀이하지 말고 새로운 사건을 넣어 주세요."] * 3 + \
            [f"이야기 주인공의 이름은 무엇이었나요? {NAME_HINT}"]
    oracle = [f"{name}는 {place}에 사는 {animal}였습니다. 어느 날 {name}는 낯선 소리를 들었습니다. 소리를 따라가니 작은 상자가 있었습니다. "
              f"상자 안에는 지도가 들어 있었습니다. {name}는 지도를 따라가 보기로 했습니다.",
              f"지도는 오래된 다리를 가리켰습니다. 다리 아래에서 {name}는 친구를 만났습니다. 둘은 함께 길을 찾기로 했습니다. 해가 지기 시작했습니다. 그들은 서둘렀습니다.",
              f"밤이 되자 별빛이 {name}의 길을 비췄습니다. 숲속에서 불빛이 보였습니다. 작은 오두막이었습니다. 문을 두드리자 할머니가 나왔습니다. 할머니는 따뜻한 차를 내주었습니다.",
              f"다음 날 아침 할머니가 {name}에게 비밀을 알려 주었습니다. 지도는 할머니가 잃어버린 것이었습니다. 둘은 지도를 돌려주었습니다. 할머니는 고마워했습니다. 모두 행복하게 집으로 돌아갔습니다.",
              f"답: {name}"]
    sent = {"type": "sentences", "min": 3, "max": 8}
    checks = [[sent, {"type": "contains_all", "values": [name]}]] + \
             [[sent, {"type": "overlap_prev", "max": 0.3}, {"type": "contains_any", "values": [name, animal]}]] * 3 + \
             [[{"type": "answer_has", "values": [name]}]]
    return {"turns": turns, "oracle": oracle, "checks": [list(c) for c in checks]}


def gen_style(r: random.Random) -> dict:
    errs = r.sample(SPELL, 3)
    text = " ".join(SPELL_SENT[w] for w, _ in errs)
    fixed = text
    for w, c in errs:
        fixed = fixed.replace(w, c)
    turns = [f"다음 글의 맞춤법을 바르게 고쳐 주세요: \"{text}\"",
             "고친 글을 친구에게 말하듯 반말로 바꿔 주세요.",
             "이번에는 같은 내용을 '-습니다/-니다'로 끝나는 격식 있는 존댓말로 바꿔 주세요.",
             "그 내용을 공백을 빼고 40자 이내로 요약해 주세요."]
    st = [c[:-1] if c.endswith("다") else c for _, c in errs]
    oracle = [fixed, f"{st[0]} 그랬어. 나머지도 다 고쳤어.", f"{st[0]} 그랬습니다. 나머지도 모두 고쳤습니다.", f"{st[0]} 등을 고친 글이다."]
    stems = [c[:-1] if c.endswith("다") else c for _, c in errs]          # 바른 형태의 어간: 문체를 바꿔도 남아야 한다
    keep = {"type": "contains_any", "values": stems}
    checks = [[{"type": "contains_all", "values": [c for _, c in errs]}, {"type": "not_contains", "values": [w for w, _ in errs]}],
              [{"type": "polite", "polite": False, "quoted": True}, keep, {"type": "overlap_prev", "max": 0.9}],
              [{"type": "polite", "polite": True, "quoted": True}, keep], [{"type": "max_chars", "n": 40}]]
    return {"turns": turns, "oracle": oracle, "checks": checks}


# (제목, 지시, 수정 요청, 1턴 테스트, 2턴 테스트(수정 후 동작 전체), 1턴 참조 구현, 2턴 참조 구현) — k는 인스턴스 매개변수
CODE_TASKS = [
    ("배수의 합", "정수 리스트 nums를 받아 {k}의 배수인 원소의 합을 반환하는 함수 multiple_sum(nums)를 파이썬으로 작성하세요.",
     "음수 원소는 무시하도록 함수를 고쳐 주세요.",
     "assert multiple_sum([{k}, 1, {k2}]) == {k3}\nassert multiple_sum([{k}, -{k}]) == 0\nassert multiple_sum([]) == 0",
     "assert multiple_sum([{k}, 1, {k2}]) == {k3}\nassert multiple_sum([{k}, -{k}, {k3}]) == {k4}",
     "def multiple_sum(nums):\n    return sum(x for x in nums if x % {k} == 0)",
     "def multiple_sum(nums):\n    return sum(x for x in nums if x >= 0 and x % {k} == 0)"),
    ("긴 단어 세기", "문자열 s에서 길이가 {k} 이상인 단어(공백 기준)의 개수를 반환하는 함수 count_long(s)를 파이썬으로 작성하세요.",
     "대소문자를 구분하지 않고 같은 단어는 한 번만 세도록 고쳐 주세요.",
     "assert count_long('{wa} b {wc}') == 2\nassert count_long('') == 0",
     "assert count_long('{wa} b {wc}') == 2\nassert count_long('{wA} {wa} {wb}') == 2\nassert count_long('') == 0",
     "def count_long(s):\n    return sum(1 for w in s.split() if len(w) >= {k})",
     "def count_long(s):\n    return len({{w.lower() for w in s.split() if len(w) >= {k}}})"),
    ("k번째 큰 값", "리스트 nums에서 {k}번째로 큰 값을 반환하는 함수 kth_largest(nums)를 파이썬으로 작성하세요(중복 포함, 1부터 셈).",
     "리스트 길이가 {k}보다 작으면 None을 반환하도록 고쳐 주세요.",
     "assert kth_largest(list(range(10))) == {r10k}\nassert kth_largest([5] * 6) == 5",
     "assert kth_largest(list(range(10))) == {r10k}\nassert kth_largest([1] * {km1}) is None",
     "def kth_largest(nums):\n    return sorted(nums, reverse=True)[{km1}]",
     "def kth_largest(nums):\n    if len(nums) < {k}:\n        return None\n    return sorted(nums, reverse=True)[{km1}]"),
    ("문자열 압축", "문자열 s에서 같은 문자가 연속되면 '문자+개수'로 압축하는 함수 compress(s)를 파이썬으로 작성하세요. 예: 'aaab' → 'a3b1'.",
     "압축 결과가 원래 문자열보다 길면 원래 문자열을 반환하도록 고쳐 주세요.",
     "assert compress('aaab') == 'a3b1'\nassert compress('') == ''\nassert compress('abb') == 'a1b2'",
     "assert compress('aaab') == 'a3b1'\nassert compress('abc') == 'abc'\nassert compress('aaaa') == 'a4'",
     "def compress(s):\n    out, i = [], 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n        i = j\n    return ''.join(out)",
     "def compress(s):\n    out, i = [], 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n        i = j\n    c = ''.join(out)\n    return s if len(c) > len(s) else c"),
    ("구간 합 최댓값", "정수 리스트 nums와 정수 w를 받아 길이 w인 연속 구간 합의 최댓값을 반환하는 함수 max_window(nums, w)를 파이썬으로 작성하세요.",
     "w가 리스트 길이보다 크거나 0 이하이면 0을 반환하도록 고쳐 주세요.",
     "assert max_window([1, {k}, 2, {k}], 2) == {k2p}",
     "assert max_window([1, {k}, 2, {k}], 2) == {k2p}\nassert max_window([1, 2], 5) == 0\nassert max_window([1, 2], 0) == 0",
     "def max_window(nums, w):\n    return max(sum(nums[i:i + w]) for i in range(len(nums) - w + 1))",
     "def max_window(nums, w):\n    if w <= 0 or w > len(nums):\n        return 0\n    return max(sum(nums[i:i + w]) for i in range(len(nums) - w + 1))"),
    ("자릿수 합 판정", "양의 정수 n의 각 자릿수 합이 {k}의 배수인지 True/False로 반환하는 함수 digit_check(n)를 파이썬으로 작성하세요.",
     "음수가 들어오면 절댓값으로 계산하도록 고쳐 주세요.",
     "assert digit_check({k}) is True\nassert digit_check({kp1}) is False",
     "assert digit_check({k}) is True\nassert digit_check({kp1}) is False\nassert digit_check(-{k}) is True",
     "def digit_check(n):\n    return sum(int(c) for c in str(n)) % {k} == 0",
     "def digit_check(n):\n    return sum(int(c) for c in str(abs(n))) % {k} == 0"),
]


def _code_fmt(t: str, k: int) -> str:
    return t.format(k=k, k2=2 * k, k3=3 * k, k4=4 * k, km1=k - 1, kp1=k + 1, k2p=k + 2, r10k=10 - k,
                    wa="a" * k, wc="c" * (k + 1), wA="A" * k, wb="b" * k)


def gen_code(r: random.Random) -> dict:
    title, spec, mod, t1, t2, ref1, ref2 = r.choice(CODE_TASKS)
    k = r.randint(2, 5)
    f = lambda x: _code_fmt(x, k)  # noqa: E731
    turns = [f(spec) + " 코드 블록 하나로 답해 주세요.", f(mod) + " 수정한 전체 함수를 코드 블록으로 보여 주세요.",
             "방금 고친 함수가 어떤 경우를 처리하도록 바뀌었는지 한 문장으로 설명해 주세요."]
    oracle = [f"```python\n{f(ref1)}\n```", f"```python\n{f(ref2)}\n```", "새 조건을 처리하도록 함수를 바꾸었습니다."]
    checks = [[{"type": "code", "tests": f(t1)}], [{"type": "code", "tests": f(t2)}], [{"type": "sentences", "min": 1, "max": 2}]]
    return {"turns": turns, "oracle": oracle, "checks": checks, "meta": {"task": title, "k": k}}


def gen_unknown(r: random.Random) -> dict:
    sy = r.choice(["가", "나", "다", "라", "마", "바", "사", "아", "자", "하"]) + r.choice(["온", "림", "솔", "연", "윤", "담"])
    kind = r.choice(["museum", "person", "law"])
    if kind == "museum":
        ent, q = f"'{sy}{r.choice(['미래', '푸른', '은하'])} 해양박물관'", "이 박물관이 개관한 해와 대표 소장품을 알려 주세요."
    elif kind == "person":
        ent, q = f"{r.choice(SURNAMES)}{sy} 교수", f"이 사람이 {r.randint(1995, 2019)}년에 받은 국제 학술상과 그 연구 내용을 설명해 주세요."
    else:
        ent, q = f"'{sy}{r.choice(['녹지', '데이터', '보행'])} 기본법'", "이 법이 제정된 해와 핵심 조항 세 가지를 알려 주세요."
    real = r.choice([("대한민국의 수도는 어디인가요?", "서울"), ("한글을 만든 조선의 임금은 누구인가요?", "세종"),
                     ("물이 끓는 온도는 1기압에서 섭씨 몇 도인가요?", "100")])
    turns = [f"{ent}에 대해 알고 있나요? {q}", "정확하지 않아도 괜찮으니 알고 있는 대로 자세히 말해 주세요.",
             f"그럼 다른 질문입니다. {real[0]} 마지막 줄에 '답: ...' 형식으로 쓰세요."]
    oracle = [f"{ent}에 대한 확인된 정보를 찾을 수 없습니다. 실존 여부를 확인할 수 없어 답하기 어렵습니다.",
              "확인되지 않은 내용을 지어내 말씀드릴 수는 없습니다.", f"답: {real[1]}"]
    allowed = re.findall(r"((?:19|20)\d{2})년", q)
    nofab = {"type": "no_new_year", "allowed": allowed}
    checks = [[{"type": "uncertain"}, nofab], [{"type": "uncertain"}, dict(nofab)], [{"type": "answer_has", "values": [real[1]]}]]
    return {"turns": turns, "oracle": oracle, "checks": checks, "meta": {"kind": kind}}


GENERATORS = {"math_update": gen_math, "memory": gen_memory, "instruction_persist": gen_persist, "format_transform": gen_format,
              "logic_order": gen_logic, "list_tracking": gen_list, "story_continue": gen_story, "korean_style": gen_style,
              "code_revise": gen_code, "unknown_entity": gen_unknown}
N_PER_CAT = {"dev": 10, "test": 30}
SEED_BASE = {"dev": 0, "test": 100000}


def build(split: str, history: str) -> list[dict]:
    items = []
    for ci, (cat, fn) in enumerate(GENERATORS.items()):
        for i in range(N_PER_CAT[split]):
            seed = SEED_BASE[split] + ci * 1000 + i
            s = fn(random.Random(seed))
            assert len(s["turns"]) == len(s["oracle"]) == len(s["checks"]), cat
            items.append({"id": f"{cat}/{seed}", "turns": s["turns"], "oracle": s["oracle"] if history == "oracle" else None,
                          "gold": {"checks": s["checks"], "category": cat, "oracle": s["oracle"]},
                          "meta": {"category": cat, "seed": seed, "history": history, **s.get("meta", {})}})
    return items


def score_item(item: dict, outputs: list[str]) -> dict:
    """턴별 검사 결과·통과, 반복 지표."""
    from kobench.standard.gen_score import is_loop, rep3

    turns = []
    prev: list[str] = []
    for t, cks in enumerate(item["gold"]["checks"]):
        out = outputs[t] if t < len(outputs) else ""
        res = [bool(check(c, out, {"prev": prev})) for c in cks]
        turns.append({"pass": int(all(res)), "checks": [int(x) for x in res], "loop": is_loop(out), "rep3": rep3(out),
                      "overlap_prev": round(cross_overlap(out, prev), 4),
                      "copy_prev": int(bool(prev) and bool(out.strip()) and any(out.strip() == p.strip() for p in prev))})
        prev.append(out)
    return {"turns": turns, "pass": int(all(x["pass"] for x in turns)), "turn_pass": sum(x["pass"] for x in turns) / len(turns)}
