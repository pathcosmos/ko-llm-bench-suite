"""커스텀 객관식 태스크 보조 함수 (KMMLU-Redux/Pro, KoBALT-700).

형식은 lm-eval KMMLU 기본(kmmlu/default)과 같게 한다: "{질문}\nA. ...\nB. ...\n정답：" 뒤에 선택지 글자의 로그우도를 비교한다.
"""
import ast

LETTERS = "ABCDEFGHIJ"


def _options(doc):
    opts = doc["options"]
    return ast.literal_eval(opts) if isinstance(opts, str) else list(opts)


def kmmlu_x_text(doc):
    opts = _options(doc)
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
    return f"{doc['question'].strip()}\n{body}\n정답："


def kmmlu_x_choices(doc):
    return list(LETTERS[: len(_options(doc))])


def kmmlu_x_target(doc):
    return int(str(doc["solution"]).strip()) - 1


def kobalt_text(doc):
    return f"{doc['Question'].strip()}\n정답："


def kobalt_target(doc):
    return LETTERS.index(doc["Answer"].strip())


# ── 한국어 PPL (bits-per-byte): wiki40b ko test ─────────────────────────────
import re as _re

_W40B_MARKERS = (("_START_ARTICLE_\n", ""), ("_START_SECTION_\n", ""), ("_START_PARAGRAPH_\n", ""), ("_NEWLINE_", "\n"))


def wiki40b_clean(raw: str) -> str:
    """wiki40b의 text는 UTF-8 바이트를 b'...' 문자열로 저장한다 → 디코딩 후 구조 표시를 일반 텍스트로."""
    text = ast.literal_eval(raw).decode("utf-8") if raw.startswith(("b'", 'b"')) else raw
    for a, b in _W40B_MARKERS:
        text = text.replace(a, b)
    return _re.sub(r"\n{3,}", "\n\n", text).strip()


def wiki40b_docs(dataset):
    """test 분할의 앞 500문서 (결정적), 정리한 본문을 page 필드로."""
    ds = dataset.select(range(min(500, len(dataset))))
    return ds.map(lambda d: {"page": wiki40b_clean(d["text"])})


def wiki40b_target(doc):
    return doc["page"]


def ppl_process_results(doc, results):
    """lm-eval wikitext와 같은 정의: 단어·바이트 수는 원문 기준."""
    (loglikelihood,) = results
    words = len(_re.split(r"\s+", doc["page"]))
    nbytes = len(doc["page"].encode("utf-8"))
    return {"word_perplexity": (loglikelihood, words), "byte_perplexity": (loglikelihood, nbytes),
            "bits_per_byte": (loglikelihood, nbytes)}
