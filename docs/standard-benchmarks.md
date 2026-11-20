# 표준 벤치마크 (`kobench/standard`)

기존 7개 트랙(Ollama 생성 기반)과 별도로, **공개 표준 벤치마크를 lm-evaluation-harness 위에서 같은 조건으로 실행**하는 모듈이다.
base 모델(사전학습만)과 instruct 모델을 모두 공정하게 비교하는 것이 목적이라 객관식은 생성 후 문자열 파싱이 아니라
**선택지 로그우도**로 채점하고, PPL은 토크나이저가 달라도 비교할 수 있도록 **bits-per-byte**로 보고한다.

## 구성

| 파일 | 역할 |
|---|---|
| `kobench/standard/run.py` | 실행기. 모델을 한 번 로드하고 태스크마다 `lm_eval.simple_evaluate`를 호출. 결과는 원자적으로 기록되고 이미 있으면 건너뜀 |
| `kobench/standard/registry.py` | `configs/standard/models.yaml`(모델·revision·base/instruct·총/활성 파라미터·백엔드), `suites.yaml`(태스크·shot·지표·출처) 로더 |
| `kobench/standard/dev_search.py` | dev 분할에서 설정 변형을 비교하고 사전 등록 규칙으로 최적 설정을 골라 `best_settings.yaml`에 기록 |
| `kobench/standard/report.py` | A/B 표 집계: lm-eval 점수, 문항 micro + bootstrap 95% CI, 기준 모델 대비 paired 차이 |
| `kobench/standard/formats.py` | 모델별 학습 형식(토크나이저·chat 형식·문서 시작 토큰) 적용 |
| `kobench/standard/evafrill_lm.py`, `evafrill_engine.py` | Mamba-2 하이브리드(EVAFRILL-Mo) lm-eval 백엔드와 증분 디코딩 엔진 |
| `kobench/standard/tasks/{dev,custom,gen}` | lm-eval에 없는 태스크(아래) |

백엔드: `vllm`(기본, BF16) 또는 `evafrill`(PyTorch 직접). 개발·검증 환경은 vLLM 0.33.0, transformers 5.20.2, lm-eval 0.4.13, H200 1장이다.

## 스위트 (`configs/standard/suites.yaml`)

| 스위트 | 태스크 | 비고 |
|---|---|---|
| `ko_mc_v1` | KoBEST(5-shot), KMMLU(5-shot), HAE-RAE, CLIcK, Belebele-ko, Global-MMLU-ko | 한국어 객관식, 로그우도 |
| `ko_mc_ext_v1` | KMMLU-Redux, KMMLU-Pro, KoBALT-700 (0-shot) | 커스텀 태스크, test만 존재, **비상업 라이선스** |
| `ppl_v1` | wiki40b_ko(test 앞 500문서), wikitext-2 | `bits_per_byte`, 낮을수록 좋음 |
| `en_mc_v1` | MMLU(5), ARC-Challenge(25), HellaSwag(10), Winogrande(5) | Open LLM Leaderboard v1 설정 |
| `gen_v1` | HRM8K(엄격 채점), IFEval-Ko | 생성, 정답 기반 규칙 채점 |
| `dev_mc_v1` | KoBEST validation, KMMLU train 과목별 40문항, Global-MMLU-Lite ko dev | **설정 선택 전용**, test와 겹치지 않음 |
| `smoke` | KoBEST COPA/BoolQ | `--limit`과 함께 환경 확인용 |

### 커스텀 태스크

- **KMMLU-Redux / KMMLU-Pro** (`LGAI-EXAONE/*`): gated 데이터셋이다. HF에서 접근 승인을 받아야 하며 cc-by-nc-nd-4.0(비상업·변경 금지)이다. 선택지 수(4~5)를 문항별로 그대로 쓴다.
- **KoBALT-700** (`snunlp/KoBALT-700`): 선택지 10개(A–J), cc-by-nc-4.0.
- **wiki40b_ko**: wiki40b 한국어 test 앞 500문서. 위키백과라 학습 데이터 오염 가능성이 있다.
- **hrm8k_strict**: lm-eval HRM8K 프롬프트를 그대로 쓰되, 원본 채점이 `\boxed{}`가 없으면 마지막 숫자를 답으로 보는 문제를 분리한다.
  `exact_match_strict`(\boxed 필수, 대표 지표), `exact_match`(원본 규칙), `format_fail`(형식 실패율)을 함께 보고한다.
- **ifeval_ko** (`allganize/IFEval-Ko`, 342문항): lm-eval IFEval 채점기 사용. 채점 의존성 `langdetect`, `immutabledict`, `nltk`(punkt_tab).

모든 커스텀 태스크는 데이터셋 revision(commit SHA)을 YAML에 고정한다.

## 실행

```bash
python -m kobench.standard.run --list                                   # 모델·스위트, revision 고정 여부
python -m kobench.standard.run --pin --models qwen3-4b-base             # 공개 모델 revision을 HF 현재 commit으로 고정
python -m kobench.standard.run --download --models qwen3-4b-base        # 고정 revision으로 내려받기
python -m kobench.standard.run --models qwen3-4b-base --suite ko_mc_v1  # 실행 (완료 태스크는 건너뜀, --force로 재실행)
python -m kobench.standard.run --models qwen3-4b-base --suite smoke --limit 50
python -m kobench.standard.run --models smollm3-3b --suite ko_mc_v1 --variant chat=1   # 설정 변형
```

- 공개 모델은 revision이 고정되지 않으면 실행을 거부한다(`--allow-unpinned`로 해제).
- 결과: `results/standard/<model>/<task>__<n>shot.json`과 문항별 `<task>__<n>shot.samples.jsonl.gz`. 변형 결과는 `results/standard/<model>/v_<variant>/`.
- 실행 메타데이터는 결과 JSON 최상위에 기록된다: `model`(경로·revision·파라미터), `variant`, `apply_chat_template`, `seed`, `versions`(라이브러리), `suite_git`(수트 commit), `configs`(lm-eval 태스크 설정, 데이터셋 revision 포함), 소요 시간.
- 모델 경로의 `${VAR:-기본값}`은 환경 변수로 바꿀 수 있다. EVAFRILL 백엔드는 원본 소스 경로를 `EVAFRILL_SRC`(또는 `$LT_SRC/EVAFRILL-Mo`)에서 찾는다.

## 설정 선택: A 표와 B 표

- **A 표(공통 프로토콜)**: 모든 모델이 같은 프롬프트·shot 수·채점 방식. 객관식에는 chat 템플릿을 쓰지 않는다.
- **B 표(최적 설정)**: 모델마다 dev에서 고른 설정. 변형 축은 `docstart=eos`(문서 앞 EOS, 사용자 모델만), `chat=1`(지시 모델의 공식 chat 템플릿), `shots=N`.

선택 규칙은 측정 전에 `configs/standard/dev_selection.yaml`에 고정했다. 세 dev 태스크 평균이 가장 높은 변형을 고르되,
A와의 차이가 0.1%p 이하면 A를 고른다. **test 결과는 설정 선택에 쓰지 않는다.**

```bash
python -m kobench.standard.dev_search run --models qwen3-4b-2507   # dev 변형 실행
python -m kobench.standard.dev_search run --public                 # 공개 모델 전체
python -m kobench.standard.dev_search select                       # configs/standard/best_settings.yaml 갱신
```

## 집계

```bash
python -m kobench.standard.report --suite ko_mc_v1 --ref jansori-bf16             # A 표
python -m kobench.standard.report --suite ko_mc_v1 --ref jansori-bf16 --table B   # B 표
```

출력은 `reports/standard/<suite>[__B].{md,csv}`이고 base/instruct 표를 나눈다. 셀 값은 다음과 같다.

- **lm-eval**: lm-eval이 보고한 점수(그룹은 lm-eval 집계 규칙).
- **micro**: 문항별 결과에서 다시 계산한 값과 문항 단위 bootstrap 95% CI(2,000회). bits_per_byte는 코퍼스 비율(−Σloglik / ln2 / Σbytes)로 계산하고 문서 단위로 재표집한다.
- **Δref**: 같은 문항 집합에서 기준 모델 대비 paired bootstrap 차이.
- **ΔA**(B 표): 같은 모델의 A 대비 paired 차이.

## 결과 파일 공개 정책

저장소에는 **요약 JSON(점수·설정·버전)만** 커밋한다. 문항별 `*.samples.jsonl.gz`는 `.gitignore`로 제외한다.
데이터셋 원문(KMMLU-Redux/Pro는 gated·재배포 금지)을 포함하고 크기도 수백 MB이기 때문이다. `--limit` 스모크 결과도 제외한다.
필요하면 위 명령으로 같은 revision에서 다시 생성할 수 있다.

## 해석 시 주의

- 공개 벤치마크는 공개 모델의 학습 데이터에 섞였을 수 있다(특히 위키백과 기반 PPL).
- base와 instruct는 다른 표로 비교한다. MoE는 총/활성 파라미터를 함께 표시한다.
- 무작위 수준: 4지선다 25%, KMMLU-Pro 20~25%, KoBALT 10%.
