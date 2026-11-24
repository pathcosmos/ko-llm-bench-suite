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
| `kobench/standard/tasks/{dev,custom,gen,uni}` | lm-eval에 없는 태스크(아래) |
| `kobench/standard/gen.py`, `gen_data.py`, `gen_score.py` | 생성 평가: vLLM offline·EVAFRILL 엔진 배치 생성, 문항 로더(`configs/standard/gen.yaml`), 규칙 채점·오동작 지표 |
| `kobench/standard/judge.py` | LLM judge(KoMT·LogicKor·K2 생성·ko-arena-hard·KoSimpleQA) |
| `kobench/standard/code_exec.py` | HumanEval+/MBPP+ 격리 실행(rlimit + 새 세션 + audit hook) |
| `kobench/standard/scenario.py` | 다턴 시나리오 벤치(절차 생성, 규칙 채점) |
| `kobench/standard/unified.py` | 통일 비교 프로토콜 U(`configs/standard/unified.yaml`) |
| `kobench/standard/gen_report.py` | 생성 dev 설정 선택(`select`·`tune`·`evafrill_ckpt`), 표(`table`), 요약 JSON(`summarize`) |
| `kobench/standard/perf.py` | 클라이언트 측 서빙 성능(TTFT·TPOT·처리량·동시 요청·prefix caching·VRAM) |
| `kobench/standard/llamacpp_lm.py`, `native/llama_loglik.cpp`, `gguf_retokenize.py` | GGUF 로그우도·PPL(libllama 직접), 학습 토크나이저 GGUF 변환 |
| `kobench/standard/final_data.py`, `final_report.py` | 모든 결과를 모아 한 장짜리 HTML 보고서 생성(`reports/final/`) |

백엔드: `vllm`(기본, BF16), `evafrill`(PyTorch 직접), `llamacpp`(GGUF). 개발·검증 환경은 vLLM 0.33.0, transformers 5.20.2, lm-eval 0.4.13, llama.cpp b12084, H200 1장이다.

## 스위트 (`configs/standard/suites.yaml`)

| 스위트 | 태스크 | 비고 |
|---|---|---|
| `ko_mc_v1` | KoBEST(5-shot), KMMLU(5-shot), HAE-RAE, CLIcK, Belebele-ko, Global-MMLU-ko | 한국어 객관식, 로그우도 |
| `ko_mc_ext_v1` | KMMLU-Redux, KMMLU-Pro, KoBALT-700 (0-shot) | 커스텀 태스크, test만 존재, **비상업 라이선스** |
| `ppl_v1` | wiki40b_ko(test 앞 500문서), wikitext-2 | `bits_per_byte`, 낮을수록 좋음 |
| `en_mc_v1` | MMLU(5), ARC-Challenge(25), HellaSwag(10), Winogrande(5) | Open LLM Leaderboard v1 설정 |
| `dev_mc_v1` | KoBEST validation, KMMLU train 과목별 40문항, Global-MMLU-Lite ko dev | **설정 선택 전용**, test와 겹치지 않음 |
| `uni_mc_v1` | KorMedMCQA·KoBBQ·TruthfulQA-ko·K2 knowledge의 cloze(선택지 문장 로그우도) | 통일 비교 U1 |

생성 스위트(`configs/standard/gen.yaml`, `python -m kobench.standard.gen`):

| 스위트 | 태스크 | 비고 |
|---|---|---|
| `gen_v1` | HRM8K(엄격), IFEval-Ko, KorMedMCQA·KoBBQ·TruthfulQA-ko·K2 knowledge(생성 객관식), KoMT, LogicKor, K2 generation, ko-arena-hard, KoSimpleQA, HumanEval+, MBPP+ | 규칙·코드 실행·judge 채점. base 모델은 코드만 |
| `scenario_v1` | `scenario_self`(앞 턴 = 모델 자신의 답), `scenario_oracle`(앞 턴 = 참조 답) | 다턴 300개, 규칙 채점 |
| `uni_v1` | hrm8k, kosimpleqa, humaneval_plus, mbpp_plus, ifeval_ko, komt, logickor, k2_generation, ko_arena_hard, scenario_self | 통일 비교 U2·U3 (chat 템플릿 없음) |
| `gen_dev_v1`, `tune_dev_v1`, `scenario_dev_v1`, `uni_dev_v1` | GSM8K-ko train 300, KorMedMCQA dev, 시나리오 dev 100 | **설정 선택·프롬프트 동작 확인 전용** |
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

## 생성 평가

```bash
python -m kobench.standard.gen run   --models <m> --suite gen_v1 --variant A [--tasks ...] [--limit N]
python -m kobench.standard.gen score --models <m> --suite gen_v1 --variant A
python -m kobench.standard.judge run [--models ...] [--variants A chosen think]; python -m kobench.standard.judge score
python -m kobench.standard.gen_report tune --stage 1|2      # dev로 모델별 운용 설정 선택 → best_settings.yaml gen 절
python -m kobench.standard.gen_report table --table A|B|think|thinkrd|U|tune
```

- 결과: `results/gen/<model>/<variant>/<task>.{jsonl,meta.json,score.json,judge.jsonl}`. instruct는 각 모델의 공식 chat 템플릿, base는 원문 이어쓰기(코드만).
- **변형** `기본[+수정자]`: `A`(greedy, thinking off, 문맥 4096), `card`(저장소 generation_config / 사용자 모델은 카드 권장값), `rep`·`rp11`·`rp12`(repetition_penalty), `fp03`, `think`(thinking on, 문맥 16384), `thinkrd`(thinking on + README 권장 샘플링), `U`(통일 비교), 수정자 `+sys`(한국어 시스템 지시).
- **채점**: HRM8K는 `\boxed{}` 필수 엄격 채점과 형식 실패를 분리하고, 생성 객관식은 '정답: X' 고정 추출 규칙(선택지를 그대로 나열한 응답은 형식 실패)을 쓴다. 모든 생성 표에 오동작 지표(반복 루프·중복 3-gram·언어 이탈·템플릿 누출·빈 답·잘림)를 함께 낸다.
- **설정 선택(v3, 측정 전 고정 `configs/standard/dev_selection.yaml`)**: dev(`tune_dev_v1`)에서 오동작 제약(루프 ≤2%, 언어 이탈 ≤2%, 누출 ≤0.5%, 잘림 ≤5%)을 만족하는 후보 중 품질 최고를 고른다. 만족 후보가 없으면 위반이 가장 작은 설정을 고르고 '설정으로 다 못 막음'으로 보고한다.
- **Judge**: Qwen3.6-35B-A3B(offline, thinking off, greedy). KoMT(MT-Bench 단일/다턴, 참조 답), LogicKor(0~10), K2 generation(1~10), ko-arena-hard(기준 llama-3.2-3b-instruct, 위치 교차 2회), KoSimpleQA(SimpleQA 채점기). judge와 같은 계열(Qwen) 모델에는 자기 선호 편향 가능성을 표시한다.
- **코드**: evalplus 테스트를 격리 실행한다. 컨테이너에서 unshare/seccomp를 쓸 수 없어 rlimit(CPU·메모리 4 GiB·파일)·새 세션·임시 cwd·`python -I -B`·audit hook(소켓·하위 프로세스·ctypes·외부 쓰기 차단)으로 막는다. HumanEval/32는 데이터셋 테스트 결함(정답 코드도 실패)이라 제외한다.

### 다턴 시나리오 (`scenario.py`)

10개 범주(수학 조건 변경, 일정 기억, 지시 유지, 형식 변환, 순서 추론, 목록 추적, 이어쓰기, 한국어 문체, 코드 수정, 모름 인정)를 시드로 절차 생성한다(test 300, dev 100, 3~5턴). 판정은 규칙으로만 한다. 참조 답은 400/400 통과하고, 빈 답이나 첫 턴 답 복사로는 시나리오를 하나도 통과하지 못함을 단위 테스트로 고정했다. 턴 통과율, 시나리오 통과율, 턴 위치별 통과율, 반복 지표(이전 답 중복·복사)를 보고한다.

## 통일 비교 프로토콜 U (`configs/standard/unified.yaml`)

base·instruct·사용자 모델 전부를 **chat 템플릿 없이 같은 원문 프롬프트**로 한 줄에 비교한다. 문구·예시·정규화·영역은 test 결과를 보기 전에 고정했다.

- **U1**(`uni_mc_v1`): 생성으로 풀던 객관식 4종을 선택지 문장 로그우도(cloze)로.
- **U2 few-shot**: HRM8K(GSM8K-ko train 예시 5개), KoSimpleQA(직접 쓴 예시 5개), 코드(base 이어쓰기 형식).
- **U3 URIAL**(Lin et al. 2023): 고정 머리말 + 대화 예시 3개 뒤에 질문. 다턴은 모델 자신의 앞 답을 같은 형식으로 이어 붙인다.
- **종합**: 태스크 점수를 정규화(객관식은 무작위 보정, judge는 /10, PPL은 최저 bpb/bpb)해 9개 영역(지식, 전문·의료, 사실성·편향, 수학, 코드, 지시 수행, 대화·생성 품질, 다턴, 언어 모델링)의 평균을 내고, 영역을 같은 가중치로 평균한다. 속도는 종합에 넣지 않고 품질 × 처리량 산점도로 따로 본다.

## 성능과 GGUF

```bash
python -m kobench.standard.perf --models <m> [--engine vllm|llamacpp|evafrill] [--concurrency 1 4 16 32 64]
python -m kobench.standard.report --suite ko_mc_v1 --table Q     # GGUF 양자화 손실 표
```

- 서버를 모델마다 하나씩 단독으로 띄우고 예열 요청은 집계에서 뺀다. W1: ko-arena 질문 128개, 출력 256 토큰 고정, 동시 1/4/16/32/64. W2: 약 2,000토큰 공유 prefix로 prefix caching on/off. 결과는 `results/perf/<model>/<engine>.json`.
- GGUF 로그우도·PPL은 llama.cpp(`llama_loglik`)로 계산한다. 토큰화·자르기는 vLLM 실행과 같게 맞춰 BF16 대비 차이를 paired CI로 본다.

## 최종 보고서

```bash
python -m kobench.standard.final_data --regen    # 표 재생성 + reports/final/data.json
python -m kobench.standard.final_report          # data.json + narrative.json(해설) → reports/final/report.html
```

[`reports/final/report.html`](../reports/final/report.html)은 사용자 모델(JANSORI, frankenstallm, EVAFRILL-Mo)과 공개 모델 30개(dense 3B급, MoE A3B급)를 비교한 한 장짜리 보고서다. 객관식·PPL, 양자화 손실, 생성(A/B/thinking), 다턴, 통일 비교 U, 서빙 성능을 담았다. 차트는 d3(cdnjs)로 그리므로 파일을 내려받아 브라우저에서 연다. 해설 문장은 `reports/final/narrative.json`에 따로 두고, 차트·표는 `data.json`에서 계산한다.

## 결과 파일 공개 정책

저장소에는 **요약 JSON(점수·설정·버전)만** 커밋한다. 문항별 `*.samples.jsonl.gz`는 `.gitignore`로 제외한다.
데이터셋 원문(KMMLU-Redux/Pro는 gated·재배포 금지)을 포함하고 크기도 수백 MB이기 때문이다. `--limit` 스모크 결과도 제외한다.
생성 결과도 같다: `results/gen`에는 `*.meta.json`(실행 설정)과 문항 결과를 뺀 `*.summary.json`(`gen_report summarize`)만 커밋하고, 모델 출력·judge 기록(`*.jsonl`)과 문항별 점수(`*.score.json`)는 커밋하지 않는다.
필요하면 위 명령으로 같은 revision에서 다시 생성할 수 있다.

## 해석 시 주의

- 공개 벤치마크는 공개 모델의 학습 데이터에 섞였을 수 있다(특히 위키백과 기반 PPL).
- base와 instruct는 다른 표로 비교한다. MoE는 총/활성 파라미터를 함께 표시한다.
- 무작위 수준: 4지선다 25%, KMMLU-Pro 20~25%, KoBALT 10%.
- 통일 비교 U의 코드 태스크는 base 이어쓰기 형식이라, 이 형식을 따르지 않는 instruct 모델(설명문을 쓰거나 docstring 뒤에 바로 `def`를 붙이는 경우)은 MBPP+가 0에 가깝게 나온다. 프롬프트 규칙은 측정 전에 고정했고 test 결과로 바꾸지 않았다.
- judge(Qwen3.6-35B-A3B)와 같은 Qwen 계열 모델의 judge 점수는 자기 선호 편향이 있을 수 있다.
