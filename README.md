# KoBench Suite (ko-llm-bench-suite)

[English](README_EN.md)

한국어 LLM 종합 벤치마크 평가 도구

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-515%20passed-brightgreen.svg)](tests/)

---

## 소개

KoBench Suite는 한국어 대형 언어 모델(LLM)의 성능을 **7개 트랙**에 걸쳐 종합 평가하는 벤치마크 도구입니다.
YAML 설정 파일만으로 새로운 모델을 추가하고 평가할 수 있으며, 코드 수정 없이 다양한 모델을 비교할 수 있습니다.

단순한 정확도 측정을 넘어, 한국어 생성 품질·심층 문화 이해·코드/수학 추론·응답 일관성·추론 속도·모델 간 직접 비교(Elo)까지 아우르는 다차원 평가를 제공합니다. 이중 LLM Judge 시스템으로 평가 편향을 줄이고, 자동 생성되는 HTML/Markdown 리포트와 시각화 차트로 결과를 한눈에 파악할 수 있습니다.

### 왜 KoBench Suite인가?

- **한국어 특화**: KoBEST, 속담/존댓말/문화 이해 등 한국어 고유 능력을 정밀 측정
- **종합 평가**: 7개 독립 트랙으로 모델의 강점과 약점을 다각도로 분석
- **공정한 비교**: 이중 Judge + Pairwise Elo로 주관적 편향 최소화
- **확장 가능**: YAML 설정만 수정하면 새 모델·새 트랙 추가 가능
- **재현 가능**: 체크포인트, 시드 고정, JSON 원본 데이터 보존

## 주요 기능

| 기능 | 설명 |
|------|------|
| **7개 평가 트랙** | KoBEST NLU, Ko-Bench 생성, 심층이해, 코드/수학, 일관성, 성능, Pairwise Elo |
| **다중 추론 백엔드** | Ollama (로컬/원격), EVAFRILL PyTorch 직접 추론, vLLM (예정) |
| **이중 Judge 시스템** | 두 개의 LLM Judge로 교차 검증 (가중 평균), 평가 편향 감소 |
| **YAML 기반 설정** | 코드 수정 없이 모델/트랙/Judge/샘플링 파라미터 구성 |
| **자동 리포트** | HTML/Markdown 리포트 + 26종 시각화 차트 자동 생성 |
| **체크포인트/재개** | 장시간 평가 중단 후 이어서 실행 가능 |
| **원격 서버 지원** | 원격 GPU 서버의 Ollama를 네트워크를 통해 활용 |
| **EVAFRILL 지원** | Mamba-2 하이브리드 아키텍처 모델의 HTTP 서버 + PyTorch 직접 추론 |

## 평가 결과 미리보기

실제 13개 모델(커스텀 5 + 베이스라인 8)을 RTX 5060 Ti 16GB에서 평가한 결과입니다.

### 종합 히트맵

전 모델 × 전 트랙의 점수를 한눈에 비교할 수 있는 히트맵입니다. 각 셀의 색상이 진할수록 높은 점수를 의미합니다.

![종합 히트맵](examples/frankenstallm/reports/visualizations/01_heatmap_overview.png)

### 레이더 비교 차트

주요 모델들의 트랙별 강점과 약점을 레이더 차트로 시각화한 것입니다. 다각형의 면적이 넓을수록 종합 성능이 우수합니다.

![레이더 비교](examples/frankenstallm/reports/visualizations/07_radar_comparison.png)

### Elo 랭킹 (Pairwise 비교)

Track 7의 모델 간 직접 대결 결과를 Bradley-Terry Elo 레이팅으로 환산한 랭킹입니다. Judge가 두 모델의 응답을 직접 비교하여 산출합니다.

![Elo 랭킹](examples/frankenstallm/reports/visualizations/05_t7_elo_ranking.png)

## 빠른 시작

### 설치

```bash
# 리포지토리 클론
git clone https://github.com/pathcosmos/ko-llm-bench-suite.git
cd ko-llm-bench-suite

# Python 패키지 설치
pip install -r requirements.txt

# 개발/테스트용 (선택)
pip install -r requirements-dev.txt

# Ollama 설치 (이미 설치되어 있으면 건너뛰기)
curl -fsSL https://ollama.com/install.sh | sh

# Judge 모델 다운로드
ollama pull qwen2.5:7b-instruct
ollama pull exaone3.5:7.8b

# 평가 대상 모델 다운로드 (예시)
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

### 실행

```bash
# 설정 파일로 전체 평가 실행
python kobench.py --config configs/examples/frankenstallm.yaml

# 특정 트랙만 실행
python kobench.py --config configs/examples/frankenstallm.yaml --tracks 1 4 6

# 기존 결과로 리포트만 생성
python kobench.py --config configs/examples/frankenstallm.yaml --report-only
```

### 설정 파일 예시

```yaml
project:
  name: "My Evaluation"
  output_dir: "./results"
  reports_dir: "./reports"

backend:
  type: ollama
  url: "http://localhost:11434"

models:
  - name: "qwen2.5:3b"
    tags: [baseline]
  - name: "gemma3:4b"
    tags: [baseline]

tracks:
  enabled: [1, 2, 3, 4, 5, 6, 7]

judge:
  dual_enabled: true
  primary:
    model: "qwen2.5:7b-instruct"
    weight: 0.6
  secondary:
    model: "exaone3.5:7.8b"
    weight: 0.4

sampling:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2048
  seed: 42
```

전체 설정 옵션은 [`configs/default.yaml`](configs/default.yaml)을 참조하세요.

## 평가 트랙

| 트랙 | 이름 | 설명 | LLM Judge |
|------|------|------|:---------:|
| T1 | Korean Bench | KoBEST 4개 태스크 (BoolQ, COPA, SentiNeg, HellaSwag) | |
| T2 | Ko-Bench | 8개 카테고리 한국어 생성 품질 평가 | O |
| T3 | Korean Deep | 심층 한국어 이해력 (속담, 존댓말, 문화 등) | O |
| T4 | Code & Math | Python 코딩 + 수학 문제 해결 | |
| T5 | Consistency | 동일 질문 반복 시 응답 일관성 | |
| T6 | Performance | 토큰 생성 속도(TPS), 레이턴시, 동시성 | |
| T7 | Pairwise Elo | 모델 쌍대비교 → Bradley-Terry Elo 레이팅 | O |

### T1: Korean Bench (KoBEST)

한국어 자연어 이해(NLU)의 기초 체력을 측정합니다. KoBEST 데이터셋의 4개 태스크 -- BoolQ(참/거짓 판단), COPA(인과 추론), SentiNeg(부정 감성 분석), HellaSwag(상황 완성) -- 에서 정확도를 측정합니다. 객관식 형태이므로 LLM Judge가 필요 없으며, 모든 모델의 한국어 기본 이해력을 공정하게 비교할 수 있는 기반 트랙입니다.

### T2: Ko-Bench (한국어 생성 품질)

창작, 요약, 번역, 분석 등 8개 카테고리에 걸쳐 한국어 텍스트 생성 품질을 평가합니다. 싱글턴과 멀티턴 대화 모두를 포함하며, LLM Judge가 유창성·정확성·완결성·지시 따르기를 기준으로 1-10점을 부여합니다. 멀티턴에서의 품질 저하(turn degradation) 패턴도 분석합니다.

### T3: Korean Deep (심층 한국어 이해)

한국어 속담 해석, 존댓말/반말 변환, 한국 문화 상식, 한자성어 등 한국어 고유의 깊은 이해력을 테스트합니다. 단순 번역으로는 획득할 수 없는 문화적 맥락 이해를 측정하며, 한국어 특화 학습 데이터의 양과 질을 간접적으로 반영합니다.

### T4: Code & Math (코드 및 수학)

Python 코딩 문제와 수학 문제를 제시하고, 코드는 실제 실행을 통한 테스트 케이스 통과 여부로, 수학은 최종 답의 정확성으로 평가합니다. 소형 모델(3B 이하)에서는 매우 도전적인 트랙으로, 모델의 논리적 추론 능력과 코드 생성 품질을 엄격하게 측정합니다.

### T5: Consistency (응답 일관성)

동일한 질문을 3회 반복하여 응답 간 의미적 일관성을 측정합니다. 코사인 유사도 기반으로 응답 변동성을 정량화하며, 높은 일관성은 모델의 안정성과 신뢰성을 나타냅니다. temperature 설정에 따른 변동성도 함께 분석합니다.

### T6: Performance (추론 성능)

토큰 생성 속도(tokens/sec), 첫 토큰 도달 시간(TTFT), 동시 요청 처리 능력을 측정합니다. 동일 하드웨어에서의 상대 비교에 초점을 맞추며, 모델 크기 대비 효율성(tokens/sec per billion parameters)을 산출하여 실용적인 배포 관점의 정보를 제공합니다.

### T7: Pairwise Elo (쌍대비교 랭킹)

모든 모델 쌍에 대해 동일 프롬프트에 대한 응답을 LLM Judge가 직접 비교합니다. Bradley-Terry 모델을 적용하여 Elo 레이팅을 산출하며, 위치 편향(position bias)을 줄이기 위해 응답 순서를 교차합니다. Track 1의 객관식 정확도와 r=0.90의 높은 상관을 보이는 것이 특징입니다.

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |
| Python | 3.10+ | 3.12 |
| RAM | 16GB | 32GB |
| GPU | 없음 (CPU 가능) | NVIDIA 16GB+ VRAM |
| 디스크 | 20GB 여유 | 50GB 여유 |

> **참고**: GPU 없이 CPU만으로도 평가가 가능하지만, 추론 속도가 크게 느려집니다.
> 13개 모델 전체 평가 기준, RTX 5060 Ti에서 약 4-6시간, CPU에서는 수일이 소요될 수 있습니다.

## 프로젝트 구조

```
ko-llm-bench-suite/
├── kobench.py                 # 메인 실행 스크립트
├── kobench/                   # 평가 프레임워크 코어 패키지
│   ├── __init__.py
│   ├── config.py              # 설정 관리
│   ├── runner.py              # Ollama API 실행 엔진
│   ├── judge.py               # 이중 LLM-as-Judge 시스템
│   ├── evafrill_runner.py     # EVAFRILL PyTorch 직접 추론
│   ├── evafrill_server.py     # EVAFRILL HTTP 추론 서버
│   ├── scoring.py             # 스코어카드 + Bradley-Terry Elo
│   ├── report.py              # HTML/Markdown 리포트 생성
│   ├── backends/              # 추론 백엔드 추상화
│   │   ├── base.py            #   InferenceBackend ABC
│   │   └── ollama.py          #   OllamaBackend 구현
│   └── tracks/                # 7개 평가 트랙
│       ├── korean_bench.py    #   T1: KoBEST
│       ├── ko_bench.py        #   T2: Ko-Bench
│       ├── korean_deep.py     #   T3: 심층이해
│       ├── code_math.py       #   T4: 코드/수학
│       ├── consistency.py     #   T5: 일관성
│       ├── performance.py     #   T6: 성능
│       └── pairwise.py        #   T7: Pairwise Elo
├── configs/                   # YAML 설정 파일
│   ├── default.yaml           #   기본 설정 (템플릿)
│   └── examples/
│       └── frankenstallm.yaml #   13모델 평가 예시
├── benchmarks/                # 벤치마크 데이터셋
│   ├── ko_bench/              #   T2 질문 데이터
│   ├── korean_deep/           #   T3 질문 데이터
│   ├── code_problems/         #   T4 코딩 문제
│   ├── math_problems/         #   T4 수학 문제
│   └── track7_prompts.json    #   T7 Pairwise 프롬프트
├── examples/
│   └── frankenstallm/         # Frankenstallm 평가 결과 아카이브
│       └── reports/
│           └── visualizations/ #  26종 시각화 차트
├── tests/                     # pytest 테스트 (515개)
│   ├── unit/
│   └── integration/
├── results/                   # 평가 결과 출력 (JSON)
├── reports/                   # 생성된 리포트 (HTML/MD)
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## 원격 Ollama 서버 사용

원격 GPU 서버의 Ollama를 사용하려면 설정 파일에서 URL과 `remote: true`를 지정합니다.

```yaml
backend:
  type: ollama
  url: "http://192.168.1.100:11434"
  remote: true
```

원격 모드에서는 모델 로드/언로드를 HTTP API를 통해 자동으로 관리합니다. 평가 종료 후 모델이 자동으로 언로드되어 VRAM을 확보합니다. 네트워크 지연이 있으므로 T6(성능 측정) 트랙의 결과는 로컬 실행과 차이가 있을 수 있습니다.

```bash
# 원격 서버 연결 확인
curl http://192.168.1.100:11434/api/tags

# 원격 서버에 모델이 설치되어 있는지 확인
curl http://192.168.1.100:11434/api/tags | python -m json.tool
```

## EVAFRILL 모드

EVAFRILL(Mamba-2 하이브리드 아키텍처) 모델은 Ollama를 통하지 않고 PyTorch로 직접 추론합니다.

```yaml
backend:
  type: evafrill
  evafrill:
    model_path: "/path/to/evafrill/model"
    device: "cuda"
```

또는 HTTP 서버 모드로 원격 실행이 가능합니다:

```bash
# 원격 서버에서 EVAFRILL HTTP 서버 시작
python -m kobench.evafrill_server --model-path /path/to/model --port 8000

# 클라이언트에서 설정
# backend:
#   type: evafrill
#   evafrill:
#     url: "http://192.168.1.100:8000"
```

## 테스트

```bash
# 전체 테스트 실행 (Ollama/GPU 불필요, mock 기반)
pytest tests/ -v

# 단위 테스트만
pytest tests/unit/ -v

# 통합 테스트만
pytest tests/integration/ -v

# 커버리지 리포트
pytest tests/ --cov=kobench --cov-report=html
```

> 515개 전체 테스트가 Ollama나 GPU 없이 mock 기반으로 동작합니다.
> 이중 Judge, HTTP 모드, 체크포인트 재개 등 모든 핵심 기능이 테스트에 포함되어 있습니다.

## 예제: Frankenstallm 평가 결과

`examples/frankenstallm/` 디렉토리에 13개 모델(커스텀 5 + 베이스라인 8)에 대한 7트랙 종합 평가 결과가 포함되어 있습니다. RTX 5060 Ti 16GB에서 실행된 결과이며, HTML/Markdown 리포트와 원본 JSON 데이터를 확인할 수 있습니다.

### Top 5 종합 랭킹

| 순위 | 모델 | T1 NLU | T2 생성 | T3 심층 | T7 Elo | 종합 |
|:----:|-------|:------:|:------:|:------:|:------:|:----:|
| 1 | EXAONE3.5:7.8b | 74.0 | 7.2 | 72.0 | 1180 | **78.2** |
| 2 | Qwen2.5:7b-instruct | 72.5 | 7.0 | 68.0 | 1150 | **75.8** |
| 3 | Gemma3:4b | 65.0 | 6.8 | 64.0 | 1080 | **68.5** |
| 4 | Qwen2.5:3b | 60.5 | 6.5 | 58.0 | 1020 | **63.2** |
| 5 | Llama3.2:3b | 58.0 | 6.2 | 52.0 | 980 | **59.8** |

### 주요 발견

- **T1과 T7의 높은 상관관계 (r=0.90)**: 객관식 NLU 정확도와 Pairwise Elo 순위가 거의 일치합니다. 기초 이해력이 뛰어난 모델이 직접 비교에서도 선호되는 경향을 보여, T1만으로도 모델의 전반적 한국어 능력을 예측할 수 있습니다.

- **T4 코드/수학의 극단적 난이도**: 3B 이하 모델 전체가 T4에서 0% 정답률을 기록했습니다. 소형 모델의 코드 생성 및 수학적 추론 능력이 실용 수준에 도달하지 못했음을 보여주며, 이 트랙이 모델 크기에 가장 민감한 변별 요소임을 확인했습니다.

- **EXAONE 3.5의 크기 대비 효율성**: EXAONE3.5:7.8b가 종합 1위를 차지하며, 파라미터 수 대비 뛰어난 효율성을 입증했습니다. 특히 T3(심층 한국어 이해)에서 한국어 특화 학습의 효과가 두드러지며, T6에서도 준수한 추론 속도를 유지합니다.

- **v1 > v2 역설**: 일부 커스텀 모델에서 v2(개선 버전)가 v1보다 낮은 점수를 기록하는 역설이 관찰되었습니다. 파인튜닝 데이터 증량이나 학습 에폭 증가가 반드시 성능 향상으로 이어지지 않을 수 있음을 시사하며, 과적합(overfitting) 또는 catastrophic forgetting의 가능성을 보여줍니다.

### 시각화 목록

`examples/frankenstallm/reports/visualizations/` 디렉토리에 26종의 차트가 포함되어 있습니다:

| 번호 | 차트 | 설명 |
|:----:|------|------|
| 01 | heatmap_overview | 전 모델 × 전 트랙 종합 히트맵 |
| 02 | t1_korean_bench | T1 KoBEST 세부 태스크별 정확도 |
| 03 | t2_ko_bench | T2 Ko-Bench 카테고리별 점수 |
| 04 | t4_code_math | T4 코드/수학 정답률 |
| 05 | t7_elo_ranking | T7 Elo 레이팅 랭킹 |
| 06 | t6_speed_t5_consistency | T6 속도 vs T5 일관성 |
| 07 | radar_comparison | 주요 모델 레이더 비교 |
| 08 | dashboard | 종합 대시보드 |
| 09 | t2_turn_degradation | T2 멀티턴 품질 저하 분석 |
| 10 | t3_korean_deep_categories | T3 카테고리별 심층 분석 |
| 11 | t4_problem_heatmap | T4 문제별 정답 히트맵 |
| 12 | t5_consistency_radar | T5 일관성 레이더 |
| 13 | t6_speed_curves | T6 속도 곡선 |
| 14 | t6_quantization | T6 양자화 영향 분석 |
| 15 | t7_h2h_winrate | T7 Head-to-Head 승률 |
| 16 | cross_track_correlation | 트랙 간 상관관계 |
| 17 | frankenstallm_lineage | Frankenstallm 계보도 |
| 18 | efficiency_frontier | 효율성 프론티어 |
| 19 | t2_turn_scatter | T2 턴 산점도 |
| 20 | t4_discriminating_problems | T4 변별력 높은 문제 |
| 21 | t6_context_limits | T6 컨텍스트 한계 |
| 22 | v1_vs_v2_paired | v1 vs v2 대응 비교 |
| 23 | korean_specialization | 한국어 특화도 분석 |
| 24 | quality_speed_tradeoff | 품질-속도 트레이드오프 |
| 25 | t5_stacked_consistency | T5 적층 일관성 |
| 26 | efficiency_bubble | 효율성 버블 차트 |

## Contributing

기여를 환영합니다! 다음과 같은 방법으로 참여할 수 있습니다:

1. **새 모델 평가**: `configs/examples/`에 YAML 설정 추가 후 결과 공유
2. **새 트랙 추가**: `kobench/tracks/` 에 TrackEvaluator 구현
3. **벤치마크 데이터 확장**: `benchmarks/` 에 새로운 질문/문제 추가
4. **버그 리포트 및 개선**: GitHub Issues 활용

```bash
# 개발 환경 설정
pip install -r requirements-dev.txt

# 테스트 실행 (PR 제출 전 필수)
pytest tests/ -v

# 코드 스타일 확인
ruff check kobench/
```

## License

[MIT](LICENSE)
