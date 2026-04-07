# KoBench Suite (ko-llm-bench-suite)

한국어 LLM 종합 벤치마크 평가 도구

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

## 소개

KoBench Suite는 한국어 대형 언어 모델(LLM)의 성능을 7개 트랙에 걸쳐 종합 평가하는 벤치마크 도구입니다. YAML 설정 파일만으로 새로운 모델을 추가하고 평가할 수 있으며, 코드 수정 없이 다양한 모델을 비교할 수 있습니다.

## 주요 기능

- **7개 평가 트랙**: 한국어 벤치마크(KoBEST), Ko-Bench, 심층이해, 코드/수학, 일관성, 성능, Pairwise Elo
- **다중 추론 백엔드**: Ollama (로컬/원격), vLLM (예정)
- **이중 Judge 시스템**: 두 개의 LLM Judge로 교차 검증 (가중 평균)
- **YAML 기반 설정**: 코드 수정 없이 모델/트랙/Judge/샘플링 파라미터 구성
- **자동 리포트**: HTML/Markdown 리포트 + 시각화 차트 자동 생성
- **체크포인트**: 중단 후 이어서 실행 가능
- **EVAFRILL 지원**: Mamba-2 하이브리드 아키텍처 모델의 PyTorch 직접 추론

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
# 설정 파일로 실행
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

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |
| Python | 3.10+ | 3.12 |
| RAM | 16GB | 32GB |
| GPU | 없음 (CPU 가능) | NVIDIA 16GB+ VRAM |
| 디스크 | 20GB 여유 | 50GB 여유 |

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
├── tests/                     # pytest 테스트 (515개)
│   ├── unit/
│   └── integration/
├── results/                   # 평가 결과 출력
├── reports/                   # 생성된 리포트
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

## 테스트

```bash
# 전체 테스트 실행 (Ollama/GPU 불필요, mock 기반)
pytest tests/ -v

# 단위 테스트만
pytest tests/unit/ -v

# 통합 테스트만
pytest tests/integration/ -v
```

## 예제: Frankenstallm 평가 결과

`examples/frankenstallm/` 디렉토리에 13개 모델(커스텀 5 + 베이스라인 8)에 대한 7트랙 종합 평가 결과가 포함되어 있습니다. RTX 5060 Ti 16GB에서 실행된 결과이며, HTML/Markdown 리포트와 원본 JSON 데이터를 확인할 수 있습니다.

---

## English

### What is KoBench Suite?

KoBench Suite is a comprehensive benchmark tool for evaluating Korean LLMs across 7 tracks: Korean NLU (KoBEST), Korean generation quality (Ko-Bench), deep Korean understanding, code/math, consistency, performance, and pairwise Elo ranking.

### Key Features

- **7 evaluation tracks** covering Korean language understanding, generation, reasoning, and performance
- **YAML-based configuration** -- add new models without changing code
- **Dual LLM Judge** system with weighted cross-validation
- **Backend abstraction** -- Ollama (local/remote), with vLLM planned
- **Checkpoint/resume** for long-running evaluations
- **Auto-generated reports** in HTML and Markdown with visualizations

### Quick Start

```bash
git clone https://github.com/pathcosmos/ko-llm-bench-suite.git
cd ko-llm-bench-suite
pip install -r requirements.txt

# Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b

# Run evaluation
python kobench.py --config configs/examples/frankenstallm.yaml
```

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v  # 515 tests, no GPU/Ollama required
```

## License

[MIT](LICENSE)
