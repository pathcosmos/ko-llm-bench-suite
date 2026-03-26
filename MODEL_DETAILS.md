# FRANKENSTALLM 3B 평가 대상 모델 상세 문서

> **작성일:** 2026-03-25 (전면 업데이트)
> **이전 버전:** 2026-03-11
> **목적:** FRANKENSTALLM 3B 종합 벤치마크에 사용되는 전체 14개 모델(테스트 13개 + Judge 1개)의 아키텍처, 양자화, 토크나이저, 성능 특성을 상세히 기록
> **환경:** Ubuntu Linux 6.8.0-106-generic, Ollama + PyTorch (GPU 추론: RTX 5060 Ti 16GB), Python 3.12
> **Judge:** gemma3:12b (Ollama, Claude CLI에서 교체)

---

## 목차

1. [모델 개요 테이블](#1-모델-개요-테이블)
2. [FRANKENSTALLM 3B 상세](#2-frankenstallm-3b-상세)
3. [비교 모델 상세](#3-비교-모델-상세)
4. [양자화 비교](#4-양자화-비교)
5. [CPU 추론 성능 참고](#5-cpu-추론-성능-참고)

---

## 1. 모델 개요 테이블

### 전체 모델 매트릭스 (2026-03-25 업데이트)

> **변경 사항:** v2 GGUF가 완전히 교체됨 (1.2B → 3.0B). v1도 토크나이저 수정 후 정상 동작. EVAFRILL-Mo-3B 추가. deepseek-r1:1.5b 추가.

| # | 모델명 | 역할 | 양자화 | 크기 | 파라미터 수 | 아키텍처 | 컨텍스트 | 임베딩 | TPS (GPU) | 상태 |
|---|--------|------|--------|------|-------------|----------|----------|--------|-----------|------|
| 1 | `frankenstallm-3b:latest` | v1 Q4_K_M | Q4_K_M | 2.0 GB | 3.2B | LLaMA | 4,096 | 3,072 | 195 | **OK** |
| 2 | `frankenstallm-3b:Q8_0` | v1 Q8_0 | Q8_0 | 3.4 GB | 3.2B | LLaMA | 4,096 | 3,072 | 115 | **OK** |
| 3 | `frankenstallm-3b-v2:latest` | v2 Q4_K_M | Q4_K_M | 1.9 GB | 3.0B | LLaMA | 4,096 | 3,072 | 193 | **OK** |
| 4 | `frankenstallm-3b-v2:Q8_0` | v2 Q8_0 | Q8_0 | 3.2 GB | 3.0B | LLaMA | 4,096 | 3,072 | 119 | **OK** |
| 5 | `qwen2.5:3b` | 비교 | Q4_K_M | 1.9 GB | 3.1B | Qwen 2 | 32,768 | 2,048 | 172 | **OK** |
| 6 | `gemma3:4b` | 비교 | Q4_K_M | 3.3 GB | 4.3B | Gemma 3 | 131,072 | 2,560 | 110 | **OK** |
| 7 | `phi4-mini` | 비교 | Q4_K_M | 2.5 GB | 3.8B | Phi-3 | 131,072 | 3,072 | 128 | **OK** |
| 8 | `exaone3.5:2.4b` | 비교 | Q4_K_M | 1.6 GB | 2.7B | EXAONE | 32,768 | 2,560 | 174 | **OK** |
| 9 | `llama3.2:3b` | 비교 | Q4_K_M | 2.0 GB | 3.2B | LLaMA | 131,072 | 3,072 | 161 | **OK** |
| 10 | `llama3.1:8b-instruct-q8_0` | 비교(상한) | Q8_0 | 8.5 GB | 8B | LLaMA | 131,072 | 4,096 | 50 | **OK** |
| 11 | `ingu627/exaone4.0:1.2b` | 비교 | Q4_K_M | 812 MB | 1.2B | EXAONE | 32,768 | 2,560 | 180 | **OK** |
| 12 | `deepseek-r1:1.5b` | 비교(추론) | Q4_K_M | 1.1 GB | 1.5B | DeepSeek | 131,072 | 1,536 | 214 | **OK** |
| 13 | `evafrill-mo-3b-slerp` | 비교(하이브리드) | BF16 | 5.9 GB | 2.94B | Mamba-2+Attn | 4,096 | 3,072 | 4.8 | **OK** |
| - | `gemma3:12b` | Judge | Q4_K_M | 8.1 GB | 12B | Gemma 3 | 131,072 | 3,840 | - | **OK** |

### 주요 관찰 사항 (업데이트)

- **v2 GGUF 완전 교체**: 이전 v2(1.2B/2,048 임베딩)가 새 v2(3.0B/3,072 임베딩)로 교체됨. v1(3.2B)과 거의 동일 아키텍처에서 토크나이저만 수정한 버전으로 추정.
- **v1 정상 동작**: 이전에 SPM 토크나이저 결함으로 SIGABRT CRASH였던 v1이 새 GGUF에서 정상 동작. 토크나이저가 수정된 것으로 보임.
- **EVAFRILL-Mo-3B**: Mamba-2 + Transformer 하이브리드 아키텍처 (24× Mamba-2 SSM + 2× Attention GQA). GGUF 변환 불가하여 PyTorch 직접 추론. KV 캐시 미지원으로 TPS 극히 낮음.
- **모델명 체계 변경**: 이전 대시 방식(`frankenstallm-3b-v2-Q4_K_M`)에서 Ollama 태그 방식(`frankenstallm-3b-v2:latest`)으로 변경.
- **Judge 교체**: Claude CLI(`claude -p`)에서 gemma3:12b(Ollama API)로 교체. 토큰 비용 절감.

---

## 2. FRANKENSTALLM 3B 상세

### 2.1 FRANKENSTALLM이란?

FRANKENSTALLM은 "Frankenstein" + "STALLM"(Small Talk And Language Learning Model)의 합성어로, **한국어에 특화된 커스텀 LLM**이다. LLaMA 아키텍처를 기반으로 구축되었으며, 여러 기술과 데이터셋을 조합하여 만들어진 실험적 모델이라는 의미를 이름에 담고 있다.

#### 핵심 특성

| 항목 | 내용 |
|------|------|
| **기반 아키텍처** | LLaMA (Meta의 Large Language Model Meta AI) |
| **목표 언어** | 한국어/영어 이중언어 |
| **파인튜닝 기법** | ORPO (Odds Ratio Preference Optimization) |
| **토크나이저** | SentencePiece Model (SPM) |
| **컨텍스트 길이** | 4,096 토큰 |
| **시스템 프롬프트** | "당신은 FRANKENSTALLM, 한국어에 특화된 AI 어시스턴트입니다. 정확하고 자연스러운 한국어로 답변해주세요." |
| **기능** | completion (대화/도구 호출 미지원) |

#### 샘플링 파라미터 (ORPO 최적화)

FRANKENSTALLM의 Modelfile에 내장된 기본 샘플링 파라미터는 ORPO 학습 환경에 맞춰 조정되었다:

```
temperature:    0.7    — 창의성과 일관성의 균형점
top_k:          50     — 상위 50개 토큰으로 후보 제한
top_p:          0.9    — 누적 확률 90%까지만 샘플링
repeat_penalty: 1.2    — 반복 생성 억제
num_predict:    512    — 최대 생성 길이
num_ctx:        4096   — 컨텍스트 윈도우
stop:           </s>   — 생성 종료 토큰
```

#### ORPO (Odds Ratio Preference Optimization) 학습

ORPO는 기존의 RLHF(Reinforcement Learning from Human Feedback)와 달리, 보상 모델(Reward Model) 없이 선호도 최적화를 수행하는 기법이다.

- **기존 RLHF 파이프라인**: SFT → Reward Model 학습 → PPO 강화학습 (3단계)
- **ORPO 파이프라인**: SFT + 선호도 최적화 동시 수행 (1단계)
- **핵심 아이디어**: Supervised Fine-Tuning 손실 함수에 Odds Ratio 기반 선호도 항을 추가하여, 선호 응답의 확률을 높이고 비선호 응답의 확률을 낮춤
- **장점**: 학습 파이프라인 단순화, 계산 비용 절감, 안정적인 수렴

이 기법은 소규모 모델에서 특히 효과적이며, FRANKENSTALLM 3B의 한국어 응답 품질 향상에 기여한 것으로 추정된다.

### 2.2 v1 vs v2 차이점 상세

#### 구조적 변경 사항

| 항목 | v1 | v2 | 변경 내용 |
|------|-----|-----|-----------|
| **파라미터 수** | 3.2B | 1.2B | 모델 크기 대폭 축소 |
| **임베딩 차원** | 3,072 | 2,048 | 임베딩 레이어 축소 |
| **Vocab 크기** | 64,000 | 64,256 | +256 바이트 폴백 토큰 |
| **파일 크기 (Q4_K_M)** | 1.9 GB | 757 MB | ~60% 감소 |
| **파일 크기 (Q8_0)** | 3.2 GB | 1.2 GB | ~63% 감소 |
| **파일 크기 (F16)** | 6.0 GB | 2.3 GB | ~62% 감소 |
| **실행 가능 여부** | 불가 (CRASH) | 가능 (OK) | 토크나이저 결함 수정 |

#### v1의 치명적 결함 — SPM 토크나이저 오류

v1 모델은 GGUF(GPT-Generated Unified Format)로의 변환 과정에서 SPM 토크나이저 직렬화에 심각한 오류가 발생했다. 이로 인해 **양자화 수준과 무관하게** Q4_K_M, Q8_0, F16 세 가지 모두에서 동일한 크래시가 발생한다.

##### 결함 1: byte_to_token 매핑 누락

```
오류: llama_vocab::byte_to_token → std::unordered_map::at → std::out_of_range
```

SPM 토크나이저는 알 수 없는 문자를 처리할 때 바이트 단위로 분해하는 "바이트 폴백(byte fallback)" 메커니즘을 사용한다. 이를 위해 0x00~0xFF까지 256개의 바이트 토큰(`<0x00>`, `<0x01>`, ..., `<0xFF>`)이 어휘에 포함되어야 한다.

v1의 Vocab 크기가 64,000인 반면 v2가 64,256인 것은 정확히 이 256개 바이트 폴백 토큰이 누락되었음을 의미한다. llama.cpp 엔진이 바이트 폴백을 시도할 때 해당 토큰을 찾지 못해 `std::out_of_range` 예외가 발생하고 프로세스가 SIGABRT로 종료된다.

##### 결함 2: 줄바꿈 토큰(`\n`) 미등록

토크나이저에 줄바꿈 문자가 정의되지 않아, 모델이 첫 번째 줄바꿈을 생성하려는 순간 즉시 크래시가 발생한다. 이는 사실상 한 줄 이상의 응답을 생성할 수 없음을 의미한다.

##### 결함 3: BOS 토큰의 EOG 플래그 미설정

시작 토큰(`<s>`, BOS)에 End-of-Generation(EOG) 플래그가 설정되지 않았다. 정상적인 LLM에서는 EOS 토큰(`</s>`)이 생성 종료를 알리지만, BOS에도 EOG가 설정되어 있어야 특정 엣지 케이스에서 무한 생성 루프를 방지할 수 있다. 이 플래그 누락은 (크래시가 먼저 발생하므로 실제로 관찰되지는 않지만) 잠재적 무한 루프 위험을 내포한다.

##### 결함 4: 채팅 템플릿 부재

Modelfile의 TEMPLATE이 `{{ .Prompt }}`로만 설정되어 있어, 시스템 프롬프트와 사용자 입력 간의 구분이 없는 raw passthrough 상태이다. 이는 대화형 사용을 사실상 불가능하게 만든다.

#### v2에서의 수정 사항

v2는 위 결함들을 모두 수정한 재변환(re-conversion) 버전이다:

1. **바이트 폴백 토큰 추가**: Vocab 64,000 → 64,256 (256개 `<0xNN>` 토큰 완전 수록)
2. **byte_to_token 매핑 정상화**: SPM 바이트 폴백 경로가 정상 동작
3. **줄바꿈 토큰 등록**: `\n` 토큰이 정상적으로 정의됨
4. **EOG 토큰 올바른 설정**: 생성 종료 조건이 정상적으로 작동
5. **모델 구조 변경**: 파라미터 수 3.2B → 1.2B, 임베딩 차원 3,072 → 2,048로 축소하여 재빌드

v2의 파라미터 수 감소는 단순한 버그 수정이 아닌 **모델 아키텍처 자체의 재설계**가 이루어졌음을 의미한다. 더 작은 모델이지만 토크나이저가 정상 동작하므로 실질적인 평가가 가능하다.

### 2.3 FRANKENSTALLM 모델 구조 요약

```
FRANKENSTALLM 3B (v2)
├── 아키텍처: LLaMA (Transformer Decoder-only)
├── 파라미터: 1.2B
├── 임베딩 차원: 2,048
├── 컨텍스트 길이: 4,096 토큰
├── Vocab 크기: 64,256 (64,000 기본 + 256 바이트 폴백)
├── 토크나이저: SentencePiece Model (SPM)
├── 학습 기법: ORPO (Odds Ratio Preference Optimization)
├── 대상 언어: 한국어/영어 이중언어
├── 기능: completion only (도구 호출, 비전 미지원)
└── 생성 종료 토큰: </s>
```

---

## 3. 비교 모델 상세

### 3.1 Qwen 2.5 3B (`qwen2.5:3b`)

| 항목 | 내용 |
|------|------|
| **개발사** | Alibaba Cloud (알리바바 클라우드) |
| **아키텍처** | Qwen 2 (자체 설계 Transformer) |
| **파라미터 수** | 3.1B |
| **Vocab 크기** | 151,936 |
| **컨텍스트 길이** | 32,768 토큰 |
| **임베딩 차원** | 2,048 |
| **양자화** | Q4_K_M (Ollama 기본) |
| **파일 크기** | 1.8 GB |
| **기능** | completion, tools (도구 호출 지원) |
| **시스템 프롬프트** | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." |

#### 특징 및 강점
- **다국어 지원**: 중국어, 영어, 한국어, 일본어 등 29개 이상 언어 지원
- **대규모 Vocab**: 151,936개 토큰으로 다국어 토크나이즈 효율이 높음
- **도구 호출**: Function calling 기능 내장
- **코드 생성**: 코드 관련 벤치마크에서 강력한 성능
- **긴 컨텍스트**: 32K 토큰 컨텍스트로 FRANKENSTALLM(4K)의 8배

#### 비교 기준 선정 이유
동일한 3B 급 모델로 다국어 처리 능력이 뛰어나며, 특히 아시아 언어(중국어, 한국어)에서의 성능이 우수하여 한국어 벤치마크의 강력한 비교 대상이 된다.

---

### 3.2 Gemma 3 4B (`gemma3:4b`)

| 항목 | 내용 |
|------|------|
| **개발사** | Google DeepMind |
| **아키텍처** | Gemma 3 (Google 자체 Transformer) |
| **파라미터 수** | 4.3B |
| **Vocab 크기** | 262,144 |
| **컨텍스트 길이** | 131,072 토큰 (128K) |
| **임베딩 차원** | 2,560 |
| **양자화** | Q4_K_M (Ollama 기본) |
| **파일 크기** | 3.2 GB |
| **기능** | completion, vision (멀티모달) |
| **생성 종료 토큰** | `<end_of_turn>` |

#### 특징 및 강점
- **초대형 Vocab**: 262,144개 토큰으로 이번 비교에서 가장 큰 어휘 크기
- **비전 기능**: 이미지 입력 처리 가능 (멀티모달)
- **초장문 컨텍스트**: 128K 토큰으로 가장 긴 컨텍스트 윈도우
- **Google 생태계**: Gemini 기술 기반, Google의 대규모 다국어 학습 데이터 활용
- **비교군 중 최대 파라미터**: 4.3B로 약간 더 큰 모델

#### 비교 기준 선정 이유
4.3B로 비교군 중 가장 큰 모델이지만, 동급(~4B) 범위 내에 있어 FRANKENSTALLM의 상한 성능 기준점 역할을 한다. Google의 다국어 학습 데이터와 멀티모달 능력은 한국어 이해도 비교에 유용하다.

---

### 3.3 Phi-4 Mini (`phi4-mini`)

| 항목 | 내용 |
|------|------|
| **개발사** | Microsoft Research |
| **아키텍처** | Phi-3 (Microsoft 자체 Transformer) |
| **파라미터 수** | 3.8B |
| **Vocab 크기** | 100,352 |
| **컨텍스트 길이** | 131,072 토큰 (128K) |
| **임베딩 차원** | 3,072 |
| **양자화** | Q4_K_M (Ollama 기본) |
| **파일 크기** | 2.4 GB |
| **기능** | completion, tools (도구 호출 지원) |

#### 특징 및 강점
- **고품질 학습 데이터**: Microsoft의 "교과서 품질(textbook-quality)" 데이터 큐레이션 철학 적용
- **추론 특화**: 수학, 코드, 논리적 추론에서 동급 대비 우수한 성능
- **도구 호출**: Function calling 기능 내장
- **긴 컨텍스트**: 128K 토큰 지원
- **가장 큰 임베딩**: 3,072차원으로 비교군 중 가장 큰 임베딩 벡터 (FRANKENSTALLM v1과 동일)

#### 비교 기준 선정 이유
Microsoft의 Phi 시리즈는 "작지만 강한" 모델의 대표격으로, 특히 코드와 수학 추론에서 뛰어난 성능을 보인다. Track 4(코드/수학) 벤치마크에서의 비교 기준점으로 적합하다. 다만 영어 중심 학습으로 한국어 성능은 상대적으로 약할 수 있다.

---

### 3.4 EXAONE 3.5 2.4B (`exaone3.5:2.4b`)

| 항목 | 내용 |
|------|------|
| **개발사** | LG AI Research (LG AI 연구원) |
| **아키텍처** | EXAONE (LG 자체 Transformer) |
| **파라미터 수** | 2.7B |
| **Vocab 크기** | 102,400 |
| **컨텍스트 길이** | 32,768 토큰 |
| **임베딩 차원** | 2,560 |
| **양자화** | Q4_K_M (Ollama 기본) |
| **파일 크기** | 1.6 GB |
| **기능** | completion |
| **시스템 프롬프트** | "You are EXAONE model from LG AI Research, a helpful assistant." |
| **생성 종료 토큰** | `[|endofturn|]` |
| **라이선스** | EXAONE AI Model License 1.1 - NC (비상업적) |

#### 특징 및 강점
- **한국어 특화**: LG AI Research가 개발한 한국어에 강한 모델
- **한국 기업 개발**: 국내 AI 연구소에서 개발된 유일한 비교 모델
- **비교군 중 최소 파라미터**: 2.7B로 가장 작은 모델
- **효율적 크기**: 1.6 GB로 가장 작은 파일 크기
- **한국어 학습 데이터**: 한국어 코퍼스를 대량 포함한 학습 데이터

#### 비교 기준 선정 이유
**한국어 벤치마크의 핵심 비교 대상**이다. LG AI Research가 한국어를 주요 타겟으로 개발했으므로, FRANKENSTALLM의 한국어 성능을 가장 공정하게 비교할 수 있는 모델이다. 파라미터 수가 2.7B로 FRANKENSTALLM v2(1.2B)보다 크지만, 한국어 특화라는 공통점이 있다.

---

### 3.5 LLaMA 3.2 3B (`llama3.2:3b`)

| 항목 | 내용 |
|------|------|
| **개발사** | Meta AI |
| **아키텍처** | LLaMA (Large Language Model Meta AI) |
| **파라미터 수** | 3.2B |
| **Vocab 크기** | 128,256 |
| **컨텍스트 길이** | 131,072 토큰 (128K) |
| **임베딩 차원** | 3,072 |
| **양자화** | Q4_K_M (Ollama 기본) |
| **파일 크기** | 1.9 GB |
| **기능** | completion, tools (도구 호출 지원) |
| **생성 종료 토큰** | `<\|start_header_id\|>`, `<\|end_header_id\|>`, `<\|eot_id\|>` |

#### 특징 및 강점
- **FRANKENSTALLM과 동일 아키텍처**: LLaMA 기반으로, 아키텍처 효과를 제거한 순수 학습 데이터/파인튜닝 차이 비교 가능
- **도구 호출**: Function calling 기능 내장
- **Meta의 대규모 학습**: 방대한 다국어 코퍼스로 사전 학습
- **커뮤니티 지원**: 가장 넓은 오픈소스 생태계와 파인튜닝 레시피
- **128K 컨텍스트**: 초장문 입력 처리 가능

#### 비교 기준 선정 이유
FRANKENSTALLM이 LLaMA 아키텍처를 기반으로 하므로, **동일 아키텍처의 원본 모델**과의 직접 비교가 가능하다. 이를 통해 FRANKENSTALLM의 커스텀 학습(ORPO, 한국어 데이터)이 원본 LLaMA 대비 어떤 성능 차이를 만들어내는지 정량적으로 평가할 수 있다. 파라미터 수(3.2B)가 FRANKENSTALLM v1과 동일하여 직접 비교에 가장 적합하다.

---

### 3.6 EVAFRILL-Mo 3B SLERP (`evafrill-mo-3b-slerp`)

> **설치 및 구동 방법**: [README.md의 "EVAFRILL-Mo 모델 설정"](README.md#evafrill-mo-모델-설정-pytorch-직접-추론) 섹션에서 소스 코드 클론, 체크포인트 다운로드, 의존성 설치, 실행 방법, 경로 변경, 트러블슈팅을 확인할 수 있다.

| 항목 | 내용 |
|------|------|
| **개발사** | pathcosmos |
| **아키텍처** | Mamba-2 + Transformer 하이브리드 (커스텀) |
| **파라미터 수** | 2.94B |
| **Vocab 크기** | 64,000 |
| **컨텍스트 길이** | 4,096 토큰 |
| **임베딩 차원** | 3,072 |
| **양자화** | BF16 (원본, 양자화 불가) |
| **파일 크기** | 5.9 GB (`model.safetensors`) |
| **기능** | completion |
| **추론 엔진** | PyTorch 직접 추론 (`evafrill_runner.py`) — Ollama/GGUF 미지원 |
| **EOS 토큰** | `</s>` |
| **GitHub** | https://github.com/pathcosmos/EVAFRILL-Mo |

#### 아키텍처 상세

26개 레이어의 하이브리드 구조. Mamba-2 SSM과 Transformer Attention 블록이 혼합 배치된다.

**하이브리드 패턴:**
```
M M M M M M M M M M M M A M M M M M M M M M M M A M
└─── 12× Mamba-2 ────┘ ↑ └──── 12× Mamba-2 ────┘ ↑
                    Attention                  Attention
```
- **M (Mamba-2 블록)**: 24개 — Selective State Space Model, 시퀀스를 재귀적으로 처리
- **A (Attention 블록)**: 2개 — Grouped-Query Attention (24 query heads, 8 KV heads)

**핵심 설정값 (`config.json`):**

| 파라미터 | 값 | 설명 |
|----------|---|------|
| `d_model` | 3,072 | 모델 히든 차원 |
| `n_layers` | 26 | 전체 레이어 수 |
| `n_heads` | 24 | Attention query 헤드 수 |
| `n_kv_heads` | 8 | GQA key-value 헤드 수 (3:1 압축) |
| `d_ffn` | 9,216 | Transformer FFN 차원 (SwiGLU) |
| `rope_theta` | 500,000 | RoPE 기저 주파수 |
| `mamba_d_state` | 128 | SSM state 차원 |
| `mamba_head_dim` | 64 | Mamba 헤드별 차원 |
| `mamba_expand` | 2 | Mamba 내부 확장 비율 (inner_dim = 6,144) |
| `mamba_conv_kernel` | 4 | Causal depth-wise Conv1D 커널 크기 |
| `mamba_n_groups` | 8 | Selective Scan B/C 행렬 그룹 수 |
| `mamba_d_ffn` | 4,608 | Mamba FFN 차원 (Nemotron-H 스타일) |
| `mamba_chunk_size` | 256 | Chunked SSD 청크 크기 |

**모델 컴포넌트:**

| 파일 | 클래스 | 역할 |
|------|--------|------|
| `model/transformer.py` | `LLM` | 최상위 디코더 모델 (임베딩 + 레이어 스택 + LM 헤드) |
| `model/transformer.py` | `TransformerBlock` | Attention + FFN 블록 (Pre-norm, RMSNorm) |
| `model/mamba_block.py` | `Mamba2Block` | Mamba-2 SSM 블록 (Selective Scan + Conv1D + 게이팅) |
| `model/attention.py` | `MultiHeadAttention` | GQA + RoPE + FlashAttention-2 지원 |
| `model/layers.py` | `RMSNorm`, `RotaryEmbedding`, `SwiGLU` | 공유 프리미티브 |
| `model/config.py` | `LMConfig` | 설정 데이터클래스 (YAML/JSON 직렬화) |

#### SLERP 머지 설명

**SLERP(Spherical Linear Interpolation)**은 두 모델의 가중치를 단위 구면(unit sphere) 위에서 보간하는 기법이다. 단순 선형 평균(`(A+B)/2`)보다 학습된 표현(learned representations)의 방향성을 더 잘 보존한다.

**체크포인트 계보:**

```
pretrain (319K steps, 55B tokens, Chinchilla 93% 최적)
    └── sft-v2 (Supervised Fine-Tuning)
            ├── dpo-r1 → dpo-r2 → dpo-r3 (Direct Preference Optimization)
            └── orpo (Odds Ratio Preference Optimization)

최종 모델:
    SLERP = sft-v2 ⊕ dpo-r2  (alpha=0.5)
```

- **Alpha = 0.5**: SFT v2와 DPO R2의 동등 보간
- **선택 이유**: SLERP 변형이 모든 체크포인트 중 가장 낮은 반복률(74.5%)을 기록
- Git LFS로 관리되는 다른 체크포인트(pretrain, dpo-r1/r3, orpo)도 사용 가능하지만, **slerp이 권장 최종 모델**

#### GGUF/Ollama 미지원 기술적 설명

llama.cpp(GGUF 런타임)는 표준 Transformer 아키텍처(Self-Attention + FFN)만 지원한다. EVAFRILL-Mo-3B의 핵심 구성 요소인 Mamba-2가 호환되지 않는 이유:

1. **Selective State Space Model (SSM)**: Mamba-2는 입력에 따라 동적으로 변하는 상태 전이 행렬(A, B, C)을 사용하는 재귀 연산. Attention의 Query-Key-Value 패턴과 근본적으로 다름
2. **KV 캐시 없음**: Transformer는 이전 토큰의 Key/Value를 캐시하여 O(1)로 다음 토큰 생성 가능. Mamba-2는 hidden state만 유지하며, `evafrill_runner.py`의 현재 구현에서는 매 토큰마다 전체 시퀀스를 forward pass → O(n) per token, O(n^2) total
3. **커스텀 CUDA 커널**: `mamba_ssm.ops.triton.ssd_combined`, `causal_conv1d` 등의 전용 커널이 필요. 없으면 PyTorch 순수 구현(`selective_scan()` 함수)으로 fallback

> **양자화 불가**: GGUF의 Q4_K_M, Q8_0 등의 양자화는 Attention 레이어 구조를 전제로 한 기법. Mamba-2의 SSM 파라미터(A_log, D, dt_bias 등)에는 적용할 수 없어 BF16 원본만 사용 가능.

#### PyTorch 추론 파이프라인 (`evafrill_runner.py`)

**로딩 시퀀스:**
```
config.json → LMConfig 생성 → LLM(cfg) 인스턴스화
→ model.safetensors 로딩 (load_safetensors)
→ bfloat16 변환 → GPU/CPU 이동 → eval 모드
→ tokenizer.json → Tokenizer 로딩
```

**생성(generate) 흐름:**
```
프롬프트 → 토크나이즈 → input_ids [1, seq_len]
→ 토큰별 루프 (max 512):
    logits = model(generated_ids)[:, -1, :]   # 마지막 토큰 logits
    → 반복 패널티 적용 (repeat_penalty=1.2)
    → 온도 스케일링 (temperature=0.7)
    → top-k(50) + top-p(0.9) 필터링
    → multinomial 샘플링
    → EOS(</s>) 검출 시 종료
→ 디코딩 → 응답 텍스트
```

**통합 방식**: `runner.py`의 `generate()`, `chat()` 함수가 모델명에 "evafrill"이 포함되면 자동으로 `evafrill_runner.generate()`로 위임. 사용자가 별도 설정할 필요 없음.

#### 필요 파일 및 디렉토리

```
/home/lanco/models/
├── EVAFRILL-Mo/                        # 모델 소스 코드 (git clone)
│   └── model/                          # 커스텀 아키텍처 Python 모듈
│       ├── config.py                   # LMConfig 데이터클래스
│       ├── transformer.py             # LLM, TransformerBlock
│       ├── mamba_block.py             # Mamba2Block, selective_scan
│       ├── attention.py               # MultiHeadAttention, RoPE
│       ├── layers.py                  # RMSNorm, RotaryEmbedding, SwiGLU
│       ├── lora.py                    # LoRA 어댑터
│       └── __init__.py                # 공개 API (LLM, LMConfig, Mamba2Block)
│
└── EVAFRILL-Mo-3B/
    └── slerp/                          # 체크포인트 (권장 최종 모델)
        ├── config.json                 # 687 B — 아키텍처 설정
        ├── model.safetensors           # 5.9 GB — 모델 가중치 (BF16)
        └── tokenizer.json              # 4.2 MB — 토크나이저 (vocab 64K)
```

> **경로 변경**: `evafrill_runner.py`의 `_EVAFRILL_SRC`(line 19)와 `EVAFRILL_CHECKPOINT`(line 31)를 수정

#### 성능 특성

| 항목 | 값 | Ollama Q4_K_M 비교 |
|------|---|---------------------|
| GPU TPS | ~4.8 | 100-200 (20-40배 느림) |
| 타임아웃 | 600초 (10분) | 120초 |
| VRAM 사용 | ~6-8 GB | 1.5-3.5 GB |
| 양자화 | BF16만 가능 | Q4_K_M, Q8_0, F16 |
| 최대 컨텍스트 | 4,096 | 4,096-131,072 |

#### 비교 기준 선정 이유

**하이브리드 아키텍처(SSM + Attention)가 순수 Transformer와 비교하여 3B 규모 한국어 LLM에서 어떤 성능 차이를 보이는지** 검증하기 위한 모델이다. Mamba-2의 이론적 강점(선형 시간 복잡도, 긴 시퀀스 효율)이 실제 한국어 생성 품질로 이어지는지, 그리고 소수의 Attention 블록(2/26)이 충분한 표현력을 제공하는지를 평가한다. 동일 개발자(pathcosmos)가 만든 모델이므로 학습 데이터/방법론의 차이를 최소화한 아키텍처 비교가 가능하다.

---

### 비교 모델 요약 비교

| 모델 | 개발사 | 한국어 지원 수준 | 코드/수학 | 도구 호출 | 비전 | 고유 강점 |
|------|--------|------------------|-----------|-----------|------|-----------|
| Qwen 2.5 3B | Alibaba | 상 (다국어 특화) | 상 | 지원 | 미지원 | 아시아 언어 강세 |
| Gemma 3 4B | Google | 중상 (다국어) | 중상 | 미지원 | 지원 | 최대 Vocab, 멀티모달 |
| Phi-4 Mini | Microsoft | 중 (영어 중심) | 최상 | 지원 | 미지원 | 추론/수학 특화 |
| EXAONE 3.5 2.4B | LG AI | 최상 (한국어 특화) | 중 | 미지원 | 미지원 | 한국어 네이티브 |
| LLaMA 3.2 3B | Meta | 중 (다국어) | 중상 | 지원 | 미지원 | 동일 아키텍처 기반 |
| EVAFRILL-Mo 3B | pathcosmos | 중 (한국어 학습) | 중 | 미지원 | 미지원 | Mamba-2 하이브리드, SSM 아키텍처 |

---

## 4. 양자화 비교

### 4.1 양자화(Quantization)란?

양자화는 모델의 가중치(weights)를 원래의 부동소수점(floating point) 표현에서 더 적은 비트 수의 표현으로 변환하는 기법이다. 이를 통해 모델 크기와 메모리 사용량을 줄이고 추론 속도를 향상시키되, 일정 수준의 정밀도 손실을 감수한다.

### 4.2 테스트 대상 양자화 수준

#### F16 (16비트 부동소수점)

| 항목 | 내용 |
|------|------|
| **비트 수** | 16비트 (IEEE 754 half-precision) |
| **압축률** | 없음 (원본 대비 2배 축소, FP32 기준) |
| **정밀도 손실** | 매우 적음 (사실상 풀 정밀도) |
| **FRANKENSTALLM v2 크기** | 2.3 GB |
| **용도** | 정밀도 기준선(baseline), 품질 최우선 시나리오 |

F16은 FP32 대비 절반의 크기이지만, 모델의 표현력은 거의 완벽하게 보존된다. 학습(training) 시 사용되는 BF16(Brain Floating Point 16)과 유사한 정밀도를 제공하며, 양자화 오류가 거의 없는 "사실상 원본"으로 간주된다.

#### Q8_0 (8비트 양자화)

| 항목 | 내용 |
|------|------|
| **비트 수** | 8비트 (블록 단위 대칭 양자화) |
| **압축률** | F16 대비 ~2배 축소 |
| **정밀도 손실** | 미미함 (~0.1~0.5% 벤치마크 저하) |
| **FRANKENSTALLM v2 크기** | 1.2 GB |
| **용도** | 품질과 효율의 균형점 |

Q8_0은 가중치를 32개 단위 블록으로 나누어, 각 블록의 최댓값으로 스케일링한 후 8비트 정수로 변환하는 방식이다. `_0`은 영점(zero-point)이 0인 대칭 양자화를 의미한다. F16 대비 정밀도 손실이 매우 작아, 대부분의 벤치마크에서 F16과 거의 동일한 결과를 보인다.

#### Q4_K_M (4비트 K-퀀트, 중간 품질)

| 항목 | 내용 |
|------|------|
| **비트 수** | 4비트 (K-Quant, Mixed precision) |
| **압축률** | F16 대비 ~3~4배 축소 |
| **정밀도 손실** | 소폭 (~1~3% 벤치마크 저하) |
| **FRANKENSTALLM v2 크기** | 757 MB |
| **용도** | 배포용 기본 양자화, 속도/크기 최적화 |

Q4_K_M은 llama.cpp에서 개발된 "K-Quant" 양자화 방식의 한 변형이다:

- **`Q4`**: 기본 4비트 양자화
- **`K`**: K-Quant 방식 (비균일 양자화, 레이어별 중요도에 따라 비트 할당 차등 적용)
- **`M`**: Medium 품질 (Small/Medium/Large 중 중간)

K-Quant은 모든 가중치를 동일하게 4비트로 변환하는 대신, 중요한 레이어(attention, output)에는 더 높은 비트를 할당하고 덜 중요한 레이어에는 더 낮은 비트를 할당하는 "혼합 정밀도(mixed precision)" 전략을 사용한다. 이를 통해 단순 Q4_0 대비 같은 평균 비트 수에서 더 높은 품질을 달성한다.

### 4.3 양자화별 크기 비교

#### FRANKENSTALLM v2

| 양자화 | 파일 크기 | F16 대비 비율 | 평균 비트/파라미터 |
|--------|-----------|---------------|---------------------|
| F16 | 2.3 GB | 100% | 16비트 |
| Q8_0 | 1.2 GB | 52% | ~8비트 |
| Q4_K_M | 757 MB | 33% | ~4.5비트 (혼합) |

#### FRANKENSTALLM v1 (참고, 실행 불가)

| 양자화 | 파일 크기 | F16 대비 비율 |
|--------|-----------|---------------|
| F16 | 6.0 GB | 100% |
| Q8_0 | 3.2 GB | 53% |
| Q4_K_M | 1.9 GB | 32% |

### 4.4 양자화가 벤치마크 성능에 미치는 영향

일반적으로 관찰되는 양자화별 벤치마크 성능 저하 패턴:

```
F16 (기준선)    ████████████████████ 100%
Q8_0            ███████████████████▌  ~99.5% (거의 차이 없음)
Q4_K_M          ██████████████████▌   ~97~99% (미세한 저하)
```

#### 예상 영향 분석

| 평가 영역 | Q4_K_M vs F16 예상 차이 | 이유 |
|-----------|-------------------------|------|
| **한국어 이해 (Track 1)** | -0.5~2% | 토큰 임베딩의 미세 정밀도 손실 |
| **개방형 생성 (Track 2, 3)** | -1~3% | 낮은 확률 토큰 선택의 변동 |
| **코드/수학 (Track 4)** | -1~3% | 정확한 수치/구문 생성 민감 |
| **일관성 (Track 5)** | -0~1% | 결정론적 디코딩 시 차이 최소 |
| **처리량 (Track 6)** | +30~50% 향상 | 작은 모델 → 빠른 추론 |

### 4.5 세 가지 양자화를 모두 테스트하는 이유

1. **품질 기준선 확보 (F16)**: 양자화 오류 없는 "진짜" 모델 성능을 측정하여, 양자화에 의한 성능 저하를 정량적으로 분리할 수 있다.

2. **실용적 배포 시나리오 (Q4_K_M)**: 실제 서비스 배포 시 가장 많이 사용되는 양자화 수준으로, 크기와 성능의 최적 균형점을 확인한다.

3. **중간 기준점 (Q8_0)**: F16과 Q4_K_M 사이의 성능 변화가 선형인지 비선형인지 확인하여, 양자화 민감도 곡선을 그릴 수 있다.

4. **비교 모델과의 공정성**: 비교 모델 5종이 모두 Q4_K_M 양자화를 사용하므로, FRANKENSTALLM의 Q4_K_M 결과와 직접 비교할 수 있다. 동시에 F16 결과로 "양자화 없는 실력"을 파악할 수 있다.

---

## 5. CPU 추론 성능 참고

### 5.1 테스트 환경

본 벤치마크는 GPU 장애로 인해 **CPU 전용 모드**로 실행되었다.

| 항목 | 사양 |
|------|------|
| **OS** | Ubuntu Linux 6.8.0-101-generic |
| **추론 엔진** | Ollama (llama.cpp 기반) |
| **실행 모드** | CPU only (GPU 미사용) |
| **타임아웃 설정** | GPU 대비 2배 적용 (config.py `_TIMEOUT_MULTIPLIER`) |

### 5.2 측정된 추론 속도

Track 6(성능 측정)에서 수집된 실측 데이터를 기반으로 한 CPU 추론 성능이다.

#### 처리량 (Tokens Per Second, TPS)

| 모델 | 양자화 | 예상 TPS | 비고 |
|------|--------|----------|------|
| FRANKENSTALLM v2 | Q4_K_M | ~13 TPS | 가장 빠른 FRANKENSTALLM 양자화 |
| FRANKENSTALLM v2 | Q8_0 | ~8~10 TPS | Q4 대비 약 30% 느림 |
| FRANKENSTALLM v2 | F16 | ~4~6 TPS | 가장 느림, 메모리 사용 최대 |
| 비교 모델 평균 | Q4_K_M | ~10~15 TPS | 모델 크기에 따라 상이 |

#### 콜드 로드 시간 (Cold Load Time)

모델이 메모리에 로딩되지 않은 상태에서 첫 번째 요청을 보낼 때까지의 지연 시간이다.

| 양자화 | 콜드 로드 시간 | 설명 |
|--------|---------------|------|
| Q4_K_M | ~2초 | 757 MB 파일 → 빠른 디스크 → 메모리 전송 |
| Q8_0 | ~5~8초 | 1.2 GB 파일 |
| F16 | ~17초 | 2.3 GB 파일 → 대용량 메모리 매핑 |

콜드 로드 시간은 모델 파일 크기에 거의 비례한다. F16의 17초는 CPU 전용 환경에서 상당한 지연으로, 사용자 체감 품질에 영향을 미칠 수 있다. Q4_K_M의 2초는 웹 서비스 배포 시에도 수용 가능한 수준이다.

### 5.3 메모리 사용 패턴

CPU 추론 시 모델 가중치는 시스템 RAM에 로드된다. Ollama는 모델을 mmap(memory-mapped file)으로 로딩하므로, 실제 물리 메모리 사용량은 접근 패턴에 따라 달라진다.

#### 예상 메모리 사용량

| 양자화 | 모델 크기 | 추론 시 최소 RAM | KV 캐시 포함 |
|--------|-----------|------------------|--------------|
| Q4_K_M | 757 MB | ~800 MB | ~1.0 GB |
| Q8_0 | 1.2 GB | ~1.3 GB | ~1.5 GB |
| F16 | 2.3 GB | ~2.5 GB | ~2.8 GB |

- **KV 캐시(Key-Value Cache)**: 컨텍스트 길이 4,096 토큰에 대한 어텐션 캐시가 추가로 할당된다. 임베딩 차원 2,048 기준으로 수백 MB 수준이다.
- **mmap 특성**: 전체 파일이 가상 메모리에 매핑되지만, 실제 물리 메모리는 접근된 페이지만 차지한다. 따라서 시스템 전체 RAM이 부족하지 않는 한 OOM(Out of Memory) 위험은 낮다.
- **모델 전환 비용**: Ollama는 새 모델 로딩 시 이전 모델을 언로드하므로, 모델 전환마다 콜드 로드 비용이 발생한다. 이를 고려하여 평가 프레임워크에서는 모델 간 10초의 쿨다운(`COOLDOWN_BETWEEN_MODELS`)을 설정하고 있다.

### 5.4 CPU 추론의 한계

CPU 전용 추론은 GPU 대비 다음과 같은 한계가 있다:

1. **낮은 처리량**: GPU 추론 대비 약 10~50배 느림 (모델 크기, GPU 사양에 따라 상이)
2. **높은 레이턴시**: TTFT(Time to First Token)가 수 초 이상 소요될 수 있음
3. **동시 요청 제한**: 동시 요청 수 증가 시 성능 급감 (Track 6에서 1/2/4 레벨 테스트)
4. **대형 모델 불리**: F16 같은 큰 모델은 메모리 대역폭 병목으로 TPS가 크게 감소
5. **벤치마크 시간**: 전체 7-트랙 평가에 수 시간 소요 (GPU 대비 수십 분)

이러한 한계에도 불구하고, **상대적 비교**에는 문제가 없다. 모든 모델이 동일한 CPU 환경에서 테스트되므로, 모델 간 성능 순위와 상대적 차이는 GPU 환경에서의 결과와 유사할 것으로 예상된다.

---

## 부록: 모델별 Modelfile 파라미터 비교

### FRANKENSTALLM (v1/v2 공통)

```
TEMPLATE: {{ .Prompt }}
SYSTEM:   당신은 FRANKENSTALLM, 한국어에 특화된 AI 어시스턴트입니다. ...
STOP:     </s>
```

### 비교 모델 파라미터

| 모델 | temperature | top_k | top_p | repeat_penalty | stop 토큰 |
|------|-------------|-------|-------|----------------|-----------|
| FRANKENSTALLM | 0.7 | 50 | 0.9 | 1.2 | `</s>` |
| Qwen 2.5 | (기본) | (기본) | (기본) | (기본) | (기본) |
| Gemma 3 | 1.0 | 64 | 0.95 | (기본) | `<end_of_turn>` |
| Phi-4 Mini | (기본) | (기본) | (기본) | (기본) | (기본) |
| EXAONE 3.5 | 1.0 | (기본) | (기본) | 1.0 | `[|endofturn|]` |
| LLaMA 3.2 | (기본) | (기본) | (기본) | (기본) | `<\|eot_id\|>` 등 |

> **참고**: 벤치마크 실행 시에는 공정한 비교를 위해 모든 모델에 동일한 샘플링 파라미터(BENCHMARK_SAMPLING: temperature=0.0, greedy decoding)를 적용한다.
