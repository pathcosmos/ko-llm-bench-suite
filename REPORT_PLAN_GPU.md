# FRANKENSTALLM 3B 최종 평가 리포트 작성 계획

> 작성: 2026-03-12
> 상태: Track 7 완료 대기 중, 리포트 구조 사전 설계

---

## 1. 리포트 목적

- FRANKENSTALLM 3B v2 (1.2B 한국어 ORPO 모델)의 **객관적 성능 위치** 파악
- 5개 비교 모델 대비 **강점/약점** 명확히 도출
- 양자화 수준별 (Q4_K_M / Q8_0 / F16) **품질-속도 트레이드오프** 분석
- 다른 환경에서 **재현 가능한** 평가 기록 남기기

---

## 2. 데이터 소스 현황

| 트랙 | 결과 파일 | 모델 수 | 비고 |
|------|-----------|---------|------|
| Track 1 (Korean Bench) | `track1_korean_bench_20260311_024226.json` | 8 | 완료 |
| Track 2 (Ko-Bench) | `track2_ko_bench_20260312_095013.json` | **6** | phi4-mini, exaone3.5 누락 주의! |
| Track 3 (Korean Deep) | 완료 대기 | 8 | — |
| Track 4 (Code & Math) | `track4_code_math_20260311_055248.json` | 8 | 완료 |
| Track 5 (Consistency) | `track5_consistency_20260311_134844.json` | 8 | 완료 |
| Track 6 (Performance) | `track6_performance_20260310_155213.json` | **11** | v1 모델 포함 (참고용) |
| Track 7 (Pairwise) | 완료 대기 | 8 | — |
| 기존 벤치마크 | `benchmark_20260310_131110.json` | — | 초기 단독 벤치마크 |

### 주의사항
- **Track 2**: phi4-mini, exaone3.5:2.4b 결과 누락 → 리포트에 명시 필요
- **Track 6**: v1 모델(3개) 포함되어 있으나 CRASH 모델이므로 별도 표기
- **Track 6**: GPU 모드 결과 (v2-Q4_K_M만 GPU, 나머지 CPU) → 비교 시 주의

---

## 3. 리포트 구조 (목차)

### 3.1 Executive Summary (경영진 요약)
- 한 페이지 핵심 결론
- FRANKENSTALLM v2의 종합 순위
- 가장 강한/약한 영역 한 줄 요약
- 권고사항

### 3.2 평가 환경

#### 하드웨어 상세
| 항목 | 상세 |
|------|------|
| **CPU** | Intel Core i5-13500 (13세대 Raptor Lake) |
| | 14코어 (6P+8E) / 20스레드, 최대 4.8GHz |
| | L1d 544KiB, L1i 704KiB, L2 11.5MiB, L3 24MiB |
| **GPU** | NVIDIA GeForce RTX 5070 Ti |
| | VRAM 16GB GDDR7, Compute Capability 12.0 |
| | Max Clock: Core 3210MHz / Memory 14001MHz |
| | TDP 300W, Driver 590.48.01, CUDA 13.1 |
| **RAM** | 64GB DDR5-5600 (2×32GB) |
| | Micron CP32G56C46U5.M16D1, ECC 없음 |
| **메인보드** | MSI MAG B760M MORTAR MAX WIFI (v3.0) |
| **OS 스토리지** | Samsung MZVL21T0HCLR 1TB NVMe (/) |
| **데이터 스토리지** | SK hynix SHPP41 2TB NVMe (/home) |
| | WD Ultrastar 14TB HDD, HGST 8TB HDD 등 추가 디스크 |
| **OS** | Ubuntu 24.04.3 LTS, Kernel 6.8.0-101-generic |
| **추론 엔진** | Ollama v0.17.4 (snap), llama.cpp 기반 |
| **Python** | 3.12 |
| **Judge LLM** | Claude (via `claude -p` CLI v2.1.74) |

#### 평가 모드
- **전 트랙 GPU 추론** (RTX 5070 Ti 16GB)
- 이전 평가 대비: CPU-only → GPU 전면 전환
- swappiness: 10 (기본 60에서 조정)
- 샘플링 파라미터: temperature=0.7, repeat_penalty=1.2, top_p=0.9
- 트랙별 실행 시간: (eval_stage*.log에서 추출, 아래 표 참조)

#### 트랙별 실행 시간 기록
| 트랙 | Stage | 모델 수 | 실행 시간 | 시작 | 비고 |
|------|-------|---------|-----------|------|------|
| Track 6 (성능) | Stage 1 | 8+2 | 채움 | 채움 | GPU 프로파일링 |
| Track 1 (Korean Bench) | Stage 1 | 8+2 | 채움 | 채움 | 자동 채점 |
| Track 4 (Code & Math) | Stage 1 | 8+2 | 채움 | 채움 | 자동 채점 |
| Track 5 (Consistency) | Stage 1 | 8+2 | 채움 | 채움 | 자동 채점 |
| Track 2 (Ko-Bench) | Stage 2 | 10 | 채움 | 채움 | Claude Judge |
| Track 3 (Korean Deep) | Stage 2 | 10 | 채움 | 채움 | Claude Judge |
| Track 7 (Pairwise Elo) | Stage 3 | 10 (45쌍) | 채움 | 채움 | Claude Judge |
| **총 실행 시간** | — | — | 채움 | — | GPU 모드 전체 |

> 참고: 이전 CPU-only 평가 총 소요 ~40시간 → GPU 모드 대비 기록 필수

### 3.3 모델 프로필
- 8개 모델 스펙 비교 테이블 (파라미터 수, 양자화, 아키텍처, vocab 크기)
- FRANKENSTALLM v2 특성: 1.2B params, LLaMA 아키텍처, SPM 토크나이저, ORPO 학습
- 비교 모델 선정 근거: 3B급 소형 모델, 한국어 지원

### 3.4 Track별 상세 결과

#### 3.4.1 Track 1: 한국어 표준 벤치마크
- **데이터**: KMMLU + KoBEST (BoolQ, COPA, SentiNeg, HellaSwag)
- **차트**: 모델별 벤치마크 정확도 히트맵
- **차트**: 모델별 종합 정확도 바 차트
- **분석**: FRANKENSTALLM이 랜덤 수준(~25%)인 원인 분석
  - 지시 이해력 부족 vs 지식 부족 구분
  - 양자화별 차이 (Q4_K_M: 25.7% vs Q8_0/F16: 22.0%)
- **핵심 발견**: exaone3.5가 97.8%로 압도적 1위

#### 3.4.2 Track 2: Ko-Bench 멀티턴 평가
- **데이터**: 8카테고리 × 10문항 × 2턴, Claude Judge 10점 만점
- **차트**: 카테고리별 모델 점수 레이더 차트
- **차트**: Turn 1 vs Turn 2 점수 비교 (멀티턴 유지력)
- **분석**: FRANKENSTALLM T1≈1.0, T2≈1.0 → 거의 무의미한 응답
- **주의**: phi4-mini, exaone3.5 결과 누락 → 6개 모델만 비교
- **핵심 발견**: gemma3:4b, qwen2.5:3b 강세

#### 3.4.3 Track 3: 한국어 심화 이해력
- **데이터**: 100문항 × 8모델, Claude Judge 채점
- **차트**: 모델별 점수 분포 박스플롯
- **분석**: 문항 유형별 (문법, 어휘, 문화, 추론 등) 세분화 가능 시 추가
- **핵심 발견**: [Track 3 완료 후 채움]

#### 3.4.4 Track 4: 코드 & 수학
- **데이터**: Python Pass@1, SQL, 디버깅, 수학 정확도
- **차트**: 영역별 모델 점수 그룹 바 차트
- **분석**: FRANKENSTALLM 전 영역 0% → 코드/수학 생성 능력 부재
  - ORPO 한국어 fine-tuning이 코드 능력을 파괴했는지?
  - 아니면 1.2B 파라미터 한계인지?
- **핵심 발견**: gemma3:4b 종합 85.8%로 1위

#### 3.4.5 Track 5: 일관성 & 강건성
- **데이터**: 6차원 (반복, 패러프레이즈, 길이, 언어, 지시, 환각)
- **차트**: 6차원 레이더 차트 (모델별 오버레이)
- **분석**:
  - 지시 준수 0% → 구조적 출력 불가 (JSON, 표, 번호 리스트)
  - 환각 탐지 10% → 존재하지 않는 정보도 생성
  - 길이 민감도 80% → 상대적으로 양호한 영역
- **핵심 발견**: qwen2.5:3b 종합 0.748로 가장 안정적

#### 3.4.6 Track 6: 성능 프로파일링
- **데이터**: Prefill tok/s, Decode tok/s, TTFT, VRAM, 동시성
- **차트**: 모델별 decode 속도 바 차트 (GPU vs CPU)
- **차트**: 동시성 레벨별 처리량 라인 차트
- **분석**:
  - v2-Q4_K_M GPU: 217 tok/s decode, TTFT 0.11s → 실용적 속도
  - v1 CPU: 9~15 tok/s → 매우 느림
  - 양자화별 속도 차이
- **핵심 발견**: GPU 사용 시 FRANKENSTALLM v2도 실용적 속도 확보

#### 3.4.7 Track 7: 쌍대비교 Elo 랭킹
- **데이터**: 28쌍 × 20프롬프트 = 560회 비교, Elo 산출
- **차트**: Elo 랭킹 바 차트
- **차트**: 승패 매트릭스 히트맵
- **분석**: [Track 7 완료 후 채움]
- **핵심 발견**: [완료 후]

### 3.5 종합 분석

#### 3.5.1 종합 랭킹
- 7개 트랙 점수를 정규화(0~1)하여 종합 점수 산출
- 가중치 방안:
  - 균등 가중 (각 14.3%)
  - 한국어 중심 가중 (Track 1,2,3에 가중)
  - 실용성 중심 가중 (Track 2,4,6에 가중)

#### 3.5.2 FRANKENSTALLM v2 심층 분석
- **강점**: 길이 민감도 양호(0.8), GPU 속도 실용적, 파일 크기 작음(757MB)
- **약점**: 벤치마크 정확도 ~25%, 코드/수학 0%, 지시 준수 0%, 환각 90%
- **근본 원인 분석**:
  - 1.2B 파라미터 한계 vs fine-tuning 품질 문제
  - SPM 토크나이저 + LLaMA 아키텍처 조합의 영향
  - ORPO 학습 데이터/하이퍼파라미터 문제 가능성
- **v1 vs v2 비교**: v1은 아예 실행 불가(SPM 결함), v2는 실행 가능하나 품질 미달

#### 3.5.3 양자화 영향 분석
- Q4_K_M vs Q8_0 vs F16 정확도 차이 (Track 1, 4 기준)
- 속도-품질 트레이드오프 (Track 6 기준)
- 결론: FRANKENSTALLM v2는 양자화 수준에 따른 품질 차이 미미 (베이스 성능이 낮아서)

#### 3.5.4 비교 모델 랭킹
- 종합 1위~5위 + 각 모델 특성 한 줄 요약
- 한국어 최강: exaone3.5:2.4b (Track 1: 97.8%)
- 코드 최강: gemma3:4b (Track 4: 85.8%)
- 종합 안정성: qwen2.5:3b

### 3.6 권고사항
- FRANKENSTALLM v3 개선 방향:
  1. 파라미터 수 확대 (3B 실제 달성)
  2. Instruction-following 학습 데이터 보강
  3. 코드/수학 데이터 혼합 학습
  4. 환각 억제 RLHF/DPO 적용
  5. BPE 토크나이저 전환 검토
- 현재 v2 활용 가능 시나리오: 간단한 한국어 텍스트 생성 (품질 무관)

### 3.7 부록
- 전체 원본 점수 테이블
- 평가 프롬프트 샘플
- 환경 재현 가이드 (README.md 참조)
- v1 모델 크래시 로그

---

## 4. 시각화 목록 (차트)

| # | 차트 | 유형 | 데이터 소스 |
|---|------|------|-------------|
| 1 | 종합 랭킹 바 차트 | 수평 바 | 전 트랙 정규화 점수 |
| 2 | Track 1 벤치마크 히트맵 | 히트맵 | Track 1 summary |
| 3 | Track 2 카테고리 레이더 | 레이더 | Track 2 summary |
| 4 | Track 2 Turn1 vs Turn2 | 그룹 바 | Track 2 summary |
| 5 | Track 3 점수 분포 | 박스플롯 | Track 3 results |
| 6 | Track 4 영역별 바 차트 | 그룹 바 | Track 4 summary |
| 7 | Track 5 6차원 레이더 | 레이더 | Track 5 summary |
| 8 | Track 6 속도 비교 | 바 차트 | Track 6 summary |
| 9 | Track 6 동시성 처리량 | 라인 | Track 6 summary |
| 10 | Track 7 Elo 랭킹 | 수평 바 | Track 7 summary |
| 11 | Track 7 승패 매트릭스 | 히트맵 | Track 7 results |
| 12 | 양자화 영향 (정확도) | 라인 | Track 1, 4 |
| 13 | 양자화 영향 (속도) | 바 | Track 6 |
| 14 | FRANKENSTALLM 강약점 | 레이더 | 전 트랙 종합 |

---

## 5. 출력 형식

### 5.1 메인 리포트
- **파일**: `reports/FINAL_REPORT.md` (Markdown, GitHub 렌더링 가능)
- 모든 차트는 인라인 이미지로 포함
- 한국어 작성

### 5.2 차트 이미지
- **디렉토리**: `reports/charts/`
- matplotlib로 생성, PNG 300dpi
- 한국어 폰트: NanumGothic

### 5.3 데이터 요약
- **파일**: `reports/summary_data.json` — 리포트에 사용된 모든 수치 정리

---

## 6. 작업 분배 (병렬 실행 가능)

### Task A: 데이터 수집 & 정규화
- 7개 트랙 결과 JSON에서 모델별 점수 추출
- 0~1 정규화 (Track별 min-max 또는 비율 기반)
- `reports/summary_data.json` 출력

### Task B: 차트 생성 스크립트
- `reports/generate_charts.py` 작성
- 14개 차트 자동 생성
- matplotlib + NanumGothic 한국어 폰트

### Task C: 리포트 본문 작성
- 3.1~3.7 섹션 Markdown 작성
- 차트 이미지 참조 삽입
- 분석 텍스트 작성

### Task D: 검증 & 마무리
- 수치 교차 검증 (리포트 ↔ 원본 JSON)
- Track 2 누락 모델 주석 확인
- GitHub push

---

## 7. 미결 사항 (Track 7 완료 후 결정)

- [ ] Track 3 최종 결과 반영
- [ ] Track 7 Elo 결과 반영
- [ ] Track 2 누락 모델(phi4-mini, exaone3.5) 재실행 여부 결정
- [ ] 종합 점수 가중치 방식 확정
- [ ] v1 vs v2 비교를 Track 6 데이터로 별도 섹션 추가할지 결정
