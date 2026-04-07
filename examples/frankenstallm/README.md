# Frankenstallm Evaluation (Legacy)

16모델 × 7트랙 평가 결과 (2026-03~04).
ko-llm-bench-suite의 사용 예시로 보존됨.

## 구조

- `results/` — 112셀 전체 결과 JSON (16모델 × 7트랙)
- `reports/` — HTML/MD 리포트 + 26개 시각화 차트
- `reports/EVALUATION_REPORT_FULL.md` — 최종 종합 보고서
- `reports/visualizations/` — 26개 PNG 시각화 차트

## 주요 결과

- **종합 1위**: Gemma3-4B (69.8점)
- **Elo 1위**: EXAONE3.5-2.4B (1,571)
- **효율성 1위**: EXAONE4-1.2B (1,621 Elo/GB)
- **T1↔T7 상관**: r=0.90

## 모델 목록 (16개)

### 자체 모델 (8)
frankenstallm-3b v1/v2 (6종 양자화) + evafrill-mo-3b-slerp

### 베이스라인 (8)
qwen2.5:3b, gemma3:4b, phi4-mini, exaone3.5:2.4b, llama3.2:3b, llama3.1:8b, exaone4.0:1.2b, deepseek-r1:1.5b
