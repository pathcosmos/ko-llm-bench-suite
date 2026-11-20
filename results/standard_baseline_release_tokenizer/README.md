# 기준선: frankenstallm을 HF 배포 tokenizer.json(byte-fallback, 학습 후 변경)으로 평가한 결과

- 실행: 2026-11-18, runs/standard_v1/own_vllm_ko_mc.retry1.log (kobench.standard.run, 당시 vLLM 기본 토크나이저 = 배포본)
- 학습 토크나이저(korean_sp)와 달리 개행·OOV 문자를 학습되지 않은 byte 토큰으로 바꾼다 (runs/format_conformance_v1/REPORT.md)
- 용도: Goal 2에서 "기존 방식" 대비 최적 설정 효과를 보여주는 기준선. 평가 표의 정식 결과로 쓰지 않는다.
