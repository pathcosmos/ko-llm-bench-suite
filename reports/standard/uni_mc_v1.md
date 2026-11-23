# 표준 벤치 결과 — uni_mc_v1 (A 표)

A 표: 공통 프로토콜(모든 모델 같은 프롬프트·shot·채점, chat 템플릿 미적용).
단위 %. 셀 = lm-eval 점수 / micro [95% CI] / Δjansori-bf16 [paired 95% CI].

| 태스크 | shot | 지표 | 출처 |
|---|---:|---|---|
| kormedmcqa_cloze | 0 | acc_norm | KorMedMCQA test 4과목(sean0042 79efd6f, 비상업), 선택지 문장 로그우도 |
| kobbq_cloze | 0 | acc_norm | KoBBQ evaluation-test (naver-ai ccb902a) |
| ko_truthful_cloze | 0 | acc | ko_truthful_qa MC1 (davidkim205 221bb55), TruthfulQA MC1 관례(acc) |
| k2_knowledge_cloze | 0 | acc_norm | K2-Eval knowledge (HAERAE-HUB 14bbbc9) |
