# 생성 벤치 gen_v1 — thinkrd 표

값: % (judge 점수는 원 척도), [95% CI]. 기준 모델 frankenstallm-bf16 대비 paired 차이는 CSV. † = judge와 같은 계열(자기 선호 가능).

## instruct

| 모델 | 총/활성 B | 설정 | hrm8k | ifeval_ko | kormedmcqa | kobbq | ko_truthful_mc1 | k2_knowledge | komt | logickor | k2_generation | ko_arena_hard | kosimpleqa | humaneval_plus | mbpp_plus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | 4.0/4.0 | thinkrd/raw | 79.9 [79.0, 80.7] | 79.2 [74.9, 83.6] | 80.1 [78.5, 81.5] | 95.3 [94.3, 96.1] | 69.0 [65.9, 72.1] | 84.0 [80.6, 87.3] | 7.05 [6.45, 7.63] † | 7.58 [6.81, 8.30] † | 4.50 [4.02, 4.99] † | 92.6 [90.4, 94.6] † | 7.5 [5.9, 9.2] † | 81.0 [74.2, 87.1] | 60.1 [55.0, 64.8] |

## 반복 지표 (루프율 % / 중복 3-gram % / 최대 길이 잘림 %)

| 모델 | 설정 | hrm8k | ifeval_ko | kormedmcqa | kobbq | ko_truthful_mc1 | k2_knowledge | komt | logickor | k2_generation | ko_arena_hard | kosimpleqa | humaneval_plus | mbpp_plus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | thinkrd | 0 / 6 / 10 | 0 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 3 / 0 | 0 / 3 / 0 | 0 / 1 / 0 | 0 / 3 / 0 | 0 / 1 / 4 | 1 / 3 / 0 | 0 / 5 / 0 |
