# 평가 프레임워크 미완료 트랙 완주 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현재 config에 정의된 13개 모델에 대해 7개 평가 트랙을 모두 완주하고, 최종 리포트를 생성한다.

**Architecture:** 트랙 간 하드 의존성 없음. 자체 채점 트랙(4,5)을 먼저, 저지 모델(gemma3:12b) 필요 트랙(2,3,7)을 나중에 실행. 체크포인트 기반 이어하기로 기존 결과 보존.

**Tech Stack:** Python 3, Ollama API, PyTorch (EVAFRILL), gemma3:12b (LLM-as-Judge), RTX 5070 Ti 16GB

---

## 현황 진단

### 모델 이름 문제
기존 결과에는 **구 모델명** (`frankenstallm-3b-v2-Q4_K_M`, `frankenstallm-3b-v2-f16`)이 사용됨.
현재 config는 **신 모델명** (`frankenstallm-3b-v2:latest`, `frankenstallm-3b-v2:Q8_0`)을 사용.
`frankenstallm-3b-v2-f16` 은 현재 config에 존재하지 않음 (제거됨).

### 트랙별 갭 분석

| Track | 결과 모델 수 | 목표 | 누락 모델 | 체크포인트 | 저지 필요 |
|-------|-------------|------|----------|-----------|----------|
| **1** | 13/13 | 13 | - | ✅ | No |
| **2** | 7 (구명) | 13 | v1×2, qwen, gemma, llama3.1, deepseek, evafrill | ✅ | **Yes** |
| **3** | 10 (구명) | 13 | v1×2, deepseek, evafrill | ✅ | **Partial** |
| **4** | **2/13** | 13 | **11개 모델** | ❌ 없음 | No |
| **5** | 10 (구명) | 13 | v1×2, deepseek, evafrill | ❌ 없음 | No |
| **6** | 13/13 | 13 | - | ✅ | No |
| **7** | 10 (구명) | 13 | v1×2, deepseek, evafrill | ✅ | **Yes** |

### 우선순위 결정 근거
1. **Track 4가 가장 위험**: 13개 중 2개만 완료, 체크포인트 없음 → 전체 재실행
2. **자체 채점 트랙 먼저**: Track 4, 5는 저지 모델 불필요 → VRAM 충돌 없이 실행 가능
3. **저지 트랙은 후순위**: Track 2, 3, 7은 gemma3:12b (7.5GB)와 테스트 모델의 VRAM 교대 필요
4. **Track 7이 가장 비쌈**: 모든 모델 쌍 비교 → 모델 추가 시 비교 수 급증

---

## 실행 순서

```
Phase 1 (자체 채점, 저지 불필요)
  Task 1: Track 4 전체 재실행 (11개 모델 누락)
  Task 2: Track 5 누락 모델 추가 (v1×2, deepseek, evafrill)

Phase 2 (저지 모델 필요)
  Task 3: Track 3 누락 모델 추가 (v1×2, deepseek, evafrill)
  Task 4: Track 2 누락 모델 추가 (6개 모델)
  Task 5: Track 7 누락 모델 추가 (v1×2, deepseek, evafrill)

Phase 3 (마무리)
  Task 6: 최종 리포트 생성
```

---

### Task 1: Track 4 — 코드/수학 평가 전체 실행

**Context:** Track 4는 13개 중 2개(llama3.1:8b, exaone4.0:1.2b)만 완료. 체크포인트 없음. 자체 채점 (Python 실행, SQL 실행, 수학 정답 비교)이라 저지 모델 불필요.

**Files:**
- Read: `eval_framework/tracks/track4_code_math.py`
- Read: `eval_framework/config.py` (ALL_MODELS 확인)
- Output: `results/track4_code_math_*.json`, `results/track4_code_math_checkpoint.json`

**예상 소요:** 모델당 ~5-10분, 13개 모델 → 약 1.5~2시간

- [ ] **Step 1: Ollama 서버 상태 확인**

```bash
curl -s http://localhost:11434/api/ps | python3 -m json.tool
```
Expected: Ollama 서버 응답. 모델 로드 상태 확인.

- [ ] **Step 2: 기존 Track 4 결과 백업**

```bash
cp results/track4_code_math_20260312_193725.json results/archive_cpu_20260312/track4_code_math_20260312_193725_backup.json 2>/dev/null || true
cp results/track4_20260312_193725.json results/archive_cpu_20260312/track4_20260312_193725_backup.json 2>/dev/null || true
```
Expected: 기존 2-모델 결과 백업됨.

- [ ] **Step 3: Track 4 실행**

```bash
python run_evaluation.py --tracks 4
```
Expected: 13개 모델 순차 실행. 각 모델마다 Python/SQL/Debug/Math 4개 카테고리 평가. 체크포인트 자동 저장.

- [ ] **Step 4: 결과 검증**

```python
import json
with open('results/track4_code_math_checkpoint.json') as f:
    data = json.load(f)
print(f"Models: {len(data.get('summary', data.get('completed_keys', [])))} completed")
```
Expected: 13개 모델 전체 완료.

- [ ] **Step 5: 커밋**

```bash
git add results/track4_*.json
git commit -m "eval: complete Track 4 (code/math) for all 13 models on GPU"
```

---

### Task 2: Track 5 — 일관성/강건성 누락 모델 추가

**Context:** 10개 모델 완료 (구 모델명). 체크포인트 없음 → 전체 재실행 필요. 자체 채점 (edit distance, Jaccard, regex). 저지 불필요.

**Files:**
- Read: `eval_framework/tracks/track5_consistency.py`
- Output: `results/track5_consistency_*.json`, `results/track5_consistency_checkpoint.json`

**예상 소요:** 모델당 ~10-15분 (반복 테스트 140+ API 콜), 13개 모델 → 약 2~3시간

- [ ] **Step 1: 기존 결과 백업**

```bash
cp results/track5_consistency_20260312_194920.json results/archive_cpu_20260312/track5_consistency_20260312_194920_backup.json 2>/dev/null || true
cp results/track5_20260312_194920.json results/archive_cpu_20260312/track5_20260312_194920_backup.json 2>/dev/null || true
```

- [ ] **Step 2: Track 5 전체 실행**

```bash
python run_evaluation.py --tracks 5
```
Expected: 13개 모델 순차 실행. Repetition (5회 반복), Paraphrase, Length, Language, Instruction, Hallucination 6개 서브테스트. 체크포인트 자동 저장.

- [ ] **Step 3: 결과 검증**

```bash
python3 -c "
import json
with open('results/track5_consistency_checkpoint.json') as f:
    data = json.load(f)
models = set()
for k in data.get('completed_keys', []):
    if ':' in k:
        models.add(k.split(':',1)[1])
print(f'Completed models: {len(models)}')
print(sorted(models))
"
```
Expected: 13개 모델 전체 완료. 현재 config 모델명으로 저장됨.

- [ ] **Step 4: 커밋**

```bash
git add results/track5_*.json
git commit -m "eval: complete Track 5 (consistency) for all 13 models on GPU"
```

---

### Task 3: Track 3 — 한국어 심화 누락 모델 추가

**Context:** 10개 모델 완료 (구 모델명). 체크포인트 존재 → **이어하기 가능하나 모델명 불일치로 신규 모델만 추가됨**. 하이브리드 채점: 일부 exact match + 일부 LLM judge (gemma3:12b). 누락: v1×2, deepseek-r1:1.5b, evafrill-mo-3b-slerp.

**Files:**
- Read: `eval_framework/tracks/track3_korean_deep.py`
- Read: `results/track3_korean_deep_checkpoint.json` (이어하기용)
- Output: `results/track3_korean_deep_*.json`

**⚠️ VRAM 주의:** gemma3:12b (~7.5GB) + 테스트 모델 (~2-4GB) 교대 로드. RTX 5070 Ti 16GB에서 충분하지만, 동시 로드 시 OOM 가능.

**예상 소요:** 4개 모델 × ~15분 → 약 1시간

- [ ] **Step 1: Track 3 누락 모델만 실행**

```bash
python run_evaluation.py --tracks 3 --models frankenstallm-3b:latest frankenstallm-3b:Q8_0 deepseek-r1:1.5b evafrill-mo-3b-slerp
```
Expected: 체크포인트에 없는 4개 모델만 실행. 기존 10개 모델 결과는 보존됨 (체크포인트 기반 skip).

- [ ] **Step 2: 결과 검증 — 전체 모델 커버리지 확인**

```bash
python3 -c "
import json, glob
latest = sorted(glob.glob('results/track3_korean_deep_2*.json'))[-1]
with open(latest) as f:
    data = json.load(f)
print(f'Total models: {len(data[\"summary\"])}')
for m in sorted(data['summary'].keys()):
    print(f'  {m}')
"
```
Expected: 14개 모델 (기존 10 구명 + 신규 4). 구명/신명 혼재는 리포트 단계에서 통합 처리.

- [ ] **Step 3: 커밋**

```bash
git add results/track3_*.json
git commit -m "eval: add 4 missing models to Track 3 (korean deep) on GPU"
```

---

### Task 4: Track 2 — Ko-Bench 누락 모델 추가

**Context:** 7개 모델 완료 (구 모델명). 체크포인트 존재 → 이어하기 가능. Claude API 대신 gemma3:12b를 LLM-as-Judge로 사용. 8개 카테고리 × 10문항 × 2턴 = 160 API 콜/모델. 누락이 가장 많은 저지 트랙.

**Files:**
- Read: `eval_framework/tracks/track2_ko_bench.py`
- Output: `results/track2_ko_bench_*.json`

**⚠️ Judge 의존:** 모든 응답에 gemma3:12b 채점 필요 → 모델 당 시간 2배.

**예상 소요:** 6개 모델 × ~20-30분 → 약 2-3시간

- [ ] **Step 1: 누락 모델만 실행**

```bash
python run_evaluation.py --tracks 2 --models frankenstallm-3b:latest frankenstallm-3b:Q8_0 qwen2.5:3b gemma3:4b llama3.1:8b-instruct-q8_0 deepseek-r1:1.5b evafrill-mo-3b-slerp
```
Expected: 7개 누락 모델 실행 (체크포인트 기반 skip으로 기존 7개 보존).

**Note:** `evafrill-mo-3b-slerp`은 Ollama 아닌 PyTorch 직접 추론. runner가 자동으로 분기 처리함.

- [ ] **Step 2: 결과 검증**

```bash
python3 -c "
import json, glob
latest = sorted(glob.glob('results/track2_ko_bench_2*.json'))[-1]
with open(latest) as f:
    data = json.load(f)
print(f'Total models: {len(data[\"summary\"])}')
"
```
Expected: 14개 모델 (기존 7 구명 + 신규 7).

- [ ] **Step 3: 커밋**

```bash
git add results/track2_*.json
git commit -m "eval: add 7 missing models to Track 2 (ko-bench) on GPU"
```

---

### Task 5: Track 7 — 쌍대비교 Elo 누락 모델 추가

**Context:** 10개 모델 완료 (구 모델명). 체크포인트 존재 → 이어하기 가능. **가장 비싼 트랙**: 모델 추가 시 기존 모든 모델과의 쌍비교 필요. 20개 프롬프트 × 양방향(position bias 제거) = 40 비교/쌍. 4개 모델 추가 시 4×10×40 = 1,600 추가 비교. 전부 gemma3:12b 저지 필요.

**Files:**
- Read: `eval_framework/tracks/track7_pairwise.py`
- Output: `results/track7_pairwise_*.json`

**⚠️ 가장 오래 걸리는 트랙.** 예상 3-5시간.

- [ ] **Step 1: 누락 모델만 실행**

```bash
python run_evaluation.py --tracks 7 --models frankenstallm-3b:latest frankenstallm-3b:Q8_0 deepseek-r1:1.5b evafrill-mo-3b-slerp
```
Expected: 4개 신규 모델과 기존 10개 모델 간의 쌍대비교 수행. Bradley-Terry Elo 재계산.

- [ ] **Step 2: 결과 검증 — Elo 스코어 확인**

```bash
python3 -c "
import json, glob
latest = sorted(glob.glob('results/track7_pairwise_2*.json'))[-1]
with open(latest) as f:
    data = json.load(f)
elo = data.get('results', {}).get('elo_scores', {})
for m, s in sorted(elo.items(), key=lambda x: -x[1].get('elo', 0)):
    print(f'{m:40s} ELO={s.get(\"elo\", \"?\"):>8}')
"
```
Expected: 14개 모델 전체 Elo 랭킹.

- [ ] **Step 3: 커밋**

```bash
git add results/track7_*.json
git commit -m "eval: add 4 missing models to Track 7 (pairwise elo) on GPU"
```

---

### Task 6: 최종 리포트 생성

**Context:** 7개 트랙 전체 결과를 통합하여 HTML + Markdown 리포트 + 스코어카드 생성.

**⚠️ 모델명 통합 이슈:** 구명(`frankenstallm-3b-v2-Q4_K_M`) ↔ 신명(`frankenstallm-3b-v2:latest`)이 혼재. 리포트 생성 전에 이름 매핑이 필요할 수 있음.

**Files:**
- Read: `eval_framework/report.py`
- Read: `eval_framework/scoring.py`
- Output: `reports/report_*.html`, `reports/report_*.md`, `results/scorecard.json`, `results/full_results_*.json`

- [ ] **Step 1: 모델명 일관성 확인**

```bash
python3 -c "
import json, glob
all_models = set()
for f in glob.glob('results/track*_2*.json'):
    if 'checkpoint' in f or 'archive' in f: continue
    with open(f) as fh:
        data = json.load(fh)
    models = data.get('summary', {}).keys()
    all_models.update(models)
print('All model names across tracks:')
for m in sorted(all_models):
    print(f'  {m}')
"
```
Expected: 구명+신명 혼재 확인. 리포트 생성이 이를 처리할 수 있는지 판단.

- [ ] **Step 2: 리포트 생성**

```bash
python run_evaluation.py --report-only
```
Expected: `reports/` 디렉토리에 HTML/Markdown 리포트 생성. `results/scorecard.json` 업데이트.

- [ ] **Step 3: 리포트 내용 검증**

리포트에서 7개 트랙 전체가 포함되었는지, 모델별 점수가 합리적인지 확인.

- [ ] **Step 4: 커밋**

```bash
git add reports/ results/scorecard.json results/full_results_*.json
git commit -m "report: generate final report with all 7 tracks, 13 models"
```

---

## 시간 예상 총표

| Task | Track | 모델 수 | 저지 | 예상 시간 |
|------|-------|---------|------|----------|
| 1 | Track 4 (Code/Math) | 13 (전체) | No | 1.5~2h |
| 2 | Track 5 (Consistency) | 13 (전체) | No | 2~3h |
| 3 | Track 3 (Korean Deep) | 4 (추가) | Partial | ~1h |
| 4 | Track 2 (Ko-Bench) | 7 (추가) | **Yes** | 2~3h |
| 5 | Track 7 (Pairwise) | 4 (추가) | **Yes** | 3~5h |
| 6 | Report | - | - | ~5min |
| **총합** | | | | **~10-14h** |

## 주의사항

1. **EVAFRILL 모델** (`evafrill-mo-3b-slerp`)은 Ollama가 아닌 PyTorch 직접 추론. `runner.py`의 `is_evafrill()` 분기로 자동 처리되지만, GPU 메모리 해제가 Ollama보다 느릴 수 있음.
2. **모델 전환 쿨다운**: `COOLDOWN_BETWEEN_MODELS = 10`초. 빈번한 전환 시 누적됨.
3. **체크포인트 재개**: Track 4, 5는 체크포인트 없으므로 처음부터 실행. 중간에 중단되면 체크포인트에서 이어하기 가능.
4. **구명/신명 혼재**: 최종 리포트에서 매핑 로직 필요 여부는 `report.py` 분석 후 판단.
