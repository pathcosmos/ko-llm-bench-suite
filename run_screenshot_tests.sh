#!/usr/bin/env bash
# ============================================================================
# FRANKENSTALLM 한국어 LLM 비교 테스트 — 스크린샷용 응답 수집
# 6개 모델 × 7개 테스트 = 42개 응답 자동 생성
# ============================================================================
set -euo pipefail

# ── 설정 ────────────────────────────────────────────────────────────────────
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OUTPUT_DIR="screenshot_outputs"
COOLDOWN=2          # 모델 간 쿨다운 (초)
NUM_PREDICT=512     # 최대 토큰 수
TEMPERATURE=0.7

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="SUMMARY_${TIMESTAMP}.md"

# ── 모델 목록 (6개) ────────────────────────────────────────────────────────
MODELS=(
    "frankenstallm-3b-v2-Q4_K_M"
    "llama3.2:3b"
    "phi4-mini"
    "qwen2.5:3b"
    "exaone3.5:2.4b"
    "gemma3:4b"
)

# ── 테스트 케이스 (7개) ────────────────────────────────────────────────────
# 배열: TEST_IDS, TEST_NAMES, TEST_PROMPTS
TEST_IDS=(T1 T2 T3 T4 T5 T6 T7)

TEST_NAMES=(
    "존댓말 전환 (경어법)"
    "속담 풀이 (사자성어/관용구)"
    "한국 문화 상식"
    "감정/뉘앙스 이해"
    "맞춤법 교정"
    "짧은 창작 (감성 글쓰기)"
    "방언 이해 (전라도 사투리)"
)

TEST_PROMPTS=(
    '"야, 이거 어디서 샀어? 진짜 좋다!" 이 문장을 직장 상사에게 쓰는 존댓말로 바꿔주세요.'
    '"소 잃고 외양간 고친다"는 무슨 뜻인가요? 한 문장으로 설명하세요.'
    '추석에 한국 사람들이 하는 전통적인 활동 3가지를 짧게 알려주세요.'
    '"괜찮아, 신경 쓰지 마." 이 말이 진심일 때와 서운할 때의 차이를 설명하세요.'
    '다음 문장의 맞춤법 오류를 찾아 고쳐주세요: "어제 회의때 얘기했던데 담주까지 되요?"'
    '비 오는 서울 골목길을 주제로 2-3문장 짧은 글을 써주세요.'
    '"아따, 거시기가 머시여? 잘 모르겄구만." 이 문장은 어느 지역 사투리이고, 표준어로 바꾸면 어떻게 되나요?'
)

# ── 유틸리티 함수 ──────────────────────────────────────────────────────────

check_ollama() {
    if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "❌ Ollama가 실행 중이지 않습니다. 'ollama serve'를 먼저 실행하세요."
        exit 1
    fi
    echo "✅ Ollama 연결 확인: ${OLLAMA_URL}"
}

check_models() {
    echo ""
    echo "📋 모델 설치 확인:"
    local available
    available=$(curl -sf "${OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
" 2>/dev/null || echo "")

    local missing=0
    for model in "${MODELS[@]}"; do
        # 모델 이름이 available 목록에 포함되는지 확인 (태그 유무 모두 매칭)
        if echo "$available" | grep -qF "$model"; then
            echo "  ✅ $model"
        else
            echo "  ❌ $model — 설치 필요: ollama pull $model"
            missing=$((missing + 1))
        fi
    done

    if [ "$missing" -gt 0 ]; then
        echo ""
        echo "⚠️  ${missing}개 모델이 없습니다. 계속 진행하면 해당 모델은 건너뜁니다."
        read -rp "계속하시겠습니까? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "중단합니다."
            exit 0
        fi
    fi
    echo ""
}

# Ollama API로 응답 생성 (Python으로 JSON 안전하게 구성)
generate_response() {
    local model="$1"
    local prompt="$2"
    local timeout="${3:-120}"

    python3 - "$model" "$prompt" "$OLLAMA_URL" "$TEMPERATURE" "$NUM_PREDICT" "$timeout" << 'PYEOF'
import sys, json, urllib.request, urllib.error

model, prompt, base_url, temp, num_predict, timeout = sys.argv[1:7]
payload = json.dumps({
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": float(temp),
        "num_predict": int(num_predict),
        "top_p": 0.9,
        "repeat_penalty": 1.2
    }
}).encode("utf-8")

try:
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(data.get("response", "[응답 없음]"))
except urllib.error.URLError as e:
    print(f"[오류: 연결 실패 — {e}]", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"[오류: {e}]", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# 안전한 파일명 생성 (모델명의 특수문자 처리)
safe_filename() {
    echo "$1" | tr ':/' '_' | tr -cd '[:alnum:]_.-'
}

# ── 메인 실행 ──────────────────────────────────────────────────────────────

main() {
    echo "============================================================"
    echo " FRANKENSTALLM 한국어 LLM 비교 테스트"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # 사전 확인
    check_ollama
    check_models

    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_DIR"

    # 요약 마크다운 헤더 작성
    cat > "$SUMMARY_FILE" << 'HEADER'
# FRANKENSTALLM 한국어 LLM 비교 테스트 결과

HEADER
    echo "**실행 시각:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "## 테스트 대상 모델" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "| # | 모델 |" >> "$SUMMARY_FILE"
    echo "|---|------|" >> "$SUMMARY_FILE"
    for i in "${!MODELS[@]}"; do
        echo "| $((i+1)) | ${MODELS[$i]} |" >> "$SUMMARY_FILE"
    done
    echo "" >> "$SUMMARY_FILE"
    echo "---" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    local total=${#MODELS[@]}
    local num_tests=${#TEST_IDS[@]}
    local completed=0
    local failed=0

    # 테스트 실행
    for t in "${!TEST_IDS[@]}"; do
        local test_id="${TEST_IDS[$t]}"
        local test_name="${TEST_NAMES[$t]}"
        local prompt="${TEST_PROMPTS[$t]}"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📝 ${test_id}: ${test_name}"
        echo "   프롬프트: ${prompt:0:60}..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # 요약에 테스트 섹션 추가
        echo "## ${test_id}: ${test_name}" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        echo "**프롬프트:** ${prompt}" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"

        for m in "${!MODELS[@]}"; do
            local model="${MODELS[$m]}"
            local safe_model
            safe_model=$(safe_filename "$model")
            local outfile="${OUTPUT_DIR}/${test_id}_${safe_model}.txt"

            echo ""
            echo "  🤖 [$(( m + 1 ))/${total}] ${model} ..."

            # 응답 생성 (실패해도 스크립트 계속 진행)
            local answer status=0
            answer=$(generate_response "$model" "$prompt" 180) || status=$?

            if [ "$status" -eq 0 ] && [ -n "$answer" ]; then
                # 개별 파일 저장
                {
                    echo "모델: ${model}"
                    echo "테스트: ${test_id} — ${test_name}"
                    echo "프롬프트: ${prompt}"
                    echo "시각: $(date '+%Y-%m-%d %H:%M:%S')"
                    echo "────────────────────────────────────"
                    echo "$answer"
                } > "$outfile"

                # 터미널 미리보기 (최대 3줄)
                local preview
                preview=$(echo "$answer" | head -3)
                echo "  ✅ 완료 — ${outfile}"
                echo "     ${preview}"

                completed=$((completed + 1))
            else
                echo "  ❌ 실패 — ${model}"
                echo "[오류: 응답 생성 실패]" > "$outfile"
                failed=$((failed + 1))
            fi

            # 요약에 응답 추가
            echo "### ${model}" >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
            echo '```' >> "$SUMMARY_FILE"
            cat "$outfile" | tail -n +6 >> "$SUMMARY_FILE"  # 헤더 제외, 응답만
            echo '```' >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"

            # 모델 간 쿨다운
            if [ "$m" -lt $((total - 1)) ]; then
                sleep "$COOLDOWN"
            fi
        done

        echo "" >> "$SUMMARY_FILE"
        echo "---" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    done

    # 최종 통계
    echo ""
    echo "============================================================"
    echo " 완료!"
    echo " ✅ 성공: ${completed} / $((total * num_tests))"
    echo " ❌ 실패: ${failed} / $((total * num_tests))"
    echo " 📁 개별 파일: ${OUTPUT_DIR}/"
    echo " 📄 요약 파일: ${SUMMARY_FILE}"
    echo "============================================================"

    # 요약 파일에 통계 추가
    echo "## 실행 통계" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "- 성공: ${completed} / $((total * num_tests))" >> "$SUMMARY_FILE"
    echo "- 실패: ${failed} / $((total * num_tests))" >> "$SUMMARY_FILE"
    echo "- 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
}

# ── 개별 스크린샷용 헬퍼 ──────────────────────────────────────────────────
# 사용법: bash run_screenshot_tests.sh single <model> <test_number>
# 예: bash run_screenshot_tests.sh single "gemma3:4b" 1
run_single() {
    local model="$1"
    local test_idx=$(( $2 - 1 ))

    if [ "$test_idx" -lt 0 ] || [ "$test_idx" -ge "${#TEST_IDS[@]}" ]; then
        echo "❌ 테스트 번호는 1-${#TEST_IDS[@]} 사이여야 합니다."
        exit 1
    fi

    local test_id="${TEST_IDS[$test_idx]}"
    local test_name="${TEST_NAMES[$test_idx]}"
    local prompt="${TEST_PROMPTS[$test_idx]}"

    check_ollama

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🤖 모델: ${model}"
    echo "📝 ${test_id}: ${test_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "프롬프트: ${prompt}"
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo ""

    generate_response "$model" "$prompt" 180
}

# ── 엔트리포인트 ──────────────────────────────────────────────────────────
if [ "${1:-}" = "single" ]; then
    if [ $# -lt 3 ]; then
        echo "사용법: $0 single <model_name> <test_number(1-7)>"
        echo ""
        echo "모델 목록:"
        for m in "${MODELS[@]}"; do echo "  - $m"; done
        echo ""
        echo "테스트 목록:"
        for i in "${!TEST_IDS[@]}"; do
            echo "  ${TEST_IDS[$i]}. ${TEST_NAMES[$i]}"
        done
        exit 1
    fi
    run_single "$2" "$3"
elif [ "${1:-}" = "list" ]; then
    echo "모델 목록:"
    for m in "${MODELS[@]}"; do echo "  - $m"; done
    echo ""
    echo "테스트 목록:"
    for i in "${!TEST_IDS[@]}"; do
        echo "  $((i+1)). ${TEST_IDS[$i]}: ${TEST_NAMES[$i]}"
    done
else
    main
fi
