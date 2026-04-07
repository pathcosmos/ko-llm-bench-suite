#!/usr/bin/env bash
# 12개 신규 테스트 × 6개 모델 = 72개 응답 수집
set -uo pipefail

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OUTPUT_DIR="screenshot_outputs_new"
COOLDOWN=2
NUM_PREDICT=256
TEMPERATURE=0.7

MODELS=(
    "frankenstallm-3b-v2-Q4_K_M"
    "llama3.2:3b"
    "phi4-mini"
    "qwen2.5:3b"
    "exaone3.5:2.4b"
    "gemma3:4b"
)

TEST_IDS=(N1 N2 N3 N4 N5 N6 N7 N8 N9 N10 N11 N12)

TEST_NAMES=(
    "단순 산술"
    "기본 지리"
    "반의어"
    "사계절 나열"
    "과학 상식"
    "문장 완성"
    "영한 번역"
    "기본 논리"
    "상식 질문"
    "패턴 인식"
    "어휘 정의"
    "조사 채우기"
)

TEST_PROMPTS=(
    '100 더하기 250은 얼마인가요?'
    '지구에서 가장 큰 대륙은 어디인가요?'
    '"뜨겁다"의 반대말은 무엇인가요?'
    '사계절의 이름을 나열하세요.'
    '물의 화학식은 무엇인가요?'
    '하늘의 색깔은 보통 ___입니다. 빈칸을 채우세요.'
    '"Thank you"를 한국어로 번역하세요.'
    '사과 3개에서 1개를 먹으면 몇 개 남나요?'
    '밤에 하늘에서 빛나는 것은 무엇인가요?'
    '월, 화, 수, 목, 다음은 무엇인가요?'
    '"학교"는 무엇을 하는 곳인가요?'
    '"나는 사과___좋아한다." 빈칸에 알맞은 조사는 무엇인가요?'
)

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
except Exception as e:
    print(f"[오류: {e}]", file=sys.stderr)
    sys.exit(1)
PYEOF
}

safe_filename() {
    echo "$1" | tr ':/' '_' | tr -cd '[:alnum:]_.-'
}

main() {
    echo "============================================================"
    echo " 신규 테스트 12개 × 6개 모델 실행"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # Ollama 확인
    if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "❌ Ollama 연결 실패"; exit 1
    fi
    echo "✅ Ollama 연결 확인"

    mkdir -p "$OUTPUT_DIR"

    local total=${#MODELS[@]}
    local completed=0
    local failed=0

    for t in "${!TEST_IDS[@]}"; do
        local test_id="${TEST_IDS[$t]}"
        local test_name="${TEST_NAMES[$t]}"
        local prompt="${TEST_PROMPTS[$t]}"

        echo ""
        echo "━━━ ${test_id}: ${test_name} ━━━"
        echo "    ${prompt}"

        for m in "${!MODELS[@]}"; do
            local model="${MODELS[$m]}"
            local safe_model
            safe_model=$(safe_filename "$model")
            local outfile="${OUTPUT_DIR}/${test_id}_${safe_model}.txt"

            local answer status=0
            answer=$(generate_response "$model" "$prompt" 120) || status=$?

            if [ "$status" -eq 0 ] && [ -n "$answer" ]; then
                {
                    echo "모델: ${model}"
                    echo "테스트: ${test_id} — ${test_name}"
                    echo "프롬프트: ${prompt}"
                    echo "────────────────────────────────────"
                    echo "$answer"
                } > "$outfile"

                local preview
                preview=$(echo "$answer" | head -1 | cut -c1-80)
                echo "  ✅ ${model}: ${preview}"
                completed=$((completed + 1))
            else
                echo "  ❌ ${model}: 실패"
                echo "[오류]" > "$outfile"
                failed=$((failed + 1))
            fi

            [ "$m" -lt $((total - 1)) ] && sleep "$COOLDOWN"
        done
    done

    echo ""
    echo "============================================================"
    echo " 완료! ✅ ${completed} ❌ ${failed} / $((total * ${#TEST_IDS[@]}))"
    echo "============================================================"
}

main
