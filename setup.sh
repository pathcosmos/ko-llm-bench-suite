#!/bin/bash
# ko-llm-bench-suite 자동 설치 스크립트

set -e

echo "================================================"
echo "  ko-llm-bench-suite 설치"
echo "================================================"

# 1. Python 의존성
echo ""
echo "[1/4] Python 패키지 설치..."
pip install -r requirements.txt

# 2. Ollama 설치 확인
echo ""
echo "[2/4] Ollama 확인..."
if command -v ollama &>/dev/null; then
    echo "  ✅ Ollama 설치됨: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
    echo "  Ollama 미설치. 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  ✅ Ollama 설치 완료"
fi

# 3. Ollama 서버 실행 확인
echo ""
echo "[3/4] Ollama 서버 확인..."
if curl -sf http://localhost:11434/ >/dev/null 2>&1; then
    echo "  ✅ Ollama 서버 실행 중"
else
    echo "  Ollama 서버 미실행. 시작합니다..."
    ollama serve &>/dev/null &
    sleep 3
    if curl -sf http://localhost:11434/ >/dev/null 2>&1; then
        echo "  ✅ Ollama 서버 시작됨"
    else
        echo "  ⚠️ Ollama 서버 시작 실패. 수동으로 'ollama serve'를 실행하세요."
    fi
fi

# 4. Judge 모델 다운로드
echo ""
echo "[4/4] Judge 모델 다운로드..."
echo "  qwen2.5:7b-instruct (Primary Judge, ~4.7GB)..."
ollama pull qwen2.5:7b-instruct 2>/dev/null || echo "  ⚠️ 다운로드 실패 (나중에 수동으로: ollama pull qwen2.5:7b-instruct)"
echo "  exaone3.5:7.8b (Secondary Judge, ~4.8GB)..."
ollama pull exaone3.5:7.8b 2>/dev/null || echo "  ⚠️ 다운로드 실패 (나중에 수동으로: ollama pull exaone3.5:7.8b)"

# 완료
echo ""
echo "================================================"
echo "  ✅ 설치 완료!"
echo ""
echo "  사용법:"
echo "    python kobench.py --config configs/default.yaml"
echo "    python kobench.py --help"
echo "================================================"
