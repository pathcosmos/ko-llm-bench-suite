"""시스템 진단 (status 서브커맨드)."""
from pathlib import Path
from kobench import config
from kobench.wizard import ui
from kobench.wizard.machine import test_ollama, get_gpu_info, get_python_info, check_dependencies


def run():
    """status 서브커맨드 실행."""
    ui.banner()

    # Step 1: System
    ui.step(1, 4, "시스템 정보")
    py = get_python_info()
    ui.success(f"Python {py['version']} ({py['executable']})")

    gpu = get_gpu_info()
    if gpu["available"]:
        ui.success(f"GPU: {gpu['name']} (VRAM: {gpu['vram_total']}, Free: {gpu['vram_free']})")
    else:
        ui.warn("GPU 감지되지 않음 (CPU 모드)")

    # Step 2: Dependencies
    ui.step(2, 4, "Python 패키지")
    deps = check_dependencies()
    missing = [k for k, v in deps.items() if not v]
    installed = [k for k, v in deps.items() if v]
    ui.success(f"{len(installed)}개 설치됨: {', '.join(installed)}")
    if missing:
        ui.fail(f"{len(missing)}개 미설치: {', '.join(missing)}")
        ui.info(f"설치: pip install {' '.join(missing)}")

    # Step 3: Ollama
    ui.step(3, 4, "Ollama 상태")
    ollama = test_ollama()
    if ollama["connected"]:
        ui.success(f"연결됨: {ollama['url']} ({ollama['version']})")
        if ollama["models"]:
            rows = [[m["name"], f"{m['size_gb']:.1f}GB"] for m in ollama["models"]]
            ui.show_table(f"Ollama 모델 ({len(rows)}개)", ["모델", "크기"], rows)
        else:
            ui.warn("설치된 모델 없음")
    else:
        ui.fail(f"연결 실패: {ollama['error']}")
        ui.info("시작: ollama serve")

    # Step 4: Previous results
    ui.step(4, 4, "이전 평가 결과")
    results_dir = config.RESULTS_DIR
    examples_dir = Path("examples/frankenstallm/results")

    result_files = list(results_dir.glob("full_results_*.json")) if results_dir.exists() else []
    example_files = list(examples_dir.glob("full_results_*.json")) if examples_dir.exists() else []

    if result_files:
        ui.success(f"results/: {len(result_files)}개 결과 파일")
    if example_files:
        ui.success(f"examples/frankenstallm/: {len(example_files)}개 (레거시)")
    if not result_files and not example_files:
        ui.info("이전 평가 결과 없음")

    # Check checkpoints
    ckpts = list(results_dir.glob("*_checkpoint.json")) if results_dir.exists() else []
    if ckpts:
        ui.info(f"체크포인트: {len(ckpts)}개 ({', '.join(c.stem for c in ckpts[:3])}...)")

    ui.divider()
