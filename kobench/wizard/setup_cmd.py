"""설치 위저드 (setup 서브커맨드)."""
import subprocess
import sys
from kobench.wizard import ui
from kobench.wizard.machine import test_ollama, get_gpu_info, check_dependencies


def run():
    """setup 서브커맨드 실행."""
    ui.banner()
    ui.console.print("[bold]설치 위저드를 시작합니다.[/]\n")

    total_steps = 5

    # Step 1: Python deps
    ui.step(1, total_steps, "Python 패키지 확인")
    deps = check_dependencies()
    missing = [k for k, v in deps.items() if not v]
    if missing:
        ui.warn(f"미설치 패키지: {', '.join(missing)}")
        if ui.confirm(f"pip install로 설치하시겠습니까?"):
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing,
                          capture_output=False)
            ui.success("패키지 설치 완료")
        else:
            ui.info(f"수동 설치: pip install {' '.join(missing)}")
    else:
        ui.success(f"모든 패키지 설치됨 ({len(deps)}개)")

    # Step 2: GPU
    ui.step(2, total_steps, "GPU 확인")
    gpu = get_gpu_info()
    if gpu["available"]:
        ui.success(f"{gpu['name']} (VRAM: {gpu['vram_total']})")
    else:
        ui.warn("GPU 없음 — CPU 모드로 실행 가능 (느림)")

    # Step 3: Ollama
    ui.step(3, total_steps, "Ollama 확인")
    ollama = test_ollama()
    if ollama["connected"]:
        ui.success(f"Ollama 실행 중 ({ollama['url']})")
    else:
        ui.fail("Ollama 미실행")
        if ui.confirm("Ollama를 시작하시겠습니까?"):
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                import time; time.sleep(3)
                ollama = test_ollama()
                if ollama["connected"]:
                    ui.success("Ollama 시작됨")
                else:
                    ui.fail("시작 실패. 수동으로 'ollama serve'를 실행하세요.")
            except FileNotFoundError:
                ui.fail("Ollama 미설치. https://ollama.com 에서 설치하세요.")
                ui.info("curl -fsSL https://ollama.com/install.sh | sh")

    # Step 4: Remote server (optional)
    ui.step(4, total_steps, "원격 서버 설정 (선택)")
    if ui.confirm("원격 Ollama 서버를 추가하시겠습니까?", default=False):
        remote_url = ui.ask("원격 서버 URL", default="http://192.168.1.100:11434")
        remote_test = test_ollama(remote_url)
        if remote_test["connected"]:
            ui.success(f"원격 서버 연결 성공 ({len(remote_test['models'])}개 모델)")
        else:
            ui.fail(f"연결 실패: {remote_test['error']}")
            ui.info("SSH 터널: ssh -NL 11434:localhost:11434 user@server &")
    else:
        ui.info("원격 서버 건너뜀")

    # Step 5: Judge models
    ui.step(5, total_steps, "Judge 모델 확인")
    if ollama["connected"]:
        model_names = [m["name"] for m in ollama["models"]]
        judges = {
            "qwen2.5:7b-instruct": "Primary Judge (~4.7GB)",
            "exaone3.5:7.8b": "Secondary Judge (~4.8GB)",
        }
        for model, desc in judges.items():
            if any(model in n for n in model_names):
                ui.success(f"{model} ({desc})")
            else:
                ui.warn(f"{model} 미설치 ({desc})")
                if ui.confirm(f"다운로드하시겠습니까?"):
                    ui.info(f"다운로드 중... (시간 소요)")
                    result = subprocess.run(["ollama", "pull", model], capture_output=False)
                    if result.returncode == 0:
                        ui.success(f"{model} 다운로드 완료")
                    else:
                        ui.fail(f"다운로드 실패. 수동: ollama pull {model}")
    else:
        ui.warn("Ollama 미연결 — Judge 모델 확인 건너뜀")

    # Summary
    ui.divider()
    ui.console.print("\n[bold green]설치 점검 완료![/]")
    ui.console.print("  다음 단계:")
    ui.console.print("  [cyan]python kobench.py config[/]  — 설정 파일 생성")
    ui.console.print("  [cyan]python kobench.py eval[/]    — 평가 실행")
    ui.console.print("  [cyan]python kobench.py status[/]  — 시스템 상태 확인")
