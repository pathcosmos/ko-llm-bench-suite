"""인터랙티브 YAML 설정 빌더 (config 서브커맨드)."""
import yaml
from pathlib import Path
from kobench.wizard import ui
from kobench.wizard.machine import test_ollama
from kobench import config


def run():
    """config 서브커맨드 실행."""
    ui.banner()
    ui.console.print("[bold]설정 파일 생성 위저드[/]\n")

    cfg = {}
    total = 6

    # Step 1: Project
    ui.step(1, total, "프로젝트 설정")
    cfg["project"] = {
        "name": ui.ask("프로젝트 이름", default="Korean LLM Evaluation"),
        "output_dir": ui.ask("결과 출력 경로", default="./results"),
        "reports_dir": ui.ask("리포트 출력 경로", default="./reports"),
    }

    # Step 2: Backend
    ui.step(2, total, "추론 백엔드")
    url = ui.ask("Ollama URL", default="http://localhost:11434")
    remote = ui.confirm("원격 서버입니까?", default=False) if "localhost" not in url else False
    cfg["backend"] = {"type": "ollama", "url": url, "remote": remote}

    # Test connection
    test = test_ollama(url)
    if test["connected"]:
        ui.success(f"연결 성공 ({len(test['models'])}개 모델)")
    else:
        ui.warn(f"연결 실패: {test['error']} (나중에 연결해도 됩니다)")

    # Step 3: Models
    ui.step(3, total, "평가 대상 모델")
    cfg["models"] = []
    if test["connected"] and test["models"]:
        available = [m["name"] for m in test["models"]]
        ui.console.print(f"  Ollama에서 {len(available)}개 모델 발견:")
        selected = ui.select_multi("평가할 모델 선택", available)
        for m in selected:
            tag = ui.ask(f"    {m} 태그", default="baseline")
            cfg["models"].append({"name": m, "tags": [tag]})
    else:
        ui.info("Ollama 미연결 — 모델을 수동으로 입력하세요")
        while True:
            name = ui.ask("모델 이름 (빈칸=완료)", default="")
            if not name:
                break
            tag = ui.ask(f"  {name} 태그", default="baseline")
            cfg["models"].append({"name": name, "tags": [tag]})

    ui.success(f"{len(cfg['models'])}개 모델 선택됨")

    # Step 4: Tracks
    ui.step(4, total, "평가 트랙 선택")
    track_names = {
        1: "Korean Bench (KoBEST)", 2: "Ko-Bench (생성 품질)",
        3: "Korean Deep (심층 이해)", 4: "Code & Math",
        5: "Consistency (일관성)", 6: "Performance (성능)",
        7: "Pairwise Elo",
    }
    scenario = ui.select_one("시나리오 선택",
        ["빠른 평가 (T1,T4,T6)", "전체 평가 (T1~T7)", "커스텀"],
        ["~30분, Judge 불필요", "~4시간, Judge 필요", "직접 선택"]
    )
    if scenario.startswith("빠른"):
        enabled = [1, 4, 6]
    elif scenario.startswith("전체"):
        enabled = [1, 2, 3, 4, 5, 6, 7]
    else:
        all_tracks = [f"T{k}: {v}" for k, v in track_names.items()]
        selected = ui.select_multi("트랙 선택", all_tracks)
        enabled = [int(s.split(":")[0][1:]) for s in selected]

    cfg["tracks"] = {"enabled": enabled, "order": enabled}
    ui.success(f"트랙: {enabled}")

    # Step 5: Judge
    ui.step(5, total, "Judge 설정")
    needs_judge = any(t in enabled for t in [2, 3, 7])
    if needs_judge:
        cfg["judge"] = {
            "dual_enabled": ui.confirm("이중 Judge 사용?", default=True),
            "primary": {
                "model": ui.ask("Primary Judge", default="qwen2.5:7b-instruct"),
                "weight": 0.6,
            },
            "secondary": {
                "model": ui.ask("Secondary Judge", default="exaone3.5:7.8b"),
                "weight": 0.4,
            },
            "timeout": 120,
        }
    else:
        ui.info("선택한 트랙에 Judge 불필요 — 기본값 사용")

    # Step 6: Save
    ui.step(6, total, "저장")

    # Add default sampling and retry
    cfg.setdefault("sampling", {
        "default": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.2, "num_predict": 512, "num_ctx": 4096},
        "benchmark": {"temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.0, "num_predict": 256, "num_ctx": 4096},
    })
    cfg.setdefault("retry", {"max_retries": 3, "backoff_base": 5, "cooldown_between_models": 10, "cooldown_between_tests": 1})

    yaml_text = yaml.dump(cfg, allow_unicode=True, default_flow_style=False, sort_keys=False)
    ui.show_yaml(yaml_text)

    save_path = ui.ask("저장 경로", default="configs/my_eval.yaml")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(yaml_text, encoding="utf-8")
    ui.success(f"저장됨: {save_path}")

    # Validate
    from kobench.config import validate_config
    errors = validate_config(cfg)
    if errors:
        ui.warn("검증 경고:")
        for e in errors:
            ui.info(f"  - {e}")
    else:
        ui.success("설정 검증 통과")

    ui.divider()
    ui.console.print(f"\n  다음: [cyan]python kobench.py eval --config {save_path}[/]")
