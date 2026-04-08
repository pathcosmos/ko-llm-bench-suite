"""인터랙티브 평가 위저드 (eval 서브커맨드)."""
from pathlib import Path
from kobench import config
from kobench.wizard import ui
from kobench.wizard.machine import test_ollama
from kobench.wizard.executor import run_tracks_interactive


def run():
    """eval 서브커맨드 실행."""
    ui.banner()
    ui.console.print("[bold]평가 위저드를 시작합니다.[/]\n")

    total = 6

    # Step 1: Config
    ui.step(1, total, "설정 로드")
    configs_dir = Path("configs")
    yamls = sorted(configs_dir.glob("**/*.yaml")) if configs_dir.exists() else []

    if yamls:
        choices = [str(y) for y in yamls] + ["기본 설정 사용"]
        config_path = ui.select_one("설정 파일 선택", choices)
        if config_path != "기본 설정 사용":
            from kobench.config import load_yaml_config, apply_yaml_to_config
            yaml_cfg = load_yaml_config(config_path)
            apply_yaml_to_config(yaml_cfg)
            ui.success(f"설정 로드: {config_path}")
        else:
            yaml_cfg = {}
            config_path = None
    else:
        ui.info("설정 파일 없음 — 기본 설정 사용")
        yaml_cfg = {}
        config_path = None

    # Step 2: Environment check
    ui.step(2, total, "환경 확인")
    ollama = test_ollama()
    if ollama["connected"]:
        ui.success(f"Ollama 연결: {ollama['url']} ({len(ollama['models'])}개 모델)")
    else:
        ui.fail(f"Ollama 미연결: {ollama['error']}")
        ui.info("'ollama serve'를 실행한 후 다시 시도하세요.")
        return

    # Step 3: Model selection
    ui.step(3, total, "모델 선택")
    available_names = [m["name"] for m in ollama["models"]]

    if yaml_cfg.get("models"):
        yaml_models = [m["name"] for m in yaml_cfg["models"]]
        ui.info(f"설정 파일에서 {len(yaml_models)}개 모델 로드")
        # Check availability
        missing = [m for m in yaml_models if not any(m in a for a in available_names)]
        present = [m for m in yaml_models if any(m in a for a in available_names)]
        for m in present:
            ui.success(m)
        for m in missing:
            ui.fail(f"{m} — 미설치")
        if missing and ui.confirm("미설치 모델을 건너뛰고 진행?"):
            models = present
        else:
            models = yaml_models
    else:
        ui.info(f"Ollama에서 {len(available_names)}개 모델 발견")
        models = ui.select_multi("평가할 모델 선택", available_names[:20])  # Limit display

    if not models:
        ui.fail("선택된 모델이 없습니다.")
        return
    ui.success(f"{len(models)}개 모델 선택")

    # Step 4: Track/Scenario
    ui.step(4, total, "평가 시나리오")
    if yaml_cfg.get("tracks", {}).get("enabled"):
        tracks = yaml_cfg["tracks"]["enabled"]
        ui.info(f"설정 파일에서 트랙 로드: {tracks}")
        if not ui.confirm("이 트랙으로 진행?"):
            tracks = _select_tracks()
    else:
        tracks = _select_tracks()

    # Estimate time
    est_min = len(models) * len(tracks) * 3  # rough: 3 min per model per track
    ui.info(f"예상 소요: ~{est_min}분 ({len(models)}모델 × {len(tracks)}트랙)")

    if not ui.confirm("평가를 시작하시겠습니까?"):
        ui.info("취소됨")
        return

    # Step 5: Execute
    ui.step(5, total, "평가 실행")
    ui.divider()

    results = run_tracks_interactive(tracks, models)

    # Step 6: Results
    ui.step(6, total, "결과")
    if results:
        _show_results_summary(results, models)

        if ui.confirm("리포트를 생성하시겠습니까?"):
            try:
                from kobench.scoring import build_scorecard, save_scorecard
                from kobench.report import generate_html_report, generate_markdown_report

                scorecard = build_scorecard(
                    {k: v.get("summary", {}) for k, v in results.items() if isinstance(v, dict)}
                )
                save_scorecard(scorecard)
                generate_html_report(results, scorecard)
                generate_markdown_report(results, scorecard)
                ui.success("리포트 생성 완료")
                ui.info(f"경로: {config.REPORTS_DIR}/")
            except Exception as e:
                ui.fail(f"리포트 생성 실패: {e}")
    else:
        ui.warn("평가 결과가 없습니다.")

    ui.divider()


def _select_tracks():
    """인터랙티브 트랙 선택."""
    scenario = ui.select_one("시나리오 선택",
        ["빠른 평가 (T1,T4,T6)", "전체 평가 (T1~T7)", "커스텀"],
        ["~30분, Judge 불필요", "~4시간, Judge 필요", "직접 선택"]
    )
    if scenario.startswith("빠른"):
        return [1, 4, 6]
    elif scenario.startswith("전체"):
        return [1, 2, 3, 4, 5, 6, 7]
    else:
        names = ["T1: KoBEST", "T2: Ko-Bench", "T3: Korean Deep",
                 "T4: Code/Math", "T5: Consistency", "T6: Performance", "T7: Pairwise Elo"]
        selected = ui.select_multi("트랙 선택", names)
        return [int(s[1]) for s in selected]


def _show_results_summary(results, models):
    """결과 요약 테이블."""
    rows = []
    for m in models:
        scores = []
        for tk, tv in sorted(results.items()):
            if isinstance(tv, dict) and "summary" in tv:
                summary = tv["summary"]
                if m in summary and isinstance(summary[m], dict):
                    vals = [v for v in summary[m].values() if isinstance(v, (int, float))]
                    avg = sum(vals) / len(vals) if vals else 0
                    scores.append(f"{avg:.2f}")
                else:
                    scores.append("—")
            else:
                scores.append("—")
        rows.append([m] + scores)

    columns = ["모델"] + [k for k in sorted(results.keys())]
    ui.show_table("평가 결과 요약", columns, rows)
