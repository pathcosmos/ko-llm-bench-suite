"""평가 실행 컨트롤러 — 진행률 + 에러 복구."""
import time
import traceback
from pathlib import Path
from kobench import config
from kobench.wizard import ui


def run_tracks_interactive(track_nums, models):
    """인터랙티브 모드로 트랙 실행 — 진행률 바 + 에러 복구.

    Returns: dict of track results
    """
    from kobench.runner import (
        wait_for_ollama, unload_all_models, save_results_incremental,
        _gpu_healthy_now, _restart_ollama,
    )

    all_results = {}

    for ti, track_num in enumerate(track_nums):
        ui.step(ti + 1, len(track_nums), f"Track {track_num} 실행")

        start = time.time()
        try:
            import importlib
            track_map = {
                1: "kobench.tracks.korean_bench", 2: "kobench.tracks.ko_bench",
                3: "kobench.tracks.korean_deep", 4: "kobench.tracks.code_math",
                5: "kobench.tracks.consistency", 6: "kobench.tracks.performance",
                7: "kobench.tracks.pairwise",
            }
            module = importlib.import_module(track_map[track_num])
            result = module.run(models=models)
            elapsed = time.time() - start

            track_key = f"track{track_num}"
            all_results[track_key] = result
            save_results_incremental(result, track_key)

            ui.success(f"Track {track_num} 완료 ({elapsed:.1f}s)")

        except KeyboardInterrupt:
            ui.warn("사용자 중단")
            choice = ui.error_choice(f"Track {track_num}", "전체", "KeyboardInterrupt")
            if choice == "abort":
                break
            elif choice == "skip_track":
                continue

        except Exception as e:
            elapsed = time.time() - start
            ui.fail(f"Track {track_num} 오류 ({elapsed:.1f}s): {e}")

            choice = ui.error_choice(f"Track {track_num}", "전체", str(e))
            if choice == "retry":
                ui.info("재시도 중...")
                try:
                    result = module.run(models=models)
                    all_results[f"track{track_num}"] = result
                    save_results_incremental(result, f"track{track_num}")
                    ui.success(f"Track {track_num} 재시도 성공")
                except Exception as e2:
                    ui.fail(f"재시도 실패: {e2}")
            elif choice == "skip_track":
                ui.info(f"Track {track_num} 건너뜀")
                continue
            elif choice == "abort":
                ui.warn("평가 중단")
                break

        # Cooldown between tracks
        if ti < len(track_nums) - 1:
            unload_all_models()
            if config.GPU_AVAILABLE and not _gpu_healthy_now():
                ui.warn("GPU 이상 감지 — Ollama 재시작")
                _restart_ollama()
            time.sleep(config.COOLDOWN_BETWEEN_MODELS)

    return all_results
