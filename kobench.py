#!/home/lanco/ai-env/bin/python3
"""
FRANKENSTALLM 3B 심화 평가 프레임워크 — 메인 실행 스크립트

Usage:
    python run_evaluation.py                  # 전체 7트랙 실행
    python run_evaluation.py --tracks 6 1     # 특정 트랙만 실행
    python run_evaluation.py --models frankenstallm-3b-v2-Q8_0 qwen2.5:3b  # 특정 모델만
    python run_evaluation.py --report-only    # 기존 결과로 리포트만 생성
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kobench import config
from kobench.runner import (
    wait_for_ollama,
    unload_all_models,
    save_results_incremental,
    _gpu_healthy_now,
    _restart_ollama,
)
from kobench.scoring import build_scorecard, save_scorecard
from kobench.report import generate_html_report, generate_markdown_report


def load_track(track_num: int):
    """트랙 모듈 동적 로드"""
    track_map = {
        1: "kobench.tracks.korean_bench",
        2: "kobench.tracks.ko_bench",
        3: "kobench.tracks.korean_deep",
        4: "kobench.tracks.code_math",
        5: "kobench.tracks.consistency",
        6: "kobench.tracks.performance",
        7: "kobench.tracks.pairwise",
    }
    import importlib
    module_name = track_map.get(track_num)
    if module_name is None:
        raise ValueError(f"알 수 없는 트랙: {track_num}")
    return importlib.import_module(module_name)


def run_tracks(track_nums: list[int], models: list[str] | None = None) -> dict:
    """지정된 트랙 순차 실행"""
    all_results = {}

    for track_num in track_nums:
        print(f"\n{'=' * 70}")
        print(f"  Track {track_num} 실행 시작")
        print(f"{'=' * 70}")

        start = time.time()
        try:
            module = load_track(track_num)
            result = module.run(models=models)
            elapsed = time.time() - start

            track_key = f"track{track_num}"
            all_results[track_key] = result

            # 결과 저장
            save_results_incremental(result, track_key)
            print(f"\n  ✅ Track {track_num} 완료 ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  ❌ Track {track_num} 실패 ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            all_results[f"track{track_num}"] = {"error": str(e)}

        # 트랙 간 쿨다운 — GPU 메모리 정리 + GPU 상태 검증
        if track_num != track_nums[-1]:
            print(f"\n  ⏳ 트랙 전환 쿨다운 ({config.COOLDOWN_BETWEEN_MODELS}s)...")
            unload_all_models()

            # GPU 드라이버 상태 확인 — EVAFRILL CUDA 실패 후 오염 감지
            if config.GPU_AVAILABLE and not _gpu_healthy_now():
                print("  ⚠ 트랙 전환 중 GPU 이상 감지 — Ollama 재시작으로 복구 시도")
                # _restart_ollama() 내부에서 GPU 리셋도 시도함
                if _restart_ollama():
                    print("  ✅ Ollama 재시작 성공")
                else:
                    print("  ❌ Ollama 재시작 실패 — 다음 트랙은 Ollama 없이 진행될 수 있음")

            time.sleep(config.COOLDOWN_BETWEEN_MODELS)

    return all_results


def load_existing_results() -> dict:
    """기존 결과 파일에서 로드 — 여러 파일의 summary/results를 병합"""
    results = {}
    for path in sorted(config.RESULTS_DIR.glob("track*_2*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            track_name = path.stem.split("_")[0]
            if track_name not in results:
                results[track_name] = data
            else:
                existing = results[track_name]
                # summary 병합 — 빈 값은 기존 데이터를 덮어쓰지 않음
                if isinstance(data.get("summary"), dict) and isinstance(existing.get("summary"), dict):
                    for model, val in data["summary"].items():
                        if val:
                            existing["summary"][model] = val
                # results 병합
                if isinstance(data.get("results"), dict) and isinstance(existing.get("results"), dict):
                    existing["results"].update(data["results"])
                elif isinstance(data.get("results"), list) and isinstance(existing.get("results"), list):
                    # 최신 파일의 모델 데이터로 교체 (채점 업데이트 반영)
                    new_models = {r.get("model") for r in data["results"]}
                    existing["results"] = [
                        r for r in existing["results"] if r.get("model") not in new_models
                    ] + data["results"]
        except Exception as e:
            print(f"  ⚠ {path.name} 로드 실패: {e}")

    # T2: list 타입 results에서 summary 재빌드
    for track_name, track_data in results.items():
        if isinstance(track_data.get("results"), list) and track_name == "track2":
            from kobench.tracks.ko_bench import _build_summary
            track_data["summary"] = _build_summary(track_data["results"])

    return results


def generate_reports(track_results: dict) -> None:
    """리포트 생성"""
    print(f"\n{'=' * 70}")
    print("  리포트 생성")
    print(f"{'=' * 70}")

    # 스코어카드 빌드
    scorecard = build_scorecard(
        {k: v.get("summary", {}) for k, v in track_results.items() if isinstance(v, dict)}
    )
    save_scorecard(scorecard)

    # HTML + Markdown
    generate_html_report(track_results, scorecard)
    generate_markdown_report(track_results, scorecard)

    # 전체 결과 JSON
    full_path = config.RESULTS_DIR / f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(track_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📁 전체 결과: {full_path}")


def main():
    parser = argparse.ArgumentParser(description="FRANKENSTALLM 3B 심화 평가 프레임워크")
    parser.add_argument("--tracks", nargs="+", type=int, default=[6, 1, 3, 2, 4, 5, 7],
                        help="실행할 트랙 번호 (기본: 6 1 3 2 4 5 7)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="평가할 모델 목록 (기본: 전체 11개)")
    parser.add_argument("--report-only", action="store_true",
                        help="기존 결과로 리포트만 생성")
    parser.add_argument("--skip-health-check", action="store_true",
                        help="Ollama health check 건너뛰기")

    args = parser.parse_args()

    print("=" * 70)
    print("  FRANKENSTALLM 3B 심화 평가 프레임워크")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  트랙: {args.tracks}")
    print(f"  모델: {args.models or config.ALL_MODELS}")
    print("=" * 70)

    if args.report_only:
        results = load_existing_results()
        if not results:
            print("❌ 기존 결과가 없습니다.")
            sys.exit(1)
        generate_reports(results)
        return

    # Ollama 연결 확인
    if not args.skip_health_check:
        print("\n🔍 Ollama 서버 연결 확인...")
        if not wait_for_ollama(max_wait=30):
            print("❌ Ollama 서버에 연결할 수 없습니다.")
            print("   ollama serve 또는 ./ollama_watchdog.sh 실행 후 다시 시도하세요.")
            sys.exit(1)
        print("  ✅ Ollama 서버 연결 성공")

    # 평가 실행
    total_start = time.time()
    track_results = run_tracks(args.tracks, args.models)

    # 리포트 생성
    generate_reports(track_results)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  전체 평가 완료: {total_elapsed / 60:.1f}분")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
