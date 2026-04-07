#!/usr/bin/env python3
"""Generate summary_data.json from all track result files."""

import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ARCHIVE_DIR = os.path.join(RESULTS_DIR, "archive_cpu_20260312")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports")

# GPU results (main results directory)
GPU_FILES = {
    "track1": "track1_korean_bench_20260312_181703.json",
    "track4": "track4_code_math_20260312_183644.json",
    "track5": "track5_consistency_20260312_192305.json",
    "track6": "track6_performance_20260312_192550.json",
}

# CPU results (archive directory) for tracks that have both CPU and GPU
CPU_FILES = {
    "track1": "track1_korean_bench_20260311_024226.json",
    "track4": "track4_code_math_20260311_055248.json",
    "track5": "track5_consistency_20260311_134844.json",
    "track6": "track6_performance_20260310_155213.json",
}

# CPU-only tracks (only available in archive)
CPU_ONLY_FILES = {
    "track2": "track2_ko_bench_20260312_095013.json",
    "track3": "track3_korean_deep_20260312_131624.json",
}

# Track 7 is CPU-only but lives in main results directory
TRACK7_FILE = "track7_pairwise_20260312_195322.json"


def load_summary(filepath):
    """Load a track result JSON and return its summary dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["summary"]


def collect_all_models(*summary_dicts):
    """Collect all unique model names across all summary dicts."""
    models = set()
    for s in summary_dicts:
        models.update(s.keys())
    return sorted(models)


def main():
    # Load GPU results
    gpu_results = {}
    for track, filename in GPU_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        gpu_results[track] = load_summary(path)
        print(f"Loaded GPU {track}: {filename} ({len(gpu_results[track])} models)")

    # Load CPU results for tracks with both CPU and GPU
    cpu_results = {}
    for track, filename in CPU_FILES.items():
        path = os.path.join(ARCHIVE_DIR, filename)
        cpu_results[track] = load_summary(path)
        print(f"Loaded CPU {track}: {filename} ({len(cpu_results[track])} models)")

    # Load CPU-only tracks
    cpu_only = {}
    for track, filename in CPU_ONLY_FILES.items():
        path = os.path.join(ARCHIVE_DIR, filename)
        cpu_only[track] = load_summary(path)
        print(f"Loaded CPU-only {track}: {filename} ({len(cpu_only[track])} models)")

    # Load track 7
    track7_path = os.path.join(RESULTS_DIR, TRACK7_FILE)
    track7_summary = load_summary(track7_path)
    print(f"Loaded CPU-only track7: {TRACK7_FILE} ({len(track7_summary)} models)")

    # Collect all model names
    all_summaries = (
        list(gpu_results.values())
        + list(cpu_results.values())
        + list(cpu_only.values())
        + [track7_summary]
    )
    models = collect_all_models(*all_summaries)

    # Build output
    output = {
        "gpu_results": gpu_results,
        "cpu_results": cpu_results,
        "track2": cpu_only["track2"],
        "track3": cpu_only["track3"],
        "track7": track7_summary,
        "models": models,
        "metadata": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Track 2,3,7 are CPU-only",
            "gpu_tracks": list(GPU_FILES.keys()),
            "cpu_tracks": list(CPU_FILES.keys()),
            "cpu_only_tracks": ["track2", "track3", "track7"],
        },
    }

    # Write output
    output_path = os.path.join(OUTPUT_DIR, "summary_data.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {output_path}")
    print(f"Models found: {models}")
    print(f"GPU tracks: {list(gpu_results.keys())}")
    print(f"CPU tracks: {list(cpu_results.keys())}")
    print(f"CPU-only tracks: track2, track3, track7")


if __name__ == "__main__":
    main()
