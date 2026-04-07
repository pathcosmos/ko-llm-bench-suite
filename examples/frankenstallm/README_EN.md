# FRANKENSTALLM 3B Evaluation Framework (English)

> This is the English companion to [README.md](README.md) (Korean). It covers the same content; refer to the Korean version for the primary documentation.

## Quick Start

```bash
git clone https://github.com/lanco/frankenstallm_test.git
cd frankenstallm_test
pip install -r requirements.txt
ollama pull qwen2.5:3b gemma3:4b phi4-mini exaone3.5:2.4b llama3.2:3b
python run_evaluation.py
```

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |
| Python | 3.12+ | 3.12 |
| RAM | 16 GB | 32 GB |
| GPU | None (CPU OK) | NVIDIA 16 GB+ VRAM |
| Disk | 20 GB free | 50 GB free |

## 7 Evaluation Tracks

| Track | Name | Description | LLM-as-Judge |
|-------|------|-------------|:---:|
| 1 | Korean Bench | KoBEST 4 tasks (BoolQ, COPA, SentiNeg, HellaSwag) | |
| 2 | KO-Bench | 8-category Korean generation quality | O |
| 3 | Korean Deep | Deep Korean comprehension | O |
| 4 | Code & Math | Coding/math problem solving | |
| 5 | Consistency | Response consistency testing | |
| 6 | Performance | Token speed, latency, concurrency | |
| 7 | Pairwise | Model pairwise comparison | O |

## EVAFRILL-Mo-3B (PyTorch Direct Inference)

EVAFRILL-Mo-3B is a **Mamba-2 + Transformer hybrid architecture** (2.94B parameters) Korean LLM. It cannot run on Ollama/GGUF because llama.cpp does not support Mamba-2 SSM operations. Instead, it runs via `eval_framework/evafrill_runner.py` using **PyTorch direct inference**.

See [MODEL_DETAILS.md](MODEL_DETAILS.md) for full architecture details.

---

## Troubleshooting

### EVAFRILL CUDA Failure Cascading to Ollama Infinite Restart (Fixed 2026-03-30)

**Symptom:**
When EVAFRILL-Mo model loading fails with `CUDA error: unknown error (cudaErrorUnknown)`, all subsequent Ollama models become unloadable. Ollama restart loops infinitely and the evaluation drifts for hours until the SSH session drops.

**Root Cause:**
`cudaErrorUnknown` is fundamentally different from CUDA OOM. It corrupts the **GPU driver-level state**, not just the process-level CUDA context. EVAFRILL loads tensors directly onto `cuda:0` via PyTorch (`evafrill_runner.py:load_model()`). When this causes a non-recoverable CUDA error, the GPU becomes unusable for **all processes sharing the same device**, including Ollama (a separate process).

**Causal Chain:**

```
EVAFRILL model.to(cuda:0) fails
  -> GPU driver corruption (cudaErrorUnknown)
  -> No cleanup code, just return False
  -> No GPU health check at track transition
  -> Ollama restart uses config.GPU_AVAILABLE=True (cached at import time)
  -> Ollama restarts in GPU mode -> GPU initialization fails again
  -> 60s wait -> 3 restart attempts -> all fail
  -> Nested retry loops -> hours of drift
  -> SSH session timeout -> connection drops
```

**Fix (3-Layer Defense):**

| Layer | File | Change |
|-------|------|--------|
| Origin | `evafrill_runner.py` | On `model.to(cuda:0)` failure: `del model` + `gc.collect()` + `torch.cuda.synchronize()` + `empty_cache()` cleanup. `gpu_is_healthy()` (nvidia-smi) diagnoses driver corruption |
| Recovery | `runner.py` | `_restart_ollama()` checks GPU state **dynamically** via `_gpu_healthy_now()` before restart (instead of static `config.GPU_AVAILABLE`). If GPU is dead: tries `nvidia-smi --gpu-reset`, falls back to `CUDA_VISIBLE_DEVICES=""` CPU mode. `switch_model()` calls `evafrill_runner.unload_model()` for CUDA cleanup after EVAFRILL failure, then attempts GPU reset **only if `_gpu_healthy_now()` detects driver corruption** |
| Transition | `run_evaluation.py` | GPU health check added at track-to-track cooldown. On anomaly: GPU reset attempted, then `_restart_ollama()` is called **regardless of reset outcome** (Ollama needs a fresh start after GPU corruption even if reset succeeds) |

**Post-Fix Behavior:**

```
EVAFRILL model.to(cuda:0) fails
  -> del model + gc.collect() + CUDA cleanup
  -> nvidia-smi checks GPU state
  -> GPU corruption detected -> nvidia-smi --gpu-reset attempted
  -> Reset succeeds: Ollama restarts in GPU mode normally
  -> Reset fails: Ollama falls back to CPU mode (slower but evaluation continues)
```

**Key New Functions:**

| Function | File | Purpose |
|----------|------|---------|
| `_cuda_cleanup()` | `evafrill_runner.py` | gc + synchronize + empty_cache + reset_peak_memory_stats after CUDA failure (each step individually try-except wrapped) |
| `gpu_is_healthy()` | `evafrill_runner.py` | Dynamic GPU driver health check via nvidia-smi |
| `_gpu_healthy_now()` | `runner.py` | Dynamic GPU check before Ollama restart |
| `_try_gpu_reset()` | `runner.py` | Runs `nvidia-smi --gpu-reset -i 0`, returns success status |
| `switch_model()` (modified) | `runner.py` | EVAFRILL failure: calls `unload_model()` + conditional GPU reset |

**Technical Background:**
- **`cudaErrorUnknown` vs `cudaErrorMemoryAllocation`**: OOM is process-level and recoverable via `empty_cache()`. Unknown error is driver-level corruption requiring GPU reset or system reboot.
- **`config.GPU_AVAILABLE`** is evaluated once at `config.py` import time. After CUDA failure it remains `True`, causing Ollama to perpetually restart in GPU mode — the direct cause of the infinite loop.
- **`nvidia-smi --gpu-reset`** only works when no CUDA processes are running. Ollama must be killed first (`pkill`) before reset.

---

## License

Private research project.
