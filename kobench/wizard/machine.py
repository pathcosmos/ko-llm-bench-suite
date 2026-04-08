"""머신 연결 테스트 유틸리티."""
import socket
import subprocess
import requests
from kobench import config


def test_ollama(url=None, timeout=5):
    """Ollama 서버 연결 테스트.
    Returns: {connected: bool, version: str, models: list, error: str}
    """
    url = url or config.OLLAMA_BASE_URL
    result = {"url": url, "connected": False, "version": "", "models": [], "error": None}
    try:
        r = requests.get(url, timeout=timeout)
        result["connected"] = r.status_code == 200
        result["version"] = r.text.strip() if r.status_code == 200 else ""
        # Get models
        tags = requests.get(f"{url}/api/tags", timeout=timeout)
        if tags.status_code == 200:
            for m in tags.json().get("models", []):
                result["models"].append({
                    "name": m.get("name", ""),
                    "size_gb": m.get("size", 0) / 1e9,
                })
    except requests.ConnectionError:
        result["error"] = "연결 거부 (Ollama 미실행?)"
    except requests.Timeout:
        result["error"] = f"타임아웃 ({timeout}초)"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_port(host, port, timeout=3):
    """TCP 포트 연결 테스트."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_gpu_info():
    """GPU 정보 조회. Returns: {available: bool, name: str, vram_total: str, vram_free: str}"""
    result = {"available": False, "name": "", "vram_total": "", "vram_free": ""}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = out.stdout.strip().split(", ")
            result["available"] = True
            result["name"] = parts[0] if len(parts) > 0 else ""
            result["vram_total"] = f"{int(parts[1])}MB" if len(parts) > 1 else ""
            result["vram_free"] = f"{int(parts[2])}MB" if len(parts) > 2 else ""
    except Exception:
        pass
    return result


def get_python_info():
    """Python 환경 정보."""
    import sys
    return {"version": sys.version.split()[0], "executable": sys.executable}


def check_dependencies():
    """필수 Python 패키지 확인. Returns: {name: installed_bool}"""
    deps = {}
    for pkg in ["requests", "pyyaml", "numpy", "scipy", "matplotlib", "pandas", "seaborn", "rich"]:
        module = pkg.replace("pyyaml", "yaml")
        try:
            __import__(module)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    return deps
