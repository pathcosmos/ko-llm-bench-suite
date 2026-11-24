"""vllm serve 진입점 + transformers LlamaConfig head_dim 검사 완화 (models.yaml relax_head_dim_check 모델용, run.py와 같은 이유).

완화는 모듈 최상위에서 적용한다: vLLM 엔진 코어는 spawn으로 뜨고 이 모듈을 __mp_main__으로 다시 import하므로
하위 프로세스에도 같은 완화가 걸린다. 서버 실행은 __main__ 보호 안에서만 한다(2026-11-21: 보호가 없어 spawn 하위 프로세스가
서버를 다시 띄우려다 'bootstrapping phase' RuntimeError로 실패).
"""
import sys

from kobench.standard.run import relax_head_dim_check

relax_head_dim_check()

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    sys.argv = ["vllm"] + sys.argv[1:]
    sys.exit(main())
