"""lm-eval LM: GGUF 모델의 로그우도를 llama.cpp(libllama)로 계산한다 (lm-eval 이름 `llamacpp_ll`).

Ollama·llama-server는 프롬프트 토큰 logprob을 주지 않는다. 그래서 libllama에 직접 연결한 도구
`$LT_BIN/llama_loglik`(kobench/standard/native/llama_loglik.cpp, env/setup.sh llamacpp 단계에서 빌드)에
토큰 ID를 넣어 계산한다. 토큰화는 BF16(vLLM) 실행과 같은 학습 토크나이저(formats.encode_text)로 하고,
문맥 자르기·rolling 창은 lm-eval VLLM 구현과 같게 둔다 → 같은 문항을 BF16과 문항 단위로 짝지어 비교할 수 있다.
생성(generate_until)은 지원하지 않는다 (객관식·PPL 전용).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window as _mdw


def loglik_binary() -> str:
    return os.environ.get("LLAMA_LOGLIK", str(Path(os.environ.get("LT_BIN", "/workspace/llm-test/bin")) / "llama_loglik"))


@register_model("llamacpp_ll")
class LlamaCppLM(TemplateLM):
    def __init__(self, gguf: str, format_id: str, max_length: int = 4096, n_ctx: int = 65536, n_batch: int = 4096, **kwargs):
        super().__init__()
        from kobench.standard import formats

        self.gguf = gguf
        self.format_id = format_id
        self._max_length = int(max_length)
        self.n_ctx, self.n_batch = int(n_ctx), int(n_batch)
        fmt = formats.model_format(format_id)
        self._eot = int(fmt["stop_token_ids"][0])
        self._formats = formats
        self.load_info = {"gguf": gguf, "format_id": format_id, "binary": loglik_binary(), "tokenizer": fmt["tokenizer"],
                          "gguf_size": os.path.getsize(gguf)}
        self.timings: list[dict] = []

    @property
    def eot_token_id(self) -> int:
        return self._eot

    @property
    def prefix_token_id(self) -> int:      # 사용자 모델은 BOS로 학습하지 않았다 → 빈 문맥 prefix = EOS (vLLM 실행과 같음)
        return self._eot

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        return self._formats.encode_text(self.format_id, string)

    def tok_decode(self, tokens) -> str:
        return self._formats.training_tokenizer(self.format_id).decode(list(tokens))

    def _run(self, groups: list[tuple[list[int], list[list[int]]]]) -> list[dict]:
        with tempfile.TemporaryDirectory(prefix="llama_loglik_") as d:
            inp, out = Path(d) / "in.jsonl", Path(d) / "out.jsonl"
            with open(inp, "w") as f:
                for ctx, conts in groups:
                    f.write(json.dumps({"ctx": ctx, "conts": conts}) + "\n")
            t0 = time.time()
            subprocess.run([loglik_binary(), "-m", self.gguf, "-i", str(inp), "-o", str(out),
                            "--n-ctx", str(self.n_ctx), "--n-batch", str(self.n_batch)], check=True)
            self.timings.append({"groups": len(groups), "seconds": round(time.time() - t0, 1),
                                 "tokens": sum(len(c) + sum(map(len, cs)) for c, cs in groups)})
            return [json.loads(l) for l in open(out)]

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
        max_ctx = self.max_length - 1                # lm-eval VLLM과 같은 자르기
        gid: dict[tuple, int] = {}                  # 같은 문맥의 연속은 한 요청으로 묶는다 (문맥 KV 한 번 계산)
        groups: list[tuple[list[int], list[list[int]]]] = []
        where: list[tuple[int, int]] = []
        for _, ctx, cont in requests:
            whole = (ctx + cont)[-max_ctx:]
            c = whole[: len(whole) - len(cont)]
            if not c:
                raise ValueError("문맥이 비었다 (연속이 최대 길이 이상)")
            k = tuple(c)
            g = gid.get(k)
            # 같은 문맥이라도 묶음이 커지면 새 묶음 (rolling PPL의 첫 창들은 모두 문맥 = prefix 토큰 하나라 수백 개가 합쳐진다;
            # llama_loglik은 묶음당 시퀀스 255개·KV n_ctx 셀 예산을 넘으면 실패한다 — 2026-11-20 첫 GGUF PPL 실행 실패 원인)
            if g is not None and (len(groups[g][1]) >= 32 or
                                  len(c) + sum(map(len, groups[g][1])) + len(cont) > self.n_ctx // 2):
                g = None
            if g is None:
                g = gid[k] = len(groups)
                groups.append((list(c), []))
            groups[g][1].append(list(cont))
            where.append((g, len(groups[g][1]) - 1))
        res = self._run(groups)
        return [(res[g]["ll"][j], bool(res[g]["greedy"][j])) for g, j in where]

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        all_w, counts = [], []
        for (string,) in [r.args for r in requests]:
            w = list(map(_mdw, get_rolling_token_windows(token_list=self.tok_encode(string), prefix_token=self.prefix_token_id,
                                                         max_seq_len=self.max_length - 2, context_len=1)))
            all_w += [(None, c, t) for c, t in w]
            counts.append(len(w))
        lls = [x[0] for x in self._loglikelihood_tokens(all_w)]
        out, i = [], 0
        for n in counts:
            out.append(sum(lls[i: i + n]))
            i += n
        return out

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("llamacpp_ll은 로그우도 전용")
