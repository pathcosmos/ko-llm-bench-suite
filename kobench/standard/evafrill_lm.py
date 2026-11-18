"""EVAFRILL-Mo (Mamba-2 + Attention 하이브리드) lm-evaluation-harness 백엔드.

- 원본 소스(pathcosmos/EVAFRILL-Mo)의 `model.transformer.LLM`을 그대로 사용한다 (수치 경로 변경 없음).
- 가중치는 strict 로드한다. 누락/초과 키가 있으면 실패한다 (원본 README의 strict=False는 오류를 숨긴다).
- 로그우도는 전체 forward 한 번으로 계산하므로 KV/SSM 캐시가 필요 없다 → 캐시 없는 원본으로도 정확하다.
- 문서 시작 토큰: 사전학습 주력 소스는 구분 토큰 없이 문서를 이어 붙였다(EV data/prepare.py, scripts/prepare_3b_data.sh).
  일부 소스만 EOS 구분 가능성(data/tokenize_extra.py). 그래서 기본은 붙이지 않고(doc_prefix_eos=False),
  붙일지는 Goal 2에서 dev로 정한다 (configs/standard/formats.yaml loglik_doc_start).
  빈 문맥(PPL 등)의 prefix는 EOS(id 2)다 — BOS로 학습한 적이 없다.
- 토크나이저는 formats.yaml의 korean_sp_unigram64k와 SHA가 같아야 한다 (로드 시 검사).
- 생성(generate_until)은 증분 디코딩 엔진(evafrill_engine.py)으로 한다.

등록 이름: "evafrill"  (lm_eval --model evafrill  또는 kobench.standard.run)
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

EOS_ID = 2  # </s> (tokenizer.json: <unk>=0, <s>=1, </s>=2, <pad>=3)


def _resolve_snapshot(path: str, revision: str | None, subfolder: str) -> Path:
    p = Path(path)
    if not p.exists():
        from huggingface_hub import snapshot_download

        allow = [f"{subfolder}/*", "tokenizer*.json"] if subfolder else ["*.json", "model.safetensors"]
        p = Path(snapshot_download(path, revision=revision, allow_patterns=allow))
    return p / subfolder if subfolder else p


def load_evafrill(model_dir: Path, src_dir: str, device: str, dtype: torch.dtype):
    """원본 소스로 모델을 만들고 safetensors를 strict 로드한다. (model, tokenizer, info) 반환."""
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from model.config import LMConfig  # type: ignore[import-not-found]
    from model.transformer import LLM  # type: ignore[import-not-found]
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    raw = json.loads((model_dir / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(LMConfig)}
    cfg = LMConfig.from_dict({k: v for k, v in raw.items() if k in fields})
    dropped = sorted(set(raw) - fields)

    model = LLM(cfg)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    # 원본 LLM.from_pretrained와 같은 처리: TransformerEngine(FP8) 키(ffn.fc1_weight 등)면 표준 키로 변환
    # (pretrain 체크포인트가 이 형식이다. README의 strict=False 로드는 이때 FFN 가중치를 조용히 빠뜨린다)
    has_te = any("_extra_state" in k or ".fc1_weight" in k for k in state)
    has_std = any(".gate_proj.weight" in k for k in state)
    te_converted = has_te and not has_std
    if te_converted:
        state = LLM._convert_te_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # RoPE 버퍼 등 비영속 버퍼는 state dict에 없을 수 있다 → 파라미터 누락만 오류로 본다
    param_names = {n for n, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    if missing_params or unexpected:
        raise RuntimeError(f"EVAFRILL 가중치 불일치: missing={missing_params[:10]} unexpected={list(unexpected)[:10]}")
    model = model.to(device=device, dtype=dtype).eval()

    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        tok_path = model_dir.parent / "tokenizer.json"
    from kobench.standard import formats

    want = formats.load_formats()["tokenizers"]["korean_sp_unigram64k"]["sha256"]
    got = formats._sha256(tok_path)
    if got != want:
        raise RuntimeError(f"EVAFRILL tokenizer SHA 불일치: {tok_path} {got[:12]} != 학습 토크나이저 {want[:12]}")
    tok = Tokenizer.from_file(str(tok_path))
    info = {
        "te_state_dict_converted": te_converted,
        "config_dropped_keys": dropped,
        "missing_buffers": [k for k in missing if k not in param_names],
        "n_params": sum(p.numel() for p in model.parameters()),
        "tokenizer": str(tok_path),
    }
    return model, tok, info


@register_model("evafrill")
class EvafrillLM(TemplateLM):
    def __init__(
        self,
        pretrained: str,
        revision: str | None = None,
        subfolder: str = "",
        src_dir: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int | str = 16,
        max_length: int = 4096,
        max_batch_tokens: int = 65536,
        doc_prefix_eos: bool = False,
        gen_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self._gen_batch_size = int(gen_batch_size)
        src_dir = src_dir or os.environ.get("EVAFRILL_SRC") or str(Path(os.environ.get("LT_SRC", "")) / "EVAFRILL-Mo")
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)
        model_dir = _resolve_snapshot(pretrained, revision, subfolder)
        self.model, self.tokenizer_obj, self.load_info = load_evafrill(model_dir, src_dir, device, self._dtype)
        self._max_length = int(max_length)
        self._batch_size = 16 if batch_size == "auto" else int(batch_size)
        self.max_batch_tokens = int(max_batch_tokens)
        self.doc_prefix_eos = bool(doc_prefix_eos) if not isinstance(doc_prefix_eos, str) else doc_prefix_eos.lower() == "true"
        self.model_dir = str(model_dir)

    # ── TemplateLM 인터페이스 ─────────────────────────────────────────────
    @property
    def eot_token_id(self) -> int:
        return EOS_ID

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        ids = self.tokenizer_obj.encode(string).ids
        if self.doc_prefix_eos and add_special_tokens is not False:
            ids = [EOS_ID] + ids
        return ids

    def tok_decode(self, tokens) -> str:
        return self.tokenizer_obj.decode(list(tokens))

    @torch.no_grad()
    def _forward_logprobs(self, batch_ids: list[list[int]]) -> list[torch.Tensor]:
        """오른쪽 패딩 배치 forward → 각 시퀀스의 log-softmax (len, V) float32 (패딩 제외)."""
        T = max(len(x) for x in batch_ids)
        inp = torch.full((len(batch_ids), T), EOS_ID, dtype=torch.long)
        for i, x in enumerate(batch_ids):
            inp[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        logits, _ = self.model(inp.to(self._device))
        return [logits[i, : len(x)] for i, x in enumerate(batch_ids)]

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, override_bs=None, **kwargs):
        # (원래 인덱스, ctx, cont) — 긴 것부터 묶어 패딩 낭비를 줄인다 (인과 모델이라 오른쪽 패딩은 앞 위치에 영향 없음)
        items = []
        for idx, (_, ctx, cont) in enumerate(requests):
            whole = (ctx + cont)[-(self.max_length + 1):]
            inp = whole[:-1]
            n_cont = len(cont)
            items.append((idx, inp, whole[-n_cont:], n_cont))
        order = sorted(range(len(items)), key=lambda i: -len(items[i][1]))
        results: list[tuple[float, bool] | None] = [None] * len(items)

        pbar = tqdm(total=len(items), disable=disable_tqdm, desc="evafrill loglikelihood")
        i = 0
        while i < len(order):
            longest = len(items[order[i]][1])
            n = max(1, min(override_bs or 10**9, self.max_batch_tokens // max(longest, 1)))
            chunk = order[i : i + n]
            outs = self._forward_logprobs([items[j][1] for j in chunk])
            for j, lg in zip(chunk, outs, strict=True):
                _, inp, cont, n_cont = items[j]
                lp = F.log_softmax(lg[-n_cont:].float(), dim=-1)  # 마지막 n_cont 위치가 연속 토큰을 예측
                tgt = torch.tensor(cont, device=lp.device)
                logprob = lp.gather(1, tgt[:, None]).sum().item()
                greedy = bool((lp.argmax(-1) == tgt).all().item())
                results[j] = (logprob, greedy)
            pbar.update(len(chunk))
            i += len(chunk)
        pbar.close()
        return results  # type: ignore[return-value]

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        out = []
        for (string,) in tqdm([r.args for r in requests], disable=disable_tqdm, desc="evafrill rolling"):
            windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string, add_special_tokens=False),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            reqs = [(None, c, t) for c, t in windows]
            out.append(sum(x[0] for x in self._loglikelihood_tokens(reqs, disable_tqdm=True)))
        return out

    # ── 생성 (증분 디코딩 엔진) ───────────────────────────────────────────
    @property
    def tokenizer_name(self) -> str:
        return "evafrill-korean_sp_unigram64k"

    def chat_template(self, chat_template: bool | str = False) -> str:
        return "user_assistant_v1"

    def apply_chat_template(self, chat_history: list[dict], add_generation_prompt: bool = True) -> str:
        """학습 형식 프롬프트 (configs/standard/formats.yaml user_assistant_v1, system은 학습처럼 버림)."""
        from kobench.standard import formats

        return formats.build_prompt("evafrill-mo-3b", chat_history)

    def _get_engine(self, max_len: int):
        from kobench.standard.evafrill_engine import EvafrillEngine

        eng = getattr(self, "_engine", None)
        if eng is None or eng.max_len < max_len:
            self._engine = EvafrillEngine(self.model, max_batch=self.gen_batch_size, max_len=max_len)
        return self._engine

    @property
    def gen_batch_size(self) -> int:
        return int(getattr(self, "_gen_batch_size", 32))

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        """lm-eval 생성 요청. greedy가 기본이고, do_sample=True면 temperature/top_p 샘플링(seed 고정)."""
        import json as _json
        from collections import defaultdict

        groups = defaultdict(list)
        for i, r in enumerate(requests):
            ctx, kw = r.args
            groups[_json.dumps(kw, sort_keys=True, default=str)].append((i, ctx, dict(kw)))
        out: list[str | None] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="evafrill generate")
        for items in groups.values():
            kw = items[0][2]
            until = kw.get("until") or []
            until = [until] if isinstance(until, str) else list(until)
            max_gen = int(kw.get("max_gen_toks", 256))
            do_sample = bool(kw.get("do_sample", False))
            temperature = float(kw.get("temperature", 0.0)) if do_sample else 0.0
            top_p = float(kw.get("top_p", 1.0)) if do_sample else 1.0
            keep = self.max_length - max_gen
            enc = [(i, self.tok_encode(ctx, add_special_tokens=False)[-keep:]) for i, ctx, _ in items]
            enc.sort(key=lambda t: -len(t[1]))
            eng = self._get_engine(max(len(e) for _, e in enc) + max_gen + 1)
            for s in range(0, len(enc), self.gen_batch_size):
                chunk = enc[s: s + self.gen_batch_size]
                res = eng.generate([e for _, e in chunk], max_gen, eos_ids=(EOS_ID,),
                                   stop_strings=until or None, decode_fn=self.tok_decode if until else None,
                                   temperature=temperature, top_p=top_p, seed=1234 if do_sample else None)
                for (i, _), r in zip(chunk, res):
                    out[i] = self.postprocess_generation(r.tokens, until)
                pbar.update(len(chunk))
        pbar.close()
        return out  # type: ignore[return-value]

    def postprocess_generation(self, tokens: list[int], until: list[str]) -> str:
        """EOS 이후를 버리고 디코딩한 뒤, lm-eval 관례대로 첫 정지 문자열 앞에서 자른다."""
        if EOS_ID in tokens:
            tokens = tokens[: tokens.index(EOS_ID)]
        text = self.tok_decode(tokens)
        for s in until:
            if s:
                text = text.split(s)[0]
        return text
