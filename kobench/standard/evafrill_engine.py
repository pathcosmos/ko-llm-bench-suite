"""EVAFRILL-Mo 증분 디코딩 엔진 (원본 소스 수정 없음, 래퍼).

원본 model/transformer.py의 forward에는 캐시가 없어 생성 시 토큰마다 전체 시퀀스를 다시 계산한다(O(n^2)).
이 엔진은
  - prefill: 원본 블록 연산을 같은 커널·같은 순서로 수행하면서 마지막 상태를 꺼낸다
      Mamba 블록 → mamba_chunk_scan_combined(return_final_states) + conv 입력의 마지막 K-1개
      어텐션 블록 → RoPE 적용된 K/V (원본 apply_rotary_emb, _standard_attention 그대로 호출)
  - decode: causal_conv1d_update + selective_state_update(z, D, dt_bias, dt_softplus) + KV 캐시 어텐션
      (어텐션 수치 경로는 원본 _standard_attention과 같다: bf16 matmul → /sqrt(d) → fp32 softmax → bf16 → matmul)
  - 배치: 시퀀스별 위치·길이 독립. 상태는 미리 할당하고, 고정 배치 decode 단계는 CUDA graph로 캡처한다.

EVAFRILL 블록 구조(표준 Mamba-2와 다름): conv1d는 x에만, 출력 y·silu(z) (RMSNormGated 없음), 블록마다 pre-norm SwiGLU.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


@dataclass
class GenResult:
    tokens: list[int]
    finish_reason: str                      # "eos" | "stop" | "length"
    prompt_len: int
    logits: list[torch.Tensor] = field(default_factory=list)   # record_logits=True일 때 단계별 (V,) bf16


class EvafrillEngine:
    def __init__(self, model, max_batch: int = 32, max_len: int = 4608, state_dtype=torch.float32,
                 batch_invariant: bool = True):
        m = model.eval()
        self.m = m
        self.batch_invariant = batch_invariant
        self.types = list(m._layer_types)
        self.device = next(m.parameters()).device
        self.dtype = next(m.parameters()).dtype
        self.mamba_layers = [l for l, t in zip(m.layers, self.types) if t == "M"]
        self.attn_layers = [l for l, t in zip(m.layers, self.types) if t == "A"]
        L0 = self.mamba_layers[0]
        self.H, self.P, self.N, self.G = L0.n_heads, L0.head_dim, L0.d_state, L0.n_groups
        self.d_inner, self.K = L0.d_inner, L0.conv1d.kernel_size[0]
        a0 = self.attn_layers[0].attn
        self.nh, self.nkv, self.hd = a0.n_heads, a0.n_kv_heads, a0.head_dim
        self.n_rep = self.nh // self.nkv
        self.max_batch, self.max_len = max_batch, max_len
        # 원본 모듈 함수·플래그 (원본 경로가 CUDA 커널을 쓰는지 확인).
        # sys.modules가 아니라 클래스가 정의된 모듈의 전역 공간을 직접 참조한다 (모듈 교체·mock에 영향받지 않게)
        self._mmod = SimpleNamespace(**type(L0).forward.__globals__)
        self._amod = SimpleNamespace(**type(a0).forward.__globals__)
        if not (self._mmod.HAS_MAMBA_KERNEL is True and self._mmod.HAS_CAUSAL_CONV1D is True):
            raise RuntimeError("원본 Mamba 블록이 CUDA 커널을 쓰지 않는다 (mamba_ssm/causal_conv1d 필요)")
        # RoPE 표: 원본 RotaryEmbedding.forward와 같은 값을 쓴다.
        #  - 원본은 길이 <= max_seq_len(4096)이면 캐시 버퍼를 쓰는데, model.to(bf16)에서 이 버퍼도 bf16으로 변환된다.
        #  - 길이 > max_seq_len이면 fp32 표를 그때 새로 만든다.
        # prefill은 원본 m.rope(T)를 그대로 호출하고, decode는 위치 < max_seq_len이면 캐시 표, 이상이면 fp32 표.
        # (한계: 4096을 넘는 순간 원본은 앞쪽 K까지 fp32 표로 다시 계산하지만, 캐시는 다시 계산할 수 없다)
        self.rope_max = m.rope.max_seq_len
        self.cos_m = m.rope._cos_cached.to(self.device)
        self.sin_m = m.rope._sin_cached.to(self.device)
        cos, sin = m.rope._build_tables(m.rope.dim, max(max_len, self.rope_max), m.rope.theta)
        self.cos, self.sin = cos.to(self.device), sin.to(self.device)
        nM, nA = len(self.mamba_layers), len(self.attn_layers)
        dev, bf = self.device, self.dtype
        self.conv = torch.zeros(nM, max_batch, self.d_inner, self.K - 1, device=dev, dtype=bf)
        self.ssm = torch.zeros(nM, max_batch, self.H, self.P, self.N, device=dev, dtype=state_dtype)
        self.kc = torch.zeros(nA, max_batch, self.nkv, self.hd, max_len, device=dev, dtype=bf)   # K^T 배치
        self.vc = torch.zeros(nA, max_batch, self.nkv, max_len, self.hd, device=dev, dtype=bf)
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._arange = torch.arange(max_len, device=dev)

    # ── prefill (시퀀스 하나, 원본 연산 순서 그대로) ──────────────────────────
    @torch.no_grad()
    def prefill(self, ids: list[int] | torch.Tensor, slot: int, all_logits: bool = False) -> torch.Tensor:
        m = self.m
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)[None]
        T = ids.shape[1]
        if T + 1 > self.max_len:
            raise ValueError(f"프롬프트 {T} > max_len {self.max_len}")
        x = m.embedding(ids)
        cos, sin = m.rope(T, self.device)                        # 원본과 같은 표 (길이에 따라 bf16 캐시 / fp32)
        mi = ai = 0
        for layer, t in zip(m.layers, self.types):
            if t == "M":
                x = self._mamba_prefill(layer, x, mi, slot)
                mi += 1
            else:
                x = self._attn_prefill(layer, x, cos, sin, ai, slot, T)
                ai += 1
        x = m.norm(x)
        logits = m.lm_head(x if all_logits else x[:, -1:])
        return logits[0]

    def _mamba_prefill(self, layer, x, mi, slot):
        # == model/mamba_block.py Mamba2Block.forward (CUDA 경로) + 마지막 상태 추출
        residual = x
        h = layer.norm(x)
        proj = layer.in_proj(h)
        z, x_ssm, Bm, Cm, dt_raw = layer._split_projection(proj)
        xt = x_ssm.transpose(1, 2)
        x_conv = causal_conv1d_fn(xt, layer.conv1d.weight.squeeze(1), bias=layer.conv1d.bias,
                                  activation="silu").transpose(1, 2)
        T, K1 = xt.shape[-1], self.K - 1
        st = xt[:, :, T - K1:] if T >= K1 else F.pad(xt, (K1 - T, 0))
        self.conv[mi, slot].copy_(st[0])
        b, L, _ = x_conv.shape
        x_heads = x_conv.reshape(b, L, layer.n_heads, layer.head_dim)
        A = -layer.A_log.exp()
        y, final = mamba_chunk_scan_combined(
            x_heads, dt_raw, A, Bm, Cm, chunk_size=layer.chunk_size, D=layer.D,
            z=z.reshape(b, L, layer.n_heads, layer.head_dim), dt_bias=layer.dt_bias, dt_softplus=True,
            return_final_states=True)
        self.ssm[mi, slot].copy_(final[0])
        y = y.reshape(b, L, layer.d_inner)
        x = residual + layer.out_proj(y)
        if layer.ffn is not None:
            x = x + layer.ffn(layer.ffn_norm(x))
        return x

    def _attn_prefill(self, layer, x, cos, sin, ai, slot, T):
        # == model/transformer.py TransformerBlock.forward + attention.py MultiHeadAttention.forward + KV 추출
        a, am = layer.attn, self._amod
        h = layer.attn_norm(x)
        qkv = a.qkv_proj(h)
        q, k, v = qkv.split([a._q_dim, a._kv_dim, a._kv_dim], dim=-1)
        q = q.view(1, T, a.n_heads, a.head_dim)
        k = k.view(1, T, a.n_kv_heads, a.head_dim)
        v = v.view(1, T, a.n_kv_heads, a.head_dim)
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        q = am.apply_rotary_emb(q, cos, sin)
        k = am.apply_rotary_emb(k, cos, sin)
        self.kc[ai, slot, :, :, :T].copy_(k[0].permute(1, 2, 0))
        self.vc[ai, slot, :, :T].copy_(v[0].permute(1, 0, 2))
        if a.use_flash and am.HAS_FLASH_ATTN and x.is_cuda:
            out = a._flash_attention(q, k, v, 1, T)
        else:
            out = a._standard_attention(q, k, v, 1, T)
        x = x + a.out_proj(out)
        x = x + layer.ffn(layer.ffn_norm(x))
        return x

    # ── decode 단계 (배치, CUDA graph 캡처 가능) ───────────────────────────
    def _step(self, tok: torch.Tensor, pos: torch.Tensor, B: int, Lb: int) -> torch.Tensor:
        m = self.m
        x = m.embedding(tok)[:, None, :]                        # (B,1,C)
        pm = pos.clamp(max=self.rope_max - 1)
        rope = (self.cos_m[pm][:, None, None, :], self.sin_m[pm][:, None, None, :],     # 위치 < 4096: 원본 캐시 표
                self.cos[pos][:, None, None, :], self.sin[pos][:, None, None, :],       # 위치 >= 4096: fp32 표
                (pos < self.rope_max)[:, None, None, None])
        mi = ai = 0
        for layer, t in zip(m.layers, self.types):
            if t == "M":
                x = self._mamba_step(layer, x, mi, B)
                mi += 1
            else:
                x = self._attn_step(layer, x, rope, ai, pos, B, Lb)
                ai += 1
        x = m.norm(x)
        return m.lm_head(x)[:, 0]                                # (B,V)

    def _mamba_step(self, layer, x, mi, B):
        H, P = layer.n_heads, layer.head_dim
        residual = x
        h = layer.norm(x)
        proj = layer.in_proj(h)
        z, x_ssm, Bm, Cm, dt_raw = layer._split_projection(proj)
        x_conv = causal_conv1d_update(x_ssm[:, 0].contiguous(), self.conv[mi, :B],
                                      layer.conv1d.weight.squeeze(1), layer.conv1d.bias, activation="silu")
        A = (-layer.A_log.exp())[:, None, None].expand(H, P, layer.d_state)
        dt = dt_raw[:, 0, :, None].expand(B, H, P)
        dt_bias = layer.dt_bias[:, None].expand(H, P)
        D = layer.D[:, None].expand(H, P)
        y = selective_state_update(self.ssm[mi, :B], x_conv.view(B, H, P), dt, A, Bm[:, 0], Cm[:, 0],
                                   D=D, z=z[:, 0].reshape(B, H, P), dt_bias=dt_bias, dt_softplus=True)
        y = y.reshape(B, 1, layer.d_inner)
        x = residual + layer.out_proj(y)
        if layer.ffn is not None:
            x = x + layer.ffn(layer.ffn_norm(x))
        return x

    def _attn_step(self, layer, x, rope, ai, pos, B, Lb):
        a = layer.attn
        h = layer.attn_norm(x)
        qkv = a.qkv_proj(h)
        q, k, v = qkv.split([a._q_dim, a._kv_dim, a._kv_dim], dim=-1)
        q = q.view(B, 1, self.nh, self.hd)
        k = k.view(B, 1, self.nkv, self.hd)
        v = v.view(B, 1, self.nkv, self.hd)
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        cm, sm, c32, s32, inside = rope
        q = torch.where(inside, self._rope(q, cm, sm), self._rope(q, c32, s32))
        k = torch.where(inside, self._rope(k, cm, sm), self._rope(k, c32, s32))
        bidx = self._arange[:B]
        kc, vc = self.kc[ai, :B], self.vc[ai, :B]
        kc[bidx, :, :, pos] = k[:, 0]                            # (B,nkv,hd) at position pos_b
        vc[bidx, :, pos] = v[:, 0]
        qh = q[:, 0].reshape(B, self.nkv, self.n_rep, self.hd)  # 헤드 h → kv h//n_rep (repeat_interleave와 같은 대응)
        scores = torch.matmul(qh, kc[..., :Lb]) / math.sqrt(self.hd)       # (B,nkv,n_rep,Lb) bf16
        mask = self._arange[:Lb][None, :] > pos[:, None]                     # 미래·빈 위치
        scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
        w = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(w, vc[:, :, :Lb])                                 # (B,nkv,n_rep,hd)
        out = out.reshape(B, 1, self.nh * self.hd)
        x = x + a.out_proj(out)
        x = x + layer.ffn(layer.ffn_norm(x))
        return x

    @staticmethod
    def _rope(x, cos, sin):
        # attention.apply_rotary_emb와 같은 식 (cos/sin만 배치별 위치)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)

    # ── CUDA graph ─────────────────────────────────────────────────────
    def _graph(self, B: int, Lb: int):
        key = (B, Lb)
        if key in self._graphs:
            return self._graphs[key]
        tok = torch.zeros(B, dtype=torch.long, device=self.device)
        pos = torch.zeros(B, dtype=torch.long, device=self.device)
        # 상태를 보존한 채 워밍업·캡처한다 (캡처 중 커널이 상태를 갱신하므로 앞뒤로 백업·복원)
        backup = [t[:, :B].clone() for t in (self.conv, self.ssm, self.kc, self.vc)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._step(tok, pos, B, Lb)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = self._step(tok, pos, B, Lb)
        for t, b in zip((self.conv, self.ssm, self.kc, self.vc), backup):
            t[:, :B].copy_(b)
        self._graphs[key] = (g, tok, pos, out)
        return self._graphs[key]

    # ── 생성 ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, eos_ids=(2,), stop_strings=None,
                 decode_fn=None, use_graph: bool = True, record_logits: bool = False,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int | None = None,
                 timings: dict | None = None, repetition_penalty: float = 1.0) -> list[GenResult]:
        B = len(prompts)
        if not 1 <= B <= self.max_batch:
            raise ValueError(f"배치 {B}는 1..{self.max_batch}")
        if stop_strings and decode_fn is None:
            raise ValueError("stop_strings에는 decode_fn이 필요하다")
        lens = [len(p) for p in prompts]
        need = max(lens) + max_new_tokens
        if need > self.max_len:
            raise ValueError(f"필요 길이 {need} > max_len {self.max_len}")
        # batch_invariant: decode를 항상 (max_batch 행, max_len 어텐션)으로 실행해 행별 결과가 배치 구성과 무관하게 같도록 한다.
        # (decode 연산은 모두 행 독립: GEMM 행, selective_state_update/causal_conv1d_update, 배치 어텐션)
        Lb = self.max_len if self.batch_invariant else self._bucket(need)
        Bx = self.max_batch if self.batch_invariant else B
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        sync = torch.cuda.synchronize
        sync()
        t0 = time.perf_counter()
        first = []
        for b, p in enumerate(prompts):
            first.append(self.prefill(p, b)[-1])
        last = torch.stack(first)                                  # (B,V)
        # repetition_penalty (HF 규칙: 프롬프트·생성에 나온 토큰의 로짓을 양수면 나누고 음수면 곱한다). 1.0이면 기존 경로 그대로
        seen = None
        if repetition_penalty != 1.0:
            seen = torch.zeros(B, last.shape[-1], dtype=torch.bool, device=last.device)
            for b, p in enumerate(prompts):
                seen[b, torch.as_tensor(p, dtype=torch.long, device=last.device)] = True
            last = self._penalize(last, seen, repetition_penalty)
        nxt = self._sample(last, temperature, top_p, gen)
        if Bx > B:                                                 # 빈 행은 더미 (토큰 0, 위치 0); 실제 행에 영향 없음
            nxt = torch.cat([nxt, torch.zeros(Bx - B, dtype=nxt.dtype, device=nxt.device)])
        sync()
        t_first = time.perf_counter()
        results = [GenResult(tokens=[], finish_reason="length", prompt_len=n) for n in lens]
        done = [False] * B
        pos = torch.tensor(lens + [0] * (Bx - B), dtype=torch.long, device=self.device)
        if use_graph:
            g, s_tok, s_pos, s_out = self._graph(Bx, Lb)
        eos = set(eos_ids)
        cur_logits = last
        for step in range(max_new_tokens):
            nx = nxt.tolist()
            for b in range(B):
                if done[b]:
                    continue
                results[b].tokens.append(nx[b])
                if record_logits:
                    results[b].logits.append(cur_logits[b].clone())
                if nx[b] in eos:
                    done[b], results[b].finish_reason = True, "eos"
                elif stop_strings:
                    text = decode_fn(results[b].tokens)
                    if any(s in text for s in stop_strings):
                        done[b], results[b].finish_reason = True, "stop"
            if all(done) or step == max_new_tokens - 1:
                break
            if use_graph:
                s_tok.copy_(nxt)
                s_pos.copy_(pos)
                g.replay()
                cur_logits = s_out
            else:
                cur_logits = self._step(nxt, pos, Bx, Lb)
            if seen is not None:
                seen[torch.arange(B, device=seen.device), nxt[:B]] = True
                cur_logits = torch.cat([self._penalize(cur_logits[:B], seen, repetition_penalty), cur_logits[B:]])
            nxt = self._sample(cur_logits, temperature, top_p, gen)
            pos += 1
        sync()
        t_end = time.perf_counter()
        if timings is not None:
            n_gen = sum(len(r.tokens) for r in results)
            timings.update({"ttft_s": t_first - t0, "total_s": t_end - t0, "decode_s": t_end - t_first,
                            "generated_tokens": n_gen, "bucket_len": Lb})
        return results

    BUCKETS = (256, 512, 1024, 2048, 4096, 4608, 8192)

    def _bucket(self, need: int) -> int:
        for b in self.BUCKETS:
            if need <= b <= self.max_len:
                return b
        return self.max_len

    @classmethod
    def _bucket_static(cls, need: int) -> int:
        """need 이상인 가장 작은 버킷 (엔진 max_len을 정할 때)."""
        return next(b for b in cls.BUCKETS if need <= b)

    @staticmethod
    def _penalize(logits, seen, penalty):
        lf = logits.float()
        return torch.where(seen, torch.where(lf > 0, lf / penalty, lf * penalty), lf).to(logits.dtype)

    @staticmethod
    def _sample(logits, temperature, top_p, gen):
        if temperature <= 0:
            return logits.argmax(-1)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            sp, si = probs.sort(-1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            probs = torch.zeros_like(probs).scatter(-1, si, sp)
        return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, generator=gen)[:, 0]


def original_greedy(model, prompt: list[int], max_new_tokens: int, eos_ids=(2,), record_logits=False):
    """원본 방식: 매 단계 전체 시퀀스를 원본 forward로 다시 계산 (캐시 없음)."""
    dev = next(model.parameters()).device
    seq = list(prompt)
    out, logs = [], []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(torch.tensor([seq], device=dev))
            last = logits[0, -1]
            t = int(last.argmax())
            out.append(t)
            if record_logits:
                logs.append(last.clone())
            seq.append(t)
            if t in eos_ids:
                break
    return out, logs
