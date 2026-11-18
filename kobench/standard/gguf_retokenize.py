"""frankenstallm GGUF의 토크나이저 메타데이터를 학습 토크나이저(korean_sp Unigram)로 바꾼 사본을 만든다.

배포 GGUF v2는 학습 후에 만든 byte-fallback SPM 토크나이저를 담고 있어, llama.cpp/Ollama의 문자열 토큰화가
학습과 다르다(configs/standard/formats.yaml eval_paths). 가중치 텐서는 바이트 그대로 복사하고 토크나이저 키만 바꾼다.
  - tokenizer.ggml.model = "t5" (llama.cpp UGM: Unigram 비터비, 연속 <unk> 병합)
  - 어휘 64,000 + 패딩 256 (가중치 어휘 64,256을 맞춤; CONTROL 타입이라 문자열 매칭에 쓰이지 않음)
  - BOS/EOS 자동 추가 없음, remove_extra_whitespaces=False, 정규화 없음
  - add_space_prefix=False (기본). llama.cpp UGM은 조각마다 앞에 ▁을 무조건 붙여서 공백으로 시작하는 입력에서 학습과 달라진다.
    대신 클라이언트가 formats.llamacpp_text(): "공백으로 시작하지 않으면 앞에 공백 하나"를 적용한다(HF Metaspace 규칙과 같음).
원본 파일은 수정하지 않는다.

python -m kobench.standard.gguf_retokenize --src <gguf> --out <gguf>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build(src: Path, out: Path, tokenizer_json: Path, gguf_py: Path, add_space_prefix: bool = False) -> dict:
    sys.path.insert(0, str(gguf_py))
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    T = gguf.GGUFValueType
    vocab = json.loads(tokenizer_json.read_text(encoding="utf-8"))["model"]["vocab"]
    reader = gguf.GGUFReader(str(src), "r")
    n_vocab_weights = len(reader.get_field("tokenizer.ggml.tokens").contents())
    tokens, scores, types = [], [], []
    for i, (piece, score) in enumerate(vocab):
        tokens.append(piece)
        scores.append(float(score))
        if piece in ("<pad>", "<s>", "</s>"):
            types.append(int(gguf.TokenType.CONTROL))
        elif piece == "<unk>":
            types.append(int(gguf.TokenType.UNKNOWN))
        else:
            types.append(int(gguf.TokenType.NORMAL))
    pad = n_vocab_weights - len(tokens)
    if pad < 0:
        raise ValueError(f"가중치 어휘 {n_vocab_weights} < 토크나이저 어휘 {len(tokens)}")
    for i in range(pad):
        tokens.append(f"<|kobench_unused_{i:03d}|>")
        scores.append(0.0)
        types.append(int(gguf.TokenType.CONTROL))
    assert tokens[:4] == ["<pad>", "<s>", "</s>", "<unk>"], tokens[:4]
    new = {
        "tokenizer.ggml.model": MetadataDetails(T.STRING, "t5"),
        "tokenizer.ggml.pre": MetadataDetails(T.STRING, "default"),
        "tokenizer.ggml.tokens": MetadataDetails(T.ARRAY, tokens, sub_type=T.STRING),
        "tokenizer.ggml.scores": MetadataDetails(T.ARRAY, scores, sub_type=T.FLOAT32),
        "tokenizer.ggml.token_type": MetadataDetails(T.ARRAY, types, sub_type=T.INT32),
        "tokenizer.ggml.padding_token_id": MetadataDetails(T.UINT32, 0),
        "tokenizer.ggml.bos_token_id": MetadataDetails(T.UINT32, 1),
        "tokenizer.ggml.eos_token_id": MetadataDetails(T.UINT32, 2),
        "tokenizer.ggml.unknown_token_id": MetadataDetails(T.UINT32, 3),
        "tokenizer.ggml.add_bos_token": MetadataDetails(T.BOOL, False),
        "tokenizer.ggml.add_eos_token": MetadataDetails(T.BOOL, False),
        "tokenizer.ggml.add_space_prefix": MetadataDetails(T.BOOL, add_space_prefix),
        "tokenizer.ggml.remove_extra_whitespaces": MetadataDetails(T.BOOL, False),
    }
    remove = [k for k in ("tokenizer.ggml.merges", "tokenizer.ggml.precompiled_charsmap") if reader.get_field(k)]
    arch = reader.get_field("general.architecture").contents()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    writer = gguf.GGUFWriter(str(tmp), arch=arch, endianess=reader.endianess)
    alignment = reader.get_field("general.alignment")
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    copy_with_new_metadata(reader, writer, new, remove)
    tmp.replace(out)
    return {"src": str(src), "out": str(out), "n_vocab_weights": n_vocab_weights, "tokenizer_vocab": len(vocab),
            "padding_tokens": pad, "removed_keys": remove, "add_space_prefix": add_space_prefix}


def main(argv=None) -> int:
    import os

    from kobench.standard import formats

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=str(Path(os.environ.get("LT_SRC", "")) / "llama.cpp" / "gguf-py"))
    ap.add_argument("--add-space-prefix", action="store_true",
                    help="기본 꺼짐: 클라이언트가 formats.llamacpp_text()로 앞 공백을 붙인다 (HF Metaspace와 같은 규칙)")
    a = ap.parse_args(argv)
    tok = formats.tokenizer_file("korean_sp_unigram64k")  # SHA 검증된 학습 토크나이저
    info = build(Path(a.src), Path(a.out), tok, Path(a.gguf_py), a.add_space_prefix)
    info["tokenizer_sha256"] = formats._sha256(tok)
    info["out_sha256"] = formats._sha256(Path(a.out))
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
