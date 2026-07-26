"""Model + tokenizer + xgrammar compiler loading, with the vocab-size pitfall
handled once, here.

The pitfall: a model's logit width (config.vocab_size) can exceed the tokenizer
vocab (tokenizer.vocab_size) because the LM head is padded. xgrammar allocates
its bitmask against the logit width, so TokenizerInfo and the bitmask must both
use config.vocab_size, NOT tokenizer.vocab_size. Getting this wrong silently
misaligns the mask and corrupts every metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import xgrammar as xgr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    compiler: xgr.GrammarCompiler
    tokenizer_info: xgr.TokenizerInfo
    full_vocab: int          # logit width == config.vocab_size
    device: torch.device
    model_id: str
    revision: str | None


def load(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    max_threads: int = 8,
) -> LoadedModel:
    torch_dtype = getattr(torch, dtype)
    print(f"[load] tokenizer + config: {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    print("[load] weights -> GPU (first run downloads ~15GB; watch nvidia-smi) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()

    # The one line that matters: logit width, not tokenizer vocab.
    full_vocab = int(cfg.vocab_size)
    print(f"[load] building xgrammar TokenizerInfo (vocab={full_vocab}; ~30-60s) ...", flush=True)
    tinfo = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=full_vocab)
    compiler = xgr.GrammarCompiler(tinfo, max_threads=max_threads)
    print("[load] ready.", flush=True)

    return LoadedModel(
        model=model,
        tokenizer=tok,
        compiler=compiler,
        tokenizer_info=tinfo,
        full_vocab=full_vocab,
        device=torch.device(device),
        model_id=model_id,
        revision=revision,
    )
