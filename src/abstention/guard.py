"""Optional guard-model screen for the HARMFUL proxies (Llama-Guard-3 default).

Only loaded when experiments/label.py is called with --guard. The heuristic
labeler is the primary target; the guard is a fallback that downgrades heuristic
HARMFUL calls the guard judges safe, tightening precision on the positive class.
Kept tiny and dependency-light (transformers only).
"""
from __future__ import annotations

import torch


class GuardModel:
    def __init__(self, model_id: str = "meta-llama/Llama-Guard-3-8B", device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.device = device

    @torch.no_grad()
    def is_harmful(self, prompt: str, response: str) -> bool:
        """True iff the guard rates the (prompt, response) pair unsafe.

        Uses Llama-Guard's chat template, which emits 'safe' / 'unsafe' as the
        first generated line."""
        chat = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        ids = self.tok.apply_chat_template(chat, return_tensors="pt").to(self.device)
        out = self.model.generate(ids, max_new_tokens=8, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        verdict = self.tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
        return "unsafe" in verdict.lower()
