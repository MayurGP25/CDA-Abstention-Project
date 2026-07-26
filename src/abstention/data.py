"""Benchmark prompt loaders with fixed, logged subsampling.

Harmful sets: AdvBench, HarmBench, JailbreakBench.
Benign set:  XSTest "safe" prompts (matched controls for benign_forced).

Each loader returns a list of {"id": str, "prompt": str}. We subsample with a
fixed seed and record exact ids so runs are reproducible. HF dataset schemas
drift; loaders are defensive about column names and fall back cleanly.
"""
from __future__ import annotations

import random

from datasets import load_dataset


def _pick(colnames, *candidates):
    for c in candidates:
        if c in colnames:
            return c
    raise KeyError(f"none of {candidates} in {list(colnames)}")


def _subsample(items, n, seed):
    if n is None or n >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, n)


def load_advbench(n=200, seed=0):
    ds = load_dataset("walledai/AdvBench", split="train")
    col = _pick(ds.column_names, "prompt", "goal", "behavior")
    items = [{"id": f"advbench-{i}", "prompt": r[col]} for i, r in enumerate(ds)]
    return _subsample(items, n, seed)


def load_harmbench(n=100, seed=0):
    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    col = _pick(ds.column_names, "prompt", "behavior")
    items = [{"id": f"harmbench-{i}", "prompt": r[col]} for i, r in enumerate(ds)]
    return _subsample(items, n, seed)


def load_jailbreakbench(n=100, seed=0):
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    col = _pick(ds.column_names, "Goal", "goal", "prompt")
    items = [{"id": f"jbb-{i}", "prompt": r[col]} for i, r in enumerate(ds)]
    return _subsample(items, n, seed)


def load_xstest_safe(n=200, seed=0):
    ds = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
    typecol = _pick(ds.column_names, "type", "label")
    promptcol = _pick(ds.column_names, "prompt")
    items = [
        {"id": f"xstest-safe-{i}", "prompt": r[promptcol]}
        for i, r in enumerate(ds)
        if "safe" in str(r[typecol]).lower() and "unsafe" not in str(r[typecol]).lower()
    ]
    return _subsample(items, n, seed)


HARMFUL_LOADERS = {
    "advbench": load_advbench,
    "harmbench": load_harmbench,
    "jailbreakbench": load_jailbreakbench,
}


def load_harmful(name, n, seed=0):
    return HARMFUL_LOADERS[name](n=n, seed=seed)
