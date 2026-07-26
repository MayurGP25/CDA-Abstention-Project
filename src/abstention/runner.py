"""Shared setup used by the experiment scripts: read configs, load a model,
build the grammar, harvest (or load cached) R, and provide prompt sets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from . import data, grammars, refusal_score, refusal_set
from .model_loader import load

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_configs():
    return read_yaml(CONFIGS / "models.yaml"), read_yaml(CONFIGS / "experiment.yaml")


def get_model_spec(models_cfg, model_key):
    for m in models_cfg["models"]:
        if m["key"] == model_key:
            return m
    raise KeyError(f"model key '{model_key}' not in models.yaml")


def setup(model_key: str):
    """Load model + grammar. Returns (lm, grammar, schema, exp_cfg, models_cfg)."""
    models_cfg, exp_cfg = load_configs()
    spec = get_model_spec(models_cfg, model_key)
    lm = load(
        spec["model_id"],
        revision=spec.get("revision"),
        dtype=spec.get("dtype", "bfloat16"),
        device=models_cfg.get("device", "cuda"),
    )
    g = exp_cfg["grammar"]
    grammar, schema = grammars.build(
        lm.compiler, g["name"], n_steps=g.get("n_steps", 5), opener=g.get("opener"),
    )
    return lm, grammar, schema, exp_cfg, models_cfg


def get_refusal_ids(lm, exp_cfg, harmful_prompts) -> list[int]:
    """Harvest R once per model and cache it to results/cache/R-<key>.json."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = lm.model_id.replace("/", "__")
    path = CACHE / f"R-{key}.json"
    if path.exists():
        blob = json.loads(path.read_text())
    else:
        rs = refusal_set.harvest(lm, [p["prompt"] for p in harmful_prompts[:60]])
        blob = {
            "ids_small": rs.ids_small,
            "ids_large": rs.ids_large,
            "coverage_small": rs.coverage_small,
            "coverage_large": rs.coverage_large,
        }
        path.write_text(json.dumps(blob, indent=2))
        print(f"[R] {lm.model_id}: |small|={len(rs.ids_small)} cov={rs.coverage_small:.3f} "
              f"| |large|={len(rs.ids_large)} cov={rs.coverage_large:.3f}")
    which = exp_cfg.get("refusal_set", "large")
    return blob[f"ids_{which}"]


def get_refusal_prefix_ids(lm):
    """Token-id lists for the multi-token refusal prefixes (sequence-level S_R)."""
    return refusal_score.encode_prefixes(lm.tokenizer)


def sr_stride(exp_cfg) -> int:
    return int(exp_cfg.get("refusal_score", {}).get("stride", 4))


def harmful_prompts(exp_cfg, bench: str):
    return data.load_harmful(bench, exp_cfg["harmful"][bench], seed=exp_cfg["seed"])


def benign_prompts(exp_cfg):
    return data.load_xstest_safe(exp_cfg["benign_n"], seed=exp_cfg["seed"])


def results_dir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d
