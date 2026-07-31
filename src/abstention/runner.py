"""Shared setup used by the experiment scripts: read configs, load a model,
build the grammar, harvest (or load cached) R, and provide prompt sets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from . import data, grammars, prompting, refusal_score, refusal_set
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


def setup(model_key: str, overrides: dict | None = None):
    """Load model + grammar. Returns (lm, grammar, schema, exp_cfg, models_cfg).

    `overrides` patches exp_cfg BEFORE the grammar is compiled, so ablation arms
    (a different grammar, anchor, or token budget) can be launched from the CLI
    without editing configs/experiment.yaml between runs -- which would otherwise
    make a multi-run script race against its own config file.
    """
    models_cfg, exp_cfg = load_configs()
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k == "grammar_name":
            exp_cfg["grammar"]["name"] = v
        else:
            exp_cfg[k] = v
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
        min_step_chars=g.get("min_step_chars", 60),
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
    return int(exp_cfg.get("refusal_score", {}).get("stride", 1))


def sr_window(exp_cfg) -> int:
    return int(exp_cfg.get("refusal_score", {}).get("window_after_tstar", 12))


def format_instruction(exp_cfg) -> str:
    """How much the prompt tells the model about the format (see prompting.py).
    Validated here rather than at first use, so a typo in the YAML or on the CLI
    fails before the model is loaded instead of 40 GB later."""
    lvl = str(exp_cfg.get("format_instruction", "none"))
    if lvl not in prompting.LEVELS:
        raise ValueError(
            f"format_instruction '{lvl}' not in {prompting.LEVELS}")
    return lvl


def anchor(exp_cfg) -> str:
    """Literal after which the grammar pins the imperative verb. Config-driven so
    it tracks the grammar: `affirmative_prefix` would need a different anchor."""
    return str(exp_cfg.get("anchor", "Step 1:"))


def run_fingerprint(lm, exp_cfg, models_cfg, schema, prompt_ids) -> dict:
    """Everything that, if changed, makes two sets of shards non-mergeable.

    collect.py hard-stops on mismatch rather than silently combining runs with
    different settings -- the failure mode that quietly ruins a result.

    The PROMPT SET is deliberately excluded from the digest. Shards are keyed by
    (condition, prompt_id), so running --n 50 after --n 5 simply adds shards; the
    existing ones stay valid because --n truncates a fixed-seed sample rather
    than resampling. Including the id list would make every change of --n look
    like an incompatible run. Anything that WOULD change the numbers for a given
    prompt (model, grammar, token budgets, anchor, stride, refusal set, seed) is
    still in the digest. The id list is recorded alongside it for provenance.

    BACKWARD COMPATIBILITY, deliberate. `format_instruction` enters the payload
    ONLY when it is not "none". Adding a key unconditionally would change the
    digest of every run already on the remote server -- including the finished
    depth runs behind Table 1 -- and check_fingerprint compares digests exactly,
    so those runs would refuse to resume and their shards would have to be
    thrown away. Omitting the default keeps the "none" digest byte-identical to
    what those manifests already hold, while every prompted arm still gets a
    distinct digest and therefore cannot be merged into them.
    """
    g = exp_cfg["grammar"]
    fmt = format_instruction(exp_cfg)
    payload = dict(
        model_id=lm.model_id,
        grammar_name=g["name"],
        schema=json.dumps(schema, sort_keys=True),
        max_new_tokens=exp_cfg["max_new_tokens"],
        free_max_new_tokens=exp_cfg.get("free_max_new_tokens", 8),
        anchor=anchor(exp_cfg),
        sr_stride=sr_stride(exp_cfg),
        sr_window=sr_window(exp_cfg),
        refusal_set=exp_cfg.get("refusal_set", "large"),
        prefix_set_version=refusal_score.PREFIX_SET_VERSION,
        prefix_set_hash=refusal_score.prefix_set_hash(),
        benign_source=exp_cfg.get("benign_source", "alpaca"),
        seed=exp_cfg["seed"],
    )
    if fmt != "none":
        payload["format_instruction"] = fmt
        # The rendered system turn, not just its label: at level `schema` the
        # text embeds the serialised schema, so two runs could share the label
        # and differ in what the model actually read.
        payload["system_prompt"] = prompting.system_prompt(fmt, schema)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    # prompt_ids is provenance only -- recorded, not gated (see docstring).
    return {"digest": digest, **payload, "prompt_ids": sorted(prompt_ids)}


def harmful_prompts(exp_cfg, bench: str):
    return data.load_harmful(bench, exp_cfg["harmful"][bench], seed=exp_cfg["seed"])


def benign_prompts(exp_cfg):
    n, seed = exp_cfg["benign_n"], exp_cfg["seed"]
    src = exp_cfg.get("benign_source", "alpaca")
    if src == "xstest":
        return data.load_xstest_safe(n, seed=seed)
    return data.load_alpaca_benign(n, seed=seed)


def results_dir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d
