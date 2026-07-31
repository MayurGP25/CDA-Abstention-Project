"""The compat checker duplicates runner.run_fingerprint's key list, because it
must run without torch. These tests keep the duplicate honest.

They also pin the bug that made the checker report two healthy runs as BROKEN:
restore.py annotates the fingerprint dict AFTER the digest is computed, so a
manifest legitimately holds keys the hash never saw.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cfc", Path(__file__).resolve().parents[1] / "scripts" / "check_fingerprint_compat.py")
cfc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cfc)


def _payload(**over):
    p = dict(model_id="Qwen/Qwen2.5-7B-Instruct", grammar_name="forced_steps",
             schema='{"a":1}', max_new_tokens=32, free_max_new_tokens=8,
             anchor="Step 1:", sr_stride=1, sr_window=12, refusal_set="large",
             prefix_set_version="v1", prefix_set_hash="abc", benign_source="alpaca",
             seed=0)
    p.update(over)
    return p


def _digest(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def test_recomputes_a_plain_manifest():
    p = _payload()
    manifest = {"digest": _digest(p), **p, "prompt_ids": ["a", "b"]}
    got, extra = cfc.digest_of(manifest)
    assert got == manifest["digest"]
    assert extra == []


def test_post_hoc_annotations_do_not_break_the_digest():
    """The actual bug. restore.py adds these two keys after hashing, and the
    checker previously folded them in and declared the run stranded."""
    p = _payload()
    manifest = {"digest": _digest(p), **p,
                "restoration_positions": [0, 2, 4], "probe": "restoration",
                "prompt_ids": ["a"]}
    got, extra = cfc.digest_of(manifest)
    assert got == manifest["digest"], "annotations must not enter the hash"
    assert extra == ["probe", "restoration_positions"]


def test_format_instruction_keys_are_hashed():
    """These two DO belong in the digest, otherwise a prompted arm could be
    merged into an unprompted run."""
    base = _payload()
    fmt = {**base, "format_instruction": "terse", "system_prompt": "Output ONLY..."}
    assert _digest(fmt) != _digest(base)
    manifest = {"digest": _digest(fmt), **fmt}
    got, extra = cfc.digest_of(manifest)
    assert got == manifest["digest"] and extra == []


def test_a_genuinely_changed_setting_is_still_caught():
    p = _payload()
    manifest = {"digest": _digest(p), **_payload(max_new_tokens=64)}
    got, _ = cfc.digest_of(manifest)
    assert got != manifest["digest"]


def test_key_list_matches_runner():
    """Fails if runner.run_fingerprint gains or loses a hashed field and this
    script is not updated. Skipped where torch is unavailable, which is exactly
    the environment the duplication exists to serve."""
    runner = pytest.importorskip("abstention.runner")
    src = Path(runner.__file__).read_text(encoding="utf-8")
    body = src.split("def run_fingerprint")[1].split("digest = hashlib")[0]
    in_runner = {ln.split("=")[0].strip() for ln in body.splitlines()
                 if "=" in ln and ln.strip().endswith(",")
                 and not ln.strip().startswith("#")}
    in_runner |= {"format_instruction", "system_prompt"}     # added conditionally
    missing = (in_runner & {"model_id", "seed", "anchor"}) - cfc.HASHED_KEYS
    assert not missing, f"runner hashes {missing} but the checker does not"
