"""Verify that the format-instruction change did NOT move the digest of runs
already on disk. Run this on the GPU box BEFORE collecting anything new.

The risk this guards: run_fingerprint() gained a field. check_fingerprint()
compares digests exactly, so if that field entered the payload unconditionally,
every finished run -- including the depth runs behind Table 1 -- would refuse to
resume and its shards would be unusable. runner.run_fingerprint only adds the
field when format_instruction != "none", but "the code says so" is not the same
as "the manifests on this machine still hash to what they claim".

This is torch-free: it re-hashes each stored manifest with the same convention
run_fingerprint uses and checks it reproduces the stored digest. It does not
need the model, the GPU, or the shards.

    python scripts/check_fingerprint_compat.py            # all runs under results/
    python scripts/check_fingerprint_compat.py results/depth
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The keys runner.run_fingerprint() actually hashes. Listed explicitly rather
# than derived by subtraction, because callers ANNOTATE the returned dict after
# the digest exists: restore.py appends `restoration_positions` and `probe`, so
# a manifest legitimately contains keys the digest never saw. Hashing by
# subtraction reported those runs BROKEN when nothing was wrong with them.
#
# Duplicated from runner.py on purpose. Importing it would pull in torch and
# xgrammar, and the point of this script is that it runs anywhere, before any
# GPU work. tests/test_fingerprint_compat.py pins the two lists together so the
# duplication cannot drift silently.
HASHED_KEYS = frozenset({
    "model_id", "grammar_name", "schema", "max_new_tokens", "free_max_new_tokens",
    "anchor", "sr_stride", "sr_window", "refusal_set", "prefix_set_version",
    "prefix_set_hash", "benign_source", "seed",
    # present only when format_instruction != "none"
    "format_instruction", "system_prompt",
})


def digest_of(manifest: dict) -> tuple[str, list[str]]:
    """Recompute the digest, and return the keys treated as post-hoc annotations
    so the caller can show its work rather than assert it."""
    payload = {k: v for k, v in manifest.items() if k in HASHED_KEYS}
    extra = sorted(k for k in manifest
                   if k not in HASHED_KEYS and k not in ("digest", "prompt_ids"))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16], extra


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        print(f"no manifests under {root} -- nothing to check")
        return 0

    bad = 0
    for m in manifests:
        blob = json.loads(m.read_text())
        recomputed, extra = digest_of(blob)
        stored = blob.get("digest")
        fmt = blob.get("format_instruction", "none")
        rel = m.relative_to(root)
        note = f"  (+annotations: {', '.join(extra)})" if extra else ""
        if stored == recomputed:
            print(f"  ok    {rel}  [{stored}] fmt={fmt}{note}")
        else:
            bad += 1
            print(f"  BROKEN {rel}\n"
                  f"         stored     ={stored}\n"
                  f"         recomputed ={recomputed}\n"
                  f"         -> this run will NOT resume; its shards are stranded.")

    print(f"\n{len(manifests) - bad}/{len(manifests)} manifests still hash to "
          f"their stored digest.")
    if bad:
        print("Do not start new collection until this is understood.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
