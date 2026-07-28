"""Crash-safe shard I/O for resumable collection.

Deliberately free of torch/xgrammar/model imports so the resume machinery is
unit-testable on a CPU box (tests/test_shards.py). This is the code that has to
survive a remote server dying at hour three, so it is tested rather than trusted.

Contract:
  * one shard per unit, written temp-then-rename, so a kill mid-write leaves a
    stray .tmp (ignored) and never a truncated .parquet;
  * completion is inferred from the filesystem, not from in-memory state, so it
    survives process restart;
  * a fingerprint of every setting that changes the numbers is stored, and a
    mismatch is a hard error rather than a silent merge of incompatible runs.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def unit_key(condition: str, prompt_id: str) -> str:
    return f"{safe(condition)}__{safe(prompt_id)}"


def write_atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write then rename. os.replace is atomic on POSIX and on Windows when the
    temp file is on the same volume (it is -- same directory)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed(shard_dir: Path) -> set[str]:
    """Units already on disk. .tmp files are NOT counted -- a half-written shard
    must be redone, not resumed past."""
    return {p.stem for p in Path(shard_dir).glob("*.parquet")}


def merge(run_dir: Path, out: Path) -> int:
    shards = sorted((Path(run_dir) / "shards").glob("*.parquet"))
    if not shards:
        print("[merge] no shards yet -- nothing to merge")
        return 0
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    write_atomic_parquet(df, Path(out))
    print(f"[merge] {len(shards)} shards -> {len(df)} rows -> {out}")
    return len(df)


class FingerprintMismatch(RuntimeError):
    def __init__(self, old: dict, new: dict, changed: list[str]):
        self.old, self.new, self.changed = old, new, changed
        super().__init__(
            f"fingerprint mismatch: existing={old.get('digest')} "
            f"this={new.get('digest')} changed={changed}")


def check_fingerprint(run_dir: Path, fp: dict, *, fresh: bool = False) -> None:
    """Write the manifest on a new run; raise FingerprintMismatch on an
    incompatible resume. `fresh=True` discards the whole run directory first."""
    run_dir = Path(run_dir)
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(fp, indent=2))
        return
    old = json.loads(manifest.read_text())
    if old.get("digest") == fp.get("digest"):
        return
    changed = [k for k in fp if k != "digest" and old.get(k) != fp.get(k)]
    raise FingerprintMismatch(old, fp, changed)


def die_on_mismatch(exc: FingerprintMismatch) -> None:
    print("\n!! FINGERPRINT MISMATCH -- refusing to resume.\n"
          f"   existing run: {exc.old.get('digest')}\n"
          f"   this run    : {exc.new.get('digest')}\n"
          f"   changed     : {', '.join(exc.changed) or '(unknown)'}\n\n"
          "   These shards were produced under different settings and must not be\n"
          "   merged. Use --fresh to discard them, or --exp <newname> for a new dir.",
          file=sys.stderr)
    sys.exit(2)
