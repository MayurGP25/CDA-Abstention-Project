"""CPU-only tests for resumable collection.

This is the code that has to survive a remote server dying mid-run, so the
failure modes are pinned explicitly rather than assumed.
"""
import json

import pandas as pd
import pytest

from abstention import shards


def _df(n=3):
    return pd.DataFrame({"pos": range(n), "sr": [-0.4] * n})


def test_unit_key_is_filesystem_safe():
    k = shards.unit_key("harmful_forced", "advbench/12: bad?")
    assert "/" not in k and ":" not in k and "?" not in k
    assert k.startswith("harmful_forced__")


def test_atomic_write_leaves_no_tmp_on_success(tmp_path):
    p = tmp_path / "u.parquet"
    shards.write_atomic_parquet(_df(), p)
    assert p.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert len(pd.read_parquet(p)) == 3


def test_half_written_tmp_is_not_counted_as_complete(tmp_path):
    """The crash case: killed mid-write. The .tmp must NOT be treated as done,
    or the unit is silently skipped and the run has a hole in it."""
    (tmp_path / "harmful_forced__a.parquet.tmp").write_bytes(b"\x00truncated")
    shards.write_atomic_parquet(_df(), tmp_path / "harmful_forced__b.parquet")
    assert shards.completed(tmp_path) == {"harmful_forced__b"}


def test_merge_concatenates_only_complete_shards(tmp_path):
    sd = tmp_path / "shards"
    sd.mkdir()
    shards.write_atomic_parquet(_df(3), sd / "a.parquet")
    shards.write_atomic_parquet(_df(4), sd / "b.parquet")
    (sd / "c.parquet.tmp").write_bytes(b"garbage")
    n = shards.merge(tmp_path, tmp_path / "out.parquet")
    assert n == 7
    assert len(pd.read_parquet(tmp_path / "out.parquet")) == 7


def test_merge_with_no_shards_is_not_an_error(tmp_path):
    (tmp_path / "shards").mkdir()
    assert shards.merge(tmp_path, tmp_path / "out.parquet") == 0


def test_fingerprint_writes_manifest_then_accepts_identical_resume(tmp_path):
    fp = {"digest": "abc123", "anchor": "Step 1:", "max_new_tokens": 32}
    shards.check_fingerprint(tmp_path / "run", fp)
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["digest"] == "abc123"
    shards.check_fingerprint(tmp_path / "run", fp)      # resume: must not raise


def test_fingerprint_mismatch_raises_and_names_what_changed(tmp_path):
    run = tmp_path / "run"
    shards.check_fingerprint(run, {"digest": "abc", "anchor": "Step 1:", "max_new_tokens": 24})
    with pytest.raises(shards.FingerprintMismatch) as ei:
        shards.check_fingerprint(run, {"digest": "xyz", "anchor": "Step 1:",
                                       "max_new_tokens": 32})
    assert "max_new_tokens" in ei.value.changed
    assert "anchor" not in ei.value.changed


def test_fresh_discards_incompatible_shards(tmp_path):
    run = tmp_path / "run"
    shards.check_fingerprint(run, {"digest": "abc"})
    (run / "shards").mkdir()
    shards.write_atomic_parquet(_df(), run / "shards" / "old.parquet")
    shards.check_fingerprint(run, {"digest": "xyz"}, fresh=True)   # must not raise
    assert shards.completed(run / "shards") == set() or not (run / "shards").exists()


def test_progress_jsonl_appends_and_survives_reopen(tmp_path):
    p = tmp_path / "progress.jsonl"
    shards.append_jsonl(p, {"unit": "a", "status": "ok"})
    shards.append_jsonl(p, {"unit": "b", "status": "failed"})
    recs = [json.loads(l) for l in p.read_text().strip().splitlines()]
    assert [r["unit"] for r in recs] == ["a", "b"]
