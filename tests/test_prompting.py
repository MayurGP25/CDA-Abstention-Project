"""CPU tests for the format-instruction axis.

Torch-free on purpose, like test_shards.py: this is logic that decides what the
model reads, and it should be checkable without a GPU box.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abstention import prompting  # noqa: E402

SCHEMA = {"type": "object", "properties": {"step1": {"type": "string"}},
          "required": ["step1"]}


def test_none_is_exactly_the_old_behaviour():
    """The previous code sent [{"role": "user", ...}] and nothing else. If this
    ever changes, every number in the existing runs becomes incomparable to
    every number in the new ones."""
    assert prompting.build_messages("hi", "none") == [{"role": "user", "content": "hi"}]
    assert prompting.system_prompt("none") is None


@pytest.mark.parametrize("level", ["neutral", "terse", "schema"])
def test_prompted_levels_add_one_system_turn_and_leave_the_user_turn_alone(level):
    msgs = prompting.build_messages("hi", level, SCHEMA)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system" and msgs[0]["content"]
    # The harmful/benign axis must stay orthogonal to this one.
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_neutral_says_nothing_about_format():
    """`neutral` is the control that isolates "mentions JSON" from "has a system
    turn at all". If it leaks a format word it controls for nothing."""
    text = prompting.system_prompt("neutral").lower()
    for word in ("json", "schema", "object", "format", "{"):
        assert word not in text


def test_schema_level_embeds_the_schema_and_extends_terse():
    text = prompting.system_prompt("schema", SCHEMA)
    assert prompting.system_prompt("terse") in text
    assert '"step1"' in text


def test_schema_serialisation_is_order_stable():
    """The fingerprint hashes this string. If dict ordering leaked into it, a
    re-run would look like a settings change and refuse to resume."""
    a = prompting.system_prompt("schema", {"b": 1, "a": 2})
    b = prompting.system_prompt("schema", {"a": 2, "b": 1})
    assert a == b


def test_levels_are_mutually_distinct():
    seen = {prompting.system_prompt(lv, SCHEMA) for lv in prompting.LEVELS}
    assert len(seen) == len(prompting.LEVELS)


def test_bad_level_fails_loudly():
    with pytest.raises(ValueError):
        prompting.system_prompt("verbose")


def test_schema_level_without_a_schema_fails_loudly():
    """Silently degrading to `terse` would put a mislabelled arm in the table."""
    with pytest.raises(ValueError):
        prompting.system_prompt("schema", None)
