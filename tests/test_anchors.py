"""CPU-only tests for the position landmarks.

The trailing-space bug guarded here cost a full collection run: matching on
'Step 1: ' found the anchor in 0 of 150 generations, because BPE glues the space
onto the following word.
"""
from abstention.anchors import anchor_hit, is_value_open

# The real running decode of a forced_steps generation, token by token, taken
# from an actual Qwen2.5-7B run.
RUNNING_DECODE = [
    '{',
    '{ "',
    '{ "step',
    '{ "step1',
    '{ "step1"',
    '{ "step1":',
    '{ "step1": "',
    '{ "step1": "Step',
    '{ "step1": "Step ',
    '{ "step1": "Step 1',
    '{ "step1": "Step 1:',            # <- anchor completes here
    '{ "step1": "Step 1: Obtain',     # <- space arrived glued to the verb
]


def test_anchor_fires_exactly_once_without_trailing_space():
    hits = [i for i, d in enumerate(RUNNING_DECODE) if anchor_hit(d, "Step 1:")]
    assert hits == [10]


def test_trailing_space_anchor_never_fires():
    """The original bug, pinned so it cannot come back."""
    assert not any(anchor_hit(d, "Step 1: ") for d in RUNNING_DECODE)


def test_anchor_survives_whitespace_formatting_variants():
    """xgrammar lets the model choose its own JSON whitespace; t* shifted 10-15
    across the real run purely because of this."""
    assert anchor_hit('{ \n\n  "step1": "Step 1:', "Step 1:")
    assert anchor_hit('{"step1":"Step 1:', "Step 1:")


def test_empty_anchor_never_fires():
    assert not anchor_hit('{ "step1": "Step 1:', "")


def test_value_open_only_at_the_value_quote():
    opens = [i for i, d in enumerate(RUNNING_DECODE) if is_value_open(d)]
    assert opens == [6]           # '{ "step1": "' -- NOT '{ "' and NOT '{ "step1"'


def test_value_open_rejects_key_quotes():
    assert not is_value_open('{ "')          # key opening
    assert not is_value_open('{ "step1"')    # key closing
    assert is_value_open('{ "step1": "')     # value opening
    assert is_value_open('{"a":"')           # no whitespace
