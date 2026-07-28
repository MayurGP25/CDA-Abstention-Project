"""Position landmarks inside a grammar-forced generation.

Pure string logic, deliberately free of torch/xgrammar so it is unit-testable on
a CPU box (see tests/test_anchors.py). This is where the bug that produced
`t* found: 0 / 150` lived, so it gets its own tests.
"""
from __future__ import annotations

import re

# A JSON string VALUE is opening: `... : "`. NOT the key's own quotes -- the
# running decode passes through `{ "` (key opens) and `{ "step1"` (key closes),
# and neither is a place where a sentence could begin.
_VALUE_OPEN = re.compile(r':\s*"$')


def anchor_hit(decoded: str, anchor: str) -> bool:
    """True when `decoded` has just completed the anchor literal.

    `anchor` must NOT carry a trailing space. BPE tokenizers attach leading
    whitespace to the FOLLOWING word, so a running decode goes

        '...Step 1:'  ->  '...Step 1: First'

    and never passes through '...Step 1: '. Matching on 'Step 1: ' therefore
    fires on exactly zero generations -- empirically 0/150 before this was fixed.
    """
    if not anchor:
        return False
    return decoded.rstrip().endswith(anchor)


def is_value_open(decoded: str) -> bool:
    """True when the context ends at a JSON string-value opening quote.

    Syntactically this is a fresh sentence start, so a refusal ("I cannot...") is
    a grammatical continuation here exactly as it is at position 0. It is the
    control point that separates 'the model stopped wanting to refuse' from
    'a refusal was no longer syntactically available'.
    """
    return bool(_VALUE_OPEN.search(decoded))
