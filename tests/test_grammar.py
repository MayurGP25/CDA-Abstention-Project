"""Schema-shape tests (CPU, no xgrammar needed) plus an optional compile test."""
import importlib.util

import pytest

from abstention.grammars import affirmative_prefix_schema, multi_step_schema


def test_affirmative_schema_shape():
    s = affirmative_prefix_schema("Sure: ")
    assert s["properties"]["answer"]["enum"] == ["Sure: "]
    assert "details" in s["properties"]
    assert s["additionalProperties"] is False


def test_multi_step_schema_has_n_forced_openers():
    s = multi_step_schema(5)
    assert list(s["properties"]) == [f"step{k}" for k in range(1, 6)]
    assert s["properties"]["step3"]["pattern"] == "^Step 3: .+"


@pytest.mark.skipif(importlib.util.find_spec("xgrammar") is None,
                    reason="xgrammar not installed (GPU box only)")
def test_compiles_with_xgrammar_tokenizer_stub():
    # Only checks that compile_json_schema accepts our schema JSON without error,
    # using a small tokenizer. Skipped on machines without xgrammar.
    import json

    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tinfo = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=tok.vocab_size)
    compiler = xgr.GrammarCompiler(tinfo)
    compiler.compile_json_schema(json.dumps(multi_step_schema(3)))
