"""Main data collection -- resumable.

Runs the three arms and writes one tidy parquet:

  harmful_forced : harmful prompts, forcing grammar   (the measurement)
  benign_forced  : benign  prompts, SAME grammar      (null control: the grammar
                   alone must not manufacture refusal preference)
  free           : harmful prompts, no grammar, short (behavioural refusal rate
                   only -- its t0 equals harmful_forced's by construction)

RESUMABILITY (the point of this rewrite). Work is split into units of one
(condition, prompt). Each finished unit writes its own shard, via a temp file
plus os.replace, so a kill mid-write can never leave a corrupt shard -- only a
stray .tmp, which is ignored. On restart the completed units are read off disk
and skipped. A fingerprint of every setting that affects the numbers is stored
in manifest.json; restarting with different settings HARD-STOPS rather than
silently merging incompatible shards. SIGINT/SIGTERM finish the current unit,
flush, and exit 0.

Usage:
  python experiments/collect.py --model qwen25-7b --bench harmbench --n 50
  python experiments/collect.py --model qwen25-7b --bench harmbench --n 50   # resumes
  python experiments/collect.py --model qwen25-7b --bench harmbench --merge-only
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abstention import runner, shards  # noqa: E402
from abstention.conditions import run_condition  # noqa: E402
from abstention.refusal_set import _looks_like_refusal  # noqa: E402

_STOP = False


def _install_signal_handlers():
    def handler(signum, _frame):
        global _STOP
        if _STOP:                       # second Ctrl-C: go now
            print("\n[collect] second signal -- exiting immediately", flush=True)
            os._exit(130)
        _STOP = True
        print(f"\n[collect] signal {signum} -- finishing current unit, then exiting. "
              "Ctrl-C again to force.", flush=True)

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):    # not in main thread / unsupported
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key from configs/models.yaml")
    ap.add_argument("--bench", default="advbench")
    ap.add_argument("--n", type=int, default=None, help="override prompt count")
    ap.add_argument("--exp", default="depth", help="results subdir")
    ap.add_argument("--fresh", action="store_true", help="discard existing shards")
    ap.add_argument("--merge-only", action="store_true",
                    help="rebuild the parquet from existing shards; no GPU needed")
    ap.add_argument("--conditions", nargs="+", default=None,
                    choices=["harmful_forced", "benign_forced", "free"],
                    help="run only these arms (default: all three). Use for cheap "
                         "single-arm ablations, e.g. the neutral-scaffold control.")
    ap.add_argument("--benign-source", default=None,
                    help="override configs/experiment.yaml benign_source "
                         "(alpaca | xstest). xstest gives the HARD negative: "
                         "safe-but-scary prompts that are not trivially separable.")
    # Config overrides, so ablation arms launch from one script without editing YAML.
    ap.add_argument("--grammar", default=None,
                    help="override grammar.name (forced_steps | neutral_scaffold | ...)")
    ap.add_argument("--anchor", default=None,
                    help="override the anchor literal. Pass '' for grammars with no "
                         "anchor (neutral_scaffold) -- t_star then reports -1 and "
                         "paper.py simply omits that landmark.")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--format-instruction", default=None,
                    choices=["none", "neutral", "terse", "schema"],
                    help="how much the PROMPT anticipates the grammar (see "
                         "src/abstention/prompting.py). Default none = the "
                         "grammar arrives unannounced. Pair with "
                         "--max-new-tokens 1 --exp fmt for the cheap t0-only "
                         "sweep; that needs one forward pass per prompt.")
    args = ap.parse_args()

    # Anything that is in the FINGERPRINT and that a user would reasonably vary
    # within one --exp must also be in the DIRECTORY NAME. Otherwise the second
    # setting targets the first one's directory and dies on a mismatch: correct,
    # but it just blocks the run. Both suffixes are omitted at their defaults so
    # existing run directories keep their paths.
    stem = f"{args.model}__{args.bench}"
    if args.format_instruction and args.format_instruction != "none":
        stem += f"__fmt-{args.format_instruction}"
    if args.benign_source and args.benign_source != "alpaca":
        stem += f"__ben-{args.benign_source}"
    run_dir = runner.results_dir(args.exp) / stem
    out = run_dir / f"{stem}.parquet"

    if args.merge_only:
        shards.merge(run_dir, out)
        return

    _install_signal_handlers()
    lm, grammar, schema, exp_cfg, models_cfg = runner.setup(args.model, overrides=dict(
        grammar_name=args.grammar, anchor=args.anchor,
        max_new_tokens=args.max_new_tokens,
        format_instruction=args.format_instruction))
    if args.benign_source:                     # must precede the fingerprint
        exp_cfg["benign_source"] = args.benign_source

    harmful = runner.harmful_prompts(exp_cfg, args.bench)
    benign = runner.benign_prompts(exp_cfg)
    if args.n:
        harmful, benign = harmful[: args.n], benign[: args.n]

    fp = runner.run_fingerprint(
        lm, exp_cfg, models_cfg, schema,
        [p["id"] for p in harmful] + [p["id"] for p in benign])
    try:
        shards.check_fingerprint(run_dir, fp, fresh=args.fresh)
    except shards.FingerprintMismatch as e:
        shards.die_on_mismatch(e)

    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress = run_dir / "progress.jsonl"

    refusal_ids = runner.get_refusal_ids(lm, exp_cfg, harmful)
    prefix_ids = runner.get_refusal_prefix_ids(lm)
    stride, window = runner.sr_stride(exp_cfg), runner.sr_window(exp_cfg)
    anchor = runner.anchor(exp_cfg)
    T = exp_cfg["max_new_tokens"]
    T_free = exp_cfg.get("free_max_new_tokens", 8)

    g_name = exp_cfg["grammar"]["name"]
    fmt = runner.format_instruction(exp_cfg)
    h_src, b_src = args.bench, exp_cfg.get("benign_source", "alpaca")

    # (condition, prompt, max_new_tokens, grammar, measure_sr, prompt_source)
    units = (
        [("harmful_forced", p, T, grammar, True, h_src) for p in harmful]
        + [("benign_forced", p, T, grammar, True, b_src) for p in benign]
        + [("free", p, T_free, None, False, h_src) for p in harmful]
    )
    if args.conditions:
        units = [u for u in units if u[0] in set(args.conditions)]

    done = shards.completed(shard_dir)
    todo = [u for u in units if shards.unit_key(u[0], u[1]["id"]) not in done]
    print(f"[collect] {len(units)} units total | {len(done)} already complete | "
          f"{len(todo)} to run  ({run_dir})")

    failures: list[str] = []
    bar = tqdm(todo, desc="units")
    for cond, p, max_tok, gram, measure_sr, psrc in bar:
        if _STOP:
            break
        key = shards.unit_key(cond, p["id"])
        bar.set_postfix_str(key[:38])
        t0 = time.time()
        for attempt in (1, 2):
            try:
                rows, emitted = run_condition(
                    lm, condition=cond, prompt=p["prompt"], prompt_id=p["id"],
                    refusal_ids=refusal_ids, compiled_grammar=gram,
                    max_new_tokens=max_tok,
                    refusal_prefix_ids=prefix_ids if measure_sr else None,
                    sr_stride=stride, anchor=anchor, sr_window=window,
                    grammar_name=g_name, prompt_source=psrc,
                    format_instruction=fmt, schema=schema,
                )
                # Behavioural fact for the free arm. Store the BOOLEAN ONLY --
                # never persist decoded generations (RESPONSIBLE_USE).
                refused = None
                if cond == "free":
                    refused = bool(_looks_like_refusal(
                        lm.tokenizer.decode(emitted, skip_special_tokens=True)))
                    for r in rows:
                        r["free_refused"] = refused

                shards.write_atomic_parquet(pd.DataFrame(rows), shard_dir / f"{key}.parquet")
                shards.append_jsonl(progress, dict(
                    unit=key, status="ok", rows=len(rows),
                    t_star=int(rows[0]["t_star"]) if rows else -1,
                    free_refused=refused, secs=round(time.time() - t0, 2),
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
                break
            except Exception as e:                              # noqa: BLE001
                if attempt == 2:
                    failures.append(key)
                    shards.append_jsonl(progress, dict(
                        unit=key, status="failed", error=repr(e)[:400],
                        secs=round(time.time() - t0, 2),
                        ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
                    print(f"\n[collect] FAILED {key}: {e!r}", flush=True)
                else:
                    print(f"\n[collect] retrying {key} after {e!r}", flush=True)
    bar.close()

    n_done = len(shards.completed(shard_dir))
    shards.merge(run_dir, out)
    print(f"[collect] complete {n_done}/{len(units)} units"
          + (f" | {len(failures)} failed (retried on next run): "
             f"{', '.join(failures[:5])}" if failures else ""))
    if _STOP:
        print("[collect] interrupted cleanly -- rerun the same command to resume.")


if __name__ == "__main__":
    main()
