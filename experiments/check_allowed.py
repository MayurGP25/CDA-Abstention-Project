"""Confirm |A_t| is exactly 5 at position 0 for every prompt, not merely on
median. The paper now claims the admitted set is identical across prompts, which
follows from the matcher being in its initial state, but a claim in a paper
should be checked against the data rather than argued from the code."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for p in (Path.cwd(), Path.cwd() / "experiments"):
    sys.path.insert(0, str(p))
from dump_all import load_all  # noqa: E402

df = load_all(Path("results/_dump_cache.parquet"), False, {"fmt2"})
d = df[(df["pos"] == 0) & (df["grammar"] == "forced_steps")
       & (df["condition"].isin(["harmful_forced", "benign_forced"]))]
g = d.groupby(["model", "fmt", "condition"])["n_allowed"].agg(
    ["min", "max", "nunique", "size"])
print(g.to_string())
bad = g[(g["min"] != 5) | (g["max"] != 5)]
print()
if bad.empty:
    print("|A_t| = 5 exactly, in every cell and every prompt. Claim holds.")
else:
    print("NOT CONSTANT -- the paper's 'same five tokens' claim needs softening:")
    print(bad.to_string())
