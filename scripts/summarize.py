"""Aggregate results/*.json: mean ± std per config, and paired t-tests of ours against each baseline
(paired by seed, since every config at a given seed shares the same real subset).

Usage: python scripts/summarize.py [--real 0.25] [--model yolo] [--metric map50_95]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=float)
    ap.add_argument("--model")
    ap.add_argument("--metric", default="map50_95")
    args = ap.parse_args()

    runs = [json.loads(p.read_text()) for p in (REPO / "results").glob("*.json")]
    runs = [r for r in runs if (args.real is None or r["real"] == args.real)
            and (args.model is None or r["model"] == args.model)]
    cells = defaultdict(dict)  # (domain, real, model) -> config -> {seed: value}
    for r in runs:
        cfg = r["synth_dir"] + (f"x{r['ratio']:g}" if r["synth"] != "none" else "") + f"/aug{r['aug']}"
        cells[(r["domain"], r["real"], r["model"])].setdefault(cfg, {})[r["seed"]] = r["test"][args.metric]

    for key in sorted(cells):
        print(f"\n== {key[0]}  real={key[1]}  model={key[2]}  metric={args.metric}")
        cfgs = cells[key]
        for cfg, by_seed in sorted(cfgs.items()):
            v = np.array(list(by_seed.values()))
            print(f"  {cfg:28s} n={len(v)}  {v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        ours = [c for c in cfgs if c.startswith("ours")]
        for o in ours:
            for base in [c for c in cfgs if not c.startswith("ours")]:
                seeds = sorted(set(cfgs[o]) & set(cfgs[base]))
                if len(seeds) >= 2:
                    a = np.array([cfgs[o][s] for s in seeds])
                    b = np.array([cfgs[base][s] for s in seeds])
                    p = ttest_rel(a, b).pvalue
                    print(f"  {o} vs {base}: Δ={np.mean(a - b):+.3f}  wins {int((a > b).sum())}/{len(seeds)}  p={p:.3f}")


if __name__ == "__main__":
    main()
