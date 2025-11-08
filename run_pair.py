#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

import day_pair_hedge as dph

SUBSET = dph.SUBSET
PDE_GRID = dph.PDE_GRID

hedge_one_day = dph.hedge_one_day

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.txt")
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--outdir", default="artifacts/pairs")
    args = ap.parse_args()

    lines = [ln.strip() for ln in Path(args.pairs).read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"No pairs found in {args.pairs}")
    if args.idx < 0 or args.idx >= len(lines):
        raise SystemExit(f"idx out of range. Have {len(lines)} pairs.")

    d0, d1 = lines[args.idx].split(",")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    bt, summary = hedge_one_day(d0, d1, SUBSET, PDE_GRID)

    base = f"{args.idx:04d}_{d0}_to_{d1}"
    csv_path = Path(args.outdir) / f"{base}.csv"
    json_path = Path(args.outdir) / f"{base}.json"

    if "pair" not in bt.columns:
        bt = bt.assign(pair=f"{d0}\u2192{d1}")
    bt.to_csv(csv_path, index=False)
    Path(json_path).write_text(json.dumps({"pair": f"{d0}\u2192{d1}", **summary}, indent=2))

    print(f"Saved:\n - {csv_path}\n - {json_path}")

if __name__ == "__main__":
    main()
