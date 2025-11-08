#!/usr/bin/env python3
import argparse
from pathlib import Path
from day_pair_hedge import business_days

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="pairs.txt")
    args = ap.parse_args()

    days = business_days(args.start, args.end)
    if len(days) < 2:
        raise SystemExit("Need at least 2 business days.")
    pairs = list(zip(days[:-1], days[1:]))
    Path(args.out).write_text("\n".join(f"{d0},{d1}" for d0, d1 in pairs))
    print(f"Wrote {len(pairs)} day-pairs to {args.out}")

if __name__ == "__main__":
    main()
