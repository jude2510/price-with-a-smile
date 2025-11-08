#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", default="artifacts/pairs")
    ap.add_argument("--out_csv", default="artifacts/rolling_hedge_pnl.csv")
    args = ap.parse_args()

    p = Path(args.pairs_dir)
    csvs = sorted(p.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs found in {p}")

    res = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)

    if all(c in res.columns for c in ("pair","pnl_pde","pnl_bsm")):
        by_pair = (res.groupby("pair")[["pnl_pde","pnl_bsm"]]
                    .mean().rename(columns={"pnl_pde":"PDE_meanPnL","pnl_bsm":"BSM_meanPnL"}))
        by_pair["cum_PDE"] = by_pair["PDE_meanPnL"].cumsum()
        by_pair["cum_BSM"] = by_pair["BSM_meanPnL"].cumsum()

        Path("artifacts").mkdir(exist_ok=True)
        plt.figure(figsize=(9,4))
        plt.plot(by_pair.index, by_pair["cum_PDE"], label="Local-Vol PDE (mean Δ-hedged PnL)")
        plt.plot(by_pair.index, by_pair["cum_BSM"], label="Black-76 sticky (mean Δ-hedged PnL)")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Cumulative mean Δ-hedged PnL")
        plt.title("Rolling 1-day hedged PnL (by pair)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("artifacts/rolling_pnl_cum.png", dpi=160)

    jsons = sorted(p.glob("*.json"))
    if jsons:
        daily = [json.loads(j.read_text()) for j in jsons]
        summary_df = pd.DataFrame(daily)
        if all(c in summary_df.columns for c in ("pair","rmse_pde","rmse_bsm")):
            plt.figure(figsize=(9,4))
            plt.plot(summary_df["pair"], summary_df["rmse_pde"], marker="o", label="PDE RMSE")
            plt.plot(summary_df["pair"], summary_df["rmse_bsm"], marker="o", label="BSM RMSE")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("RMSE (price space)")
            plt.title("Daily RMSE vs model")
            plt.legend()
            plt.tight_layout()
            plt.savefig("artifacts/rolling_rmse.png", dpi=160)

    print(f"Saved:\n - {args.out_csv}")
    if (Path('artifacts')/'rolling_pnl_cum.png').exists():
        print(" - artifacts/rolling_pnl_cum.png")
    if (Path('artifacts')/'rolling_rmse.png').exists():
        print(" - artifacts/rolling_rmse.png")

if __name__ == "__main__":
    main()
