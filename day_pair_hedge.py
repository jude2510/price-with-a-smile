
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Callable
from scipy.stats import norm

# --- Global configuration for all hedging runs ---
SUBSET = dict(max_expiries=8, m_band=(0.95, 1.05))
PDE_GRID = dict(nS=501, nT=600, span=(0.45, 1.75))

PayoffCP = Literal["C","P"]

@dataclass
class GridSpec:
    S_min: float
    S_max: float
    nS: int
    nT: int

def make_spot_grid(S0: float, K: float, T: float,
                   nS: int=401, nT: int=400,
                   span: Tuple[float,float]=(0.5,1.6)) -> GridSpec:
    Sref = K if K>0 else S0
    S_min = max(1e-8, Sref * span[0])
    S_max = Sref * span[1]
    return GridSpec(S_min=S_min, S_max=S_max, nS=nS, nT=nT)

def crank_nicolson_localvol(
    cp: PayoffCP,
    K: float,
    T: float,
    S0: float,
    lv_func: Callable[[float, np.ndarray], np.ndarray],
    r: float = 0.0,
    q: float = 0.0,
    grid: Optional[GridSpec] = None
):
    """
    European option under local vol via Crank-Nicholson on S-grid.
    PDE: u_t + 0.5*sigma_loc(t,S)^2*S^2*u_SS + (r-q)*S*u_S - r*u = 0, u(T,S)=payoff
    Returns (price, delta) at S0.
    """
    if T<= 0:
        # epiry: price is intrinsic
        if cp == "C":
            return max(S0 - K, 0.0), (1.0 if S0 > K else 0.0)
        else:
            return max(K - S0, 0.0), (-1.0 if S0 < K else 0.0)
        
    if grid is None:
        grid = make_spot_grid(S0, K, T)

    S_min, S_max, nS, nT = grid.S_min, grid.S_max, grid.nS, grid.nT
    S = np.linspace(S_min, S_max, nS)
    dS = S[1] - S[0]
    dt = T / nT if nT > 0 else 0.0

    # Terminal payoff
    if cp == "C":
        U = np.maximum(S - K, 0.0)
        def bc_left(t): return 0.0
        def bc_right(t): return S_max - K * np.exp(-r * (T - t)) if r != 0 else S_max - K
    else:
        U = np.maximum(K - S, 0.0)
        def bc_left(t): return K * np.exp(-r * (T - t)) - S_min if r != 0 else K - S_min
        def bc_right(t): return 0.0

    # Precompute S index around S0 for delta
    j0 = int(np.clip(np.searchsorted(S, S0), 1, nS-2))

    # March backwards
    for n in range(nT):
        t_next = T - n * dt # known time (terminal at start)
        t_curr = t_next - dt # target time
        t_mid = max(t_curr, 0.0) + 0.5 * dt

        sigma = np.asarray(lv_func(t_mid, S), dtype=float)
        sigma2 = np.clip(sigma, 1e-8, 5.0)**2

        # CN coefficients (standard)
        Acoef = 0.25 * dt * ( (sigma2*(S**2))/(dS**2) - ((r - q)*S)/dS )
        Bcoef = -0.5 * dt * ( (sigma2*(S**2))/(dS**2) + r )
        Ccoef = 0.25 * dt * ( (sigma2*(S**2))/(dS**2) + ((r - q)*S)/dS )

        # Implicit (A) tridiagonal for interior nodes 1..nS-2
        lower = -Acoef[2:-1].copy()         # length nS-3
        diag  = (1.0 - Bcoef[1:-1]).copy()  # length nS-2
        upper = -Ccoef[1:-2].copy()         # length nS-3

        # Explicit (B) application to previous U
        RHS = np.empty_like(U)
        RHS[1:-1] = (Acoef[1:-1]*U[:-2] + (1.0 + Bcoef[1:-1])*U[1:-1] + Ccoef[1:-1]*U[2:])

        # Dirichlet boundary contributions
        left_val = bc_left(t_curr)
        right_val = bc_right(t_curr)
        RHS[1] -= (-Acoef[1]) * left_val      # because lower diag coeff in A multiplies U[0]
        RHS[-2] -= (-Ccoef[-2]) * right_val   # because upper diag coeff in A multiplies U[-1]

        # Set boundary rows (Dirichlet)
        RHS[0] = left_val
        RHS[-1] = right_val

        # Solve interior by Thomas algorithm
        U_new = U.copy()
        # forward elimination (on interior system)
        for i in range(len(lower)):
            m = lower[i] / diag[i]
            diag[i+1] -= m * upper[i]
            RHS[i+2] -= m * RHS[i+1]
        # back substitution
        U_new[-2] = RHS[-2] / diag[-1]
        for i in range(len(lower)-1, -1, -1):
            U_new[i+1] = (RHS[i+1] - upper[i] * U_new[i+2]) / diag[i]

        # apply Dirichlet BCs
        U_new[0] = left_val
        U_new[-1] = right_val
        U = U_new

    price = float(np.interp(S0, S, U))
    delta = float((U[j0+1] - U[j0-1]) / (2.0 * dS))
    return price, delta     

class BiLinearLV:
    def __init__(self, lv_parquet_path: str):
        df = pd.read_parquet(lv_parquet_path)
        self.Ts = np.array(sorted(df["T"].unique()))
        Kmap = {}
        Sigmap = {}
        for T in self.Ts:
            row = df[df["T"]==T].sort_values("K")
            Kmap[T] = row["K"].to_numpy()
            Sigmap[T] = row["sigma_loc"].to_numpy()
        # pad to rectangular grid
        self.Ks = np.unique(np.concatenate([v for v in Kmap.values()]))
        self.SIG = np.empty((len(self.Ts), len(self.Ks)))
        self.SIG[:]=np.nan
        for i, T in enumerate(self.Ts):
            Ks, Sg = Kmap[T], Sigmap[T]
            self.SIG[i,:] = np.interp(self.Ks, Ks, Sg, left=Sg[0], right=Sg[-1])
        self.SIG = np.clip(self.SIG, 1e-6, 5.0)
        self.Tmin, self.Tmax = float(self.Ts[0]), float(self.Ts[-1])
        self.Kmin, self.Kmax = float(self.Ks[0]), float(self.Ks[-1])

    def __call__(self, t, S):
        """
        Vectorized bilinear interpolation:
        - t: scalar maturity
        - S: scalar or 1D array of spot nodes (we treat S along the 'K' axis of the LV grid)
        Returns array with same shape as S (or scalar if S is a scalar)
        """
        # clamp t and compute time weights
        t = float(np.clip(float(t), self.Tmin, self.Tmax))
        i = np.searchsorted(self.Ts, t)
        i0 = max(i-1, 0); i1 = min(i, len(self.Ts)-1)
        t0, t1 = self.Ts[i0], self.Ts[i1]
        w_t = 0.0 if t1==t0 else (t - t0) / (t1 - t0)

        # prepare S as array and clamp to grid bounds
        S_arr = np.asarray(S, dtype=float)
        S_arr = np.clip(S_arr, self.Kmin, self.Kmax)

        # interpolate in K on the two time slices then blend across t
        s0 = np.interp(S_arr, self.Ks, self.SIG[i0,:], left=self.SIG[i0,0], right=self.SIG[i0,-1])
        s1 = np.interp(S_arr, self.Ks, self.SIG[i1,:], left=self.SIG[i1,0], right=self.SIG[i1,-1])
        sig = s0 * (1 - w_t) + s1 * w_t

        # clamp small/huge values for numerical stability
        sig = np.clip(sig, 1e-6, 5.0)

        # return scalar if input was scalar
        return float(sig) if np.ndim(S) == 0 else sig   

def business_days(start_iso: str, end_iso: str):
    # Simple Mon-Fri generator
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start_iso, end_iso)]

# Single function for one-day hedge
def hedge_one_day(asof0: str, asof1: str, SUBSET: dict, PDE_GRID: dict):
    """
    Uses day-0 prices, IVs, and LV to compute day-1 delta hedged PnL.
    Returns: (bt_df, summary_dict)
    """
    SUBSET = dict(max_expiries=8, m_band=(0.95, 1.05))
    PDE_GRID = dict(nS=501, nT=600, span=(0.45,1.75))

    # load both days
    day0, iv0, lv0 = load_day(asof0)
    day1, iv1, lv1 = load_day(asof1)

    S0 = float(day0["S"].median())
    S1_med = float(day1["S"].median()) # just for info

    F0 = F_curve_from_ivtab(iv0)
    F1 = F_curve_from_ivtab(iv1)
    lvf0 = BiLinearLV(day_files(asof0)["lvparq"])
    lvf1 = BiLinearLV(day_files(asof1)["lvparq"])

    # universe selection for day0 (liquidity + compute)
    m = day0["K"] / S0
    univ0 = (day0[(m>=SUBSET["m_band"][0]) & (m<=SUBSET["m_band"][1])].copy())
    # recompute T
    a0 = pd.to_datetime(asof0)
    univ0["T"] = (pd.to_datetime(univ0["expiry"]) - a0).dt.days / 365.0
    
    # first N expiries
    exps = sorted(univ0["T"].unique())[:SUBSET["max_expiries"]]
    univ0 = univ0[univ0["T"].isin(exps)].copy()
    # optional spread filter
    #if "max_rel_spread" in SUBSET:
    #    rel_spread = (univ0["ask"] - univ0["bid"]) / univ0["mid"].clip(lower=1e-8)
    #   univ0 = univ0[rel_spread <= SUBSET["max_rel_spread"]].copy()

    univ0 = univ0.sort_values(["T","K","cp"]).reset_index(drop=True)

    # build IV lookup from day0
    lookup_sigma_df = build_iv_lookup(iv0)

    # loop and compute PnL
    rows = []
    hT = 1.0 /365.0  # 1 day time step
    for r in univ0.itertuples(index=False):
        cp, K, T, mid0 = r.cp, float(r.K), float(r.T), float(r.mid)
        if T <= hT or K <= 0.0 or mid0 <= 0.0:
            continue

        # rq anchored at day0 center (so Δ is a proper ∂/∂S)
        rq0 = math.log(F0(T) / S0) / T

        # PDE price & bumped delta
        price0_pde = price_pde_only(cp, K, T, S0, lvf0, rq0, PDE_GRID)
        delta0_pde = pde_delta_bump(cp, K, T, S0, lvf0, rq0, PDE_GRID, eps=1e-3)

        # find same contract at day1 (match cp and K, choose closest T)
        d1 = day1[(day1["cp"]==cp) & (np.isclose(day1["K"], K))].copy()
        if d1.empty:
            continue
        a1 = pd.to_datetime(asof1)
        d1["T1"] = (pd.to_datetime(d1["expiry"]) - a1).dt.days / 365.0
        T1_target = max(T - hT, 1e-6)
        j = int((d1["T1"] - T1_target).abs().idxmin())
        mid1 = float(d1.loc[j, "mid"])
        S1 = float(d1.loc[j, "S"])
        T1 = float(d1.loc[j, "T1"])

        # BSM sticky baseline: sigma, df from day0 IV table
        sigma0, df0 = lookup_sigma_df(cp, K, T)
        if (sigma0 is None) or (df0 is None) or (sigma0 <= 0.0):
            continue

        price0_bsm = black_price(cp, F0(T), K, T, sigma0, df0)
        delta0_bsm = black_delta_spot(cp, F0(T), K, T, sigma0, df0, S0)

        # evolve one day: LV priced with day1 lvf1 and rq1
        # sticky-sigma priced with same sigma0 but F evolved with rq0
        rq1 = math.log(F1(T1) / S1) / T1
        price1_pde = price_pde_only(cp, K, T1, S1, lvf1, rq1, PDE_GRID)

        F0_T1 = S1 * math.exp(rq0 * T1) # sticky sigma calendar theta
        price1_bsm = black_price(cp, F0_T1, K, T1, sigma0, df0)

        # Delta-hedged PnL using day0 delta
        pnl_pde = (mid1 - mid0) - delta0_pde * (S1 - S0)
        pnl_bsm = (mid1 - mid0) - delta0_bsm * (S1 - S0)

        rows.append({
            "asof0": asof0, "asof1": asof1,
            "cp": cp, "K": K, "T": T,
            "price0_pde": price0_pde, "delta0_pde": delta0_pde,
            "price0_bsm": price0_bsm, "delta0_bsm": delta0_bsm,
            "S0": S0, "S1": S1, "T1": T1,
            "price1_pde": price1_pde, "price1_bsm": price1_bsm,
            "pnl_pde": pnl_pde, "pnl_bsm": pnl_bsm
        })

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {"rmse_pde": np.nan, "mae_pde": np.nan, "rmse_bsm": np.nan, "mae_bsm": np.nan, "N": 0}
    
    # bucketing for summaries
    F_for_bucket = F0 # bucket by day0 forward
    bt["m"] = bt["K"] / bt["T"].apply(F_for_bucket)
    bt["T_bucket"] = bt["T"].apply(bucket_T)
    bt["m_bucket"] = bt["m"].apply(bucket_m)

    # overall summary
    rmse_pde, mae_pde = _rmse_mae(bt["pnl_pde"])
    rmse_bsm, mae_bsm = _rmse_mae(bt["pnl_bsm"])
    summary = {
        "rmse_pde": rmse_pde,
        "mae_pde": mae_pde,
        "rmse_bsm": rmse_bsm,
        "mae_bsm": mae_bsm,
        "N": len(bt)
    }
    return bt, summary

def load_day(asof):
    files = day_files(asof)
    day = pd.read_csv(files["quotes"])
    iv = pd.read_csv(files["ivtab"])
    lv = pd.read_parquet(files["lvparq"])
    return day, iv, lv

def day_files(asof):
    ymd = asof.replace("-", "")
    return dict(
        quotes=f"data/quotes/{ymd}_quotes.csv",
        ivtab=f"artifacts/ivtable_{asof}.csv",
        lvparq=f"artifacts/local_vol_surface_{asof}.parquet",
        fwds=f"artifacts/forwards_{asof}.csv",
    )

# ------Build foward curve F(T) as a function ------
def F_curve_from_ivtab(ivtab: pd.DataFrame):
    # F in ivtable is per-row; we'll median by rounded T to be robust
    tmp = ivtab.copy()
    tmp["T_key"] = (tmp["T"]*10000).round().astype(int)
    F_by_T = tmp.groupby("T_key")["F"].median().reset_index()
    grid_T = F_by_T["T_key"].to_numpy() / 10000.0
    grid_F = F_by_T["F"].to_numpy()

    # piecewise linear in T
    def F_of_T(t):
        t = np.clip(t, grid_T.min(), grid_T.max())
        return float(np.interp(t, grid_T, grid_F))
    return F_of_T

# Single function for IV lookup from ivtable
def build_iv_lookup(ivtab: pd.DataFrame):
    base = (
        ivtab.assign(
            T_key=(ivtab["T"] * 10000).round().astype(int),
            K_key=(ivtab["K"] * 10).round().astype(int),
        )
        .set_index(["cp", "T_key", "K_key"])
    )
    def lookup(cp, K, T):
        key = (cp, int(round(T * 10000)), int(round(K)))
        if key in base.index:
            return float(base.loc[key, "iv"]), float(base.loc[key, "df"])
        # fallback: nearest strike within same tenor bucket
        sl = ivtab[(ivtab["cp"] == cp) & ((ivtab["T"] * 10000).round() == int(round(T * 10000)))]
        if sl.empty:
            return None, None
        j = (sl["K"] - K).abs().idxmin()
        return float(sl.loc[j, "iv"]), float(sl.loc[j, "df"])
    return lookup

# Robust bump-and-reprice delta for the PDE

def price_pde_only(cp, K, T, S0, lv_func, rq, grid_kwargs):
    """Return PDE price only (no delta), handy for bump-and-reprice."""
    price, _delta = crank_nicolson_localvol(
        cp = cp,
        K  = float(K),
        T  = float(T),
        S0 = float(S0),
        lv_func = lv_func,
        r = rq,
        q = 0.0,
        grid = make_spot_grid(S0, K, T, **grid_kwargs)
    )
    return float(price)

def pde_delta_bump(cp, K, T, S0, lv_func, rq, grid_kwargs, eps=1e-3):
    """Central finite-difference delta using PDE prices."""
    S_up = S0 * (1.0 + eps)
    S_down = S0 * (1.0 - eps)
    p_up = price_pde_only(cp, K, T, S_up, lv_func, rq, grid_kwargs)
    p_down = price_pde_only(cp, K, T, S_down, lv_func, rq, grid_kwargs)
    delta = (p_up - p_down) / (2.0 * S0 * eps)
    return float(delta)

# BSM function to compute price and delta
def black_price(cp: str, F: float, K: float, T: float, sigma: float, df: float):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0 or df <= 0:
        return max(df*(F - K), 0.0) if cp == "C" else max(df*(K - F), 0.0)
    vol_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) / vol_sqrtT) + 0.5 * vol_sqrtT
    d2 = d1 - vol_sqrtT
    sgn = 1.0 if cp == "C" else -1.0
    price = df * (sgn * (F * norm.cdf(sgn * d1) - K * norm.cdf(sgn * d2)))
    return price

def black_delta_spot(cp: str, F: float, K: float, T: float, sigma: float, df: float, S: float):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0 or df <= 0 or S <= 0:
        return 0.0
    vol_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) / vol_sqrtT) + 0.5 * vol_sqrtT
    sgn = 1.0 if cp == "C" else -1.0
    dP_dF = df * norm.cdf(sgn * d1) * sgn
    dF_dS = F / S
    delta = dP_dF * dF_dS
    return delta

def bucket_T(T):
    if T < 0.010: return "<1.0%"
    if T < 0.020: return "1.0-2.0%"
    if T < 0.030: return "2.0-3.0%"
    if T < 0.040: return "3.0-4.0%"
    return ">=4.0%"
def bucket_m(m):
    if m < 0.98: return "<0.98"
    if m < 1.00: return "0.98-1.00"
    if m < 1.02: return "1.00-1.02"
    return ">=1.02"

def _rmse_mae(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2))), float(np.mean(np.abs(x)))