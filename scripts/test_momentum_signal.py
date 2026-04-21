"""Smoke test for TSMomentum and XSMomentum signals.

Four synthetic commodities with known dynamics. Parameters are chosen
with a high signal-to-noise ratio so the mechanism is clearly testable;
they are NOT representative of real commodity markets (real Sharpe
ratios are rarely > 1).

  UP      : +25% annualized drift, 12% vol
  FLAT    :   0% drift,             12% vol
  DOWN    : -25% drift,             12% vol
  UP_VOL  : +25% drift,             24% vol

Verifies:
  1. TSMOM signs match drift directions (UP > FLAT > DOWN)
  2. XSMOM z-scores rank UP > FLAT > DOWN
  3. Formula check: signal output at any date equals ret_cum / vol
     with inputs hand-computed from the raw return series
  4. With risk_adjust=False, the signal equals raw ret_cum
  5. Signal is temporally stable — rarely flips sign for UP or DOWN
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.signals.base import SignalData
from src.signals.momentum import TSMomentumSignal, XSMomentumSignal


def make_continuous(
    symbol: str,
    annual_drift: float,
    annual_vol: float,
    n_days: int = 500,
    start: str = "2022-01-03",
    init_price: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a continuous return series with target drift and vol."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    rets = rng.normal(daily_mu, daily_sig, n_days)
    rets[0] = 0.0
    price_cont = init_price * np.cumprod(1.0 + rets)
    return pd.DataFrame({
        "date": dates,
        "return": rets,
        "price_cont": price_cont,
        "gen1": price_cont,
    })


def main() -> int:
    universe = {
        "UP":     make_continuous("UP",     +0.25, 0.12, seed=1),
        "FLAT":   make_continuous("FLAT",    0.00, 0.12, seed=2),
        "DOWN":   make_continuous("DOWN",   -0.25, 0.12, seed=3),
        "UP_VOL": make_continuous("UP_VOL", +0.25, 0.24, seed=4),
    }
    data = SignalData(continuous=universe)

    # ------------------------------------------------------------------
    # [1] Time-series momentum, window-averaged
    # ------------------------------------------------------------------
    tsmom = TSMomentumSignal(lookback_days=252, skip_days=21, risk_adjust=True)
    raw = tsmom.run(data, standardize="none").dropna()
    if raw.empty:
        print("  FAIL: TSMOM produced no valid dates "
              "(need > lookback + skip days of history)")
        return 1

    # Mean over eligible window — more robust than a single date, since
    # overlapping windows make successive signal values highly correlated.
    mean_sig = raw.mean()
    print(f"[1] TSMOM (risk-adjusted), mean over {len(raw)} eligible dates:")
    for sym, val in mean_sig.items():
        print(f"      {sym:8s} = {val:+.3f}")

    if mean_sig["UP"] <= 0:
        print(f"  FAIL: UP should average positive TSMOM, got {mean_sig['UP']:+.3f}")
        return 1
    if mean_sig["DOWN"] >= 0:
        print(f"  FAIL: DOWN should average negative TSMOM, got {mean_sig['DOWN']:+.3f}")
        return 1
    if not (mean_sig["UP"] > mean_sig["FLAT"] > mean_sig["DOWN"]):
        print(f"  FAIL: UP > FLAT > DOWN ordering violated")
        return 1
    print("    Sign and ordering checks ok")

    # ------------------------------------------------------------------
    # [2] Cross-sectional momentum: z-score rankings
    # ------------------------------------------------------------------
    xsmom = XSMomentumSignal(lookback_days=252, skip_days=21, risk_adjust=True)
    z = xsmom.run(data, standardize="zscore").dropna()
    z_mean = z.mean()
    print(f"\n[2] XSMOM z-scores, mean over {len(z)} eligible dates:")
    for sym, val in z_mean.items():
        print(f"      {sym:8s} = {val:+.3f}")

    if not (z_mean["UP"] > z_mean["FLAT"] > z_mean["DOWN"]):
        print("  FAIL: UP > FLAT > DOWN z-score ordering violated")
        return 1
    print("    UP > FLAT > DOWN ordering ok")

    # ------------------------------------------------------------------
    # [3] Formula check: signal output equals ret_cum / vol, computed
    # independently from the input data.
    # ------------------------------------------------------------------
    print(f"\n[3] Formula check (signal output vs. hand-computed):")
    lookback, skip = 252, 21

    # Hand-compute for UP_VOL at the last eligible date:
    df = universe["UP_VOL"].set_index("date").sort_index()
    n = len(df)
    end_idx   = n - 1 - skip              # index of end-of-window price
    start_idx = end_idx - lookback        # index of start-of-window price
    end_price   = df["price_cont"].iloc[end_idx]
    start_price = df["price_cont"].iloc[start_idx]
    ret_cum_manual = end_price / start_price - 1.0

    # Vol over the window: std of daily returns from start_idx+1 to end_idx
    window_returns = df["return"].iloc[start_idx + 1: end_idx + 1]
    vol_manual = window_returns.std() * np.sqrt(252)
    expected_score = ret_cum_manual / vol_manual

    actual_score = raw["UP_VOL"].iloc[-1]
    print(f"      ret_cum (manual) = {ret_cum_manual:+.4f}")
    print(f"      vol     (manual) = {vol_manual:.4f}")
    print(f"      expected score   = {expected_score:+.4f}")
    print(f"      actual score     = {actual_score:+.4f}")
    if abs(actual_score - expected_score) > 1e-6:
        print(f"  FAIL: signal doesn't match formula (diff "
              f"{abs(actual_score - expected_score):.2e})")
        return 1
    print("    Signal output matches ret_cum / vol formula")

    # ------------------------------------------------------------------
    # [4] Without risk adjustment, signal equals raw cumulative return.
    # ------------------------------------------------------------------
    tsmom_raw = TSMomentumSignal(
        lookback_days=252, skip_days=21, risk_adjust=False,
    )
    no_adj = tsmom_raw.run(data, standardize="none").dropna()
    actual_raw = no_adj["UP_VOL"].iloc[-1]
    print(f"\n[4] Without risk adjustment:")
    print(f"      signal (UP_VOL, last) = {actual_raw:+.4f}")
    print(f"      ret_cum manual        = {ret_cum_manual:+.4f}")
    if abs(actual_raw - ret_cum_manual) > 1e-6:
        print(f"  FAIL: signal (no risk-adj) should equal ret_cum")
        return 1
    print("    Non-risk-adjusted signal equals raw cumulative return")

    # ------------------------------------------------------------------
    # [5] Stability: UP is mostly positive, DOWN mostly negative
    # ------------------------------------------------------------------
    up_pct_pos = (raw["UP"].dropna() > 0).mean()
    down_pct_neg = (raw["DOWN"].dropna() < 0).mean()
    print(f"\n[5] Signal stability over eligible window:")
    print(f"      UP     positive on {up_pct_pos:.1%} of days")
    print(f"      DOWN   negative on {down_pct_neg:.1%} of days")
    if up_pct_pos < 0.90 or down_pct_neg < 0.90:
        print("  FAIL: momentum should flip signs rarely at these SNRs")
        return 1

    print("\n  All momentum-signal checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
