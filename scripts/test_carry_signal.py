"""Smoke test for CarrySignal.

Simulates three commodities with known curve shapes and verifies:
  1. Raw carry signs match expectations (positive for backwardation,
     negative for contango).
  2. Raw magnitudes match theoretical carry within noise tolerance.
  3. Cross-sectional z-scores correctly rank backwardation > flat > contango.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.signals.base import SignalData
from src.signals.carry import CarrySignal


def make_curve(
    symbol: str,
    spread_pct: float,
    start: str = "2023-01-03",
    n_days: int = 252,
    init_price: float = 75.0,
    daily_vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate gen1/gen2 with a constant *proportional* spread.

    spread_pct > 0 => contango   (gen2 > gen1)  => negative carry
    spread_pct < 0 => backwardation              => positive carry
    spread_pct = 0 => flat curve                 => zero carry
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    rets = rng.normal(0.0, daily_vol, n_days)
    rets[0] = 0.0
    gen1 = init_price * np.cumprod(1.0 + rets)
    gen2 = gen1 * (1.0 + spread_pct)

    return pd.DataFrame({"date": dates, "gen1": gen1, "gen2": gen2})


def main() -> int:
    # Three commodities, same underlying price dynamics but different curves.
    universe = {
        "BACK": make_curve("BACK", spread_pct=-0.025, seed=1),  # -2.5% (back.)
        "FLAT": make_curve("FLAT", spread_pct=+0.000, seed=2),  #  0.0%
        "CONT": make_curve("CONT", spread_pct=+0.025, seed=3),  # +2.5% (cont.)
    }
    data = SignalData(raw_futures=universe)

    signal = CarrySignal(smoothing_window=5, annualization_factor=12.0)

    # [1] Raw signal — should recover theoretical carry
    raw = signal.run(data, standardize="none")
    latest = raw.iloc[-1]
    print("[1] Raw annualized carry (last date):")
    for sym, val in latest.items():
        print(f"      {sym}  = {val:+.4f}")

    # Theoretical annualized carry = (1 - (1 + spread)) / (1 + spread) * 12
    #                              = -spread / (1 + spread) * 12
    expectations = {
        "BACK": -(-0.025) / (1 + -0.025) * 12,   # ≈ +0.3077
        "FLAT":  0.0,
        "CONT": -(+0.025) / (1 + +0.025) * 12,   # ≈ -0.2927
    }
    for sym, expected in expectations.items():
        actual = latest[sym]
        if abs(actual - expected) > 0.01:
            print(f"  FAIL: {sym} carry {actual:+.4f} != expected {expected:+.4f}")
            return 1
    print("    Raw carry magnitudes match theory within tolerance.")

    # [2] Signs are right
    print("\n[2] Sign check:")
    if not (latest["BACK"] > 0 > latest["CONT"]):
        print(f"  FAIL: BACK should be positive, CONT negative")
        return 1
    if abs(latest["FLAT"]) > 0.01:
        print(f"  FAIL: FLAT should be ~0, got {latest['FLAT']:+.4f}")
        return 1
    print(f"    BACK ({latest['BACK']:+.3f}) > FLAT ({latest['FLAT']:+.3f}) "
          f"> CONT ({latest['CONT']:+.3f})  ok")

    # [3] Standardized signal — z-scores should mirror the raw ranking
    zscored = signal.run(data, standardize="zscore")
    z_latest = zscored.iloc[-1]
    print("\n[3] Cross-sectional z-scores (last date):")
    for sym, val in z_latest.items():
        print(f"      {sym}  = {val:+.3f}")
    ranking = z_latest.sort_values(ascending=False).index.tolist()
    if ranking != ["BACK", "FLAT", "CONT"]:
        print(f"  FAIL: wrong ranking {ranking}, expected [BACK, FLAT, CONT]")
        return 1
    print("    Ranking: backwardation > flat > contango  ok")

    # [4] Rank standardization should also work (symmetric around 0)
    ranked = signal.run(data, standardize="rank")
    r_latest = ranked.iloc[-1]
    print("\n[4] Rank-standardized signal (last date):")
    for sym, val in r_latest.items():
        print(f"      {sym}  = {val:+.3f}")
    if not (r_latest["BACK"] > r_latest["FLAT"] > r_latest["CONT"]):
        print("  FAIL: rank ordering wrong")
        return 1
    # For 3 items, expect exactly -1, 0, +1
    if abs(r_latest["BACK"] - 1.0) > 1e-9 or abs(r_latest["CONT"] + 1.0) > 1e-9:
        print(f"  FAIL: rank bounds wrong, got {r_latest.to_dict()}")
        return 1
    if abs(r_latest["FLAT"]) > 1e-9:
        print(f"  FAIL: middle rank should be 0, got {r_latest['FLAT']:+.4f}")
        return 1

    # [5] Time-series stability — signal shouldn't flip sign from noise
    back_series = raw["BACK"].dropna()
    if (back_series < 0).mean() > 0.05:
        print(f"  FAIL: BACK carry went negative too often "
              f"({(back_series < 0).mean():.1%} of days)")
        return 1
    print("\n[5] BACK carry stayed positive on "
          f"{(back_series > 0).mean():.1%} of days  ok")

    print("\n  All carry-signal checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
