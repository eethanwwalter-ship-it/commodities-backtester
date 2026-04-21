"""Smoke test for FundamentalSurpriseSignal.

Simulates weekly EIA inventory data with a known large draw injected
at a specific date, and verifies:
  [1] The surprise is detected on the correct print date
  [2] The sign is correct (draw → positive score for crude)
  [3] Exponential decay between prints works as expected
  [4] Multiple series contribute to the same commodity additively
  [5] A commodity not in the weight map remains NaN
  [6] Warm-up period before first usable surprise produces NaN
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.signals.base import SignalData
from src.signals.fundamental import (
    FundamentalSurpriseSignal,
    FundamentalSurpriseConfig,
)


def make_weekly_eia(
    series_name: str,
    n_weeks: int = 80,
    start: str = "2022-01-05",      # a Wednesday
    base_level: float = 400_000.0,  # kbbl
    weekly_noise_std: float = 2_000.0,
    shock_week: int = 50,           # inject a large draw at this week
    shock_size: float = -15_000.0,  # kbbl (negative = draw)
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate weekly EIA inventory series with a known shock.

    Generates a random walk of inventory levels with small weekly noise,
    then injects a single large draw at `shock_week`. The signal should
    detect this as a large negative surprise on that date.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_weeks, freq="7D")
    changes = rng.normal(0.0, weekly_noise_std, n_weeks)
    changes[0] = 0.0
    changes[shock_week] += shock_size  # inject the draw
    levels = base_level + np.cumsum(changes)
    return pd.DataFrame({
        "date": dates,
        "series_id": series_name,
        "value": levels,
    })


def make_continuous_stub(
    symbol: str,
    start: str = "2022-01-03",
    n_days: int = 600,
) -> pd.DataFrame:
    """Minimal continuous series stub — just provides the business-day grid."""
    dates = pd.bdate_range(start=start, periods=n_days)
    return pd.DataFrame({
        "date": dates,
        "return": 0.0,
        "price_cont": 100.0,
        "gen1": 100.0,
    })


def main() -> int:
    # Two EIA series, each with a shock at week 50 (different seeds).
    crude_stocks = make_weekly_eia("crude_stocks", shock_week=50, seed=1)
    gasoline_stocks = make_weekly_eia("gasoline_stocks", shock_week=50, seed=2)

    # Minimal continuous stubs so the signal has a business-day grid
    continuous = {
        "CL": make_continuous_stub("CL"),
        "XB": make_continuous_stub("XB"),
        "UNUSED": make_continuous_stub("UNUSED"),
    }

    data = SignalData(
        eia={"crude_stocks": crude_stocks, "gasoline_stocks": gasoline_stocks},
        continuous=continuous,
    )

    # Simple weight map: crude_stocks → CL, gasoline_stocks → XB
    weights = {
        ("crude_stocks", "CL"):      1.0,
        ("gasoline_stocks", "XB"):   1.0,
    }
    config = FundamentalSurpriseConfig(
        forecast_window=8,
        decay_halflife=3.0,
        series_weights=weights,
    )
    signal = FundamentalSurpriseSignal(config=config)
    sig = signal.run(data, standardize="none")

    # The shock is at week 50; find the actual date
    shock_date = crude_stocks.iloc[50]["date"]
    print(f"Shock date (week 50): {shock_date.date()}")

    # [1] Signal should fire on the shock date for CL
    # Find the closest business day on or after shock_date in our index
    shock_bday = sig.index[sig.index >= shock_date][0]
    shock_val_cl = sig.loc[shock_bday, "CL"]
    shock_val_xb = sig.loc[shock_bday, "XB"]
    print(f"\n[1] Signal on shock date ({shock_bday.date()}):")
    print(f"      CL = {shock_val_cl:+.3f}")
    print(f"      XB = {shock_val_xb:+.3f}")

    if not (shock_val_cl > 1.5):
        print(f"  FAIL: CL should show strong positive surprise (draw is bullish), "
              f"got {shock_val_cl:+.3f}")
        return 1
    if not (shock_val_xb > 1.5):
        print(f"  FAIL: XB should also show strong positive surprise, "
              f"got {shock_val_xb:+.3f}")
        return 1
    print("    Both commodities show strong positive surprise on shock date  ok")

    # [2] Sign check: a draw (negative change) should produce POSITIVE score
    #     (bullish). Already checked above, but be explicit.
    if shock_val_cl <= 0:
        print("  FAIL: draw should be bullish (positive score)")
        return 1
    print("\n[2] Sign convention: draw → positive score  ok")

    # [3] Exponential decay: the signal should be weaker 3 business days
    #     after the shock than on the shock date itself.
    decay_check_date = sig.index[sig.index >= shock_bday][3]  # 3 bdays later
    decayed_cl = sig.loc[decay_check_date, "CL"]
    print(f"\n[3] Exponential decay check:")
    print(f"      CL on shock date:     {shock_val_cl:+.3f}")
    print(f"      CL 3 bdays later:     {decayed_cl:+.3f}")
    print(f"      Ratio: {decayed_cl / shock_val_cl:.3f} "
          f"(expect ~0.5 for halflife=3)")

    if not (0.0 < decayed_cl < shock_val_cl):
        print(f"  FAIL: signal should decay but stay positive")
        return 1
    # With halflife=3, after 3 days the ratio should be ~0.5
    ratio = decayed_cl / shock_val_cl
    if not (0.3 < ratio < 0.7):
        print(f"  FAIL: decay ratio {ratio:.3f} not near 0.5")
        return 1
    print("    Decay ratio is near 0.5 at halflife  ok")

    # [4] Before the shock, signal should be mild on average (individual
    #     weeks can be noisy — that's realistic for EIA data). The shock
    #     should clearly stand out vs the median pre-shock magnitude.
    pre_shock = sig.loc[sig.index < shock_bday, "CL"].dropna()
    if len(pre_shock) > 0:
        pre_median = pre_shock.abs().median()
        print(f"\n[4] Pre-shock median |CL signal|: {pre_median:.3f}")
        print(f"    Shock signal:                 {shock_val_cl:.3f}")
        if not (shock_val_cl > pre_median * 1.5):
            print("  FAIL: shock should clearly exceed median pre-shock level")
            return 1
        print("    Shock is well above typical pre-shock levels  ok")

    # [5] UNUSED commodity should be NaN throughout
    if "UNUSED" in sig.columns and sig["UNUSED"].notna().any():
        print("\n  FAIL: UNUSED commodity should be NaN (not in weight map)")
        return 1
    print("\n[5] UNUSED commodity is NaN throughout  ok")

    # [6] Warm-up: with forecast_window=8, the first ~9 weeks should
    #     produce no surprise (need 1 diff + 8 rolling window). Check
    #     that the first few weeks of business days are NaN for CL.
    first_print = crude_stocks.iloc[0]["date"]
    early_window = sig.loc[
        sig.index < first_print + pd.Timedelta(weeks=10), "CL"
    ]
    early_nan_frac = early_window.isna().mean()
    print(f"\n[6] Warm-up (first 10 weeks): {early_nan_frac:.0%} NaN")
    if early_nan_frac < 0.5:
        print("  FAIL: early window should be mostly NaN during warm-up")
        return 1
    print("    Warm-up period produces NaN as expected  ok")

    # [7] Additive contribution test: if we add crude_stocks → XB with
    #     weight=0.5, the XB signal should increase at the shock date.
    weights_multi = {
        ("crude_stocks", "CL"):    1.0,
        ("crude_stocks", "XB"):    0.5,  # now crude also hits XB
        ("gasoline_stocks", "XB"): 1.0,
    }
    config_multi = FundamentalSurpriseConfig(
        forecast_window=8,
        decay_halflife=3.0,
        series_weights=weights_multi,
    )
    sig_multi = FundamentalSurpriseSignal(config=config_multi).run(
        data, standardize="none",
    )
    xb_single = sig.loc[shock_bday, "XB"]
    xb_multi = sig_multi.loc[shock_bday, "XB"]
    print(f"\n[7] Additive contribution test at shock date:")
    print(f"      XB (gasoline only):          {xb_single:+.3f}")
    print(f"      XB (gasoline + 0.5*crude):   {xb_multi:+.3f}")
    if not (xb_multi > xb_single):
        print("  FAIL: adding crude contribution should increase XB signal")
        return 1
    print("    Additional series contribution increases signal  ok")

    print("\n  All fundamental-surprise checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
