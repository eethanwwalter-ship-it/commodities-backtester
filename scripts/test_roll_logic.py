"""Smoke test: verify roll logic on synthetic futures data.

Simulates a year of front/second-month prices with clean monthly rolls
and confirms that:
  1. Raw rolls are detected correctly (~12 per year for monthly contracts)
  2. Continuous returns don't contain the roll-yield jumps
  3. A naive stitched series DOES contain them (sanity check that we're
     actually fixing a real problem)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.universe.roll import (
    detect_raw_roll_dates,
    build_calendar_roll_schedule,
    build_continuous_returns,
)


def make_synthetic_futures(
    start: str = "2023-01-03",
    n_days: int = 252,
    init_price: float = 75.0,
    daily_vol: float = 0.005,          # 0.5% daily — calm regime
    roll_yield_per_month: float = 2.00,  # $2 contango — stressed market
    seed: int = 42,
) -> pd.DataFrame:
    # Note on params: for the data-driven roll detector to work reliably,
    # the forward-curve spread must dominate one day of market movement.
    # These synthetic params set spread/level (~2.7%) well above daily vol
    # (0.5%). In real crude data the separation is often narrower, in which
    # case calendar-based detection (exchange contract expiry tables) is
    # more robust — see the note in detect_raw_roll_dates().
    """Simulate gen1/gen2 with monthly rolls and a realistic contango spread.

    Each month gen2 = gen1 + roll_yield. On the first business day of each
    month, the "front contract" expires: gen1 jumps up to what was gen2,
    and gen2 jumps up by another roll_yield. The underlying market price
    follows a GBM-style random walk — real P&L.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    gen1 = np.empty(n_days)
    gen2 = np.empty(n_days)
    gen1[0] = init_price
    gen2[0] = init_price + roll_yield_per_month

    prev_month = dates[0].month
    for i in range(1, n_days):
        ret = rng.normal(0.0, daily_vol)  # real market move
        if dates[i].month != prev_month:
            # Contract expired overnight; generics shift down by one
            gen1[i] = gen2[i - 1] * (1.0 + ret)
            gen2[i] = gen1[i] + roll_yield_per_month
            prev_month = dates[i].month
        else:
            gen1[i] = gen1[i - 1] * (1.0 + ret)
            gen2[i] = gen1[i] + roll_yield_per_month

    return pd.DataFrame({"date": dates, "gen1": gen1, "gen2": gen2})


def main() -> int:
    prices = make_synthetic_futures()
    print(f"Synthetic data: {len(prices)} days, "
          f"{prices['date'].min().date()} to {prices['date'].max().date()}")

    # 1. Roll detection
    raw_rolls = detect_raw_roll_dates(prices)
    print(f"\n[1] Detected {len(raw_rolls)} raw roll dates "
          f"(expect ~11-12 for one year of monthly contracts)")
    for rd in raw_rolls[:5]:
        print(f"      {rd.date()}")
    if not (8 <= len(raw_rolls) <= 14):
        print("    FAIL: unexpected number of rolls")
        return 1

    # 2. Continuous return series
    schedule = build_calendar_roll_schedule(prices, "TEST", days_before_expiry=0)
    cont = build_continuous_returns(prices, schedule)
    n_roll_days = int(schedule.schedule["is_roll_day"].sum())
    print(f"\n[2] Continuous series: {len(cont)} obs, {n_roll_days} roll days")

    # 3. Compare continuous returns vs naive stitched returns
    naive_returns = prices["gen1"].pct_change().fillna(0.0)
    cont_returns = cont["return"]

    # Isolate roll-day returns — where naive stitching contaminates with the
    # ~50¢ downward jump when gen1 expires and is replaced with the cheaper
    # next contract.
    is_month_start = prices["date"].dt.month != prices["date"].shift(1).dt.month
    naive_roll_rets = naive_returns[is_month_start]
    print(f"\n[3] On contract-change days (naive series):")
    print(f"      mean return = {naive_roll_rets.mean():+.4f}")
    print(f"      std  return = {naive_roll_rets.std():.4f}")

    roll_day_mask = cont["date"].isin(schedule.schedule.loc[schedule.schedule["is_roll_day"], "date"])
    cont_roll_rets = cont_returns[roll_day_mask]
    print(f"    On roll days (continuous series):")
    print(f"      mean return = {cont_roll_rets.mean():+.4f}")
    print(f"      std  return = {cont_roll_rets.std():.4f}")

    # The continuous series should have roll-day returns that look like
    # ordinary daily returns (~1.5% vol, zero mean). The naive series will
    # have a strong negative mean from the repeated downward jumps.
    if abs(naive_roll_rets.mean()) < 0.002:
        print("    FAIL: naive series didn't show roll-yield contamination — "
              "synthetic data is wrong")
        return 1
    if abs(cont_roll_rets.mean()) > 0.005:
        print("    FAIL: continuous series still has roll-yield bias")
        return 1

    # 4. Cumulative performance sanity check
    naive_total = (1.0 + naive_returns).prod() - 1.0
    cont_total  = (1.0 + cont_returns).prod()  - 1.0
    print(f"\n[4] Cumulative return over year:")
    print(f"      naive   = {naive_total:+.2%}  (contaminated)")
    print(f"      cont    = {cont_total:+.2%}  (clean)")

    print("\n  All roll-logic checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
