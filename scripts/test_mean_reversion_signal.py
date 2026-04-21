"""Smoke test for SpreadMeanReversionSignal.

Simulates cointegrated pairs with a deliberately stretched spread and
verifies:
  [1] The signal correctly identifies the rich vs cheap leg
  [2] Contributions from multiple spreads sum correctly for a
      commodity appearing in two pairs
  [3] log_ratio and difference kinds both produce sensible output
  [4] Symbols not in any spread appear as NaN (not spuriously zero)
  [5] Warm-up period correctly produces NaN
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.signals.base import SignalData
from src.signals.mean_reversion import Spread, SpreadMeanReversionSignal


def make_cointegrated_pair(
    sym_a: str,
    sym_b: str,
    n_days: int = 400,
    start: str = "2022-01-03",
    init_price: float = 80.0,
    common_vol: float = 0.01,
    idio_vol: float = 0.003,
    stretch_magnitude: float = 0.10,
    stretch_start: int = 200,
    stretch_end: int = 250,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two cointegrated continuous series that briefly diverge.

    Both commodities share a common-factor return path, with small idio
    noise. Between `stretch_start` and `stretch_end`, sym_a gets a
    positive boost making its log ratio vs sym_b climb above its usual
    zero mean. After the window closes, the pair drifts together again.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    common = rng.normal(0.0, common_vol, n_days)
    idio_a = rng.normal(0.0, idio_vol, n_days)
    idio_b = rng.normal(0.0, idio_vol, n_days)

    ret_a = common + idio_a
    ret_b = common + idio_b

    # Inject a stretch in sym_a: small daily boost during the window,
    # then mean-revert over the following window.
    boost = stretch_magnitude / (stretch_end - stretch_start)
    ret_a[stretch_start:stretch_end] += boost

    ret_a[0] = 0.0
    ret_b[0] = 0.0
    price_a = init_price * np.cumprod(1.0 + ret_a)
    price_b = init_price * np.cumprod(1.0 + ret_b)

    df_a = pd.DataFrame({
        "date": dates, "return": ret_a, "price_cont": price_a, "gen1": price_a,
    })
    df_b = pd.DataFrame({
        "date": dates, "return": ret_b, "price_cont": price_b, "gen1": price_b,
    })
    return df_a, df_b


def main() -> int:
    # Build two independent cointegrated pairs, plus a third commodity
    # that's in NO spread (should remain NaN in output).
    a1, b1 = make_cointegrated_pair(
        "WTI_LIKE", "BRENT_LIKE",
        stretch_start=200, stretch_end=250, seed=1,
    )
    a2, b2 = make_cointegrated_pair(
        "HO_LIKE", "RB_LIKE",
        stretch_start=150, stretch_end=200, seed=2,
    )
    c1, _ = make_cointegrated_pair("UNUSED", "_IGNORE", seed=99)

    continuous = {
        "WTI_LIKE":   a1,
        "BRENT_LIKE": b1,
        "HO_LIKE":    a2,
        "RB_LIKE":    b2,
        "UNUSED":     c1,
    }
    raw_futures = dict(continuous)  # same frames; gen1 column is present
    data = SignalData(continuous=continuous, raw_futures=raw_futures)

    spreads = [
        Spread("WTI_LIKE", "BRENT_LIKE", kind="log_ratio", label="WTI-Brent"),
        Spread("HO_LIKE",  "RB_LIKE",    kind="log_ratio", label="HO-RB"),
    ]
    signal = SpreadMeanReversionSignal(spreads=spreads, lookback=60)
    sig = signal.run(data, standardize="none")

    # Each pair is checked at its own peak-stretch date (a few days after
    # its stretch_end, once the rolling window has fully absorbed the move).

    # [1] WTI-Brent (stretch 200..250) — peak at ~day 253
    wti_peak_date = sig.index[253]
    wti_peak = sig.loc[wti_peak_date]
    print(f"[1] WTI-Brent signal at peak ({wti_peak_date.date()}):")
    for sym in ["WTI_LIKE", "BRENT_LIKE"]:
        print(f"      {sym:<12} = {wti_peak[sym]:+.3f}")
    if not (wti_peak["WTI_LIKE"] < -1.0):
        print(f"  FAIL: WTI_LIKE should be < -1 at peak, got {wti_peak['WTI_LIKE']:+.3f}")
        return 1
    if not (wti_peak["BRENT_LIKE"] > 1.0):
        print(f"  FAIL: BRENT_LIKE should be > +1, got {wti_peak['BRENT_LIKE']:+.3f}")
        return 1
    if not np.isclose(wti_peak["WTI_LIKE"], -wti_peak["BRENT_LIKE"], atol=1e-9):
        print("  FAIL: WTI/Brent contributions should be equal and opposite")
        return 1
    print("    Rich leg short, cheap leg long, contributions sum to zero  ok")

    # [2] HO-RB (stretch 150..200) — peak at ~day 203
    ho_peak_date = sig.index[203]
    ho_peak = sig.loc[ho_peak_date]
    print(f"\n[2] HO-RB signal at its own peak ({ho_peak_date.date()}):")
    for sym in ["HO_LIKE", "RB_LIKE"]:
        print(f"      {sym:<12} = {ho_peak[sym]:+.3f}")
    if not (ho_peak["HO_LIKE"] < -1.0):
        print(f"  FAIL: HO_LIKE should be < -1 at its peak, got {ho_peak['HO_LIKE']:+.3f}")
        return 1
    if not (ho_peak["RB_LIKE"] > 1.0):
        print(f"  FAIL: RB_LIKE should be > +1, got {ho_peak['RB_LIKE']:+.3f}")
        return 1
    print("    HO-RB pair also shows expected rich/cheap signal  ok")

    # [3] UNUSED is in no spread: must be NaN throughout
    if "UNUSED" in sig.columns and sig["UNUSED"].notna().any():
        print("  FAIL: UNUSED commodity should be NaN throughout (not in any spread)")
        return 1
    print("    UNUSED commodity (not in any spread) is NaN throughout  ok")

    # [5] Warm-up: with min_periods=30 (lookback=60 * 0.5), the first
    #     29 obs should be NaN (index 29 is the first with enough data).
    warmup_end = 29
    if not sig.iloc[:warmup_end]["WTI_LIKE"].isna().all():
        print(f"  FAIL: warm-up should be NaN for first {warmup_end} obs")
        return 1
    if sig.iloc[warmup_end:warmup_end + 5]["WTI_LIKE"].isna().all():
        print(f"  FAIL: signal should start producing values at obs {warmup_end}")
        return 1
    print(f"    First {warmup_end} obs are NaN (warm-up)  ok")

    # [6] Post-reversion: by the time we're well past stretch_end + lookback/2,
    #     the spread should have mostly reverted and z-score should be close
    #     to zero. Check at end of series.
    end_row = sig.iloc[-1]
    print(f"\n[6] Signal at end of series ({sig.index[-1].date()}) — "
          "expect mild magnitudes after reversion:")
    for sym in ["WTI_LIKE", "BRENT_LIKE", "HO_LIKE", "RB_LIKE"]:
        print(f"      {sym:<12} = {end_row[sym]:+.3f}")

    # [7] Overlapping-commodity test: create a spread where one symbol
    #     appears in two different pairs.  Its contribution should be
    #     the sum of contributions from each pair.
    print("\n[7] Multi-pair contribution sum test:")
    multi_spreads = [
        Spread("WTI_LIKE", "BRENT_LIKE", kind="log_ratio"),
        Spread("WTI_LIKE", "HO_LIKE",    kind="log_ratio"),
    ]
    single_spread_ab = SpreadMeanReversionSignal(
        spreads=[multi_spreads[0]], lookback=60,
    ).run(data, standardize="none")
    single_spread_ac = SpreadMeanReversionSignal(
        spreads=[multi_spreads[1]], lookback=60,
    ).run(data, standardize="none")
    combined = SpreadMeanReversionSignal(
        spreads=multi_spreads, lookback=60,
    ).run(data, standardize="none")

    # On the WTI peak date, WTI contribution in combined should equal sum
    # of the two individual pair contributions.
    wti_from_ab = single_spread_ab.loc[wti_peak_date, "WTI_LIKE"]
    wti_from_ac = single_spread_ac.loc[wti_peak_date, "WTI_LIKE"]
    wti_combined = combined.loc[wti_peak_date, "WTI_LIKE"]
    print(f"      WTI from (WTI-Brent):  {wti_from_ab:+.3f}")
    print(f"      WTI from (WTI-HO):     {wti_from_ac:+.3f}")
    print(f"      WTI combined:          {wti_combined:+.3f}")
    print(f"      Sum of individual:     {wti_from_ab + wti_from_ac:+.3f}")
    if not np.isclose(wti_combined, wti_from_ab + wti_from_ac, atol=1e-9):
        print("  FAIL: combined signal should equal sum of individual contributions")
        return 1
    print("    Contributions sum correctly across pairs  ok")

    # [8] `difference` kind sanity check — should also produce reasonable
    #     output for same-unit legs.
    diff_spreads = [Spread("WTI_LIKE", "BRENT_LIKE", kind="difference")]
    diff_sig = SpreadMeanReversionSignal(
        spreads=diff_spreads, lookback=60,
    ).run(data, standardize="none")
    diff_peak = diff_sig.loc[wti_peak_date]
    print(f"\n[8] `difference` kind on WTI-Brent at peak date:")
    print(f"      WTI_LIKE   = {diff_peak['WTI_LIKE']:+.3f}")
    print(f"      BRENT_LIKE = {diff_peak['BRENT_LIKE']:+.3f}")
    if not (diff_peak["WTI_LIKE"] < -1.0 and diff_peak["BRENT_LIKE"] > 1.0):
        print("  FAIL: difference kind should also identify rich/cheap correctly")
        return 1
    print("    `difference` kind produces consistent signal  ok")

    print("\n  All mean-reversion checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
