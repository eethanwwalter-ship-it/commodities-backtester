"""Position sizer — converts composite scores into dollar positions.

The core idea: a signal score tells you DIRECTION and CONVICTION, but
not HOW MUCH to trade. Position sizing translates scores into positions
that target a consistent level of portfolio risk.

Vol-targeting
-------------
The standard approach at systematic funds:

    position_i(t) = score_i(t) * (target_vol / realized_vol_i(t))

Where:
  - score_i(t) is the composite signal for commodity i at date t
  - target_vol is the annualized volatility you want each position to
    contribute (e.g., 10% annualized)
  - realized_vol_i(t) is the trailing realized volatility of commodity i

This normalizes risk across commodities: a position in volatile nat gas
is mechanically smaller (in dollar terms) than a position in calmer
crude, because both are scaled to contribute the same vol to the book.

Without vol-targeting, a uniform $1M position in NG and CL would make
NG dominate your P&L variance simply because it moves more — your
"signal quality" becomes irrelevant vs. your vol exposure. This is
the single most important step in systematic portfolio construction.

Output
------
Positions are expressed as "vol-adjusted score" — dimensionless units.
To convert to actual contract counts, the analytics layer multiplies by
(capital * target_vol) / (contract_size * price * daily_vol * sqrt(252)).
This separation keeps the portfolio-construction layer independent of
AUM.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def size_positions(
    scores: pd.DataFrame,
    returns: dict[str, pd.DataFrame],
    vol_lookback: int = 63,
    vol_floor: float = 0.05,
    max_position: float | None = None,
) -> pd.DataFrame:
    """Convert composite scores into vol-targeted positions.

    Parameters
    ----------
    scores
        Wide DataFrame (index=date, cols=symbols) of composite signal
        scores from the combiner.
    returns
        Mapping of symbol → DataFrame with [date, return] columns.
        Used to estimate trailing realized volatility.
    vol_lookback
        Business-day window for trailing vol estimate. 63 ≈ 3 months.
    vol_floor
        Minimum annualized vol used in scaling, to prevent positions
        from exploding during quiet periods. 5% annualized is a
        reasonable floor for energy futures.
    max_position
        If set, clip positions at ±max_position after vol-scaling.
        Acts as a crude gross-exposure limit.

    Returns
    -------
    Wide DataFrame of vol-adjusted positions (same shape as scores).
    Positive = long, negative = short.
    """
    # Build realized-vol estimates for each symbol
    vol_estimates: dict[str, pd.Series] = {}
    for symbol, df in returns.items():
        if "date" not in df.columns or "return" not in df.columns:
            continue
        s = (
            df[["date", "return"]]
            .dropna()
            .drop_duplicates(subset="date")
            .set_index("date")["return"]
            .sort_index()
        )
        daily_vol = s.rolling(vol_lookback, min_periods=max(10, vol_lookback // 3)).std(ddof=0)
        ann_vol = (daily_vol * np.sqrt(252)).clip(lower=vol_floor)
        vol_estimates[symbol] = ann_vol

    if not vol_estimates:
        raise ValueError("No symbols had usable return data for vol estimation")

    # Build a wide vol DataFrame aligned to scores
    vol_wide = pd.DataFrame(vol_estimates).reindex(
        index=scores.index, columns=scores.columns,
    )
    # Forward-fill vol estimates to cover any gaps (weekends, holidays)
    vol_wide = vol_wide.ffill()

    # Vol-target: score / vol.  A score of +1 in a 20% vol commodity
    # gets a position of 5 (= 1/0.20), while the same score in a 40%
    # vol commodity gets 2.5.  Both contribute equally to portfolio vol.
    positions = scores / vol_wide.where(vol_wide > 0)

    if max_position is not None:
        positions = positions.clip(lower=-max_position, upper=max_position)

    positions.index.name = "date"

    logger.info(
        "Sized %d symbols, vol_lookback=%d, vol_floor=%.2f",
        len(scores.columns), vol_lookback, vol_floor,
    )
    return positions
