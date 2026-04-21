"""Continuous futures return series construction.

Background
----------
Futures contracts expire. To get a multi-year price series we must stitch
consecutive contracts. Naive stitching — just concatenating prices — creates
artificial returns at every roll because the expiring and next contract
trade at different prices (the "roll yield").

Example: Feb WTI expires at $78, Mar WTI trades at $79. If you stitch the
prices directly, you get a +$1 move that isn't a real P&L opportunity.
Run a momentum signal on that contaminated series and you'll find "edge"
that evaporates in production.

Method
------
We use within-contract returns only. For each day t:
    - Normal day:  return_t = gen1_t / gen1_{t-1} - 1
    - Roll day:    return_t = gen1_t / gen2_{t-1} - 1

On a roll day, the contract that was "gen2" yesterday becomes "gen1"
today — they're the same physical contract. Computing the return from
yesterday's gen2 to today's gen1 is therefore a valid within-contract
return that captures the market move without the roll-yield artifact.

Detecting roll dates
--------------------
Bloomberg's generic tickers auto-roll on contract expiry. We actually want
to roll EARLIER than expiry (typically 5 business days before) to stay in
the more liquid contract. So we:

1. Detect when the underlying contract changes by looking at the jump in
   gen1 relative to gen2 — when gen1_t approximately equals gen2_{t-1},
   the underlying has rolled.
2. Back up `days_before_expiry` business days from each detected roll to
   get our actual roll date.

For production use, you'd use the exchange's contract calendar directly
(first notice day, last trading day) rather than detecting from data.
The data-driven approach here is robust enough for research and avoids
having to maintain a contract calendar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RollSchedule:
    """Per-commodity schedule flagging which business days are roll days."""
    symbol: str
    schedule: pd.DataFrame  # columns: [date, is_roll_day]


def detect_raw_roll_dates(
    prices: pd.DataFrame,
    ratio_threshold: float = 0.5,
) -> list[pd.Timestamp]:
    """Detect the dates on which the underlying contract rolled.

    Intuition: on a roll day today's gen1 *is* what was gen2 yesterday,
    modulo one day of market movement. So:

        gap      = gen1_t - gen2_{t-1}             ≈ one day of market move
        spread   = gen2_{t-1} - gen1_{t-1}         ≈ the forward-curve slope

    On a normal day `gap ≈ -spread` (gen1 barely moves, so it's still
    ~`spread` away from gen2 yesterday). On a roll day `gap ≈ 0` because
    gen1 jumped up to where gen2 was.

    Detector: flag roll when |gap| / |spread| < ratio_threshold.

    This is robust to any spread size — contango or backwardation — as
    long as the spread is materially larger than one day's return. For
    markets where the spread is small relative to daily vol, a calendar-
    based detector (using contract expiry tables) is more reliable.
    """
    if "gen1" not in prices.columns or "gen2" not in prices.columns:
        raise ValueError("prices must contain 'gen1' and 'gen2' columns")

    df = (
        prices[["date", "gen1", "gen2"]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(df) < 3:
        return []

    spread_prev = (df["gen2"].shift(1) - df["gen1"].shift(1)).abs()
    gap = (df["gen1"] - df["gen2"].shift(1)).abs()

    # Guard against tiny/zero spreads (stale data, curve-inversion points)
    safe_spread = spread_prev.where(spread_prev > 1e-9)
    ratio = gap / safe_spread

    mask = ratio < ratio_threshold
    return df.loc[mask.fillna(False), "date"].tolist()


def build_calendar_roll_schedule(
    prices: pd.DataFrame,
    symbol: str,
    days_before_expiry: int = 0,
) -> RollSchedule:
    """Build a calendar-based roll schedule from price data.

    Flags the "natural" roll date — the day on which Bloomberg's generic
    tickers shift to a new underlying contract. On that day, today's gen1
    is the same physical contract as yesterday's gen2, so we compute the
    return as gen1_t / gen2_{t-1} - 1 to avoid the roll-yield jump that
    contaminates naively stitched series.

    `days_before_expiry` is currently unused (kept for API stability). A
    future extension will add true early-rolling — holding gen2 for N days
    before the natural roll — which requires switching return computation
    to gen2-on-gen2 over that window, not shifting this roll flag.
    """
    if days_before_expiry != 0:
        logger.warning(
            "%s: early-roll (days_before_expiry=%d) not yet implemented; "
            "using natural roll date",
            symbol, days_before_expiry,
        )

    raw_rolls = detect_raw_roll_dates(prices)
    if not raw_rolls:
        logger.warning("%s: no roll dates detected", symbol)
        schedule = (
            prices[["date"]]
            .assign(is_roll_day=False)
            .sort_values("date")
            .reset_index(drop=True)
        )
        return RollSchedule(symbol=symbol, schedule=schedule)

    roll_set = set(raw_rolls)
    schedule = (
        prices[["date"]]
        .drop_duplicates()
        .assign(is_roll_day=lambda d: d["date"].isin(roll_set))
        .sort_values("date")
        .reset_index(drop=True)
    )
    logger.info("%s: %d roll dates", symbol, len(roll_set))
    return RollSchedule(symbol=symbol, schedule=schedule)


def build_continuous_returns(
    prices: pd.DataFrame,
    roll: RollSchedule,
) -> pd.DataFrame:
    """Build a roll-adjusted continuous return series.

    Parameters
    ----------
    prices
        Wide-format price DataFrame with columns [date, gen1, gen2, ...].
    roll
        A RollSchedule flagging which dates are roll days.

    Returns
    -------
    DataFrame with columns:
        date          — observation date
        gen1          — raw front-month price (for reference)
        return        — roll-adjusted daily return
        price_cont    — cumulative continuous price index (starts at 100)
    """
    df = (
        prices[["date", "gen1", "gen2"]]
        .merge(roll.schedule[["date", "is_roll_day"]], on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["is_roll_day"] = df["is_roll_day"].fillna(False)
    df["gen1_prev"] = df["gen1"].shift(1)
    df["gen2_prev"] = df["gen2"].shift(1)

    # Default: within-gen1 return
    df["return"] = df["gen1"] / df["gen1_prev"] - 1.0
    # Roll days: yesterday's gen2 -> today's gen1 (same physical contract)
    mask = df["is_roll_day"] & df["gen2_prev"].notna()
    df.loc[mask, "return"] = df.loc[mask, "gen1"] / df.loc[mask, "gen2_prev"] - 1.0

    df["return"] = df["return"].fillna(0.0)
    df["price_cont"] = 100.0 * (1.0 + df["return"]).cumprod()

    return df[["date", "gen1", "return", "price_cont"]].reset_index(drop=True)
