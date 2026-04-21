"""Carry (roll-yield) signal.

Theory
------
For a futures contract, the slope of the forward curve between the front
and second-nearby contract tells you the roll yield — the P&L you earn
(or pay) just from the passage of time as the curve rolls down (or up).

Annualized carry, front-vs-second:

        carry = (F_near - F_far) / F_far * (periods_per_year / periods_between)

    Backwardation (F_near > F_far): positive carry — long holder earns the
        spread by rolling into a cheaper contract. This typically happens
        when physical supply is tight.

    Contango (F_near < F_far): negative carry — long holder pays the
        spread rolling up the curve. Typically signals comfortable supply
        or high storage costs.

Why it's an alpha signal
------------------------
Empirically, across a wide range of commodities, backwardated contracts
have outperformed contangoed ones over long samples. Three standard
explanations:
  1. Hedging-pressure / normal-backwardation theory: producers hedge
     their output, pushing deferred prices down relative to spot, and
     speculators demand a risk premium for taking the other side.
  2. Inventory theory: backwardation reflects scarcity; scarce commodities
     subsequently outperform as spot prices rise to clear the market.
  3. Market segmentation: each commodity's forward curve reflects its own
     storage cost and convenience yield, creating cross-sectional spreads
     that systematic traders can harvest.

None of these are perfect; carry alone is a weak signal with meaningful
drawdowns. It's a staple building block in a multi-signal portfolio.

Implementation
--------------
For monthly contracts, `periods_per_year = 12` and `periods_between = 1`.
Strictly, periods_between should use the actual expiry dates of gen1 and
gen2 (not always exactly one month apart — natural gas and crude have
monthly expiries, but some ags are quarterly). The annualization factor
is stored per-symbol in the universe config; defaults to 12.

Output
------
Raw signal = annualized carry per commodity per date. Standardization
via `Signal.run()` cross-sectionally z-scores so the signal is
directly comparable to other signals (momentum, mean-reversion).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import Signal, SignalData

logger = logging.getLogger(__name__)


class CarrySignal(Signal):
    """Front-vs-second annualized carry.

    Parameters
    ----------
    smoothing_window
        Rolling mean window (in business days) applied to raw carry before
        output. Carry spreads are noisy day-to-day; a 5-day smooth is
        typically enough to damp microstructure noise without introducing
        material lag. Set to 1 to disable.
    annualization_factor
        Periods per year for the spread. 12 for monthly contracts (default).
        For quarterly contracts, set to 4.
    """
    name = "carry"

    def __init__(
        self,
        smoothing_window: int = 5,
        annualization_factor: float = 12.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        self.smoothing_window = smoothing_window
        self.annualization_factor = annualization_factor

    def compute(self, data: SignalData) -> pd.DataFrame:
        if not data.raw_futures:
            raise ValueError(
                "CarrySignal requires raw_futures (gen1, gen2 per symbol)"
            )

        series_by_symbol: dict[str, pd.Series] = {}
        for symbol, df in data.raw_futures.items():
            if not {"date", "gen1", "gen2"}.issubset(df.columns):
                logger.warning(
                    "%s: missing gen1/gen2, skipping", symbol,
                )
                continue

            s = (
                df[["date", "gen1", "gen2"]]
                .dropna()
                .drop_duplicates(subset="date")
                .set_index("date")
                .sort_index()
            )
            # Guard against zero/negative far price (would produce inf/-inf)
            far = s["gen2"].where(s["gen2"] > 0)
            carry = (s["gen1"] - far) / far * self.annualization_factor

            if self.smoothing_window > 1:
                carry = carry.rolling(
                    self.smoothing_window, min_periods=1,
                ).mean()

            series_by_symbol[symbol] = carry.rename(symbol)

        if not series_by_symbol:
            raise ValueError("No symbols had usable gen1/gen2 data")

        wide = pd.concat(series_by_symbol.values(), axis=1).sort_index()
        wide.index.name = "date"
        return wide
