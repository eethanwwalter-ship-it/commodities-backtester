"""Risk manager — drawdown-based position scaling.

At a multi-manager pod like Millennium, each PM has a drawdown limit.
Breach it and you get cut — either your positions are reduced or you're
out. This module simulates that constraint by scaling positions down
as the strategy approaches its drawdown limit.

Mechanism
---------
The risk scalar is a function of how much drawdown budget remains:

    remaining = max_drawdown - current_drawdown
    scalar    = min(1.0, remaining / (max_drawdown * buffer_fraction))

When the strategy is near its high-water mark, scalar = 1.0 (full risk).
As drawdown deepens, the scalar declines linearly. At the drawdown limit,
scalar = 0 and all positions are closed.

`buffer_fraction` controls how aggressively risk is cut. At 0.5, risk
starts declining at 50% of the drawdown limit. At 1.0, risk only
declines at the limit itself (binary — full risk then immediate stop).
0.5 is a reasonable default.

This is a simplified version of what real risk systems do. Production
systems also enforce: sector concentration limits, single-name limits,
gross and net exposure caps, VaR limits, and correlation-adjusted risk.
Those are left as extensions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration.

    Attributes
    ----------
    max_drawdown
        Maximum tolerated drawdown as a fraction of cumulative P&L.
        0.10 = 10% drawdown limit.
    buffer_fraction
        Fraction of max_drawdown at which risk scaling begins. At 0.5,
        positions start shrinking when drawdown reaches 50% of the limit.
    position_limit
        Maximum absolute position per commodity after risk scaling.
        Acts as a hard cap on concentration. None = no limit.
    """
    max_drawdown: float = 0.10
    buffer_fraction: float = 0.5
    position_limit: float | None = None


def apply_risk_management(
    positions: pd.DataFrame,
    portfolio_returns: pd.Series,
    config: RiskConfig,
) -> pd.DataFrame:
    """Scale positions based on drawdown proximity.

    Parameters
    ----------
    positions
        Wide DataFrame of raw positions (from sizer).
    portfolio_returns
        Series of daily portfolio returns (index=date). Used to track
        cumulative P&L and drawdown. Must be aligned with positions.
    config
        Risk management configuration.

    Returns
    -------
    Risk-adjusted positions (same shape as input).
    """
    cum_return = (1.0 + portfolio_returns).cumprod()
    running_max = cum_return.cummax()
    drawdown = 1.0 - cum_return / running_max

    # Risk scalar: 1.0 when drawdown is small, declining linearly as
    # drawdown approaches the limit, 0.0 at the limit.
    buffer = config.max_drawdown * config.buffer_fraction
    remaining = (config.max_drawdown - drawdown).clip(lower=0.0)
    scalar = (remaining / buffer).clip(upper=1.0)

    # Apply scalar to all positions
    scaled = positions.mul(scalar, axis=0)

    if config.position_limit is not None:
        scaled = scaled.clip(
            lower=-config.position_limit,
            upper=config.position_limit,
        )

    # Log drawdown events
    max_dd = drawdown.max()
    n_scaled = (scalar < 1.0).sum()
    if n_scaled > 0:
        logger.info(
            "Risk scaling active on %d/%d days, max drawdown: %.2f%%",
            n_scaled, len(scalar), max_dd * 100,
        )

    scaled.index.name = "date"
    return scaled
