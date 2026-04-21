"""Backtester — end-to-end simulation engine.

Orchestrates: signals → combiner → position sizer → risk manager → P&L.

The critical timing convention:
  - Signal computed at close of day T uses data through T.
  - Position implied by signal at T is ENTERED at close of T (or open of T+1).
  - Return earned on day T+1 is applied to the position set at T.

This one-day lag is non-negotiable. Without it you have lookahead bias
and your backtest is worthless. The lag is enforced here by shifting
positions forward by one day before multiplying by returns.

P&L calculation
---------------
Daily P&L for commodity i:

    pnl_i(t) = position_i(t-1) * return_i(t)

where position_i(t-1) is the vol-adjusted, risk-scaled position
determined using information through day t-1.

Total portfolio return on day t = sum of pnl_i(t) across commodities.

The positions are in "vol-adjusted score" units (not dollars), so the
P&L is also in those units. To convert to dollar P&L, multiply by
capital * target_vol_per_position. This separation keeps the backtest
independent of AUM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..signals.base import Signal, SignalData
from .combiner import combine_signals
from .risk import RiskConfig, apply_risk_management
from .sizer import size_positions

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Full backtest configuration."""
    # Signal combination
    signal_weights: dict[str, float] | None = None
    combine_method: Literal["equal_weight", "custom_weight"] = "equal_weight"

    # Position sizing
    vol_lookback: int = 63
    vol_floor: float = 0.05
    max_position: float | None = None

    # Risk management
    risk: RiskConfig = field(default_factory=RiskConfig)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    # Daily time series
    portfolio_returns: pd.Series       # daily portfolio return
    cumulative_returns: pd.Series      # cumulative (1+r).cumprod()
    positions: pd.DataFrame            # final risk-adjusted positions
    positions_pre_risk: pd.DataFrame   # positions before risk scaling
    signal_scores: pd.DataFrame        # composite signal scores

    # Per-signal raw outputs (for decomposition)
    signal_outputs: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Summary statistics
    stats: dict[str, float] = field(default_factory=dict)


def run_backtest(
    signals: list[Signal],
    data: SignalData,
    config: BacktestConfig | None = None,
    signal_standardize: Literal["zscore", "rank", "none"] = "zscore",
) -> BacktestResult:
    """Run a full backtest.

    Parameters
    ----------
    signals
        List of Signal instances to run and combine.
    data
        SignalData bundle with raw_futures, continuous, eia data.
    config
        Backtest configuration. Defaults are sensible for energy futures.
    signal_standardize
        How to standardize each signal before combination.

    Returns
    -------
    BacktestResult with all time series and summary stats.
    """
    if config is None:
        config = BacktestConfig()

    # Step 1: Run each signal
    signal_outputs: dict[str, pd.DataFrame] = {}
    for sig in signals:
        std = getattr(sig, "preferred_standardize", signal_standardize)
        logger.info("Running signal: %s (standardize=%s)", sig.name, std)
        output = sig.run(data, standardize=std)
        signal_outputs[sig.name] = output

    # Step 2: Combine signals
    if config.combine_method == "custom_weight" and config.signal_weights:
        composite = combine_signals(
            signal_outputs,
            method="custom_weight",
            weights=config.signal_weights,
        )
    else:
        composite = combine_signals(signal_outputs, method="equal_weight")

    # Step 3: Size positions (vol-targeting)
    returns_dict = {
        sym: df for sym, df in data.continuous.items()
        if "return" in df.columns
    }
    positions_raw = size_positions(
        composite,
        returns_dict,
        vol_lookback=config.vol_lookback,
        vol_floor=config.vol_floor,
        max_position=config.max_position,
    )

    # Step 4: Compute initial portfolio return (before risk management)
    # to feed into the risk manager. Uses the ONE-DAY LAG.
    returns_wide = _build_returns_wide(data.continuous, positions_raw.index)
    portfolio_ret_raw = _compute_portfolio_returns(positions_raw, returns_wide)

    # Step 5: Apply risk management
    positions_final = apply_risk_management(
        positions_raw, portfolio_ret_raw, config.risk,
    )

    # Step 6: Recompute portfolio return with risk-adjusted positions
    portfolio_returns = _compute_portfolio_returns(positions_final, returns_wide)
    cumulative = (1.0 + portfolio_returns).cumprod()

    # Step 7: Compute summary statistics
    stats = _compute_stats(portfolio_returns, cumulative)

    return BacktestResult(
        portfolio_returns=portfolio_returns,
        cumulative_returns=cumulative,
        positions=positions_final,
        positions_pre_risk=positions_raw,
        signal_scores=composite,
        signal_outputs=signal_outputs,
        stats=stats,
    )


def _build_returns_wide(
    continuous: dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a wide returns DataFrame aligned to the position index."""
    series = {}
    for sym, df in continuous.items():
        if "return" in df.columns and "date" in df.columns:
            s = (
                df[["date", "return"]]
                .dropna()
                .drop_duplicates(subset="date")
                .set_index("date")["return"]
                .sort_index()
            )
            series[sym] = s
    return pd.DataFrame(series).reindex(date_index).fillna(0.0)


def _compute_portfolio_returns(
    positions: pd.DataFrame,
    returns_wide: pd.DataFrame,
) -> pd.Series:
    """Compute daily portfolio returns with proper one-day lag.

    Position at close of T earns the return from T to T+1. So we shift
    positions forward by one day before multiplying by returns.
    """
    # Align columns
    common_cols = sorted(set(positions.columns) & set(returns_wide.columns))
    if not common_cols:
        raise ValueError("No overlapping symbols between positions and returns")

    pos = positions[common_cols]
    ret = returns_wide[common_cols].reindex(pos.index).fillna(0.0)

    # LAG: position from yesterday × return today
    lagged_pos = pos.shift(1).fillna(0.0)
    daily_pnl = (lagged_pos * ret).sum(axis=1)

    # Normalize by gross exposure to get a return (not raw P&L in score units).
    # This makes the return interpretable as "return per unit of gross risk."
    gross = lagged_pos.abs().sum(axis=1).replace(0, np.nan)
    portfolio_return = daily_pnl / gross
    portfolio_return = portfolio_return.fillna(0.0)
    portfolio_return.index.name = "date"
    return portfolio_return


def _compute_stats(
    returns: pd.Series,
    cumulative: pd.Series,
) -> dict[str, float]:
    """Compute standard backtest summary statistics."""
    valid = returns.dropna()
    if len(valid) < 2:
        return {}

    n_years = len(valid) / 252.0
    ann_return = cumulative.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    ann_vol = valid.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Drawdown
    running_max = cumulative.cummax()
    drawdown = 1.0 - cumulative / running_max
    max_drawdown = drawdown.max()

    # Calmar ratio
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0.0

    # Hit rate
    hit_rate = (valid > 0).mean()

    # Daily turnover (mean absolute change in positions, not computed here
    # since we don't have dollar positions — would need contract specs)

    # Skew and kurtosis
    skew = valid.skew()
    kurtosis = valid.kurtosis()

    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "skewness": skew,
        "excess_kurtosis": kurtosis,
        "n_days": len(valid),
        "n_years": n_years,
    }
