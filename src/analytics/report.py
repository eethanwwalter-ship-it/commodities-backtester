"""Analytics and reporting for backtest results.

Generates summary statistics, per-signal attribution, drawdown analysis,
and correlation structure. Outputs a text report that can be pasted into
a research note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio.backtest import BacktestResult


def generate_report(result: BacktestResult) -> str:
    """Generate a full text report from backtest results."""
    lines = []
    lines.append("=" * 70)
    lines.append("SYSTEMATIC COMMODITIES STRATEGY — BACKTEST REPORT")
    lines.append("=" * 70)

    # Summary stats
    s = result.stats
    lines.append("")
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Period:              {s.get('n_years', 0):.1f} years ({s.get('n_days', 0)} trading days)")
    lines.append(f"  Annualized return:   {s.get('annualized_return', 0):>8.2%}")
    lines.append(f"  Annualized vol:      {s.get('annualized_vol', 0):>8.2%}")
    lines.append(f"  Sharpe ratio:        {s.get('sharpe_ratio', 0):>8.2f}")
    lines.append(f"  Max drawdown:        {s.get('max_drawdown', 0):>8.2%}")
    lines.append(f"  Calmar ratio:        {s.get('calmar_ratio', 0):>8.2f}")
    lines.append(f"  Hit rate:            {s.get('hit_rate', 0):>8.2%}")
    lines.append(f"  Skewness:            {s.get('skewness', 0):>8.3f}")
    lines.append(f"  Excess kurtosis:     {s.get('excess_kurtosis', 0):>8.3f}")

    # Annual returns
    lines.append("")
    lines.append("ANNUAL RETURNS")
    lines.append("-" * 40)
    yearly = result.portfolio_returns.groupby(
        result.portfolio_returns.index.year
    ).sum()
    for year, ret in yearly.items():
        bar = "+" * int(min(abs(ret) * 20, 40)) if ret > 0 else "-" * int(min(abs(ret) * 20, 40))
        lines.append(f"  {year}:  {ret:>+7.2%}  {bar}")

    # Drawdown analysis
    lines.append("")
    lines.append("DRAWDOWN ANALYSIS")
    lines.append("-" * 40)
    cum = result.cumulative_returns
    running_max = cum.cummax()
    drawdown = 1.0 - cum / running_max
    top_dd = _top_drawdowns(drawdown, cum, n=5)
    for i, dd in enumerate(top_dd):
        lines.append(f"  #{i+1}:  {dd['depth']:>7.2%}  "
                      f"({dd['peak_date']} to {dd['trough_date']}, "
                      f"{dd['duration_days']} days)")

    # Signal correlation
    lines.append("")
    lines.append("SIGNAL CORRELATION MATRIX")
    lines.append("-" * 40)
    sig_returns = _compute_signal_returns(result)
    if sig_returns is not None and len(sig_returns.columns) > 1:
        corr = sig_returns.corr()
        header = "         " + "  ".join(f"{c:>8s}" for c in corr.columns)
        lines.append(header)
        for name, row in corr.iterrows():
            vals = "  ".join(f"{v:>8.2f}" for v in row)
            lines.append(f"  {name:>6s}  {vals}")
    else:
        lines.append("  (insufficient data for correlation)")

    # Per-signal contribution
    lines.append("")
    lines.append("PER-SIGNAL AVERAGE SCORES (full sample)")
    lines.append("-" * 40)
    for name, df in result.signal_outputs.items():
        mean_scores = df.mean()
        active = mean_scores.dropna()
        if len(active) > 0:
            scores_str = ", ".join(f"{sym}: {v:+.3f}" for sym, v in active.items())
            lines.append(f"  {name:>25s}:  {scores_str}")

    # Current positioning
    lines.append("")
    lines.append("CURRENT POSITIONS (last date)")
    lines.append("-" * 40)
    last_pos = result.positions.iloc[-1]
    for sym, pos in last_pos.items():
        if pd.notna(pos) and abs(pos) > 0.001:
            direction = "LONG " if pos > 0 else "SHORT"
            lines.append(f"  {sym}:  {direction}  {abs(pos):.3f}")

    # Monthly return heatmap
    lines.append("")
    lines.append("MONTHLY RETURNS")
    lines.append("-" * 40)
    monthly = result.portfolio_returns.groupby(
        [result.portfolio_returns.index.year,
         result.portfolio_returns.index.month]
    ).sum()
    monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["year", "month"])
    monthly_df = monthly.unstack(level="month")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    header = "       " + "  ".join(f"{m:>6s}" for m in months) + "    TOTAL"
    lines.append(header)
    for year in monthly_df.index:
        row_vals = []
        for m in range(1, 13):
            val = monthly_df.loc[year, m] if m in monthly_df.columns else np.nan
            if pd.notna(val):
                row_vals.append(f"{val:>+6.1%}")
            else:
                row_vals.append(f"{'':>6s}")
        yr_total = yearly.get(year, 0)
        lines.append(f"  {year}  {'  '.join(row_vals)}  {yr_total:>+7.1%}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def _top_drawdowns(
    drawdown: pd.Series,
    cumulative: pd.Series,
    n: int = 5,
) -> list[dict]:
    """Find the top N drawdown episodes."""
    results = []
    dd = drawdown.copy()

    for _ in range(n):
        if dd.max() < 0.001:
            break
        trough_idx = dd.idxmax()
        trough_val = dd[trough_idx]

        # Walk backward to find peak
        peak_idx = cumulative.loc[:trough_idx].idxmax()

        # Walk forward to find recovery (or end of series)
        post_trough = cumulative.loc[trough_idx:]
        recovery_mask = post_trough >= cumulative.loc[peak_idx]
        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()
        else:
            recovery_idx = cumulative.index[-1]

        duration = (pd.Timestamp(trough_idx) - pd.Timestamp(peak_idx)).days

        results.append({
            "depth": trough_val,
            "peak_date": peak_idx.strftime("%Y-%m-%d"),
            "trough_date": trough_idx.strftime("%Y-%m-%d"),
            "recovery_date": recovery_idx.strftime("%Y-%m-%d"),
            "duration_days": duration,
        })

        # Zero out this episode so we find the next one
        dd.loc[peak_idx:recovery_idx] = 0.0

    return results


def _compute_signal_returns(result: BacktestResult) -> pd.DataFrame | None:
    """Approximate per-signal return contribution.

    Uses each signal's scores as pseudo-positions and multiplies by
    asset returns to get a rough attribution. Not exact (ignores
    vol-targeting and risk scaling) but directionally useful.
    """
    if not result.signal_outputs:
        return None

    # Get asset returns
    ret_series = {}
    # Pull returns from positions index
    pos = result.positions
    port_ret = result.portfolio_returns

    # Simple approach: correlation of each signal's composite score
    # changes with portfolio returns
    signal_scores = {}
    for name, df in result.signal_outputs.items():
        # Average score across commodities as a proxy for signal direction
        avg = df.mean(axis=1).dropna()
        signal_scores[name] = avg

    if not signal_scores:
        return None

    combined = pd.DataFrame(signal_scores)
    # Compute rolling changes as proxy for signal-driven returns
    combined = combined.diff().dropna()

    # Align with portfolio returns
    common = combined.index.intersection(port_ret.index)
    if len(common) < 50:
        return None

    return combined.loc[common]


