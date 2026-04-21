# Systematic Energy Futures: A Multi-Signal Backtesting Framework

**Author:** Ethan Tanner
**Date:** April 2026

---

## Overview

This note describes the design and preliminary results of a modular backtesting framework for systematic commodities strategies across the energy futures complex. The framework implements four independent signal families — carry, momentum, spread mean-reversion, and fundamental surprise — with vol-targeted position sizing and drawdown-based risk management. The backtest covers five energy futures contracts (WTI, Brent, natural gas, heating oil, RBOB gasoline) from January 2015 through April 2026.

## Universe and Data

The universe consists of five NYMEX/ICE energy futures contracts spanning crude oil, natural gas, and refined products. Daily front-month and second-nearby prices were sourced from Bloomberg; weekly fundamental data (crude and product inventories, refinery utilization, crude imports) from the EIA API.

Continuous return series are constructed using within-contract returns only. On roll days — detected from the behavior of Bloomberg's generic tickers — the daily return is computed from yesterday's second-nearby price to today's front-month price, eliminating the artificial roll-yield jump that contaminates naively stitched series. This distinction is critical: a momentum signal run on unadjusted prices would misidentify the systematic contango bleed in natural gas as a genuine downtrend.

## Signals

**Carry (roll yield).** The annualized spread between front and second-nearby contracts, smoothed over five business days. Backwardation (front above deferred) produces a positive score. Empirically, backwardated commodities have outperformed contangoed ones over long samples, consistent with hedging-pressure and inventory theories.

**Time-series momentum.** Trailing 12-month risk-adjusted return with a one-month skip (to avoid short-term reversal). Each commodity is scored on its own merit — the book can run net long or net short depending on how many contracts are trending. Standardized as a raw risk-adjusted return (not cross-sectionally demeaned) so the signal captures absolute trend direction.

**Cross-sectional momentum.** Same raw computation as time-series momentum, but cross-sectionally z-scored. Ranks commodities by relative performance: long the winners, short the losers. Always approximately market-neutral. This and time-series momentum are correlated but not identical — TSMOM captures directional exposure, XSMOM captures rotation.

**Spread mean-reversion.** Rolling z-score on inter-commodity log-price ratios: WTI–Brent (geographic arbitrage) and heating oil–RBOB (seasonal crack spread). A z-score above +2 on WTI–Brent indicates WTI is historically rich vs Brent, generating a short-WTI / long-Brent signal. Contributions from multiple spreads sum per commodity.

**Fundamental surprise.** Week-over-week change in EIA weekly petroleum inventories vs a rolling eight-week forecast, standardized by rolling surprise volatility. A larger-than-expected draw (negative surprise) is bullish for crude. The signal fires on the Wednesday print date and decays exponentially with a three-business-day half-life. Inventory prints are mapped to specific commodities via a weight matrix: crude stocks affect CL and CO, gasoline stocks affect XB, distillate stocks affect HO, and refinery utilization is bearish for refined products.

## Portfolio Construction

Signals are combined using equal weights and converted to positions via volatility targeting. Each commodity's position is scaled by the inverse of its trailing 63-day realized volatility, so a unit of signal conviction in volatile natural gas produces a mechanically smaller dollar position than the same conviction in calmer crude. This ensures risk contribution is proportional to signal strength, not to underlying volatility.

A drawdown-based risk manager linearly scales positions toward zero as cumulative drawdown approaches a 15% limit, with scaling beginning at 50% of the limit. This simulates the loss-limit constraints typical of multi-manager pod structures.

All positions are lagged by one day: the signal computed at the close of day T determines the position held from T+1. This is enforced mechanically in the backtester and verified in the test suite via exact numerical comparison of the lagged-position P&L formula.

## Results (In-Sample, Before Costs)

| Metric | Value |
|---|---|
| Period | Jan 2015 – Apr 2026 (11.3 years) |
| Annualized return | 50.3% |
| Annualized volatility | 19.9% |
| Sharpe ratio | 2.53 |
| Maximum drawdown | 18.4% |
| Hit rate | 37.5% |

These results carry important caveats. The Sharpe ratio is inflated by several factors that would compress returns in production:

- **Zero transaction costs.** Energy futures bid-ask spreads are 1–2 bps, and daily rebalancing across five contracts accumulates meaningful drag.
- **No slippage.** The backtest assumes execution at closing prices. In practice, crossing the spread and market impact would reduce returns.
- **Signal scale imbalance.** Time-series momentum scores are on a different scale than the z-scored signals, causing it to dominate the equal-weight combination. A production implementation would normalize signal contributions more carefully.
- **In-sample only.** All parameters (lookback windows, smoothing, decay half-lives) were set a priori based on literature defaults, not optimized — but the full sample was visible during development.

A realistic post-cost Sharpe estimate for this type of strategy is 0.5–1.0.

## Notable Episodes

The backtest's worst drawdown (18.4%, Nov 2024 – Jun 2025) and the April 2020 WTI dislocation are both worth examining. WTI crude traded below zero on April 20, 2020, producing a meaningless -306% single-day return in the continuous series. This was capped at -25% for the backtest — a simplification that a production system would handle through exchange-mandated position limits and margin calls. The ability to identify and handle such data artifacts is itself a useful skill for systematic trading.

## Known Simplifications

Several design choices were made for tractability that would need to be refined for production use:

- **Roll detection is data-driven.** The detector compares gen1/gen2 price behavior to infer roll dates. This works when the forward-curve spread clearly dominates daily volatility, but fails in tight-spread regimes. A production system would use exchange contract calendars (first notice day, last trading day) directly.
- **No early rolling.** The framework rolls on the natural expiry date. In practice, liquidity dries up before expiry, and most systematic traders roll 3–5 days early to minimize market impact. The hook exists in the code but is not yet implemented.
- **EIA surprise uses a rolling forecast, not analyst consensus.** Bloomberg's SURV function provides pre-print analyst estimates, which would produce a cleaner surprise signal. The rolling-mean approach is a reasonable proxy but noisier.
- **Volatility floor.** A 5% annualized floor prevents position blow-up during quiet periods, but the choice of floor level has material impact on the strategy's behavior and is not optimized.

## Repository

The full codebase, including six smoke tests with synthetic data and the real-data backtest, is available at: github.com/eethanwwalter-ship-it/commodities-backtester

## Architecture

```
src/
  data/         Bloomberg & EIA clients, Parquet storage
  universe/     Contract specs, roll logic, continuous series
  signals/      Carry, momentum (TS + XS), mean-reversion, fundamental
  portfolio/    Signal combiner, vol-targeting sizer, risk manager, backtester
  analytics/    Report generation
```
