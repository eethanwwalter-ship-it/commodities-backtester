# Commodities Systematic Backtester

A modular Python framework for researching and backtesting systematic signals
across the energy futures complex (WTI, Brent, natural gas, heating oil,
gasoline). Designed to demonstrate the analytical stack of a commodities-
focused systematic pod.

## Signals

- **Carry / roll yield** — annualized front-vs-second curve slope
- **Time-series momentum** — trailing 12M risk-adjusted return (Moskowitz-Ooi-Pedersen)
- **Cross-sectional momentum** — relative performance ranking across the universe
- **Spread mean-reversion** — rolling z-score on WTI-Brent and HO-RB spreads
- **Fundamental surprise** — EIA weekly inventory prints vs rolling forecast, with exponential decay

## Portfolio Construction

- Equal-weight signal combination with NaN-aware re-weighting
- Volatility-targeted position sizing (inverse realized vol)
- Drawdown-based risk scaling (linear reduction as drawdown approaches limit)
- One-day lag enforced mechanically (verified in test suite)

## Universe

| Symbol | Contract | Exchange |
|--------|----------|----------|
| CL | WTI Crude Oil | NYMEX |
| CO | Brent Crude Oil | ICE |
| NG | Henry Hub Natural Gas | NYMEX |
| HO | NY Harbor ULSD (Heating Oil) | NYMEX |
| XB | RBOB Gasoline | NYMEX |

## Setup

1. Install dependencies: pip install -r requirements.txt
2. Get a free EIA API key at https://www.eia.gov/opendata/register.php
3. Export it: export EIA_API_KEY=your_key_here
4. Pull EIA data: python scripts/build_dataset.py --skip-bloomberg
5. For futures prices, either run build_dataset.py with a Bloomberg terminal, or export from Bloomberg Excel (BDH) and convert with python scripts/convert_excel.py
6. Run the backtest: python scripts/run_real_backtest.py
7. Generate analytics report: python scripts/run_report.py

## Roll Methodology

Continuous return series use within-contract returns only. On roll days,
the return is computed from yesterday s second-nearby to today s front-month,
the same physical contract before and after the generic pointer shifts.
This eliminates the roll-yield contamination that destroys naive backtests.

## Directory Layout

    config/         Universe configuration (YAML)
    src/
      data/         Bloomberg and EIA clients, local Parquet storage
      universe/     Contract specs and roll logic
      signals/      Signal library (carry, momentum, mean-reversion, fundamental)
      portfolio/    Combiner, vol-targeting sizer, risk manager, backtester
      analytics/    Report generation
    scripts/        CLI entry points, smoke tests, data conversion
    data/           Generated Parquet files (gitignored)

## Testing

Six smoke tests verify the pipeline end-to-end on synthetic data:

    python scripts/test_roll_logic.py
    python scripts/test_carry_signal.py
    python scripts/test_momentum_signal.py
    python scripts/test_mean_reversion_signal.py
    python scripts/test_fundamental_signal.py
    python scripts/test_backtest.py

## Research Note

See RESEARCH_NOTE.md for a full write-up of methodology, results, and known limitations.


