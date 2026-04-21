# Commodities Systematic Backtester

A modular Python framework for researching and backtesting systematic signals
across the energy futures complex (WTI, Brent, natural gas, heating oil,
gasoline). Designed to demonstrate the analytical stack of a commodities-
focused systematic pod.

## Architecture

```
Data layer → Universe & roll logic → Signal modules → Portfolio construction → Analytics
```

This repo currently implements the first two layers. Signals, portfolio
construction, and analytics plug in on top without changing the data
interface.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install `blpapi` from Bloomberg (required for live data pulls — not on PyPI):
   https://www.bloomberg.com/professional/support/api-library/

3. Get a free EIA API key (https://www.eia.gov/opendata/register.php) and
   export it:
   ```bash
   export EIA_API_KEY=your_key_here
   ```

4. Build the initial dataset (run with Bloomberg terminal open):
   ```bash
   python scripts/build_dataset.py --config config/universe.yaml
   ```

## Universe

Five energy futures contracts covering the crude complex and refined products:

| Symbol | Contract | Exchange |
|--------|----------|----------|
| CL | WTI Crude Oil | NYMEX |
| CO | Brent Crude Oil | ICE |
| NG | Henry Hub Natural Gas | NYMEX |
| HO | NY Harbor ULSD (Heating Oil) | NYMEX |
| XB | RBOB Gasoline | NYMEX |

## Roll methodology

Futures contracts expire, so building a long price history requires stitching
consecutive contracts together. Naive stitching contaminates returns with the
roll yield (the spread between expiring and next contract), which destroys
the backtest.

This framework builds continuous return series using **within-contract
returns only**. On roll days, the return is computed from yesterday's
second-nearby generic to today's front generic — because those are the same
physical contract before and after the generic pointer shifts. This
preserves economic realism while avoiding roll contamination.

## Directory layout

```
config/         Universe configuration (YAML)
src/
  data/         Bloomberg & EIA clients, local Parquet storage
  universe/     Contract specs and roll logic
  signals/      Signal library (carry, momentum, mean-reversion, fundamental)
  portfolio/    Combiner, vol-targeting sizer, risk manager, backtester
scripts/        CLI entry points and smoke tests
data/           Generated Parquet files (gitignored)
```

## Roadmap

- [x] Data layer (Bloomberg + EIA + local storage)
- [x] Universe & roll logic
- [x] Signal layer base class (`Signal`, `SignalData`)
- [x] Carry / roll-yield signal
- [x] Momentum signals (time-series + cross-sectional)
- [x] Mean-reversion on inter-commodity spreads
- [x] Fundamental-surprise signal (EIA inventories)
- [x] Portfolio construction: signal combiner, vol-targeting, risk manager
- [x] Backtester engine with one-day lag enforcement
- [ ] Analytics: signal decay, per-signal attribution, correlation heatmaps
