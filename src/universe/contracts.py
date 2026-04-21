"""Contract specifications and universe loading."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml


@dataclass(frozen=True)
class ContractSpec:
    """Metadata for a single commodity futures contract."""
    symbol: str                  # internal symbol, e.g. "CL"
    name: str                    # human-readable, e.g. "WTI Crude Oil"
    exchange: str
    bloomberg_root: str          # generic ticker root, e.g. "CL"
    bloomberg_yellow_key: str    # e.g. "Comdty"
    contract_months: list[str]   # Bloomberg month codes (F G H J K M N Q U V X Z)
    tick_size: float
    contract_size: float
    currency: str
    sector: str
    sub_sector: str

    def generic_ticker(self, n: int) -> str:
        """Return Bloomberg generic ticker for the Nth contract (1-indexed).

        The Nth generic always points at the Nth-nearby contract. When the
        front contract expires, the generics shift down by one automatically
        (tomorrow's "CL1" is today's "CL2"). This is why we pull gen1+gen2
        and handle roll returns ourselves.
        """
        if n < 1:
            raise ValueError("generic index must be >= 1")
        return f"{self.bloomberg_root}{n} {self.bloomberg_yellow_key}"


@dataclass(frozen=True)
class Universe:
    """Full universe with data settings."""
    contracts: dict[str, ContractSpec]
    start_date: str
    max_contracts: int
    roll_method: str
    roll_days_before_expiry: int

    def __iter__(self) -> Iterator[ContractSpec]:
        return iter(self.contracts.values())

    def __getitem__(self, symbol: str) -> ContractSpec:
        return self.contracts[symbol]

    def symbols(self) -> list[str]:
        return list(self.contracts.keys())


def load_universe(path: str | Path) -> Universe:
    """Load universe configuration from YAML."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    contracts = {
        sym: ContractSpec(symbol=sym, **spec)
        for sym, spec in cfg["commodities"].items()
    }
    ds = cfg["data_settings"]
    start = ds["start_date"]
    if not isinstance(start, str):
        start = start.isoformat()

    return Universe(
        contracts=contracts,
        start_date=start,
        max_contracts=int(ds["max_contracts"]),
        roll_method=ds["roll_method"],
        roll_days_before_expiry=int(ds["roll_days_before_expiry"]),
    )
