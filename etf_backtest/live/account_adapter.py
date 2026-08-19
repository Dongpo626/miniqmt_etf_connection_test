"""Convert one broker snapshot into the existing price-free strategy account view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.live.state import BrokerAssetSnapshot, BrokerPositionSnapshot
from etf_backtest.strategy.context import AccountPositionView, AccountView


@dataclass(frozen=True, slots=True)
class AdaptedAccountState:
    account_view: AccountView
    current_weights_by_symbol: Mapping[str, Decimal]
    total_asset: Decimal
    positions_by_symbol: Mapping[str, BrokerPositionSnapshot]


def adapt_broker_account(
    *,
    asset: BrokerAssetSnapshot,
    positions: Sequence[BrokerPositionSnapshot],
    symbols: Sequence[str],
) -> AdaptedAccountState:
    """Build the strategy view while retaining complete broker position snapshots."""

    frozen_symbols = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
    if not frozen_symbols:
        raise ValueError("symbols must not be empty")
    if asset.total_asset <= 0:
        raise ValueError("total_asset must be positive")
    position_map: dict[str, BrokerPositionSnapshot] = {}
    for position in positions:
        if position.symbol in position_map:
            raise ValueError(f"duplicate broker position: {position.symbol}")
        position_map[position.symbol] = position
    for symbol in frozen_symbols:
        position_map.setdefault(
            symbol,
            BrokerPositionSnapshot(
                symbol=symbol,
                total_quantity=0,
                available_quantity=0,
                today_buy_quantity=0,
                market_value=Decimal("0"),
                turnover_rule=TurnoverRule.T1,
                captured_at=asset.captured_at,
            ),
        )
    strategy_positions = {
        symbol: AccountPositionView(
            symbol=symbol,
            turnover_rule=position_map[symbol].turnover_rule,
            total_quantity=position_map[symbol].total_quantity,
            available_quantity=position_map[symbol].available_quantity,
            today_buy_quantity=position_map[symbol].today_buy_quantity,
        )
        for symbol in frozen_symbols
    }
    weights = {
        symbol: position_map[symbol].market_value / asset.total_asset for symbol in frozen_symbols
    }
    return AdaptedAccountState(
        account_view=AccountView(cash=asset.available_cash, positions=strategy_positions),
        current_weights_by_symbol=MappingProxyType(weights),
        total_asset=asset.total_asset,
        positions_by_symbol=MappingProxyType(dict(sorted(position_map.items()))),
    )


__all__ = ["AdaptedAccountState", "adapt_broker_account"]
