"""Daily engine test doubles and deterministic three-frame fixture."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.config.schema import FeeConfig, SlippageConfig
from etf_backtest.core.account import Account
from etf_backtest.core.engine import BacktestEngine
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.fill import FillModel
from etf_backtest.core.market import (
    EtfCategory,
    EtfInfo,
    EtfTradingRule,
    Exchange,
    FrameKey,
    IndexBarView,
    MarketBar,
    MarketBarView,
    MarketFrame,
    TurnoverRule,
)
from etf_backtest.core.order import RuleCheckResult, RuleReasonCode
from etf_backtest.core.order_generator import OrderGenerator
from etf_backtest.core.position import Position
from etf_backtest.core.slippage import SlippageModel
from etf_backtest.core.target import NO_REBALANCE, TargetPortfolio
from etf_backtest.strategy.base import BaseStrategy

DATES = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
SYMBOL = "SH.510300"


class FakeCalendar:
    def next_trading_day(self, trade_date: date) -> date:
        return DATES[DATES.index(trade_date) + 1]


class FakePortal:
    def __init__(self) -> None:
        self.symbols = (SYMBOL,)
        self.etf_infos = (
            EtfInfo(
                symbol=SYMBOL,
                exchange=Exchange.SSE,
                name="ETF",
                primary_category="domestic",
                fund_type="stock",
                list_date=date(2012, 1, 1),
                delist_date=None,
                current_status="active",
            ),
        )
        self.trading_calendar = FakeCalendar()
        closes = (Decimal("10"), Decimal("10"), Decimal("20"))
        pre_closes = (Decimal("10"), Decimal("10"), Decimal("20"))
        self._frames = tuple(
            MarketFrame.from_bars(
                FrameKey(trade_date=trade_date, calendar_version="test-v1"),
                (
                    MarketBar(
                        source_record_key=f"raw:SH.510300:{trade_date.isoformat()}",
                        symbol=SYMBOL,
                        trade_date=trade_date,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        pre_close=pre_close,
                        volume=100000,
                        amount=close * 100000,
                        suspended=False,
                    ),
                ),
            )
            for trade_date, close, pre_close in zip(DATES, closes, pre_closes, strict=True)
        )
        self._views = tuple(
            MarketBarView(
                source_record_key=f"front:SH.510300:{trade_date.isoformat()}",
                symbol=SYMBOL,
                trade_date=trade_date,
                open=front_close,
                high=front_close,
                low=front_close,
                close=front_close,
                volume=100000,
                suspended=False,
            )
            for trade_date, front_close in zip(
                DATES,
                (Decimal("100"), Decimal("200"), Decimal("300")),
                strict=True,
            )
        )

    def execution_frames(self, start_date: date, end_date: date):
        return tuple(frame for frame in self._frames if start_date <= frame.trade_date <= end_date)

    def views_through(
        self,
        as_of_date: date,
        *,
        symbols=None,
        lookback_trading_days=None,
    ):
        selected = set(self.symbols if symbols is None else symbols)
        eligible_dates = [trade_date for trade_date in DATES if trade_date <= as_of_date]
        if lookback_trading_days is not None:
            eligible_dates = eligible_dates[-lookback_trading_days:]
        return tuple(
            view
            for view in self._views
            if view.trade_date in eligible_dates and view.symbol in selected
        )

    def share_history_through(self, as_of_date: date, *, symbols=None):
        selected = self.symbols if symbols is None else tuple(symbols)
        return {
            symbol: {
                trade_date: Decimal(index * 100)
                for index, trade_date in enumerate(DATES, start=1)
                if trade_date <= as_of_date
            }
            for symbol in selected
        }

    def huijin_ratios_as_of(self, as_of_date: date, *, symbols=None):
        selected = self.symbols if symbols is None else tuple(symbols)
        return {
            symbol: {
                "中央汇金投资有限责任公司": (
                    date(2023, 12, 31),
                    Decimal("0.0971"),
                )
            }
            for symbol in selected
            if as_of_date > date(2023, 12, 31)
        }

    def combined_huijin_ratios_as_of(self, as_of_date: date, *, symbols=None):
        selected = self.symbols if symbols is None else tuple(symbols)
        return {
            symbol: (date(2023, 12, 31), Decimal("0.1096"))
            for symbol in selected
            if as_of_date > date(2023, 12, 31)
        }

    def index_history_through(self, as_of_date: date, *, lookback_trading_days=None):
        bars = tuple(
            IndexBarView(
                index_code="000001.SH",
                trade_date=trade_date,
                open=Decimal("3000"),
                high=Decimal("3100") + Decimal(index),
                low=Decimal("2900"),
                close=Decimal("3000") + Decimal(index),
                pre_close=None,
                pct_change=None,
                source_system="TUSHARE",
            )
            for index, trade_date in enumerate(DATES)
            if trade_date <= as_of_date
        )
        if lookback_trading_days is not None:
            bars = bars[-lookback_trading_days:]
        return {"000001.SH": bars}


class RecordingResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []

    def resolve(self, symbol: str, trade_date: date) -> EtfTradingRule:
        self.calls.append((symbol, trade_date))
        return EtfTradingRule(
            symbol=symbol,
            etf_category=EtfCategory.DOMESTIC_STOCK_ETF,
            turnover_rule=TurnoverRule.T1,
            price_limit_ratio=Decimal("0.10"),
        )


class ApproveAllWholeLots:
    def __init__(self) -> None:
        self.sell_available: list[int] = []

    def approve_batch(self, *, orders, account, **_kwargs):
        results = []
        for order in orders:
            if order.side.value == "SELL":
                self.sell_available.append(account.position_for(order.symbol).available_quantity)
            approved = order.requested_quantity // 100 * 100
            results.append(
                RuleCheckResult(
                    order_id=order.order_id,
                    requested_quantity=order.requested_quantity,
                    approved_quantity=approved,
                    passed=approved > 0,
                    reason_code=(
                        RuleReasonCode.APPROVED if approved > 0 else RuleReasonCode.LOT_SIZE
                    ),
                    message="approved" if approved > 0 else "below lot",
                )
            )
        return tuple(results)


class RotateInThenCashStrategy(BaseStrategy):
    def __init__(self) -> None:
        self.signal_dates: list[date] = []
        self.visible_dates: list[tuple[date, ...]] = []
        self.contexts = []
        self.no_rebalance_frames: set[int] = set()

    @property
    def required_history_trading_days(self) -> int:
        return 21

    def should_generate_target(self, frame_index: int) -> bool:
        return True

    def _generate_target(self, *, signal_date, market_history, context, **_kwargs):
        self.signal_dates.append(signal_date)
        self.visible_dates.append(tuple(view.trade_date for view in market_history))
        self.contexts.append(context)
        if context.frame_index in self.no_rebalance_frames:
            return NO_REBALANCE
        if context.frame_index == 0:
            return TargetPortfolio(weights={SYMBOL: Decimal("1")})
        return TargetPortfolio(weights={})


@pytest.fixture
def engine_components():
    portal = FakePortal()
    resolver = RecordingResolver()
    rule_engine = ApproveAllWholeLots()
    strategy = RotateInThenCashStrategy()
    account = Account(
        cash=Decimal("10000"),
        positions={SYMBOL: Position(symbol=SYMBOL, turnover_rule=TurnoverRule.T1)},
    )
    engine = BacktestEngine(
        portal=portal,
        account=account,
        strategy=strategy,
        rule_resolver=resolver,
        rule_engine=rule_engine,
        order_generator=OrderGenerator(),
        fill_model=FillModel(
            fee_model=FeeModel(
                FeeConfig(
                    commission_rate=Decimal("0"),
                    minimum_commission=Decimal("0"),
                )
            ),
            slippage_model=SlippageModel(SlippageConfig(rate=Decimal("0"))),
        ),
    )
    return engine, portal, resolver, rule_engine, strategy
