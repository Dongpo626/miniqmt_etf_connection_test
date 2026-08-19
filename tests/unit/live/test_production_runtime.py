import importlib.util
from pathlib import Path
from queue import Queue
from typing import cast
from unittest.mock import Mock

import pytest

import etf_backtest
import etf_backtest.application
import etf_backtest.core
import etf_backtest.live
import etf_backtest.live_cli
import etf_backtest.strategy
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.broker.miniqmt import MiniQmtBrokerGateway
from etf_backtest.live.config import load_live_config
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.market.xtdata import XtDataQuoteProvider
from etf_backtest.live_cli import build_production_runtime

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"


def test_missing_xtquant_is_delayed_until_production_adapter_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    assert all(
        module is not None
        for module in (
            etf_backtest,
            etf_backtest.application,
            etf_backtest.core,
            etf_backtest.live,
            etf_backtest.live_cli,
            etf_backtest.strategy,
        )
    )
    if importlib.util.find_spec("xtquant") is not None:
        pytest.skip("target environment provides xtquant")
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        MiniQmtBrokerGateway(
            userdata_path=Path("C:/missing"),
            session_id=1,
            account_id="paper-1",
            event_queue=Queue(),
        )
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        XtDataQuoteProvider()


def test_production_builder_wires_existing_rule_components_without_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "paper-1")
    monkeypatch.setenv("QMT_LIVE_MYSQL_PASSWORD", "state-secret")
    monkeypatch.setenv("QMT_MYSQL_PASSWORD", "strategy-secret")
    config = load_live_config(CONFIG)
    broker = Mock(spec=BrokerGateway)
    quotes = Mock(spec=QuoteProvider)
    broker_factory = Mock(return_value=cast(BrokerGateway, broker))
    quote_factory = Mock(return_value=cast(QuoteProvider, quotes))

    runtime = build_production_runtime(
        config,
        broker_factory=broker_factory,
        quote_factory=quote_factory,
    )

    assert runtime.engine.jobs is runtime.jobs
    assert runtime.engine.broker is broker
    assert runtime.engine.quote_provider is quotes
    assert runtime.event_consumer is not None
    assert runtime.jobs.signal_evaluator.__class__.__name__ == "RuleSignalService"
    broker_factory.assert_called_once()
    assert "model" not in vars(runtime.jobs)
    runtime.state_engine.dispose()
