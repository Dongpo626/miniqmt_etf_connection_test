from etf_backtest.live.state import QueryResult


def test_query_result_distinguishes_successful_empty_and_failure() -> None:
    empty = QueryResult[str](success=True)
    failed = QueryResult[str](success=False, error="broker unavailable")

    assert empty.success and empty.records == () and empty.error is None
    assert not failed.success and failed.records == () and failed.error == "broker unavailable"
