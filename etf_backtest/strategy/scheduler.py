"""Deterministic SSE-frame decision schedules."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PeriodicDecisionScheduler:
    """Decide on frame 0 and every fixed number of trading frames after it."""

    every_trading_days: int

    def __post_init__(self) -> None:
        if type(self.every_trading_days) is not int or self.every_trading_days <= 0:
            raise ValueError("every_trading_days must be a positive integer")

    def should_decide(self, frame_index: int) -> bool:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        return frame_index % self.every_trading_days == 0


class EveryTradingDayScheduler(PeriodicDecisionScheduler):
    def __init__(self) -> None:
        super().__init__(every_trading_days=1)


__all__ = ["EveryTradingDayScheduler", "PeriodicDecisionScheduler"]
