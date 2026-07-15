"""
Simulation Time Engine — manages wall-clock and simulated time,
step advance, and real-time pacing.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional


class SimulationClock:
    """
    Controls simulation time.

    Parameters
    ----------
    start_time : datetime
        Simulated start instant.
    dt_seconds : float
        Duration of each simulation step in simulated seconds.
    speed_factor : float
        Wall-clock acceleration.  ``1.0`` → real-time.
        ``0`` or negative → as-fast-as-possible.
    """

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        dt_seconds: float = 60.0,
        speed_factor: float = 0.0,
    ) -> None:
        self._sim_time: datetime = start_time or datetime(2024, 1, 1, 0, 0, 0)
        self._dt = timedelta(seconds=dt_seconds)
        self._speed_factor = speed_factor
        self._step: int = 0
        self._wall_start: float = time.monotonic()
        self._paused: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def now(self) -> datetime:
        """Current simulated timestamp."""
        return self._sim_time

    @property
    def step(self) -> int:
        """Number of steps elapsed since simulation start."""
        return self._step

    @property
    def dt(self) -> timedelta:
        return self._dt

    @property
    def dt_seconds(self) -> float:
        return self._dt.total_seconds()

    # ------------------------------------------------------------------
    # Advance
    # ------------------------------------------------------------------

    def tick(self) -> datetime:
        """
        Advance the clock by one time step.

        If ``speed_factor > 0``, this method sleeps to align with the
        requested wall-clock pacing.

        Returns the *new* simulated timestamp.
        """
        if self._paused:
            return self._sim_time

        if self._speed_factor > 0:
            expected_wall = (
                self._wall_start
                + (self._step + 1) * self._dt.total_seconds() / self._speed_factor
            )
            sleep_s = expected_wall - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

        self._sim_time += self._dt
        self._step += 1
        return self._sim_time

    def reset(self, start_time: Optional[datetime] = None) -> None:
        if start_time is not None:
            self._sim_time = start_time
        self._step = 0
        self._wall_start = time.monotonic()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._wall_start = time.monotonic() - self._step * self._dt.total_seconds()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def elapsed_simulated(self) -> timedelta:
        return self._sim_time - (self._sim_time - self._step * self._dt)

    def __repr__(self) -> str:
        return (
            f"SimulationClock(now={self._sim_time.isoformat()}, "
            f"step={self._step}, dt={self._dt.total_seconds()}s)"
        )
