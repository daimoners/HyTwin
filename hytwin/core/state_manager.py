"""
State Manager — centralised, timestamped state store for all
components in the digital twin.  Acts as the single source of truth.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


StateKey = str          # e.g. "wind_turbine.wt1.power_kw"
StateRecord = Tuple[datetime, Any]


class StateManager:
    """
    Thread-safe key-value store with history.

    Keys follow the naming convention::

        <component_type>.<component_id>.<variable_name>

    Examples::

        state.set("wind_turbine.wt1.power_kw", 320.5, ts)
        val, ts = state.get("wind_turbine.wt1.power_kw")
        hist = state.history("electrolyzer.el1.h2_flow_kg_s", last=50)
    """

    def __init__(self, max_history: int = 8_640) -> None:
        """
        Parameters
        ----------
        max_history : int
            Maximum number of historical records kept per key.
            Default covers 24 h at 10-second steps.
        """
        self._data: Dict[StateKey, StateRecord] = {}
        self._history: Dict[StateKey, List[StateRecord]] = defaultdict(list)
        self._max_history = max_history
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(self, key: StateKey, value: Any, timestamp: Optional[datetime] = None) -> None:
        ts = timestamp or datetime.utcnow()
        with self._lock:
            self._data[key] = (ts, value)
            hist = self._history[key]
            hist.append((ts, value))
            if len(hist) > self._max_history:
                hist.pop(0)

    def update(self, mapping: Dict[StateKey, Any], timestamp: Optional[datetime] = None) -> None:
        """Batch update multiple keys with the same timestamp."""
        ts = timestamp or datetime.utcnow()
        with self._lock:
            for key, value in mapping.items():
                self.set(key, value, ts)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: StateKey, default: Any = None) -> Tuple[Optional[datetime], Any]:
        """Return ``(timestamp, value)`` for *key*, or ``(None, default)``."""
        with self._lock:
            return self._data.get(key, (None, default))

    def value(self, key: StateKey, default: Any = None) -> Any:
        """Convenience — return only the value."""
        _, v = self.get(key, default)
        return v

    def history(
        self,
        key: StateKey,
        last: Optional[int] = None,
    ) -> List[StateRecord]:
        """Return list of ``(timestamp, value)`` tuples."""
        with self._lock:
            h = list(self._history.get(key, []))
        return h[-last:] if last else h

    def snapshot(self, prefix: Optional[str] = None) -> Dict[StateKey, Any]:
        """Return a flat dict of the latest values, optionally filtered by prefix."""
        with self._lock:
            if prefix:
                return {k: v for k, (_, v) in self._data.items() if k.startswith(prefix)}
            return {k: v for k, (_, v) in self._data.items()}

    def keys(self, prefix: Optional[str] = None) -> List[StateKey]:
        with self._lock:
            if prefix:
                return [k for k in self._data if k.startswith(prefix)]
            return list(self._data.keys())

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._history.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __repr__(self) -> str:
        with self._lock:
            return f"StateManager({len(self._data)} keys)"
