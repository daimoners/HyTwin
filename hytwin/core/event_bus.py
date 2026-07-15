"""
Event Bus — lightweight publish/subscribe system for decoupled
communication between grid components, sensors, and the digital twin.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A single simulation or system event."""
    topic: str
    payload: Any
    source: str = ""
    timestamp: Optional[datetime] = field(default_factory=datetime.utcnow)


EventHandler = Callable[[Event], None]


class EventBus:
    """
    Thread-safe pub/sub event bus.

    Usage::

        bus = EventBus()

        @bus.subscribe("sensor.power")
        def handle(event):
            print(event.payload)

        bus.publish(Event("sensor.power", {"value": 42.0}, source="pv_panel_1"))
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._history: List[Event] = []
        self._max_history = 10_000

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, topic: str) -> Callable[[EventHandler], EventHandler]:
        """Decorator-style subscription."""
        def decorator(fn: EventHandler) -> EventHandler:
            self.add_handler(topic, fn)
            return fn
        return decorator

    def add_handler(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers[topic].append(handler)
            logger.debug("Handler registered for topic '%s'", topic)

    def remove_handler(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers[topic] = [
                h for h in self._handlers[topic] if h is not handler
            ]

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Dispatch *event* to all subscribed handlers (synchronous)."""
        with self._lock:
            handlers = list(self._handlers.get(event.topic, []))
            # wildcard handlers
            handlers += list(self._handlers.get("*", []))
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Error in handler for topic '%s': %s", event.topic, exc
                )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_history(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        with self._lock:
            events = self._history if topic is None else [
                e for e in self._history if e.topic == topic
            ]
        return events[-limit:]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def topics(self) -> List[str]:
        with self._lock:
            return list(self._handlers.keys())


# Module-level singleton — subsystems can share or create their own
_global_bus: Optional[EventBus] = None


def get_global_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
