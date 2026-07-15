"""
Base model — abstract interface every physics model must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class ModelState:
    """Snapshot of a model's output at one time step."""
    timestamp: datetime
    component_id: str
    values: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)


class BaseModel(ABC):
    """
    Abstract physics model.

    Every concrete model exposes:
    - ``component_id``  — unique string identifier
    - ``step(dt, context)`` — compute one time step, return ``ModelState``
    - ``reset()``          — reinitialise internal state
    - ``state``            — last computed ``ModelState``
    """

    def __init__(self, component_id: str, params: Dict[str, Any]) -> None:
        self.component_id = component_id
        self.params = params
        self._state: ModelState | None = None

    @property
    def state(self) -> ModelState | None:
        return self._state

    @abstractmethod
    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        Advance the model by *dt* seconds.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        context : dict
            External inputs (e.g. weather, control actions).

        Returns
        -------
        ModelState
            Current output variables.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reinitialise all mutable state."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.component_id!r})"
