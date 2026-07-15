"""
Component Registry — catalogue of all instantiated grid components.
Provides lookup, iteration, and dependency injection helpers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """
    Typed component registry.

    Usage::

        reg = Registry()
        reg.register("wt1", my_wind_turbine)
        reg.register("el1", my_electrolyzer)

        for comp in reg.by_type(WindTurbineModel):
            comp.step(...)
    """

    def __init__(self) -> None:
        self._store: Dict[str, object] = {}

    def register(self, component_id: str, component: object) -> None:
        if component_id in self._store:
            raise KeyError(f"Component '{component_id}' already registered.")
        self._store[component_id] = component

    def unregister(self, component_id: str) -> None:
        self._store.pop(component_id, None)

    def get(self, component_id: str) -> object:
        try:
            return self._store[component_id]
        except KeyError:
            raise KeyError(f"Component '{component_id}' not found in registry.")

    def by_type(self, component_type: Type[T]) -> List[T]:
        return [c for c in self._store.values() if isinstance(c, component_type)]

    def all(self) -> Dict[str, object]:
        return dict(self._store)

    def ids(self) -> List[str]:
        return list(self._store.keys())

    def __iter__(self) -> Iterator[object]:
        return iter(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, component_id: str) -> bool:
        return component_id in self._store

    def __repr__(self) -> str:
        return f"Registry({list(self._store.keys())})"
