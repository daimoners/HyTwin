"""
RNG utilities
=============
Physics models, weather, and sensors draw randomness (measurement noise,
weather variability, outage timing, ...).  Historically this was done via the
global ``numpy.random.*`` state, which works for a single sequential
simulation but is not safe when **two independent simulations run
concurrently in the same process** — e.g. the network dashboard's live
simulation thread and a background RL training job both stepping their own
``NetworkTwin`` at once.  Sharing one global stream between them means
neither stays reproducible, and (worse) their draws can interleave and
silently corrupt each other's sequence.

The fix: give every randomness-consuming component an optional
``numpy.random.Generator`` of its own. A component tree constructed with an
explicit generator (``NetworkTwin(topology, seed=...)``) gets a fully
isolated, thread-safe, reproducible-per-instance stream, obtained by
``Generator.spawn()``-ing independent children down the object tree.

Components constructed *without* an explicit generator (``rng=None``, the
default everywhere) fall back to :class:`_GlobalRNGProxy`, which forwards to
the legacy global ``numpy.random`` functions — preserving byte-identical
behaviour (and the ``numpy.random.seed()``-based reproducibility contract)
for every call site that predates this module and was never updated to pass
a generator (single-site ``SimulationEngine``, demos, experiments).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

RNGLike = Union[np.random.Generator, "_GlobalRNGProxy"]


class _GlobalRNGProxy:
    """Forwards draws to the legacy global ``numpy.random`` state."""

    def normal(self, *a, **k):
        return np.random.normal(*a, **k)

    def uniform(self, *a, **k):
        return np.random.uniform(*a, **k)

    def random(self, *a, **k):
        return np.random.random(*a, **k)

    def exponential(self, *a, **k):
        return np.random.exponential(*a, **k)

    def integers(self, low, high=None, **k):
        # np.random.Generator uses .integers(); the legacy module uses .randint().
        return np.random.randint(low, high, **k)


_GLOBAL_RNG_PROXY = _GlobalRNGProxy()


def resolve_rng(rng: Optional[np.random.Generator]) -> RNGLike:
    """Return *rng* if given, else the legacy-global-RNG proxy."""
    return rng if rng is not None else _GLOBAL_RNG_PROXY


def spawn_generators(
    rng: Optional[np.random.Generator], n: int
) -> List[Optional[np.random.Generator]]:
    """
    Split *rng* into *n* statistically independent child generators, one per
    sub-component of a composite object (e.g. one per site, one per sensor).

    If *rng* is ``None`` (the caller hasn't opted into isolated streams),
    returns ``[None] * n`` so every child falls back to the legacy global
    proxy via :func:`resolve_rng` — spawning is a no-op in that case.
    """
    if rng is None:
        return [None] * n
    return list(rng.spawn(n))


def spawn_one(rng: Optional[np.random.Generator]) -> Optional[np.random.Generator]:
    """Spawn a single independent child generator (or ``None`` if *rng* is ``None``)."""
    if rng is None:
        return None
    return rng.spawn(1)[0]
