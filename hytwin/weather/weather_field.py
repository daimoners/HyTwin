"""
Weather Field
=============
A spatially distributed weather generator: one :class:`WeatherModel` per
network site, each with its own local climatology (latitude, Weibull wind
parameters, cloud cover, temperature regime, …), PLUS a shared **synoptic
disturbance layer** that ties nearby sites' weather together the way real
frontal systems do — a storm crossing Puglia should also be felt (to a
lesser extent) in Campania an hour later, and strongly correlate with a
simultaneous storm in Basilicata, while a calm high-pressure system over
Lombardy has no bearing on what Sicily is doing that day.

Design
------
* **Local layer** (per site): each site's ``WeatherModel`` draws its own
  independent generator (spawned from the ``rng`` passed to
  :class:`WeatherField`, if any) — this is what lets a live simulation and a
  concurrent RL training job (each owning its own ``NetworkTwin`` /
  ``WeatherField``) run without their weather streams interfering with one
  another.  Passing no ``rng`` keeps the legacy global-RNG behaviour for
  un-migrated callers.
* **Regional/synoptic layer** (shared across sites): a single standardised
  disturbance index ``z`` is drawn *jointly* for all sites each step, with
  spatial correlation ``corr(i,j) = exp(-distance_km(i,j) / correlation_length_km)``
  (nearby sites highly correlated, distant sites nearly independent) and
  temporal persistence ``phi = exp(-dt / synoptic_tau_hours)`` (a frontal
  system lingers for roughly a day, not a single 10-minute step). Positive
  ``z`` (frontal/stormy) nudges wind speed AND cloud cover UP together at
  each site; negative ``z`` (anticyclonic/calm) nudges both DOWN — see
  :meth:`WeatherModel.step`.  This shared draw uses its own generator (spawned
  once from ``rng``), separate from every site's local generator, since by
  definition it must be a single joint draw shared across sites.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .weather_model import WeatherModel
from ..core.rng import spawn_generators, spawn_one, resolve_rng

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points [km]."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(max(0.0, a)))


class WeatherField:
    """
    Collection of per-site :class:`WeatherModel` instances with a spatially
    correlated synoptic layer tying them together.

    Parameters
    ----------
    site_weather_params : dict[str, dict]
        Mapping ``site_id -> WeatherModel kwargs`` — must include
        ``latitude_deg``/``longitude_deg`` per site (already the case when
        built via :meth:`from_topology`) for the spatial correlation to be
        meaningful; sites missing coordinates are treated as co-located
        (fully correlated with each other) as a harmless fallback.
    rng : numpy.random.Generator, optional
        If given, every site's ``WeatherModel`` gets an independent generator
        spawned from it, and the shared synoptic layer gets its own separate
        spawned generator. If ``None`` (default), everything falls back to
        the legacy global-RNG stream.
    correlation_length_km : float
        e-folding distance of the spatial correlation — sites this far apart
        share ~37% of the synoptic signal's variance. Default 300 km, roughly
        the scale of an Italian synoptic weather system.
    synoptic_tau_hours : float
        Persistence timescale of the shared disturbance (how long a
        frontal/anticyclonic regime tends to linger). Default 18 h.
    dt_seconds : float
        Step duration used to derive the synoptic AR(1) coefficient from
        *synoptic_tau_hours*. This is a soft climatological parameter (not a
        hard physical constant), so an approximate value is fine even if the
        scenario's actual ``dt`` differs somewhat. Default 600 s.

    Usage
    -----
    >>> field = WeatherField.from_topology(topology, rng=np.random.default_rng(42))
    >>> per_site = field.step(timestamp)   # {site_id: weather_dict}
    """

    def __init__(
        self,
        site_weather_params: Dict[str, Dict[str, Any]],
        rng: Optional[np.random.Generator] = None,
        correlation_length_km: float = 300.0,
        synoptic_tau_hours: float = 18.0,
        dt_seconds: float = 600.0,
    ) -> None:
        self._params = dict(site_weather_params)
        site_ids = list(self._params.keys())
        n = len(site_ids)
        self._site_ids: List[str] = site_ids

        # Local layer: one independent generator per site.
        child_rngs = spawn_generators(rng, n)
        self._models: Dict[str, WeatherModel] = {
            site_id: WeatherModel(**self._params[site_id], rng=child)
            for site_id, child in zip(site_ids, child_rngs)
        }

        # Regional/synoptic layer: one shared generator for the joint draw.
        self._regional_rng = resolve_rng(spawn_one(rng))
        self._phi = math.exp(-float(dt_seconds) / (max(1e-6, synoptic_tau_hours) * 3600.0))
        self._synoptic_z = np.zeros(n)
        self._chol = self._build_cholesky(site_ids, correlation_length_km)

        logger.info(
            "WeatherField built for %d site(s) (synoptic phi=%.3f, corr_length=%.0fkm)",
            n, self._phi, correlation_length_km,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_cholesky(self, site_ids: List[str], correlation_length_km: float) -> np.ndarray:
        n = len(site_ids)
        if n <= 1:
            return np.eye(max(n, 1))
        lats = [float(self._params[s].get("latitude_deg", 41.9)) for s in site_ids]
        lons = [float(self._params[s].get("longitude_deg", 12.5)) for s in site_ids]
        sigma = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = _haversine_km(lats[i], lons[i], lats[j], lons[j])
                rho = math.exp(-dist / max(1.0, correlation_length_km))
                sigma[i, j] = sigma[j, i] = rho
        sigma += 1e-6 * np.eye(n)  # numerical regularisation
        try:
            return np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:  # pragma: no cover — should not happen post-regularisation
            logger.warning("Synoptic correlation matrix not positive-definite; falling back to independent sites")
            return np.eye(n)

    @classmethod
    def from_topology(cls, topology, rng: Optional[np.random.Generator] = None, **kwargs) -> "WeatherField":
        """Build a field from a NetworkTopology (uses each site's weather + location)."""
        return cls({
            site_id: dict(site.weather_params)
            for site_id, site in topology.sites.items()
        }, rng=rng, **kwargs)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _advance_synoptic(self) -> np.ndarray:
        n = len(self._site_ids)
        if n == 0:
            return np.zeros(0)
        eps = np.asarray(self._regional_rng.normal(size=n), dtype=float)
        self._synoptic_z = self._phi * self._synoptic_z + math.sqrt(max(0.0, 1 - self._phi ** 2)) * (self._chol @ eps)
        return self._synoptic_z

    def step(self, timestamp: datetime) -> Dict[str, Dict[str, Any]]:
        """
        Advance every site's weather by one step, coupled through the shared
        spatially-correlated synoptic disturbance.

        Returns
        -------
        dict
            ``{site_id: weather_dict}`` — each value has the same shape as a
            single ``WeatherModel.step()`` output.
        """
        z = self._advance_synoptic()
        return {
            site_id: self._models[site_id].step(timestamp, synoptic=float(z[i]))
            for i, site_id in enumerate(self._site_ids)
        }

    def step_site(self, site_id: str, timestamp: datetime) -> Dict[str, Any]:
        """Advance and return the weather for a single site (no synoptic coupling in isolation)."""
        return self._models[site_id].step(timestamp)

    # ------------------------------------------------------------------
    # Access / lifecycle
    # ------------------------------------------------------------------

    def model(self, site_id: str) -> WeatherModel:
        return self._models[site_id]

    @property
    def site_ids(self):
        return list(self._models.keys())

    @property
    def synoptic_state(self) -> Dict[str, float]:
        """Current per-site synoptic disturbance index (diagnostic use)."""
        return {sid: float(z) for sid, z in zip(self._site_ids, self._synoptic_z)}

    def reset(self) -> None:
        for model in self._models.values():
            model.reset()
        self._synoptic_z = np.zeros(len(self._site_ids))

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return f"WeatherField(sites={self.site_ids})"
