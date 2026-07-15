"""
Scenario
========
Defines a complete simulation scenario: grid topology, weather parameters,
sensor configuration, and a controllable schedule.

Scenarios can be defined programmatically or loaded from YAML.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class Scenario:
    """
    Container for everything needed to run one simulation.

    Parameters
    ----------
    name : str
    grid_config : dict        — component specs for GridTwin
    weather_params : dict     — kwargs for WeatherModel
    sensor_config : list      — list of sensor spec dicts
    start_time : datetime
    dt_seconds : float
    episode_steps : int
    """

    def __init__(
        self,
        name: str,
        grid_config: Dict[str, Any],
        weather_params: Optional[Dict[str, Any]] = None,
        sensor_config: Optional[List[Dict[str, Any]]] = None,
        start_time: Optional[datetime] = None,
        dt_seconds: float = 60.0,
        episode_steps: int = 1440,
    ) -> None:
        self.name = name
        self.grid_config = grid_config
        self.weather_params = weather_params or {}
        self.sensor_config = sensor_config or []
        self.start_time = start_time or datetime(2024, 6, 15, 0, 0)
        self.dt_seconds = dt_seconds
        self.episode_steps = episode_steps

        # Multi-site network topology (None for legacy single-plant scenarios).
        # Populated by from_yaml when a top-level ``network:`` block is present.
        self.network = None  # type: Optional[NetworkTopology]

        self._agent: Optional[Any] = None
        self._schedule: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    @property
    def is_multi_site(self) -> bool:
        """True if this scenario describes a multi-site network."""
        return self.network is not None and self.network.is_multi_site

    def topology(self):
        """
        Return the :class:`NetworkTopology` for this scenario.

        For legacy single-plant scenarios a 1-site / 0-link topology is built
        on the fly from ``grid_config`` / ``weather_params`` so that network
        code can treat every scenario uniformly.
        """
        if self.network is not None:
            return self.network
        from ..network.topology import NetworkTopology
        return NetworkTopology.single_site(
            grid_config=self.grid_config,
            weather_params=self.weather_params,
            sensor_config=self.sensor_config,
        )

    # ------------------------------------------------------------------
    # Control actions
    # ------------------------------------------------------------------

    def set_agent(self, agent) -> None:
        """Attach an RL agent; its predict() provides control actions."""
        self._agent = agent

    def set_schedule(self, fn: Callable) -> None:
        """
        Attach a rule-based schedule function.
        fn(step, timestamp, grid_state) → dict of control dicts.
        """
        self._schedule = fn

    def get_control_actions(
        self,
        step: int,
        ts: datetime,
        grid_state,
    ) -> Dict[str, Any]:
        """Resolve control actions for this step."""
        if self._agent is not None and grid_state is not None:
            # Import here to avoid circular
            from ..digital_twin.grid_twin import GridTwin
            obs = grid_state  # will be converted by caller
            try:
                from ..rl.environment import OBS_MAX, OBS_MIN
                import numpy as np
                obs_vec = _grid_state_to_obs(grid_state, OBS_MAX, OBS_MIN)
                action, _ = self._agent.predict(obs_vec, deterministic=True)
                return self._action_to_control(action)
            except Exception as e:
                logger.warning("Agent predict failed: %s", e)

        if self._schedule is not None:
            return self._schedule(step, ts, grid_state)

        # Default: electrolyzer 50 %, fuel cell 30 %, no DR
        return self._default_actions()

    def _default_actions(self) -> Dict[str, Any]:
        return {
            "_el_fraction": 0.5,
            "_fc_fraction": 0.3,
            "_dr_fraction": 0.0,
        }

    def _action_to_control(self, action) -> Dict[str, Any]:
        """Convert raw action array to per-component control dicts."""
        import numpy as np
        a = np.clip(action, 0.0, 1.0)
        return {
            "_el_fraction": float(a[0]),
            "_fc_fraction": float(a[1]) if len(a) > 1 else 0.3,
            "_dr_fraction": float(a[2]) if len(a) > 2 else 0.0,
        }

    # ------------------------------------------------------------------
    # YAML persistence
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        st = data.get("start_time")
        if isinstance(st, str):
            st = datetime.fromisoformat(st)

        # ── Multi-site network scenario ────────────────────────────────────
        if "network" in data:
            from ..network.topology import NetworkTopology
            topology = NetworkTopology.from_config(data["network"])
            # Populate the legacy single-site fields from the first site so
            # that code paths still expecting scenario.grid_config keep working
            # (and so a network scenario degrades gracefully if run through the
            # current single-site engine before the NetworkTwin is wired in).
            first = topology.sites[topology.site_ids[0]]
            scenario = cls(
                name=data.get("name", Path(path).stem),
                grid_config=dict(first.grid_config),
                weather_params=dict(first.weather_params),
                sensor_config=list(first.sensor_config),
                start_time=st,
                dt_seconds=float(data.get("dt_seconds", 60.0)),
                episode_steps=int(data.get("episode_steps", 1440)),
            )
            scenario.network = topology
            return scenario

        # ── Legacy single-plant scenario ───────────────────────────────────
        grid_config = dict(data["grid"])
        # Forward optional top-level sections into grid_config so GridTwin sees them
        for extra_key in ("energy_cost",):
            if extra_key in data:
                grid_config[extra_key] = data[extra_key]
        return cls(
            name=data.get("name", Path(path).stem),
            grid_config=grid_config,
            weather_params=data.get("weather", {}),
            sensor_config=data.get("sensors", []),
            start_time=st,
            dt_seconds=float(data.get("dt_seconds", 60.0)),
            episode_steps=int(data.get("episode_steps", 1440)),
        )

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "dt_seconds": self.dt_seconds,
            "episode_steps": self.episode_steps,
            "grid": self.grid_config,
            "weather": self.weather_params,
            "sensors": self.sensor_config,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _grid_state_to_obs(gs, obs_max, obs_min):
    import numpy as np
    raw = np.array([
        gs.wind_power_kw, gs.pv_power_kw,
        gs.electrolyzer_power_kw, gs.fuel_cell_power_kw,
        gs.load_kw, gs.grid_exchange_kw,
        gs.h2_production_kg_h, gs.h2_consumption_kg_h,
        gs.h2_storage_kg, gs.h2_soc, gs.h2_pressure_bar,
        gs.renewable_fraction, gs.grid_self_sufficiency, gs.overall_health,
    ], dtype=np.float32)
    mid = (obs_max + obs_min) / 2.0
    scale = (obs_max - obs_min) / 2.0 + 1e-9
    return (raw - mid) / scale
