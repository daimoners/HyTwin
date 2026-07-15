"""
Reward Functions
================
Modular, composable reward functions for the H2 grid RL agent.

Each function receives the ``GridState`` and returns a scalar reward component.
The total reward is a weighted sum of components (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..digital_twin.grid_twin import GridState


@dataclass
class RewardConfig:
    """Weights for each reward component."""
    w_self_sufficiency: float = 1.0      # maximise self-sufficiency
    w_renewable_fraction: float = 0.8    # maximise renewable share
    w_grid_import_penalty: float = -0.5  # penalise grid import cost
    w_h2_soc_deviation: float = -0.3     # penalise tank over/under fill
    w_h2_soc_target: float = 0.5         # target SOC (0-1)
    w_anomaly_penalty: float = -1.0      # penalise system anomalies
    w_curtailment_penalty: float = -0.2  # penalise renewable waste
    w_demand_met: float = 1.2            # reward meeting load demand
    grid_import_price_eur_kwh: float = 0.15
    grid_export_price_eur_kwh: float = 0.05


def compute_reward(
    gs: GridState,
    prev_gs: Optional[GridState] = None,
    config: Optional[RewardConfig] = None,
) -> tuple[float, Dict[str, float]]:
    """
    Compute the total reward and a breakdown dict.

    Parameters
    ----------
    gs : GridState — current step state
    prev_gs : GridState, optional — previous step state (for differentials)
    config : RewardConfig, optional

    Returns
    -------
    total_reward : float
    components : dict[str, float] — individual terms for logging
    """
    cfg = config or RewardConfig()

    # --- Self-sufficiency (want ~1) ---
    r_self = cfg.w_self_sufficiency * gs.grid_self_sufficiency

    # --- Renewable fraction (want ~1) ---
    r_renew = cfg.w_renewable_fraction * gs.renewable_fraction

    # --- Grid import/export economics ---
    if gs.grid_exchange_kw > 0:
        # Importing – cost
        r_grid = cfg.w_grid_import_penalty * (
            gs.grid_exchange_kw * cfg.grid_import_price_eur_kwh / 60.0
        )
    else:
        # Exporting – small revenue
        r_grid = abs(gs.grid_exchange_kw) * cfg.grid_export_price_eur_kwh / 60.0

    # --- H2 SOC target deviation ---
    soc_error = abs(gs.h2_soc - cfg.w_h2_soc_target)
    r_soc = cfg.w_h2_soc_deviation * soc_error ** 2

    # --- Demand met ---
    supply_kw = gs.wind_power_kw + gs.pv_power_kw + gs.fuel_cell_power_kw
    demand_met = min(1.0, supply_kw / (gs.load_kw + 1e-9))
    r_demand = cfg.w_demand_met * demand_met

    # --- Anomaly penalty ---
    r_anomaly = cfg.w_anomaly_penalty * (1.0 - gs.overall_health)

    # --- Curtailment penalty (renewable generated but not used) ---
    available_renewable = gs.wind_power_kw + gs.pv_power_kw
    used_renewable = max(0.0, available_renewable + gs.grid_exchange_kw)
    curtailment = max(0.0, available_renewable - used_renewable) / (available_renewable + 1e-9)
    r_curtail = cfg.w_curtailment_penalty * curtailment

    total = r_self + r_renew + r_grid + r_soc + r_demand + r_anomaly + r_curtail

    components = {
        "self_sufficiency": r_self,
        "renewable_fraction": r_renew,
        "grid_exchange": r_grid,
        "h2_soc": r_soc,
        "demand_met": r_demand,
        "anomaly": r_anomaly,
        "curtailment": r_curtail,
        "total": total,
    }

    return total, components
