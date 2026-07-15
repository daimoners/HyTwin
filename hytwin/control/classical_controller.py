"""
Classical (Rule-Based) Controller
===================================
A cost-aware, heuristic controller for the H2 grid.

Strategy
--------
The controller observes the current system state and the electricity
buy price from the EnergyCostModel, then decides:

1. **High-price + high H2 SOC** → maximise FC dispatch, minimise grid import
2. **Low-price + low H2 SOC** → run electrolyzer, accept grid import to charge
3. **Surplus renewable** → always run electrolyzer up to SOC ceiling
4. **Grid unavailable** → shift entirely to local FC + renewables
5. **Demand response** → activate load reduction when grid is both
   expensive *and* unavailable

The controller also applies ramp-rate smoothing and SOC guard-rails
(min 10 %, max 90 %) to protect the H₂ tank.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from ..digital_twin.grid_twin import GridState
from ..models.energy_cost import EnergyCostModel


class ClassicalController:
    """
    Rule-based controller for the H2 grid.

    Parameters
    ----------
    grid_config : dict
        The same grid config dict passed to GridTwin (used to read component
        rated powers for setpoint scaling).
    cost_model : EnergyCostModel, optional
        If provided, price-aware dispatch is enabled.
    soc_target : float
        H₂ SOC target (0–1). Default 0.50.
    soc_min : float
        FC dispatch is disabled when SOC falls below this. Default 0.10.
    soc_max : float
        Electrolyzer is fully cut when SOC exceeds this. Default 0.90.
    price_high_threshold : float
        Buy price [€/kWh] above which we avoid grid import. Default 0.20.
    price_low_threshold : float
        Buy price [€/kWh] below which we allow extra grid import for H₂. Default 0.11.
    surplus_threshold_kw : float
        Net renewable surplus [kW] above which we definitely run the EL. Default 30.
    deficit_threshold_kw : float
        Net deficit [kW] below which we activate FC dispatch. Default –20.
    name : str
        Controller name (for logging / reporting). Default 'ClassicalController'.
    """

    def __init__(
        self,
        grid_config: Dict[str, Any],
        cost_model: Optional[EnergyCostModel] = None,
        soc_target: float = 0.50,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        price_high_threshold: float = 0.20,
        price_low_threshold: float = 0.11,
        surplus_threshold_kw: float = 30.0,
        deficit_threshold_kw: float = -20.0,
        name: str = "ClassicalController",
    ) -> None:
        self.name = name
        self._cost_model = cost_model
        self._soc_target = soc_target
        self._soc_min = soc_min
        self._soc_max = soc_max
        self._price_high = price_high_threshold
        self._price_low = price_low_threshold
        self._surplus_thr = surplus_threshold_kw
        self._deficit_thr = deficit_threshold_kw

        # Extract rated powers from config
        self._el_rated: Dict[str, float] = {}
        self._fc_rated: Dict[str, float] = {}
        self._gc_max_import: Dict[str, float] = {}
        cfg = grid_config.get("grid", grid_config)  # handle nested or flat
        for ec in cfg.get("electrolyzers", []):
            self._el_rated[ec["id"]] = float(ec["params"]["rated_power_kw"])
        for fc in cfg.get("fuel_cells", []):
            self._fc_rated[fc["id"]] = float(fc["params"]["rated_power_kw"])
        for gc in cfg.get("grid_connections", []):
            self._gc_max_import[gc["id"]] = float(
                gc["params"].get("max_import_kw", 800.0)
            )

        # Previous setpoints (for smooth transitions)
        self._prev_el_sp: Dict[str, float] = {k: 0.0 for k in self._el_rated}
        self._prev_fc_sp: Dict[str, float] = {k: 0.0 for k in self._fc_rated}

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_actions(
        self,
        grid_state: GridState,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute control actions for the current grid state.

        Returns
        -------
        dict
            {component_id: {param: value, ...}} — compatible with GridTwin.step().
        """
        ts = timestamp or grid_state.timestamp
        soc = grid_state.h2_soc
        renewable_kw = grid_state.wind_power_kw + grid_state.pv_power_kw
        load_kw = grid_state.load_kw
        net_renewable = renewable_kw - load_kw   # positive → surplus

        # Current price (or default mid-range)
        price = (
            self._cost_model.get_buy_price(ts)
            if self._cost_model is not None
            else 0.15
        )
        grid_available = grid_state.grid_available
        price_is_high = price >= self._price_high
        price_is_low = price <= self._price_low

        actions: Dict[str, Dict[str, Any]] = {}

        # ---- Electrolyzer dispatch ----------------------------------------
        el_sp_target = self._compute_el_setpoint(
            net_renewable, soc, price_is_high, price_is_low, grid_available
        )
        for el_id, rated_kw in self._el_rated.items():
            # Smooth ramp (±50 % per step — physical model enforces its own limits)
            prev = self._prev_el_sp.get(el_id, 0.0)
            el_sp = float(_ramp(prev, el_sp_target, 0.50))
            self._prev_el_sp[el_id] = el_sp
            actions[el_id] = {"power_setpoint_kw": el_sp * rated_kw}

        # ---- Fuel-cell dispatch --------------------------------------------
        fc_sp_target = self._compute_fc_setpoint(
            net_renewable, soc, price_is_high, grid_available
        )
        for fc_id, rated_kw in self._fc_rated.items():
            prev = self._prev_fc_sp.get(fc_id, 0.0)
            fc_sp = float(_ramp(prev, fc_sp_target, 0.50))
            self._prev_fc_sp[fc_id] = fc_sp
            actions[fc_id] = {"power_setpoint_kw": fc_sp * rated_kw}

        # ---- Grid connection dispatch --------------------------------------
        for gc_id, max_import in self._gc_max_import.items():
            grid_import_sp = self._compute_grid_setpoint(
                net_renewable, soc, price_is_high, price_is_low,
                grid_available, max_import,
                el_sp_target, fc_sp_target,
            )
            actions[gc_id] = {"power_setpoint_kw": grid_import_sp}

        # ---- Demand response -----------------------------------------------
        dr_factor = 0.0
        if (not grid_available) or (price_is_high and net_renewable < self._deficit_thr):
            # Curtail load by up to 20 %
            dr_factor = min(0.20, abs(net_renewable) / (load_kw + 1e-9))
        for ld in grid_state.anomaly_nodes:
            pass  # future: per-node load shedding
        for node_id in _all_load_ids(actions, grid_state):
            actions[node_id] = actions.get(node_id, {})
            actions[node_id]["demand_response"] = dr_factor

        return actions

    # ------------------------------------------------------------------
    # Setpoint helpers
    # ------------------------------------------------------------------

    def _compute_el_setpoint(
        self,
        net_renewable: float,
        soc: float,
        price_high: bool,
        price_low: bool,
        grid_avail: bool,
    ) -> float:
        """Return electrolyzer setpoint [0-1]."""
        if soc >= self._soc_max:
            return 0.0          # tank full — no H₂ production
        if price_high and net_renewable <= self._surplus_thr:
            return 0.0          # expensive + no surplus → skip EL
        if net_renewable > self._surplus_thr:
            # Scale EL to consume the surplus
            return min(1.0, net_renewable / max(1.0, sum(self._el_rated.values())))
        if price_low and soc < self._soc_target:
            # Cheap energy — run EL to build up H₂ reserve
            soc_deficit = self._soc_target - soc
            return min(0.6, soc_deficit * 2.0)
        return 0.0

    def _compute_fc_setpoint(
        self,
        net_renewable: float,
        soc: float,
        price_high: bool,
        grid_avail: bool,
    ) -> float:
        """Return fuel-cell setpoint [0-1]."""
        if soc <= self._soc_min:
            return 0.0          # protect tank
        if net_renewable < self._deficit_thr:
            # Deficit → use FC to fill gap (up to SOC headroom)
            soc_headroom = (soc - self._soc_min) / max(1e-9, 1 - self._soc_min)
            deficit_fraction = min(1.0, abs(net_renewable) / 200.0)
            return min(1.0, deficit_fraction * soc_headroom)
        if price_high and soc > self._soc_target and not grid_avail:
            # High price + grid down → dispatch FC
            soc_surplus = (soc - self._soc_target) / max(1e-9, 1 - self._soc_target)
            return min(0.7, soc_surplus)
        if price_high and soc > self._soc_target + 0.1:
            return min(0.5, (soc - self._soc_target))
        return 0.0

    def _compute_grid_setpoint(
        self,
        net_renewable: float,
        soc: float,
        price_high: bool,
        price_low: bool,
        grid_avail: bool,
        max_import: float,
        el_sp: float,
        fc_sp: float,
    ) -> float:
        """Return grid import setpoint [kW]; negative = export."""
        if not grid_avail:
            return 0.0
        if price_high:
            # Minimise import — only buy if nothing else can cover
            deficit = max(0.0, -net_renewable - fc_sp * sum(self._fc_rated.values()))
            return min(deficit * 0.5, max_import * 0.2)
        if price_low and soc < self._soc_target:
            # Buy cheap grid power to fill the electrolyzer
            el_demand = el_sp * sum(self._el_rated.values())
            shortfall = max(0.0, el_demand - max(0.0, net_renewable))
            return min(shortfall, max_import * 0.5)
        # Normal operation: cover deficit up to 60 % of max import
        deficit = max(0.0, -net_renewable)
        return min(deficit, max_import * 0.6)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prev_el_sp = {k: 0.0 for k in self._el_rated}
        self._prev_fc_sp = {k: 0.0 for k in self._fc_rated}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ramp(current: float, target: float, max_step: float) -> float:
    """Ramp-rate limiter: move current towards target by at most max_step."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + max_step * (1 if delta > 0 else -1)


def _all_load_ids(
    actions: Dict[str, Any], grid_state: GridState
) -> list:
    """
    Return node IDs that appear to be loads (not already in actions).
    We detect them by absence in the controller-owned id sets so this
    won't duplicate existing action entries.
    """
    # The controller doesn't know load IDs directly; we skip load DR
    # for now (actions dict is updated inline in compute_actions).
    return []
