"""
Fixed Baseline Policy Controller
================================
Deterministic controller used as baseline when no adaptive control is desired.

Policy goals
------------
- Keep electrolyzers always active at a low baseline load.
- On renewable surplus, split power approximately 50/50 between:
  (a) export to grid and (b) H2 production.
- Keep fuel-cell H2 usage relatively low (limited dispatch only on deficit).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ..digital_twin.grid_twin import GridState


class FixedPolicyController:
    """Simple fixed policy baseline controller."""

    def __init__(
        self,
        grid_config: Dict[str, Any],
        el_base_fraction: float = 0.20,
        fc_max_fraction: float = 0.15,
        fc_support_fraction: float = 0.30,
        split_surplus_to_grid: float = 0.50,
        split_surplus_to_h2: float = 0.50,
        fc_soc_min: float = 0.25,
        name: str = "FixedPolicyController",
    ) -> None:
        self.name = name
        cfg = grid_config.get("grid", grid_config)

        self._el_rated: Dict[str, float] = {}
        self._fc_rated: Dict[str, float] = {}
        self._gc_max_import: Dict[str, float] = {}
        self._gc_max_export: Dict[str, float] = {}
        self._load_ids = []

        for ec in cfg.get("electrolyzers", []):
            self._el_rated[ec["id"]] = float(ec["params"]["rated_power_kw"])
        for fc in cfg.get("fuel_cells", []):
            self._fc_rated[fc["id"]] = float(fc["params"]["rated_power_kw"])
        for gc in cfg.get("grid_connections", []):
            self._gc_max_import[gc["id"]] = float(gc["params"].get("max_import_kw", 800.0))
            self._gc_max_export[gc["id"]] = float(gc["params"].get("max_export_kw", 200.0))
        for ld in cfg.get("loads", []):
            self._load_ids.append(ld["id"])

        self._el_base_fraction = float(el_base_fraction)
        self._fc_max_fraction = float(fc_max_fraction)
        self._fc_support_fraction = float(fc_support_fraction)
        self._split_grid = float(split_surplus_to_grid)
        self._split_h2 = float(split_surplus_to_h2)
        self._fc_soc_min = float(fc_soc_min)

    def compute_actions(
        self,
        grid_state: GridState,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        renewable_kw = grid_state.wind_power_kw + grid_state.pv_power_kw
        load_kw = grid_state.load_kw
        net_surplus_kw = renewable_kw - load_kw

        total_el = sum(self._el_rated.values())
        total_fc = sum(self._fc_rated.values())
        base_el_kw = self._el_base_fraction * total_el

        if net_surplus_kw > 0.0:
            el_target_kw = max(base_el_kw, self._split_h2 * net_surplus_kw)
            grid_target_kw = -self._split_grid * net_surplus_kw
            fc_target_kw = 0.0
        else:
            # Keep EL running at baseline even in deficit (fixed policy).
            el_target_kw = base_el_kw
            deficit_with_el = max(0.0, load_kw + el_target_kw - renewable_kw)

            if grid_state.h2_soc >= self._fc_soc_min and total_fc > 0.0:
                fc_target_kw = min(
                    self._fc_max_fraction * total_fc,
                    self._fc_support_fraction * deficit_with_el,
                )
            else:
                fc_target_kw = 0.0

            grid_target_kw = max(0.0, deficit_with_el - fc_target_kw)

        actions: Dict[str, Dict[str, Any]] = {}

        if total_el > 0:
            for el_id, rated in self._el_rated.items():
                p = min(rated, el_target_kw * rated / total_el)
                actions[el_id] = {"power_setpoint_kw": p}

        if total_fc > 0:
            for fc_id, rated in self._fc_rated.items():
                p = min(rated, fc_target_kw * rated / total_fc)
                actions[fc_id] = {"power_setpoint_kw": p}

        if self._gc_max_import:
            if grid_target_kw >= 0.0:
                total_imp = sum(self._gc_max_import.values())
                for gc_id, cap in self._gc_max_import.items():
                    p = min(cap, grid_target_kw * cap / (total_imp + 1e-9))
                    actions[gc_id] = {"power_setpoint_kw": p}
            else:
                total_exp = sum(self._gc_max_export.values())
                needed = abs(grid_target_kw)
                for gc_id, cap in self._gc_max_export.items():
                    p = min(cap, needed * cap / (total_exp + 1e-9))
                    actions[gc_id] = {"power_setpoint_kw": -p}

        for ld_id in self._load_ids:
            actions[ld_id] = {"demand_response": 0.0}

        return actions

    def reset(self) -> None:
        pass
