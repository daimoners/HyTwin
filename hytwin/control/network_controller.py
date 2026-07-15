"""
Network Classical Controller
============================
Rule-based controller for a **multi-site** H2 network.  It is a thin
orchestrator that instantiates one :class:`ClassicalController` per site
(reusing the existing single-site logic unchanged) and returns a combined
``actions_by_site`` mapping consumable by :meth:`NetworkTwin.step`.

Separation of concerns
----------------------
* **Site-local assets** (electrolyzers, fuel cells, demand response) are driven
  by each site's own cost-aware :class:`ClassicalController`, using that site's
  market-zone price model.
* **Inter-node exchange** (electric lines + H₂ pipelines) and the **national
  grid slack** are handled by the NetworkTwin's greedy dispatch (F3) — the
  controller does not need to micro-manage them.

Because the controller reacts to the *previous* step's state (a feedback loop,
exactly as the single-site engine does), :meth:`compute_actions` takes the last
:class:`NetworkState`; on the cold-start step it returns empty per-site actions,
which makes the NetworkTwin fall back to its naive defaults for that one step.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .classical_controller import ClassicalController


class _RngSafeCostModel:
    """
    Read-only price proxy over an :class:`EnergyCostModel` whose price queries
    do **not** perturb the RNG or the model's internal day/spike state.

    The authoritative advancement of the cost model still happens inside
    ``GridTwin.step``; the controller only *peeks* at prices.  This keeps the
    weather/price/outage RNG streams identical whether or not a controller is
    attached, which is what makes a no-control vs rule-based (vs RL) comparison
    fair — the same reproducibility guarantee the single-site harness relies on.

    Snapshots the cost model's **own** generator state (``cm._rng.bit_generator``)
    when it owns an isolated ``numpy.random.Generator`` — the normal case since
    ``NetworkTwin(seed=...)`` gives every site's ``EnergyCostModel`` one (see
    ``hytwin.core.rng``) — and falls back to snapshotting the legacy global
    ``numpy.random`` state for cost models that were never migrated (``rng=None``
    call sites).  Getting this right matters: with a real per-instance
    generator, restoring the *global* state instead of the model's own would
    silently fail to undo the peek's draws, permanently perturbing the site's
    future prices every time a forecast/lookahead peeked at them.
    """

    def __init__(self, cost_model) -> None:
        self._cm = cost_model

    def _peek(self, fn, ts):
        if self._cm is None:
            return 0.15
        cm = self._cm
        saved = (cm._current_day, cm._daily_factor, cm._spike_remaining, cm._spike_active)
        rng = getattr(cm, "_rng", None)
        isolated = hasattr(rng, "bit_generator")
        rng_snapshot = rng.bit_generator.state if isolated else np.random.get_state()
        try:
            return fn(ts)
        finally:
            (cm._current_day, cm._daily_factor, cm._spike_remaining, cm._spike_active) = saved
            if isolated:
                rng.bit_generator.state = rng_snapshot
            else:
                np.random.set_state(rng_snapshot)

    def get_buy_price(self, ts: datetime) -> float:
        return self._peek(self._cm.get_buy_price, ts) if self._cm else 0.15

    def get_sell_price(self, ts: datetime) -> float:
        return self._peek(self._cm.get_sell_price, ts) if self._cm else 0.15 * 0.28


class NetworkClassicalController:
    """
    Per-site rule-based control for a network.

    Build it from a live :class:`~hytwin.network.network_twin.NetworkTwin` so it
    can share each site's cost model (for price-aware dispatch):

    >>> ctrl = NetworkClassicalController.from_network(network_twin)
    >>> twin.run(144, start, dt,
    ...          actions_provider=lambda i, ts, prev: ctrl.compute_actions(prev, ts))
    """

    name = "NetworkClassicalController"

    def __init__(
        self,
        site_controllers: Dict[str, ClassicalController],
        fc_rated_by_site: Optional[Dict[str, float]] = None,
        elec_neighbors: Optional[Dict[str, list]] = None,
        peak_price_threshold: float = 0.22,
        soc_floor_for_fc: float = 0.20,
        peak_shave_margin_kw: float = 5.0,
    ) -> None:
        self._controllers = site_controllers
        self._fc_rated = fc_rated_by_site or {}
        # {site_id: [(neighbor_id, electric_line_capacity_kw), ...]}
        self._elec_neighbors = elec_neighbors or {}
        self._peak_price = float(peak_price_threshold)
        self._soc_floor = float(soc_floor_for_fc)
        self._peak_margin = float(peak_shave_margin_kw)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_network(cls, network_twin, **kwargs) -> "NetworkClassicalController":
        """
        Build one ClassicalController per site, sharing that site's cost model.

        Extra ``kwargs`` are forwarded to every per-site ClassicalController
        (e.g. ``soc_target``, ``price_high_threshold``).
        """
        controllers: Dict[str, ClassicalController] = {}
        fc_rated_by_site: Dict[str, float] = {}
        # Separate overlay kwargs (network-level) from per-site controller kwargs.
        overlay_keys = ("peak_price_threshold", "soc_floor_for_fc", "peak_shave_margin_kw")
        overlay_kwargs = {k: kwargs.pop(k) for k in overlay_keys if k in kwargs}

        for sid, spec in network_twin.topology.sites.items():
            cost_model = network_twin.site(sid).cost_model
            controllers[sid] = ClassicalController(
                grid_config=spec.grid_config,
                cost_model=_RngSafeCostModel(cost_model),
                name=f"Classical[{sid}]",
                **kwargs,
            )
            fc_rated_by_site[sid] = sum(
                float(fc["params"]["rated_power_kw"])
                for fc in spec.grid_config.get("fuel_cells", [])
            )

        # Electric-line neighbours + capacities (to reserve renewable surplus
        # for efficient electric transport instead of lossy H₂ round-trips).
        from ..network.topology import LinkType
        elec_neighbors: Dict[str, list] = {sid: [] for sid in controllers}
        for lid, link in network_twin.topology.links.items():
            if link.link_type != LinkType.ELECTRIC_LINE:
                continue
            cap = network_twin._line_models[lid].max_power_kw
            elec_neighbors.setdefault(link.from_site, []).append((link.to_site, cap))
            elec_neighbors.setdefault(link.to_site, []).append((link.from_site, cap))

        return cls(
            controllers,
            fc_rated_by_site=fc_rated_by_site,
            elec_neighbors=elec_neighbors,
            **overlay_kwargs,
        )

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_actions(
        self,
        prev_state: Optional[Any],
        timestamp: datetime,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Return ``{site_id: {component_id: {param: value}}}`` for this step.

        Parameters
        ----------
        prev_state : NetworkState or None
            The previous step's network state (feedback).  ``None`` on the
            first step → empty actions (NetworkTwin uses naive defaults).
        timestamp : datetime
        """
        if prev_state is None:
            return {}

        actions_by_site: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for sid, ctrl in self._controllers.items():
            node = prev_state.nodes.get(sid)
            if node is None or node.grid_state is None:
                continue
            actions = ctrl.compute_actions(node.grid_state, timestamp)
            self._apply_network_overlay(sid, node, actions, prev_state)
            actions_by_site[sid] = actions
        return actions_by_site

    def _apply_network_overlay(self, sid, node, actions, prev_state) -> None:
        """
        Two network-aware corrections to the single-site rule set:

        1. **Electrolyzer = surplus not exportable by wire.**  Moving energy to
           a deficit neighbour over an electric line loses ~6 %; converting it
           to H₂ and back (electrolysis → pipeline → fuel cell) loses ~68 %.
           So a producer should first let its surplus flow to neighbours over
           the wires (the F3 dispatch does this from the leftover net) and only
           make H₂ from the surplus that neighbours *cannot* absorb.  We cap the
           electrolyzers at ``surplus − exportable_to_neighbours``.

        2. **Green-H₂-first fuel cell.**  Discharge the FC to cover local deficit
           using stored (surplus-made) H₂ before importing from the grid.  The
           single-site rule only discharges on grid outages, so without this it
           would hoard H₂ and never pay it back in network mode.
        """
        gs = node.grid_state
        renewable_kw = gs.wind_power_kw + gs.pv_power_kw
        surplus_kw = max(0.0, renewable_kw - gs.load_kw)
        price = float(getattr(gs, "energy_price_eur_kwh", 0.15))
        soc = float(node.h2_soc)

        # Estimate how much of the surplus neighbours could absorb over the
        # electric lines (their raw deficit, capped by line capacity).
        exportable_kw = 0.0
        if prev_state is not None:
            for nbr, cap_kw in self._elec_neighbors.get(sid, []):
                nn = prev_state.nodes.get(nbr)
                if nn is None:
                    continue
                nbr_deficit = max(0.0, nn.load_kw - nn.renewable_kw)
                exportable_kw += min(nbr_deficit, cap_kw)
        surplus_for_h2 = max(0.0, surplus_kw - exportable_kw)

        # ---- 1. Cap electrolyzers at the non-exportable surplus ----
        el_ids = [nid for nid in actions if "el" in nid and nid.startswith(f"{sid}_")]
        if el_ids:
            share = surplus_for_h2 / len(el_ids)
            for nid in el_ids:
                cur = float(actions[nid].get("power_setpoint_kw", 0.0))
                actions[nid]["power_setpoint_kw"] = min(cur, share)

        # ---- 2. Green-H₂-first fuel-cell dispatch ----
        # Use stored (green, surplus-made) H₂ to cover any local deficit before
        # importing from the national grid, whenever SOC is above the floor.
        # Above the peak-price threshold we dispatch more aggressively (deeper
        # into the reserve); off-peak we only tap the comfort band above target.
        fc_cap = self._fc_rated.get(sid, 0.0)
        if fc_cap <= 0.0 or soc <= self._soc_floor:
            return
        deficit_kw = max(0.0, gs.load_kw - renewable_kw)
        if deficit_kw <= 0.0:
            return
        if price >= self._peak_price:
            usable = (soc - self._soc_floor) / max(1e-9, 1.0 - self._soc_floor)
        else:
            # off-peak: only spend H₂ held above the target SOC band
            usable = max(0.0, (soc - 0.5) / max(1e-9, 1.0 - 0.5))
        fc_target = min(deficit_kw + self._peak_margin, fc_cap * usable)
        fc_ids = [nid for nid in actions if "fc" in nid and nid.startswith(f"{sid}_")]
        if not fc_ids or fc_target <= 0.0:
            return
        share = fc_target / len(fc_ids)
        for nid in fc_ids:
            prev = float(actions.get(nid, {}).get("power_setpoint_kw", 0.0))
            actions.setdefault(nid, {})["power_setpoint_kw"] = max(prev, share)

    def reset(self) -> None:
        for ctrl in self._controllers.values():
            ctrl.reset()
