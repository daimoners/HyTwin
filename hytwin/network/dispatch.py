"""
Network Dispatch (L1 — greedy merit-order)
==========================================
Explicit inter-node dispatch for the multi-site network.  This is the piece
that replaces the single-site *implicit residual* balance with a real,
energy-conserving allocation of flows over the interconnections.

Two independent greedy passes (order matters: H2 first, since pipeline
compression adds electric demand that the electric pass must then cover):

  1. :func:`dispatch_h2_pipelines` — move H₂ from high-SOC sites to low-SOC
     sites through pipelines (storage balancing), charging/discharging the
     site tanks and accounting compression energy as electric load.
  2. :func:`dispatch_electric_lines` — cover electric deficits from neighbour
     surpluses through electric lines, respecting capacity and losses.

Both are deterministic and side-effecting on the passed-in model/tank objects,
and both return structured per-link / per-site results for the NetworkTwin to
assemble into a :class:`~hytwin.network.network_state.NetworkState`.

The greedy policy is intentionally simple and transparent; it is the L1
baseline that a later LP/QP optimiser (F7) can replace behind the same API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

from .network_state import LinkFlow
from .topology import LinkType, NetworkTopology


# ------------------------------------------------------------------
# Tank helpers (a site may have several tanks)
# ------------------------------------------------------------------

def site_soc(tanks: List[Any]) -> float:
    """Capacity-weighted mean SOC across a site's tanks."""
    cap = sum(t.max_capacity_kg for t in tanks)
    if cap <= 0:
        return 0.0
    stored = sum(t.soc * t.max_capacity_kg for t in tanks)
    return stored / cap


def site_capacity_kg(tanks: List[Any]) -> float:
    return sum(t.max_capacity_kg for t in tanks)


def site_discharge(tanks: List[Any], kg: float, dt: float) -> float:
    """Discharge *kg* spread across tanks; return actual kg removed."""
    if kg <= 0 or not tanks:
        return 0.0
    n = len(tanks)
    removed = 0.0
    for t in tanks:
        removed += t.discharge(kg / n, dt)
    return removed


def site_charge(tanks: List[Any], kg: float, dt: float) -> float:
    """Charge *kg* spread across tanks; return actual kg stored."""
    if kg <= 0 or not tanks:
        return 0.0
    n = len(tanks)
    stored = 0.0
    for t in tanks:
        stored += t.charge(kg / n, dt)
    return stored


# ------------------------------------------------------------------
# H2 pipeline dispatch
# ------------------------------------------------------------------

def dispatch_h2_pipelines(
    topology: NetworkTopology,
    pipe_models: Dict[str, Any],
    tanks_by_site: Dict[str, List[Any]],
    dt: float,
    ts: datetime,
    soc_target: float = 0.5,
) -> Tuple[Dict[str, LinkFlow], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Move H₂ from high-SOC to low-SOC sites through the pipelines.

    Every pipeline is stepped each call (so in-transit gas keeps flowing even
    when no new injection is requested).

    Returns
    -------
    link_flows : dict[link_id -> LinkFlow]
    compressor_load_kw : dict[site_id -> kW]   (electric load at sending site)
    pipe_out_kg_h : dict[site_id -> kg/h]      (injected out)
    pipe_in_kg_h : dict[site_id -> kg/h]       (delivered in)
    """
    dt_h = dt / 3600.0
    link_flows: Dict[str, LinkFlow] = {}
    compressor_load_kw: Dict[str, float] = {s: 0.0 for s in topology.site_ids}
    pipe_out_kg_h: Dict[str, float] = {s: 0.0 for s in topology.site_ids}
    pipe_in_kg_h: Dict[str, float] = {s: 0.0 for s in topology.site_ids}

    for link in topology.links_by_type(LinkType.H2_PIPELINE):
        model = pipe_models[link.id]
        src, dst = link.from_site, link.to_site
        src_tanks = tanks_by_site.get(src, [])
        dst_tanks = tanks_by_site.get(dst, [])

        # Decide new injection from the SOC imbalance (source above target,
        # destination below target).
        inject_kg = 0.0
        soc_src = site_soc(src_tanks)
        soc_dst = site_soc(dst_tanks)
        if soc_src > soc_target and soc_dst < soc_target:
            spare_kg = (soc_src - soc_target) * site_capacity_kg(src_tanks)
            need_kg = (soc_target - soc_dst) * site_capacity_kg(dst_tanks)
            cap_kg = model.max_flow_kg_h * dt_h
            inject_kg = max(0.0, min(spare_kg, need_kg, cap_kg))

        # Withdraw from source tank(s) for the new injection.
        actual_injected_kg = site_discharge(src_tanks, inject_kg, dt) if inject_kg > 0 else 0.0
        flow_request_kg_h = actual_injected_kg / dt_h if dt_h > 0 else 0.0

        # Step the pipeline (also flushes previously injected gas).
        st = model.step(dt, {"flow_request_kg_h": flow_request_kg_h, "timestamp": ts})
        delivered_kg_h = st["delivered_kg_h"]
        delivered_kg = delivered_kg_h * dt_h

        # Charge destination tank(s) with delivered gas.
        site_charge(dst_tanks, delivered_kg, dt)

        compressor_load_kw[src] += st["compressor_power_kw"]
        pipe_out_kg_h[src] += st["flow_kg_h"]
        pipe_in_kg_h[dst] += delivered_kg_h

        link_flows[link.id] = LinkFlow(
            link_id=link.id,
            link_type=link.link_type.value,
            from_site=src,
            to_site=dst,
            flow=st["flow_kg_h"],
            delivered=delivered_kg_h,
            loss=0.0,
            utilization=st["utilization"],
            available=True,
        )

    return link_flows, compressor_load_kw, pipe_out_kg_h, pipe_in_kg_h


# ------------------------------------------------------------------
# Electric line dispatch
# ------------------------------------------------------------------

def dispatch_electric_lines(
    topology: NetworkTopology,
    line_models: Dict[str, Any],
    net_by_site: Dict[str, float],
    dt: float,
    ts: datetime,
) -> Tuple[Dict[str, LinkFlow], Dict[str, float], Dict[str, float]]:
    """
    Cover electric deficits from neighbour surpluses over the lines.

    ``net_by_site`` (surplus > 0, deficit < 0) is **mutated** in place to
    reflect the transfers.  Every line is stepped each call.

    Returns
    -------
    link_flows : dict[link_id -> LinkFlow]
    link_in_kw : dict[site_id -> kW]    (power delivered in from lines)
    link_out_kw : dict[site_id -> kW]   (power injected out onto lines)
    """
    link_flows: Dict[str, LinkFlow] = {}
    link_in_kw: Dict[str, float] = {s: 0.0 for s in topology.site_ids}
    link_out_kw: Dict[str, float] = {s: 0.0 for s in topology.site_ids}

    for link in topology.links_by_type(LinkType.ELECTRIC_LINE):
        model = line_models[link.id]
        a, b = link.from_site, link.to_site
        loss_frac = model.loss_fraction

        # Pick sender (surplus) and receiver (deficit) if such a pair exists.
        inject = 0.0
        direction = 0     # +1 = a->b, -1 = b->a
        if net_by_site[a] > 1e-6 and net_by_site[b] < -1e-6:
            sender, receiver, direction = a, b, +1
        elif net_by_site[b] > 1e-6 and net_by_site[a] < -1e-6:
            sender, receiver, direction = b, a, -1
        else:
            sender = receiver = None

        if sender is not None:
            need = -net_by_site[receiver]                 # deficit magnitude (kW)
            # Inject at sender so that `need` is delivered after losses.
            inject = min(net_by_site[sender], need / (1.0 - loss_frac + 1e-9))
            inject = max(0.0, inject)

        request_kw = direction * inject
        st = model.step(dt, {"power_request_kw": request_kw, "timestamp": ts})

        actual_inject = abs(st["flow_kw"])
        delivered = abs(st["delivered_kw"])
        if actual_inject > 1e-9 and sender is not None:
            net_by_site[sender] -= actual_inject
            net_by_site[receiver] += delivered
            link_out_kw[sender] += actual_inject
            link_in_kw[receiver] += delivered

        link_flows[link.id] = LinkFlow(
            link_id=link.id,
            link_type=link.link_type.value,
            from_site=a,
            to_site=b,
            flow=st["flow_kw"],
            delivered=st["delivered_kw"],
            loss=st["loss_kw"],
            utilization=st["utilization"],
            available=bool(st["available"] > 0.5),
        )

    return link_flows, link_in_kw, link_out_kw
