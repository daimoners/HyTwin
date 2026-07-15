"""
hytwin.network
==============
Multi-site network layer: promotes HyTwin from a single-plant digital twin
to a geographically distributed H2 energy network (see
``docs/08_network_layer_plan.md``).

Phase F1 exposes the structural topology (sites + links).  Later phases add
the ``NetworkTwin`` orchestrator, link physics models and network dispatch.
"""

from .topology import (
    Location,
    LinkType,
    SiteSpec,
    LinkSpec,
    NetworkTopology,
    haversine_km,
)
from .network_state import NodeState, LinkFlow, NetworkState
from .network_twin import NetworkTwin

__all__ = [
    "Location",
    "LinkType",
    "SiteSpec",
    "LinkSpec",
    "NetworkTopology",
    "haversine_km",
    "NodeState",
    "LinkFlow",
    "NetworkState",
    "NetworkTwin",
]
