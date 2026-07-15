"""
test_network_twin.py
====================
End-to-end tests for the F3 NetworkTwin + explicit dispatch: energy-balance
closure, inter-node flow coherence, and system KPIs on the Italian pilot.
Run: pytest tests/test_network_twin.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.network.network_twin import NetworkTwin

PILOT_YAML = ROOT / "config" / "italy_network_pilot.yaml"
DT = 600.0
START = datetime(2024, 6, 15, 0, 0)


@pytest.fixture
def twin():
    np.random.seed(1234)
    sc = Scenario.from_yaml(PILOT_YAML)
    return NetworkTwin(sc.topology())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_builds_all_sites_and_links(twin):
    assert set(twin.topology.site_ids) == {"foggia", "napoli", "milano"}
    assert twin.network_state is None  # not stepped yet


def test_single_step_runs(twin):
    ns = twin.step(DT, START)
    assert ns is not None
    assert set(ns.nodes.keys()) == {"foggia", "napoli", "milano"}
    # 4 links (2 pipelines + 2 electric lines)
    assert len(ns.links) == 4


# ---------------------------------------------------------------------------
# Energy-balance closure — the core F3 milestone
# ---------------------------------------------------------------------------

def _assert_site_balance_closes(node, eps=1e-6):
    """gen + link_in + import  ==  dem + link_out + export + curtailed - unmet."""
    supply = node.generation_kw + node.link_import_kw + node.grid_import_kw
    sink = (node.demand_kw + node.link_export_kw
            + node.grid_export_kw + node.curtailed_kw - node.unmet_kw)
    assert abs(supply - sink) < eps, (
        f"balance broken at {node.site_id}: supply={supply} sink={sink}"
    )


def test_energy_balance_closes_every_step(twin):
    ts = START
    for _ in range(144):  # full 24 h
        ns = twin.step(DT, ts)
        for node in ns.nodes.values():
            _assert_site_balance_closes(node)
        ts += timedelta(seconds=DT)


def test_no_negative_flows_or_socs(twin):
    ts = START
    for _ in range(72):
        ns = twin.step(DT, ts)
        for n in ns.nodes.values():
            assert n.grid_import_kw >= -1e-9
            assert n.grid_export_kw >= -1e-9
            assert n.unmet_kw >= -1e-9
            assert n.curtailed_kw >= -1e-9
            assert 0.0 - 1e-9 <= n.h2_soc <= 1.0 + 1e-9
        ts += timedelta(seconds=DT)


# ---------------------------------------------------------------------------
# Inter-node flow coherence
# ---------------------------------------------------------------------------

def test_electric_link_delivered_leq_flow(twin):
    ts = START
    for _ in range(48):
        ns = twin.step(DT, ts)
        for l in ns.links.values():
            if l.link_type == "electric_line":
                # delivered magnitude never exceeds injected magnitude (losses)
                assert abs(l.delivered) <= abs(l.flow) + 1e-6
        ts += timedelta(seconds=DT)


def test_link_import_matches_delivery_direction(twin):
    """Sum of electric link deliveries in == sum of per-node link_import."""
    ts = START
    for _ in range(24):
        ns = twin.step(DT, ts)
        total_delivered = sum(
            abs(l.delivered) for l in ns.links.values() if l.link_type == "electric_line"
        )
        total_node_in = sum(n.link_import_kw for n in ns.nodes.values())
        assert total_node_in == pytest.approx(total_delivered, abs=1e-6)
        ts += timedelta(seconds=DT)


# ---------------------------------------------------------------------------
# System KPIs
# ---------------------------------------------------------------------------

def test_system_kpis_sane(twin):
    ns = twin.run(48, START, DT)[-1]
    assert ns.total_load_kw > 0
    assert 0.0 <= ns.network_self_sufficiency <= 1.0
    assert 0.0 <= ns.network_renewable_fraction <= 1.0
    assert 0.0 <= ns.reliability_index <= 1.0
    assert ns.cumulative_cost_eur == pytest.approx(
        sum(h.total_cost_eur_step for h in twin.history()), rel=1e-6
    )


def test_as_dict_serialisable(twin):
    ns = twin.step(DT, START)
    d = ns.as_dict()
    assert "nodes" in d and "links" in d
    assert set(d["nodes"].keys()) == {"foggia", "napoli", "milano"}


def test_reset_clears_state(twin):
    twin.run(10, START, DT)
    twin.reset()
    assert twin.network_state is None
    assert twin.history() == []


# ---------------------------------------------------------------------------
# Inter-node value: pipeline actually moves H2 to a low-SOC consumer
# ---------------------------------------------------------------------------

def test_pipeline_moves_h2_over_horizon(twin):
    """Over a day, at least some H2 should be delivered through the pipelines."""
    total_pipeline_delivery = 0.0
    ts = START
    for _ in range(144):
        ns = twin.step(DT, ts)
        total_pipeline_delivery += ns.inter_node_h2_kg_h * (DT / 3600.0)
        ts += timedelta(seconds=DT)
    assert total_pipeline_delivery > 0.0
