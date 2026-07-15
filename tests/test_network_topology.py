"""
test_network_topology.py
========================
Unit tests for the F1 network layer: topology parsing, backward-compat
single-site wrapping, per-site weather field, and Scenario YAML integration.
Run: pytest tests/test_network_topology.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.network.topology import (
    NetworkTopology,
    LinkType,
    SiteSpec,
    Location,
    haversine_km,
)
from hytwin.weather.weather_field import WeatherField
from hytwin.simulation.scenario import Scenario

PILOT_YAML = ROOT / "config" / "italy_network_pilot.yaml"
LEGACY_YAML = ROOT / "config" / "advanced_grid.yaml"


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def test_haversine_known_distance():
    # Foggia -> Milano is roughly 560-600 km great-circle.
    d = haversine_km(41.46, 15.55, 45.46, 9.19)
    assert 500.0 < d < 700.0


def test_haversine_zero():
    assert haversine_km(45.0, 9.0, 45.0, 9.0) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Topology construction from config
# ---------------------------------------------------------------------------

def _mini_network_cfg():
    return {
        "sites": [
            {
                "id": "a",
                "location": {"name": "A", "lat": 41.0, "lon": 15.0, "alt_m": 50},
                "weather": {"weibull_c": 7.0},
                "energy_cost": {"f1_price": 0.24},
                "grid": {"loads": [{"id": "a_load", "params": {"base_load_kw": 100.0}}]},
                "sensors": [],
            },
            {
                "id": "b",
                "location": {"name": "B", "lat": 45.0, "lon": 9.0, "alt_m": 120},
                "weather": {"weibull_c": 5.0},
                "grid": {"loads": [{"id": "b_load", "params": {"base_load_kw": 300.0}}]},
            },
        ],
        "links": [
            {"id": "ab_h2", "type": "h2_pipeline", "from": "a", "to": "b",
             "params": {"max_flow_kg_h": 300.0}},
            {"id": "ab_el", "type": "electric_line", "from": "a", "to": "b",
             "params": {"length_km": 999.0, "max_power_mw": 50.0}},
        ],
    }


def test_topology_from_config_basic():
    topo = NetworkTopology.from_config(_mini_network_cfg())
    assert topo.is_multi_site
    assert set(topo.site_ids) == {"a", "b"}
    assert len(topo.links) == 2
    assert topo.links["ab_h2"].link_type == LinkType.H2_PIPELINE
    assert topo.links["ab_el"].link_type == LinkType.ELECTRIC_LINE


def test_site_weather_gets_location_geodata():
    topo = NetworkTopology.from_config(_mini_network_cfg())
    site_a = topo.sites["a"]
    assert site_a.weather_params["latitude_deg"] == 41.0
    assert site_a.weather_params["longitude_deg"] == 15.0
    assert site_a.weather_params["altitude_m"] == 50.0
    # explicit weather param preserved
    assert site_a.weather_params["weibull_c"] == 7.0


def test_energy_cost_forwarded_into_grid_config():
    topo = NetworkTopology.from_config(_mini_network_cfg())
    assert topo.sites["a"].grid_config["energy_cost"]["f1_price"] == 0.24


def test_link_length_autocomputed_when_missing():
    topo = NetworkTopology.from_config(_mini_network_cfg())
    # h2 link had no length -> auto-computed from coordinates
    assert topo.links["ab_h2"].length_km is not None
    assert topo.links["ab_h2"].length_km > 0
    # electric link had explicit length -> preserved
    assert topo.links["ab_el"].length_km == 999.0


def test_neighbors_and_links_of():
    topo = NetworkTopology.from_config(_mini_network_cfg())
    assert topo.neighbors("a") == ["b"]
    assert {l.id for l in topo.links_of("a")} == {"ab_h2", "ab_el"}
    assert topo.links_by_type(LinkType.H2_PIPELINE)[0].id == "ab_h2"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_unknown_site_reference_raises():
    cfg = _mini_network_cfg()
    cfg["links"][0]["to"] = "ghost"
    with pytest.raises(ValueError):
        NetworkTopology.from_config(cfg)


def test_self_link_raises():
    cfg = _mini_network_cfg()
    cfg["links"][0]["to"] = "a"
    with pytest.raises(ValueError):
        NetworkTopology.from_config(cfg)


def test_duplicate_site_raises():
    cfg = _mini_network_cfg()
    cfg["sites"][1]["id"] = "a"
    with pytest.raises(ValueError):
        NetworkTopology.from_config(cfg)


def test_unknown_link_type_raises():
    cfg = _mini_network_cfg()
    cfg["links"][0]["type"] = "teleporter"
    with pytest.raises(ValueError):
        NetworkTopology.from_config(cfg)


# ---------------------------------------------------------------------------
# Backward-compat single-site wrapping
# ---------------------------------------------------------------------------

def test_single_site_wrapper():
    grid = {"loads": [{"id": "l1", "params": {"base_load_kw": 100.0}}]}
    weather = {"latitude_deg": 40.5, "longitude_deg": 14.8, "altitude_m": 50.0}
    topo = NetworkTopology.single_site(grid, weather, sensor_config=[])
    assert not topo.is_multi_site
    assert len(topo.sites) == 1
    assert len(topo.links) == 0
    site = topo.sites[topo.site_ids[0]]
    assert site.location.lat == 40.5


# ---------------------------------------------------------------------------
# WeatherField
# ---------------------------------------------------------------------------

def test_weather_field_per_site_output():
    topo = NetworkTopology.from_config(_mini_network_cfg())
    field = WeatherField.from_topology(topo)
    assert len(field) == 2
    out = field.step(datetime(2024, 6, 15, 12, 0))
    assert set(out.keys()) == {"a", "b"}
    for site_id, w in out.items():
        assert "wind_speed_ms" in w or "wind_ms" in w
        assert "ghi_wm2" in w
    field.reset()


def test_weather_field_sites_differ():
    # Two sites with very different latitude/cloud should not be identical.
    topo = NetworkTopology.from_config(_mini_network_cfg())
    field = WeatherField.from_topology(topo)
    ts = datetime(2024, 6, 15, 12, 0)
    out = field.step(ts)
    # GHI at same instant differs because latitudes differ (41 vs 45).
    assert out["a"]["ghi_wm2"] != out["b"]["ghi_wm2"]


# ---------------------------------------------------------------------------
# Scenario YAML integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PILOT_YAML.exists(), reason="pilot yaml missing")
def test_scenario_loads_multi_site_yaml():
    sc = Scenario.from_yaml(PILOT_YAML)
    assert sc.is_multi_site
    assert sc.network is not None
    assert set(sc.network.site_ids) == {"foggia", "napoli", "milano"}
    # 4 links: 2 H2 pipelines + 2 electric lines
    assert len(sc.network.links) == 4
    assert len(sc.network.links_by_type(LinkType.H2_PIPELINE)) == 2
    assert len(sc.network.links_by_type(LinkType.ELECTRIC_LINE)) == 2
    # legacy single-site fields populated from first site (backward-compat)
    assert sc.grid_config  # non-empty


@pytest.mark.skipif(not LEGACY_YAML.exists(), reason="legacy yaml missing")
def test_scenario_legacy_single_site_still_works():
    sc = Scenario.from_yaml(LEGACY_YAML)
    assert not sc.is_multi_site
    assert sc.network is None
    # topology() synthesises a 1-site network on demand
    topo = sc.topology()
    assert not topo.is_multi_site
    assert len(topo.sites) == 1
