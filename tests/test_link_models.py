"""
test_link_models.py
===================
Unit tests for the F2 inter-site link physics models:
ElectricLineModel and H2PipelineModel.
Run: pytest tests/test_link_models.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.models.electric_line import ElectricLineModel
from hytwin.models.h2_pipeline import H2PipelineModel

TS = datetime(2024, 6, 15, 12, 0)


# ===========================================================================
# ElectricLineModel
# ===========================================================================

def test_electric_line_loss_fraction_from_length():
    line = ElectricLineModel("l1", {"max_power_mw": 50, "length_km": 1000, "loss_per_1000km": 0.06})
    assert line.loss_fraction == pytest.approx(0.06)
    line2 = ElectricLineModel("l2", {"max_power_mw": 50, "length_km": 500, "loss_per_1000km": 0.06})
    assert line2.loss_fraction == pytest.approx(0.03)


def test_electric_line_delivered_less_than_flow():
    line = ElectricLineModel("l1", {"max_power_mw": 50, "length_km": 500, "loss_per_1000km": 0.06})
    st = line.step(600, {"power_request_kw": 10_000.0, "timestamp": TS})
    assert st["flow_kw"] == pytest.approx(10_000.0)
    # 3% loss over 500 km
    assert st["delivered_kw"] == pytest.approx(9_700.0, rel=1e-6)
    assert st["loss_kw"] == pytest.approx(300.0, rel=1e-6)


def test_electric_line_capacity_clamp():
    line = ElectricLineModel("l1", {"max_power_mw": 40})  # 40 MW = 40000 kW
    st = line.step(600, {"power_request_kw": 999_999.0, "timestamp": TS})
    assert st["flow_kw"] == pytest.approx(40_000.0)
    assert st["utilization"] == pytest.approx(1.0, rel=1e-6)


def test_electric_line_reverse_flow_when_bidirectional():
    line = ElectricLineModel("l1", {"max_power_mw": 50, "length_km": 0.0, "bidirectional": True})
    st = line.step(600, {"power_request_kw": -5_000.0, "timestamp": TS})
    assert st["flow_kw"] == pytest.approx(-5_000.0)
    assert st["delivered_kw"] < 0  # sign preserved


def test_electric_line_reverse_blocked_when_unidirectional():
    line = ElectricLineModel("l1", {"max_power_mw": 50, "bidirectional": False})
    st = line.step(600, {"power_request_kw": -5_000.0, "timestamp": TS})
    assert st["flow_kw"] == pytest.approx(0.0)


def test_electric_line_ramp_limit():
    # ramp 10 kW/s over 600 s => max delta 6000 kW per step
    line = ElectricLineModel("l1", {"max_power_mw": 50, "ramp_rate_kw_s": 10.0})
    st = line.step(600, {"power_request_kw": 40_000.0, "timestamp": TS})
    assert st["flow_kw"] == pytest.approx(6_000.0)


def test_electric_line_reset():
    line = ElectricLineModel("l1", {"max_power_mw": 50})
    line.step(600, {"power_request_kw": 10_000.0, "timestamp": TS})
    line.reset()
    assert line.state is None
    assert line._flow_kw == 0.0


# ===========================================================================
# H2PipelineModel
# ===========================================================================

def test_h2_pipeline_capacity_clamp():
    pipe = H2PipelineModel("p1", {"max_flow_kg_h": 300.0, "length_km": 0.0})
    st = pipe.step(600, {"flow_request_kg_h": 9999.0, "timestamp": TS})
    assert st["flow_kg_h"] == pytest.approx(300.0)


def test_h2_pipeline_compression_energy():
    # 300 kg/h at 2 kWh/kg = 600 kW compressor load
    pipe = H2PipelineModel("p1", {"max_flow_kg_h": 300.0, "length_km": 0.0,
                                   "compressor_spec_kwh_per_kg": 2.0})
    st = pipe.step(600, {"flow_request_kg_h": 300.0, "timestamp": TS})
    assert st["compressor_power_kw"] == pytest.approx(600.0, rel=1e-6)


def test_h2_pipeline_reverse_blocked():
    pipe = H2PipelineModel("p1", {"max_flow_kg_h": 300.0})
    st = pipe.step(600, {"flow_request_kg_h": -100.0, "timestamp": TS})
    assert st["flow_kg_h"] == pytest.approx(0.0)


def test_h2_pipeline_transport_delay():
    # length 54 km, velocity 15 m/s -> 3600 s delay -> 6 steps of 600 s
    pipe = H2PipelineModel("p1", {"max_flow_kg_h": 300.0, "length_km": 54.0,
                                   "transport_velocity_ms": 15.0, "line_pack_capacity_kg": 10_000.0})
    delay_steps = int(round(pipe.delay_s / 600.0))
    assert delay_steps == 6
    # Inject for one step, then zero — delivery should appear only after the delay.
    st0 = pipe.step(600, {"flow_request_kg_h": 300.0, "timestamp": TS})
    assert st0["delivered_kg_h"] == pytest.approx(0.0)
    delivered_seen = []
    for _ in range(delay_steps + 1):
        st = pipe.step(600, {"flow_request_kg_h": 0.0, "timestamp": TS})
        delivered_seen.append(st["delivered_kg_h"])
    # Some delivery must have occurred by now (the initial slug arrives).
    assert max(delivered_seen) > 0.0


def test_h2_pipeline_mass_conservation():
    # Everything injected must eventually be delivered (lossless transport).
    pipe = H2PipelineModel("p1", {"max_flow_kg_h": 300.0, "length_km": 30.0,
                                   "transport_velocity_ms": 15.0, "line_pack_capacity_kg": 10_000.0})
    injected = 0.0
    for _ in range(5):
        st = pipe.step(600, {"flow_request_kg_h": 200.0, "timestamp": TS})
        injected += st["flow_kg_h"] * (600 / 3600.0)
    # Flush the pipeline
    for _ in range(20):
        pipe.step(600, {"flow_request_kg_h": 0.0, "timestamp": TS})
    delivered = pipe.state["delivered_total_kg"]
    assert delivered == pytest.approx(injected, rel=1e-6)
    assert pipe.linepack_kg == pytest.approx(0.0, abs=1e-9)


def test_h2_pipeline_linepack_overflow_throttle():
    # Tiny line-pack forces injection throttling.
    pipe = H2PipelineModel("p1", {"max_flow_kg_h": 300.0, "length_km": 100.0,
                                   "transport_velocity_ms": 5.0, "line_pack_capacity_kg": 20.0})
    seen = []
    for _ in range(10):
        st = pipe.step(600, {"flow_request_kg_h": 300.0, "timestamp": TS})
        seen.append(st["flow_kg_h"])
        assert pipe.linepack_kg <= 20.0 + 1e-9
    # At least one step must have been throttled below the full request.
    assert min(seen) < 300.0


def test_h2_pipeline_reset():
    pipe = H2PipelineModel("p1", {"max_flow_kg_h": 300.0, "length_km": 50.0})
    pipe.step(600, {"flow_request_kg_h": 200.0, "timestamp": TS})
    pipe.reset()
    assert pipe.state is None
    assert pipe.linepack_kg == 0.0
