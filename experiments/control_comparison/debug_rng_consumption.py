#!/usr/bin/env python3
"""
Debug script to trace RNG consumption in ClassicalController
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.models.energy_cost import EnergyCostModel
from hytwin.control.classical_controller import ClassicalController
from hytwin.digital_twin.grid_twin import GridState


def print_rng_state_hash(state_tuple, label: str = ""):
    """Print a hash of RNG state for comparison."""
    state_array = state_tuple[1]
    state_hash = hash(tuple(state_array[:10]))  # Hash first 10 elements
    print(f"  [{label}] RNG state hash: {state_hash}")


config_path = ROOT / "config" / "advanced_stress.yaml"
scenario = Scenario.from_yaml(str(config_path))

print("=" * 80)
print("RNG CONSUMPTION TRACE")
print("=" * 80)

# Step 1: Create cost model
print("\n1. Creating EnergyCostModel...")
np.random.seed(42)
rng_s1 = np.random.get_state()
print_rng_state_hash(rng_s1, "after seed(42)")

cost_model = EnergyCostModel(scenario.grid_config.get("energy_cost", {}))
rng_s2 = np.random.get_state()
print_rng_state_hash(rng_s2, "after EnergyCostModel()")

if np.array_equal(rng_s1[1], rng_s2[1]):
    print("  ✓ No RNG consumed by EnergyCostModel.__init__()")
else:
    print("  ✗ RNG WAS consumed by EnergyCostModel.__init__()")

# Step 2: Create controller
print("\n2. Creating ClassicalController...")
np.random.seed(42)
rng_s3 = np.random.get_state()
print_rng_state_hash(rng_s3, "after seed(42)")

controller = ClassicalController(scenario.grid_config, cost_model)
rng_s4 = np.random.get_state()
print_rng_state_hash(rng_s4, "after ClassicalController()")

if np.array_equal(rng_s3[1], rng_s4[1]):
    print("  ✓ No RNG consumed by ClassicalController.__init__()")
else:
    print("  ✗ RNG WAS consumed by ClassicalController.__init__()")

# Step 3: Create GridState
print("\n3. Creating GridState...")
gs = GridState(
    timestamp=datetime(2024, 1, 1, 12, 0, 0),
    wind_power_kw=100.0,
    pv_power_kw=50.0,
    load_kw=200.0,
    fuel_cell_power_kw=50.0,
    electrolyzer_power_kw=0.0,
    h2_soc=0.5,
    h2_storage_kg=1000.0,
    grid_available=True,
    grid_exchange_kw=0.0,
    grid_connection_kw=0.0,
    renewable_fraction=0.75,
    grid_self_sufficiency=0.75,
)

print("  ✓ GridState created (pure data, no RNG)")

# Step 4: Call controller.compute_actions()
print("\n4. Calling controller.compute_actions()...")

np.random.seed(42)
cost_model = EnergyCostModel(scenario.grid_config.get("energy_cost", {}))
controller = ClassicalController(scenario.grid_config, cost_model)

rng_s5 = np.random.get_state()
print_rng_state_hash(rng_s5, "BEFORE compute_actions()")

actions = controller.compute_actions(gs, timestamp=datetime(2024, 1, 1, 12, 0, 0))

rng_s6 = np.random.get_state()
print_rng_state_hash(rng_s6, "AFTER compute_actions()")

if np.array_equal(rng_s5[1], rng_s6[1]):
    print("  ✓ No RNG consumed by compute_actions()")
else:
    print("  ✗ RNG WAS consumed by compute_actions()")
    print(f"\n  First 10 state values BEFORE: {rng_s5[1][:10]}")
    print(f"  First 10 state values AFTER:  {rng_s6[1][:10]}")

# Step 5: Call again to see if second call consumes RNG
print("\n5. Calling compute_actions() AGAIN...")

rng_s7 = np.random.get_state()
print_rng_state_hash(rng_s7, "BEFORE second compute_actions()")

actions2 = controller.compute_actions(gs, timestamp=datetime(2024, 1, 1, 12, 0, 0))

rng_s8 = np.random.get_state()
print_rng_state_hash(rng_s8, "AFTER second compute_actions()")

if np.array_equal(rng_s7[1], rng_s8[1]):
    print("  ✓ No RNG consumed by second compute_actions()")
else:
    print("  ✗ RNG WAS consumed by second compute_actions()")

print(f"\nActions from call 1: {list(actions.keys())[:3]}...")
print(f"Actions from call 2: {list(actions2.keys())[:3]}...")

print("\n" + "=" * 80)
