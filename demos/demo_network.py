"""
demo_network.py
================
End-to-end demo of the multi-node network layer: builds a geolocated
multi-site H2 network from YAML, runs it forward in time, and compares
the "no control" baseline against the rule-based classical controller
(and, if a compatible trained model is present, the RL controller too)
under identical weather/price conditions.

This is the network/"grid" counterpart to the single-site demos
(demo_simulation.py, demo_sensors.py, demo_digital_twin.py,
demo_rl_training.py) — see docs/09_usage_guide.md for the full picture,
including how to run the same comparison from the dashboard.

Usage
-----
    python demos/demo_network.py [--config config/italy_network_pilot.yaml]
                                  [--steps 144] [--seed 42] [--rl-model PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.network.compare import compare_controllers


def main() -> None:
    parser = argparse.ArgumentParser(description="HyTwin — multi-node network demo")
    parser.add_argument("--config", default=str(ROOT / "config" / "italy_network_pilot.yaml"),
                         help="Path to a network-layer scenario YAML (default: italy_network_pilot.yaml)")
    parser.add_argument("--steps", type=int, default=144,
                         help="Simulation steps to run per strategy (default: 144 = 24h at dt=600s)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed — identical across strategies for a fair comparison")
    parser.add_argument("--rl-model", default=None,
                         help="Optional path (no .zip) to a trained network RL model to include")
    args = parser.parse_args()

    topo = Scenario.from_yaml(args.config).topology()
    print(f"Network: {len(topo.sites)} site(s) — {', '.join(topo.site_ids)}")
    print(f"Links:   {len(topo.links)}")
    print(f"Running {args.steps} steps per strategy (seed={args.seed})...\n")

    strategies = {"none": "none", "classical": "classical"}
    if args.rl_model:
        from hytwin.control.network_rl_controller import NetworkRLController
        strategies["rl"] = NetworkRLController.factory(args.rl_model)

    results = compare_controllers(topo, steps=args.steps, seed=args.seed, strategies=strategies)

    header = f"{'strategy':<12}" + "".join(f"{k:>18}" for k in next(iter(results.values())))
    print(header)
    print("-" * len(header))
    for name, kpis in results.items():
        row = f"{name:<12}" + "".join(f"{v:>18.3f}" for v in kpis.values())
        print(row)


if __name__ == "__main__":
    main()
