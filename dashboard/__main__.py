"""Entry point for `python -m dashboard`."""
import argparse
from dashboard import run

parser = argparse.ArgumentParser(description="HyTwin 2.0 Real-Time Dashboard")
parser.add_argument("--config",  default=None,    help="Path to YAML config (default: config/advanced_grid.yaml)")
parser.add_argument("--port",    type=int, default=8050, help="HTTP port (default: 8050)")
parser.add_argument("--speed",   type=float, default=0.0, help="Speed factor: 0=max, 1=real-time, N=N× (default: 0)")
parser.add_argument("--dt",      type=float, default=600.0, help="Simulation step in seconds (default: 600)")
parser.add_argument("--seed",    type=int, default=42, help="Random seed (default: 42)")
args = parser.parse_args()

run(config_path=args.config, dt_seconds=args.dt, speed_factor=args.speed, port=args.port, seed=args.seed)
