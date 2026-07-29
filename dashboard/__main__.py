"""Entry point for `python -m dashboard` — HyTwin Network Control Room."""
import argparse
from dashboard.network_app import run_network_dashboard, DEFAULT_CONFIG, DEFAULT_RL_MODEL

parser = argparse.ArgumentParser(description="HyTwin — Italian H2 Network Control Room")
parser.add_argument("--config",       default=DEFAULT_CONFIG,   help="Path to network scenario YAML")
parser.add_argument("--port",         type=int,   default=8060, help="HTTP port (default: 8060)")
parser.add_argument("--speed-factor", type=float, default=0.0,  help="Speed: 0=max, 1=real-time, N=N×")
parser.add_argument("--dt",           type=float, default=600.0,help="Simulation step [s] (default: 600)")
parser.add_argument("--seed",         type=int,   default=42,   help="Random seed (default: 42)")
parser.add_argument("--rl-model",     default=DEFAULT_RL_MODEL, help="Path to trained RL model (no .zip)")
args = parser.parse_args()

run_network_dashboard(
    config_path=args.config,
    dt_seconds=args.dt,
    speed_factor=args.speed_factor,
    port=args.port,
    seed=args.seed,
    rl_model_path=args.rl_model,
)
