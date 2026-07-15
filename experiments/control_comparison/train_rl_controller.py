#!/usr/bin/env python3
"""
Train an RL model for control comparison experiments.

Outputs:
- output/rl_models/control_compare/final_model.zip
- output/rl_models/control_compare/best_model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.rl.advanced_environment import AdvancedH2GridEnv, AdvancedRewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO model for scenario comparison")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "advanced_stress.yaml"),
        help="Scenario YAML path",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episode-steps", type=int, default=144)
    parser.add_argument("--dt", type=int, default=600)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-window", type=int, default=24)
    parser.add_argument("--forecast-horizon", type=int, default=6)
    parser.add_argument("--forecast-step-mult", type=int, default=1)
    parser.add_argument(
        "--lookahead-hours",
        type=float,
        default=None,
        help=(
            "Lookahead horizon in hours. Overrides --forecast-horizon. "
            "Example: --lookahead-hours 12 with --dt 3600 sets forecast_horizon=12"
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--vec-env", default="subproc", choices=["dummy", "subproc"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--legacy-mlp", action="store_true", help="Use legacy feedforward PPO policy")
    parser.add_argument(
        "--load-model",
        default=None,
        help="Path to a saved model (.zip) to continue training from (for fine-tuning)",
    )
    parser.add_argument(
        "--no-h2-incentives",
        action="store_true",
        help="Use reduced (not zero) H\u2082 incentive weights during fine-tuning to avoid catastrophic forgetting",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "output" / "rl_models" / "control_compare"),
        help="Model output directory",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 and sb3-contrib are required. Install dependencies from requirements.txt"
        ) from exc

    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] Requested device='cuda' but CUDA is not available. Falling back to CPU.")
        resolved_device = "cpu"
    else:
        resolved_device = args.device

    if args.lookahead_hours is not None:
        resolved_forecast_horizon = max(1, round(args.lookahead_hours * 3600.0 / max(1, args.dt)))
    else:
        resolved_forecast_horizon = int(args.forecast_horizon)

    scenario = Scenario.from_yaml(args.config)
    h2_weight_scale = 0.25 if args.no_h2_incentives else 1.0
    env_kwargs = dict(
        grid_config=scenario.grid_config,
        weather_params=scenario.weather_params,
        cost_params=scenario.grid_config.get("energy_cost", {}),
        dt_seconds=float(args.dt),
        episode_length=int(args.episode_steps),
        history_window=int(args.history_window),
        forecast_horizon=int(resolved_forecast_horizon),
        forecast_step_multiplier=int(args.forecast_step_mult),
        reward_config=AdvancedRewardConfig(
            w_unmet_demand_penalty=-10.0,
            w_h2_depletion_rate_penalty=-5.5,
            w_soc_smoothing=1.4,
            w_renewable_fraction=1.6,
            w_operational_stability=1.0,
            w_demand_response_penalty=-1.2,
            # H₂ incentives: scaled down (not fully removed) in stage-2 fine-tuning
            w_h2_fc_usage=1.5 * h2_weight_scale,
            w_h2_el_usage=1.2 * h2_weight_scale,
            w_h2_accumulation=0.8 * h2_weight_scale,
            w_h2_waste_penalty=-1.5 * h2_weight_scale,
        ),
    )

    print("=" * 70)
    print("HyTwin 2.0 — RL training for control comparison")
    print("=" * 70)
    print(f"config        : {args.config}")
    print(f"timesteps     : {args.timesteps}")
    print(f"episode_steps : {args.episode_steps}")
    print(f"n_envs        : {args.n_envs}")
    print(f"vec_env       : {args.vec_env}")
    print(f"device        : {args.device} (resolved: {resolved_device})")
    print(f"history_window: {args.history_window}")
    if args.lookahead_hours is not None:
        print(f"lookahead_hrs : {args.lookahead_hours}h")
    print(f"forecast_hor  : {resolved_forecast_horizon}")
    from hytwin.rl.forecast_utils import N_FORECAST_FEATURES_PER_STEP as _N_FCAST_FEAT
    obs_dim = 19 + int(args.history_window) * 7 + int(resolved_forecast_horizon) * _N_FCAST_FEAT
    print(f"obs_dim       : {obs_dim} (19 base + hist + forecast)")
    print(f"n_steps       : {args.n_steps}")
    print(f"batch_size    : {args.batch_size}")
    print(f"n_epochs      : {args.n_epochs}")
    print(f"outdir        : {outdir}")
    print(f"load_model    : {args.load_model or 'None (new training)'}")
    print(f"h2_incentives : {'REDUCED (stage-2 fine-tune)' if args.no_h2_incentives else 'FULL (stage-1)'}")

    rollout_steps_per_update = int(args.n_steps) * int(args.n_envs)
    approx_policy_updates = max(1, int(args.timesteps) // max(1, rollout_steps_per_update))
    steps_per_env = int(args.timesteps) / max(1, int(args.n_envs))
    episode_coverage = steps_per_env / max(1, int(args.episode_steps))
    print(f"rollout/update: {rollout_steps_per_update} env-steps")
    print(f"approx_updates: {approx_policy_updates}")
    print(f"steps/env      : {steps_per_env:.1f} ({episode_coverage:.2f} episodes/env)")
    if approx_policy_updates < 20:
        print("[WARN] Very few policy updates: training likely insufficient for stable H2 behavior.")
        print("       Increase --timesteps (recommended >= 200k for dt=3600, long-horizon setup).")
    if episode_coverage < 1.0:
        print("[WARN] Each env sees less than one full episode on average.")
        print("       Increase --timesteps or reduce --episode-steps.")

    def make_env():
        return Monitor(AdvancedH2GridEnv(**env_kwargs))

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else None
    train_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
    )
    eval_env = Monitor(AdvancedH2GridEnv(**env_kwargs))

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(outdir),
            eval_freq=max(args.timesteps // 20, 5000),
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=max(args.timesteps // 10, 10_000),
            save_path=str(outdir / "checkpoints"),
            name_prefix="ppo_compare",
            verbose=0,
        ),
    ]

    PolicyClass = PPO if args.legacy_mlp else RecurrentPPO
    policy_name = "MlpPolicy" if args.legacy_mlp else "MlpLstmPolicy"
    ppo_kwargs = dict(
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        device=resolved_device,
        tensorboard_log=str(ROOT / "output" / "tb_logs"),
    )

    if args.load_model is not None:
        load_path = args.load_model
        if not load_path.endswith(".zip"):
            load_path = load_path + ".zip"
        print(f"Loading model from: {load_path}")
        model = PolicyClass.load(
            load_path,
            env=train_env,
            **{k: v for k, v in ppo_kwargs.items() if k != "seed"},
        )
    else:
        model = PolicyClass(policy_name, train_env, **ppo_kwargs)

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    final_path = outdir / "final_model"
    model.save(str(final_path))

    print("\nTraining complete")
    print(f"Final model : {final_path}.zip")
    print(f"Best model  : {outdir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
