"""
demo_rl_training.py
===================
End-to-end RL training demo for HyTwin 2.0.

Steps
-----
  1. Create H2GridEnv from default_grid.yaml
  2. Validate the environment with gymnasium.utils.env_checker
  3. Run a random-policy baseline (returns baseline reward)
  4. Train a PPO agent for a short run (configurable via --timesteps)
  5. Evaluate the trained agent vs the random baseline
  6. Plot training reward curve + evaluation episode

Usage
-----
    python demos/demo_rl_training.py [--timesteps 20000] [--eval-eps 3] [--plot] [--save <path>]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HyTwin 2.0 — RL training demo")
    p.add_argument("--timesteps", type=int, default=20_000,
                   help="Total training timesteps (default: 20,000)")
    p.add_argument("--eval-eps", type=int, default=3,
                   help="Evaluation episodes after training (default: 3)")
    p.add_argument("--algorithm", default="PPO",
                   choices=["PPO", "SAC", "TD3", "DDPG"],
                   help="SB3 algorithm (default: PPO)")
    p.add_argument("--config", default=str(ROOT / "config" / "default_grid.yaml"))
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save", default="")
    p.add_argument("--model-save", default="",
                   help="Path to save trained model (optional)")
    return p.parse_args()


# ---------------------------------------------------------------------------
def _build_env(config_path: str, seed: int = 42):
    """Import env and build from YAML."""
    import yaml
    from hytwin.rl.environment import H2GridEnv

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    grid_cfg      = cfg.get("grid", {})
    weather_cfg   = cfg.get("weather", {})
    reward_cfg    = {}                      # use defaults

    env = H2GridEnv(
        grid_config=grid_cfg,
        weather_params=weather_cfg,
        reward_config=reward_cfg,
        dt_seconds=cfg.get("dt_seconds", 600),
        episode_length=cfg.get("episode_steps", 144),
    )
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
def _random_baseline(env, n_episodes: int = 3) -> float:
    """Run n_episodes with random actions → return mean total reward."""
    rewards_total = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards_total.append(ep_reward)
    return float(np.mean(rewards_total))


# ---------------------------------------------------------------------------
def _train(env, algorithm: str, timesteps: int, out_dir: Path):
    """Train with Stable-Baselines3, return (model, tensorboard_log_dir)."""
    try:
        from stable_baselines3 import PPO, SAC, TD3, DDPG
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback
        )
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        print(f"  [ERROR] stable-baselines3 not installed: {exc}")
        print("          Run: pip install stable-baselines3[extra]")
        sys.exit(1)

    algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}
    AlgoCls = algo_map[algorithm]

    log_dir = out_dir / "tb_logs"
    ckpt_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Wrap env for SB3
    train_env = Monitor(env)
    vec_env   = DummyVecEnv([lambda: train_env])

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1000, timesteps // 10),
        save_path=str(ckpt_dir),
        name_prefix=algorithm.lower(),
        verbose=0,
    )

    model = AlgoCls(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=42,
    )

    print(f"\n  Training {algorithm} for {timesteps:,} timesteps …")
    t0 = time.perf_counter()
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_cb,
        progress_bar=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Training completed in {elapsed:.1f} s")
    return model


# ---------------------------------------------------------------------------
def _evaluate(model, env, n_episodes: int) -> tuple[float, list]:
    """Evaluate trained model → (mean_reward, list of episode reward curves)."""
    ep_rewards_list = []
    total_rewards   = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        ep_curve = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            ep_curve.append(reward)
            done = terminated or truncated
        ep_rewards_list.append(ep_curve)
        total_rewards.append(ep_rew)
        print(f"    Episode {ep+1}: total reward = {ep_rew:.2f}")

    return float(np.mean(total_rewards)), ep_rewards_list


# ---------------------------------------------------------------------------
def _plot_results(random_mean: float, trained_mean: float,
                  ep_curves: list, save_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("HyTwin 2.0 — RL Training Demo", fontsize=13, fontweight="bold")

    # ── Per-step reward for each eval episode ────────────────────────────────
    ax = axes[0]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, curve in enumerate(ep_curves):
        ax.plot(curve, alpha=0.8, color=colors[i % len(colors)], label=f"Ep {i+1}")
    ax.set_title("Trained Agent — Per-Step Reward")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Random vs Trained comparison bar ────────────────────────────────────
    ax = axes[1]
    labels  = ["Random policy", "Trained agent"]
    values  = [random_mean, trained_mean]
    colors2 = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, values, color=colors2, width=0.4, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=10)
    ax.set_title("Mean Episode Reward Comparison")
    ax.set_ylabel("Total Episode Reward")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    print("=" * 64)
    print("  HyTwin 2.0 — RL Training Demo")
    print("=" * 64)
    print(f"  Algorithm  : {args.algorithm}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  Eval eps   : {args.eval_eps}")
    print(f"  Config     : {args.config}")
    print()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    out_dir = ROOT / "output" / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Validate environment ──────────────────────────────────────────────
    print("  Validating Gymnasium environment …")
    try:
        from gymnasium.utils.env_checker import check_env
        env_check = _build_env(str(config_path), seed=0)
        check_env(env_check, warn=True)
        print("  check_env: PASSED\n")
        env_check.close()
    except Exception as exc:
        print(f"  check_env: WARNING — {exc}\n")

    # ── 2. Random baseline ───────────────────────────────────────────────────
    print("  Running random-policy baseline …")
    env_rand = _build_env(str(config_path), seed=1)
    random_mean = _random_baseline(env_rand, n_episodes=2)
    env_rand.close()
    print(f"  Random baseline mean reward = {random_mean:.2f}\n")

    # ── 3. Train ─────────────────────────────────────────────────────────────
    env_train = _build_env(str(config_path), seed=42)
    model = _train(env_train, args.algorithm, args.timesteps, out_dir)
    env_train.close()

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    print(f"\n  Evaluating trained agent ({args.eval_eps} episodes) …")
    env_eval = _build_env(str(config_path), seed=99)
    trained_mean, ep_curves = _evaluate(model, env_eval, args.eval_eps)
    env_eval.close()
    print(f"\n  Trained agent mean reward = {trained_mean:.2f}")
    improvement = (trained_mean - random_mean) / (abs(random_mean) + 1e-9) * 100
    print(f"  Improvement vs random     = {improvement:+.1f}%")

    # ── 5. Save model ────────────────────────────────────────────────────────
    if args.model_save:
        model.save(args.model_save)
        print(f"  Model saved → {args.model_save}")
    else:
        default_path = str(out_dir / f"{args.algorithm.lower()}_h2grid")
        model.save(default_path)
        print(f"  Model saved → {default_path}")

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    if args.plot or args.save:
        save_path = args.save if args.save else str(out_dir / "demo_rl_training.png")
        _plot_results(random_mean, trained_mean, ep_curves, save_path)
    else:
        print("\n  (pass --plot or --save <path> to generate charts)")

    print("=" * 64)


if __name__ == "__main__":
    main()
