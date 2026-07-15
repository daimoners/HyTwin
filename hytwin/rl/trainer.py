"""
RL Trainer
==========
High-level training pipeline for the H2 grid RL agent using
Stable-Baselines3.  Supports:
  - Multiple algorithms (PPO, SAC, TD3, DDPG)
  - Callbacks: checkpoint, evaluation, progress logging
  - Training, evaluation, and inference phases
  - Model save / load
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np

from .environment import H2GridEnv
from .rewards import RewardConfig

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO, SAC, TD3, DDPG
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, CallbackList,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed; RL trainer unavailable.")

try:
    from sb3_contrib import RecurrentPPO
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False

ALGORITHMS = {
    "ppo": "PPO",
    "recurrent_ppo": "RecurrentPPO",
    "sac": "SAC",
    "td3": "TD3",
    "ddpg": "DDPG",
}


class RLTrainer:
    """
    Trains a reinforcement learning agent on the H2 grid environment.

    Parameters
    ----------
    grid_config : dict         — grid topology (passed to H2GridEnv)
    algorithm : str            — 'ppo' | 'sac' | 'td3' | 'ddpg'
    policy : str               — SB3 policy string, default 'MlpPolicy'
    total_timesteps : int      — total training steps
    eval_freq : int            — evaluation frequency (steps)
    n_eval_episodes : int      — episodes per evaluation
    save_dir : str | Path      — directory to save checkpoints & final model
    weather_params : dict      — passed to WeatherModel
    dt_seconds : float         — simulation step size [s]
    episode_length : int       — steps per episode
    reward_config : RewardConfig
    algo_kwargs : dict         — extra kwargs for SB3 algorithm constructor
    """

    def __init__(
        self,
        grid_config: Dict[str, Any],
        algorithm: str = "ppo",
        policy: str = "MlpPolicy",
        env_cls: Type[H2GridEnv] = H2GridEnv,
        env_kwargs: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 200_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_dir: str | Path = "./models",
        weather_params: Optional[Dict[str, Any]] = None,
        dt_seconds: float = 60.0,
        episode_length: int = 1440,
        reward_config: Optional[RewardConfig] = None,
        algo_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RLTrainer.")

        self._grid_config = grid_config
        self._algo_name = algorithm.lower()
        self._policy = policy
        self._env_cls = env_cls
        self._env_kwargs = env_kwargs or {}
        self._total_steps = total_timesteps
        self._eval_freq = eval_freq
        self._n_eval = n_eval_episodes
        self._save_dir = Path(save_dir)
        self._weather_params = weather_params or {}
        self._dt = dt_seconds
        self._ep_len = episode_length
        self._reward_cfg = reward_config or RewardConfig()
        self._algo_kwargs = algo_kwargs or {}

        self._model = None
        self._save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    def _make_env(self, seed: int = 0) -> H2GridEnv:
        env = self._env_cls(
            grid_config=self._grid_config,
            weather_params=self._weather_params,
            dt_seconds=self._dt,
            episode_length=self._ep_len,
            reward_config=self._reward_cfg,
            **self._env_kwargs,
        )
        return Monitor(env)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, verbose: int = 1) -> None:
        """Run the full training loop."""
        logger.info("Starting RL training — algorithm=%s, steps=%d",
                    self._algo_name, self._total_steps)

        train_env = self._make_env(seed=42)
        eval_env = self._make_env(seed=100)

        algo_cls = self._get_algo()
        policy_name = self._resolve_policy_name()

        self._model = algo_cls(
            policy_name,
            train_env,
            verbose=verbose,
            **self._algo_kwargs,
        )

        callbacks = []
        # Checkpoint every N steps
        ckpt_cb = CheckpointCallback(
            save_freq=max(self._eval_freq, 10_000),
            save_path=str(self._save_dir / "checkpoints"),
            name_prefix=f"h2grid_{self._algo_name}",
        )
        callbacks.append(ckpt_cb)

        # Evaluation callback
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(self._save_dir / "best"),
            log_path=str(self._save_dir / "logs"),
            eval_freq=self._eval_freq,
            n_eval_episodes=self._n_eval,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)

        self._model.learn(
            total_timesteps=self._total_steps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )

        final_path = self._save_dir / f"h2grid_{self._algo_name}_final"
        self._model.save(str(final_path))
        logger.info("Training complete. Model saved to %s", final_path)

    # ------------------------------------------------------------------
    # Evaluation / inference
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained model."""
        if self._model is None:
            raise RuntimeError("No model — call train() or load() first.")
        eval_env = self._make_env(seed=999)
        mean_r, std_r = evaluate_policy(
            self._model, eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )
        logger.info("Evaluation: mean_reward=%.2f ± %.2f", mean_r, std_r)
        return {"mean_reward": mean_r, "std_reward": std_r}

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Run inference (predict action for observation)."""
        if self._model is None:
            raise RuntimeError("No model — call train() or load() first.")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> Path:
        if self._model is None:
            raise RuntimeError("No model to save.")
        p = Path(path) if path else self._save_dir / f"h2grid_{self._algo_name}"
        self._model.save(str(p))
        return p

    def load(self, path: str) -> None:
        algo_cls = self._get_algo()
        env = self._make_env()
        self._model = algo_cls.load(path, env=env)
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_algo(self):
        mapping = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}
        if _SB3_CONTRIB_AVAILABLE:
            mapping["recurrent_ppo"] = RecurrentPPO
        algo_cls = mapping.get(self._algo_name)
        if algo_cls is None:
            raise ValueError(
                f"Unknown algorithm '{self._algo_name}'. Choose from: {list(mapping.keys())}"
            )
        return algo_cls

    def _resolve_policy_name(self) -> str:
        if self._algo_name == "recurrent_ppo":
            return self._policy if self._policy != "MlpPolicy" else "MlpLstmPolicy"
        return self._policy
