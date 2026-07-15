"""
Network RL Trainer
==================
Thin helper to train a PPO agent on :class:`NetworkH2GridEnv` and save it, so a
:class:`~hytwin.control.network_rl_controller.NetworkRLController` can then be
compared against the traditional controller on identical conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .network_environment import NetworkH2GridEnv, NetworkRewardConfig

logger = logging.getLogger(__name__)


def make_env(topology, dt_seconds: float = 600.0, episode_steps: int = 144,
             reward_config: Optional[NetworkRewardConfig] = None) -> NetworkH2GridEnv:
    return NetworkH2GridEnv(
        topology, dt_seconds=dt_seconds, episode_steps=episode_steps,
        reward_config=reward_config,
    )


def train_network_agent(
    topology,
    timesteps: int = 100_000,
    dt_seconds: float = 600.0,
    episode_steps: int = 144,
    save_path: Optional[str] = None,
    seed: int = 0,
    n_steps: int = 288,
    reward_config: Optional[NetworkRewardConfig] = None,
    verbose: int = 0,
    callback=None,
):
    """
    Train a PPO agent on the network env.  Returns the trained model.

    Requires stable-baselines3.  ``save_path`` (without extension) saves a
    ``.zip`` loadable by :class:`NetworkRLController`.

    Parameters
    ----------
    callback : stable_baselines3.common.callbacks.BaseCallback, optional
        Passed through to ``model.learn()`` — used by the dashboard's
        ``TrainingWorker`` to report live progress and to abort early
        (a callback whose ``_on_step`` returns ``False`` stops training).
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:  # pragma: no cover
        raise ImportError("stable-baselines3 is required to train the network agent") from e

    env = Monitor(make_env(topology, dt_seconds, episode_steps, reward_config))
    model = PPO(
        "MlpPolicy", env,
        n_steps=n_steps, batch_size=min(64, n_steps),
        gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
        learning_rate=3e-4, seed=seed, verbose=verbose,
    )
    model.learn(total_timesteps=int(timesteps), callback=callback)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info("Saved network RL model to %s.zip", save_path)
    return model
