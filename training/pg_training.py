#!/usr/bin/env python3
"""
Policy-Gradient Training Script (PPO / A2C / REINFORCE)
for Nairobi CBD Protest Navigation
"""

import os, time, json, numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from environment.custom_env import NairobiCBDProtestEnv

gym.register(
    id="NairobiProtestEnv-v0",
    entry_point="environment.custom_env:NairobiCBDProtestEnv",
)

# 1.  Choose algorithm (PPO, A2C, REINFORCE)

ALGO = "A2C"          # <--- change to "A2C" or "REINFORCE" (REINFORCE = A2C w/o baseline)
POLICY_TYPE = "MlpPolicy"

# 2.  Hyper-parameter grids
HP_GRID: Dict[str, Dict] = {
    "ppo_default": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
    "a2c_default": {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
    "reinforce": {            # A2C without baseline (REINFORCE)
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "policy_kwargs": dict(net_arch=[256, 256]),
        "use_sde": False,     # turn off baseline
    },
}

# 3.  Training manager (minimal clone of DQNTrainingManager)

class PGTrainingManager:
    def __init__(self, log_dir: str, model_dir: str):
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def create_env(self, n_envs: int = 8):
        def _make():
            env = make_vec_env(
                "NairobiProtestEnv-v0",
                n_envs=n_envs,
                seed=42,
                env_kwargs={"render_mode": None},
            )
            return VecNormalize(env, norm_obs=True, norm_reward=True)

        return _make()

    def train(self, name: str, hp: Dict, total_timesteps: int = 150_000):
        print(f"\n{'='*60}")
        print(f"Training {ALGO} with config: {name}")
        print(f"{'='*60}")

        train_env = self.create_env()
        eval_env  = self.create_env()

        # Map algorithm choice
        if ALGO == "PPO":
            Model = PPO
        elif ALGO == "A2C":
            Model = A2C
        else:                       # REINFORCE ≈ A2C w/o baseline
            Model = A2C

        model = Model(
            POLICY_TYPE,
            train_env,
            verbose=1,
            tensorboard_log=str(self.log_dir / "tensorboard"),
            **hp,
        )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir / f"{name}_best"),
            log_path=str(self.log_dir),
            eval_freq=total_timesteps // 10,
            n_eval_episodes=10,
            deterministic=True,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=total_timesteps // 5,
            save_path=str(self.model_dir),
            name_prefix=f"{name}_ckpt",
        )

        start = time.time()
        model.learn(total_timesteps, callback=CallbackList([eval_cb, ckpt_cb]))
        elapsed = time.time() - start

        final_path = self.model_dir / f"{name}_final"
        model.save(final_path)

        print(f"{ALGO} {name} finished in {elapsed:.1f}s.")
        return model

# ------------------------------------------------------------------
# 4.  CLI / quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(HP_GRID.keys()), default="ppo_default")
    parser.add_argument("--steps", type=int, default=150_000)
    args = parser.parse_args()

    mgr = PGTrainingManager(
        log_dir=f"logs/{ALGO.lower()}",
        model_dir=f"models/{ALGO.lower()}",
    )
    mgr.train(args.config, HP_GRID[args.config], args.steps)
