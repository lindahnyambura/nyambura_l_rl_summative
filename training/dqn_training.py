#!/usr/bin/env python3
"""
DQN Training Script for Nairobi CBD Protest Navigation
Value-Based Reinforcement Learning using Stable Baselines3

This script implements Deep Q-Network (DQN) training for the protest navigation environment.
Includes hyperparameter tuning, model evaluation, and performance monitoring.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import torch

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold, 
    CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

# Add environment to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))
from environment.custom_env import NairobiCBDProtestEnv


class DQNTrainingManager:
    """
    Manages DQN training with hyperparameter tuning and performance monitoring
    """
    
    def __init__(self, log_dir: str = "logs/dqn", model_dir: str = "models/dqn"):
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.training_history = []
        self.best_mean_reward = -np.inf
        
        # Hyperparameter configurations to test
        self.hyperparameter_configs = {
            'recovery_config': {
                'learning_rate': 1e-4,
                'buffer_size': 100000,
                'learning_starts': 5000,
                'batch_size': 128,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 500,
                'exploration_fraction': 0.5,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'max_grad_norm': 10,
                'policy_kwargs': dict(net_arch=[256, 256, 128]),
            },
            'robust_exploration': {
                'learning_rate': 3e-4,
                'buffer_size': 200000, # Increased buffer size for better exploration
                'learning_starts': 10000, # Increased learning starts, more random exploration
                'batch_size': 128,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 1000,
                'exploration_fraction': 0.8, # longer exploration phase
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.1, # higher final epsilon for more exploration
                'max_grad_norm': 10,
                'policy_kwargs': {
                    'net_arch': [256, 256, 128], # Deeper network for better representation
                    'activation_fn': torch.nn.ReLU,
                    'optimizer_class': torch.optim.AdamW,
                    'optimizer_kwargs': {'weight_decay': 1e-5},
                },
            },
            'prioritized_replay': {
                'learning_rate': 3e-4,
                'buffer_size': 200000,
                'learning_starts': 10000,
                'batch_size': 128,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 1000,
                'exploration_fraction': 0.8,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.1,
                'max_grad_norm': 10,
                'policy_kwargs': {
                    'net_arch': [256, 256, 128],
                    'activation_fn': torch.nn.ReLU,
                    'optimizer_class': torch.optim.AdamW,
                    'optimizer_kwargs': {'weight_decay': 1e-5},
                },
                'replay_buffer_class': 'stable_baselines3.dqn.prioritized_replay_buffer.PrioritizedReplayBuffer',
                'replay_buffer_kwargs': {
                    'alpha': 0.6,  # Prioritization exponent
                    'beta': 0.4,   # Importance sampling exponent
                }
            }
        }
    
    def create_monitored_env(self, env_id: str = "NairobiCBD-v0", 
                           n_envs: int = 1, normalize: bool = True) -> VecNormalize:
        """Create monitored vectorized environment for training"""
        
        def make_env():
            env = NairobiCBDProtestEnv(render_mode=None, grid_size=(100, 100))
            env = Monitor(env, str(self.log_dir))
            return env
        
        # Create vectorized environment
        if n_envs == 1:
            vec_env = DummyVecEnv([make_env])
        else:
            vec_env = make_vec_env(make_env, n_envs=n_envs)
        
        # Normalize observations and rewards
        if normalize:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, 
                                 clip_obs=10.0, clip_reward=10.0)
        vec_env = VecFrameStack(vec_env, n_stack=3)
        return vec_env
    
    def create_callbacks(self, eval_env, model_name: str, 
                        total_timesteps: int) -> CallbackList:
        """Create training callbacks for monitoring and checkpointing"""
        # early stopping if reward plateaus
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=2000,  # Adjust based on reward scale
            verbose=1
        )

        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir / f"{model_name}_best"),
            log_path=str(self.log_dir),
            eval_freq=max(total_timesteps // 20, 1000),
            deterministic=False,
            render=False,
            n_eval_episodes=20,
            callback_after_eval=stop_callback,
            verbose=1
        )
        
        # Add cirriculum learning callback
        class CirriculumLearningCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.progress_remaining = 1.0

            def _on_step(self) -> bool:
                self.progress_remaining = 1.0  - (self.num_timesteps / self.locals['total_timesteps'])
                if self.training_env is not None:
                    for env in self.training_env.envs:
                        if hasattr(env.unwrapped, 'set_difficulty'):
                            difficulty = min(1.0, 1.5 - self.progress_remaining)
                            env.unwrapped.set_difficulty(difficulty)
                return True
        
        return CallbackList([eval_callback, CirriculumLearningCallback()])
    
    def train_single_config(self, config_name: str, config: Dict, 
                          total_timesteps: int = 100000, 
                          verbose: int = 1) -> Tuple[DQN, Dict]:
        """Train DQN with a specific hyperparameter configuration"""
        
        print(f"\n{'='*60}")
        print(f"Training DQN with configuration: {config_name}")
        print(f"{'='*60}")
        
        # Create environments
        train_env = self.create_monitored_env()
        eval_env = self.create_monitored_env()
        
        # Create model
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=verbose,
            device='auto',
            tensorboard_log=str(self.log_dir / "tensorboard"),
            **config
        )
        
        # Create callbacks
        model_name = f"dqn_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        callbacks = self.create_callbacks(eval_env, model_name, total_timesteps)
        
        # Record training start time
        start_time = time.time()
        
        # Train the model
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=100,
                tb_log_name=f"DQN_{config_name}",
                reset_num_timesteps=False
            )
            
            training_time = time.time() - start_time
            
            # Save final model
            final_model_path = self.model_dir / f"{model_name}_final"
            model.save(str(final_model_path))
            
            # Evaluate final performance
            mean_reward, std_reward = self.evaluate_model(model, eval_env, n_episodes=20)
            
            # Training results
            results = {
                'config_name': config_name,
                'config': config,
                'training_time': training_time,
                'total_timesteps': total_timesteps,
                'final_mean_reward': mean_reward,
                'final_std_reward': std_reward,
                'model_path': str(final_model_path)
            }
            
            # Update best model tracking  
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = self.model_dir / "dqn_best_overall"
                model.save(str(best_model_path))
                print(f"New best model saved! Mean reward: {mean_reward:.2f}")
            
            self.training_history.append(results)
            
            print(f"\nTraining completed for {config_name}")
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Final performance: {mean_reward:.2f} ± {std_reward:.2f}")
            
            return model, results
            
        except Exception as e:
            print(f"Training failed for {config_name}: {str(e)}")
            return None, {'error': str(e), 'config_name': config_name}
        
        finally:
            train_env.close()
            eval_env.close()
    
    def evaluate_model(self, model: DQN, env, n_episodes: int = 20) -> Tuple[float, float]:
        """Evaluate trained model performance"""
        metrics = {
            'rewards': [],
            'lengths': [],
            'coverage': [],
            'success_rate': 0,
            'hazard_exposure': [],
            'edge_time': []
        }

        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            ep_metrics = {
                'reward': 0,
                'length': 0,
                'visited': set(),
                'hazard_steps': 0,
                'edge_steps': 0
            }
            
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs,reward, done, _, _ = env.step(action)

                # update metrics
                ep_metrics['reward'] += reward
                ep_metrics['length'] += 1

                # track visited cells
                cell = (int(env.agent_pos[0]), int(env.agent_pos[1]))
                ep_metrics['visited'].add(cell)

                # track hazards
                if (env._get_tear_gas_intensity(*env.agent_pos) > 0.1 or
                    env._get_water_cannon_intensity(*env.agent_pos) > 0.1):
                    ep_metrics['hazard_steps'] += 1

                # track edge hugging
                edge_dist = min(
                    env.agent_pos[0],
                    env.grid_width - env.agent_pos[0],
                    env.agent_pos[1],
                    env.grid_height - env.agent_pos[1]
                )
                if edge_dist < 10:
                    ep_metrics['edge_steps'] += 1

            # store episode metrics
            metrics['rewards'].append(ep_metrics['reward'])
            metrics['lengths'].append(ep_metrics['length'])
            metrics['coverage'].append(len(ep_metrics['visited']) / (env.grid_width * env.grid_height))
            metrics['hazard_exposure'].append(ep_metrics['hazard_steps'] / ep_metrics['length'])
            metrics['edge_time'].append(ep_metrics['edge_steps'] / ep_metrics['length'])

            if ep_metrics['reward'] > 500:
                metrics['success_rate'] += 1
        
        # calculate final metrics
        metrics['success_rate'] /= n_episodes
        results = {
            'mean_reward': np.mean(metrics['rewards']),
            'std_reward': np.std(metrics['rewards']),
            'mean_coverage': np.mean(metrics['coverage']),
            'success_rate': metrics['success_rate'],
            'mean_hazard_exposure': np.mean(metrics['hazard_exposure']),
            'mean_edge_time': np.mean(metrics['edge_time'])
        }

        print("\n=== Comprehensive Evaluation ===")
        print(f"Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
        print(f"Map Coverage: {results['mean_coverage']:.1%}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Hazard Exposure: {results['mean_hazard_exposure']:.1%} of steps")
        print(f"Edge Time: {results['mean_edge_time']:.1%} of steps")
    
        return results
    
    def hyperparameter_search(self, total_timesteps: int = 100000) -> Dict:
        """Run hyperparameter search across all configurations"""
        
        print("Starting DQN Hyperparameter Search")
        print(f"Total configurations to test: {len(self.hyperparameter_configs)}")
        print(f"Timesteps per configuration: {total_timesteps}")
        
        search_start_time = time.time()
        
        for config_name, config in self.hyperparameter_configs.items():
            try:
                model, results = self.train_single_config(
                    config_name, config, total_timesteps
                )
                
                # Save intermediate results
                self.save_training_history()
                
            except KeyboardInterrupt:
                print("\nHyperparameter search interrupted by user")
                break
            except Exception as e:
                print(f"Error in configuration {config_name}: {str(e)}")
                continue
        
        search_time = time.time() - search_start_time
        
        # Analyze results
        analysis = self.analyze_hyperparameter_results()
        analysis['total_search_time'] = search_time
        
        print(f"\nHyperparameter search completed in {search_time:.2f} seconds")
        
        return analysis
    
    def analyze_hyperparameter_results(self) -> Dict:
        """Analyze hyperparameter search results"""
        if not self.training_history:
            return {'error': 'No training history available'}
        
        # Filter successful runs
        successful_runs = [r for r in self.training_history if 'error' not in r]
        
        if not successful_runs:
            return {'error': 'No successful training runs'}
        
        # Find best configuration
        best_run = max(successful_runs, key=lambda x: x['final_mean_reward'])
        
        # Calculate statistics
        rewards = [r['final_mean_reward'] for r in successful_runs]
        training_times = [r['training_time'] for r in successful_runs]
        
        analysis = {
            'best_config': best_run['config_name'],
            'best_reward': best_run['final_mean_reward'],
            'best_config_details': best_run['config'],
            'num_successful_runs': len(successful_runs),
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'training_time_stats': {
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'total': np.sum(training_times)
            },
            'all_results': successful_runs
        }
        
        return analysis
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history_file = self.log_dir / "dqn_training_history.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Training history saved to {history_file}")
    
    def create_performance_plots(self):
        """Create performance comparison plots"""
        if not self.training_history:
            print("No training history available for plotting")
            return
        
        successful_runs = [r for r in self.training_history if 'error' not in r]
        
        if len(successful_runs) < 2:
            print("Need at least 2 successful runs for comparison plots")
            return
        
        # Performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        configs = [r['config_name'] for r in successful_runs]
        rewards = [r['final_mean_reward'] for r in successful_runs]
        std_rewards = [r['final_std_reward'] for r in successful_runs]
        training_times = [r['training_time'] for r in successful_runs]
        
        # Reward comparison
        ax1.bar(configs, rewards, yerr=std_rewards, capsize=5)
        ax1.set_title('Final Mean Reward by Configuration')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Training time comparison
        ax2.bar(configs, training_times)
        ax2.set_title('Training Time by Configuration')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Reward vs Training time scatter
        ax3.scatter(training_times, rewards)
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Final Mean Reward')
        ax3.set_title('Reward vs Training Time')
        
        # Hyperparameter correlation (learning rate vs reward)
        learning_rates = [r['config']['learning_rate'] for r in successful_runs]
        ax4.scatter(learning_rates, rewards)
        ax4.set_xlabel('Learning Rate')
        ax4.set_ylabel('Final Mean Reward')
        ax4.set_title('Learning Rate vs Reward')
        ax4.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / "dqn_performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance plots saved to {plot_path}")
    
    def load_best_model(self) -> Optional[DQN]:
        """Load the best performing model"""
        best_model_path = self.model_dir / "dqn_best_overall.zip"
        
        if best_model_path.exists():
            try:
                model = DQN.load(str(best_model_path))
                print(f"Best model loaded from {best_model_path}")
                return model
            except Exception as e:
                print(f"Error loading best model: {str(e)}")
                return None
        else:
            print("No best model found")
            return None


def main():
    """Main training function"""
    print("DQN Training for Nairobi CBD Protest Navigation")
    print("=" * 60)
    
    # Initialize training manager
    trainer = DQNTrainingManager()
    
    # Train only robust_exploration
    timesteps = 150000
    model, results = trainer.train_single_config(
        "robust_exploration", 
        trainer.hyperparameter_configs["robust_exploration"], 
        total_timesteps=timesteps
    )
    print(results)
    trainer.save_training_history()
    print("\nDQN training completed!")
    print(f"Models saved in: {trainer.model_dir}")
    print(f"Logs saved in: {trainer.log_dir}")

if __name__ == "__main__":
    main()