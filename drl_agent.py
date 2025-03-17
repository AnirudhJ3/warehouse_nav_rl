import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
from warehouse_env import WarehouseEnv

class DRLAgent:
    def __init__(self, config=None):
        self.config = {
            'total_timesteps': 1_000_000,
            'n_steps': 2048,
            'batch_size': 64,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'n_epochs': 10,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'eval_freq': 10000,
            'n_eval_episodes': 10
        } if config is None else config
        
        # Initialize environment
        self.env = DummyVecEnv([lambda: WarehouseEnv()])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        
        # Initialize evaluation environment
        self.eval_env = DummyVecEnv([lambda: WarehouseEnv()])
        self.eval_env = VecNormalize(self.eval_env, norm_obs=True, norm_reward=True)
        
        # Initialize PPO agent
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            n_epochs=self.config['n_epochs'],
            verbose=1
        )
        
    def train(self, use_wandb=True):
        """Train the DRL agent."""
        if use_wandb:
            run = wandb.init(
                project="warehouse_navigation",
                config=self.config,
                sync_tensorboard=True
            )
            
            # Create evaluation callback
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path="./logs/best_model",
                log_path="./logs/results",
                eval_freq=self.config['eval_freq'],
                n_eval_episodes=self.config['n_eval_episodes'],
                deterministic=True,
                render=False
            )
            
            # Create WandB callback
            wandb_callback = WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            )
            
            callbacks = [eval_callback, wandb_callback]
        else:
            callbacks = None
            
        # Train the agent
        self.model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=callbacks
        )
        
        # Save the final model
        self.model.save("final_model")
        
        if use_wandb:
            wandb.finish()
            
    def evaluate(self, n_episodes=100):
        """Evaluate the trained agent."""
        rewards = []
        successes = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()[0]
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                episode_reward += reward
                
                if done:
                    successes.append(info[0]['is_success'])
                    
            rewards.append(episode_reward)
            
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(successes),
            'n_episodes': n_episodes
        }
        
    def save(self, path):
        """Save the trained model and environment normalization stats."""
        self.model.save(path)
        self.env.save(f"{path}_env_stats.pkl")
        
    def load(self, path):
        """Load a trained model and environment normalization stats."""
        self.model = PPO.load(path, env=self.env)
        self.env = VecNormalize.load(f"{path}_env_stats.pkl", self.env)
        self.env.training = False
        self.env.norm_reward = False 