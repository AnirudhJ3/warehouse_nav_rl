#!/usr/bin/env python3

from drl_agent import DRLAgent
import argparse
import os

def main(args):
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Training configuration
    config = {
        'total_timesteps': args.timesteps,
        'n_steps': 2048,
        'batch_size': 64,
        'learning_rate': args.learning_rate,
        'gamma': 0.99,
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'eval_freq': 10000,
        'n_eval_episodes': 10
    }
    
    # Initialize and train agent
    agent = DRLAgent(config)
    agent.train(use_wandb=args.use_wandb)
    
    # Evaluate trained agent
    eval_results = agent.evaluate(n_episodes=100)
    print("\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL agent for warehouse navigation")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                      help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate for the PPO algorithm")
    parser.add_argument("--use-wandb", action="store_true",
                      help="Use Weights & Biases for logging")
    args = parser.parse_args()
    
    main(args) 