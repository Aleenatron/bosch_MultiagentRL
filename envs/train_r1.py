import gymnasium as gym
import sys
import os

# Add envs/ directory to Python path before importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.multiagent_driving import MultiAgentDrivingEnv

# Create the environment
env = make_vec_env(lambda: MultiAgentDrivingEnv(), n_envs=1)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train for 10,000 steps
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_multiagent")

print("Training complete!")
