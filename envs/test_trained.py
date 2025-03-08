import gymnasium as gym
import os
import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.multiagent_driving import MultiAgentDrivingEnv

# Load the trained model
model_path = "ppo_multiagent.zip"
assert os.path.exists(model_path), "No trained model found! Run train_r1.py first."
# Create environment
env = MultiAgentDrivingEnv(num_agents=2)

# Load the trained model 
model = PPO.load(model_path)

# Reset environment
obs, _ = env.reset()
done = False

plt.ion()  # Enable interactive mode for live rendering

while not done:
    action, _ = model.predict(obs)  # Get action from trained policy
    obs, rewards, done, _, _ = env.step(action)
    env.render()  # Visualize the movement
