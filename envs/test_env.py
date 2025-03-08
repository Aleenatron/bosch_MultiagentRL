import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.multiagent_driving import MultiAgentDrivingEnv

# Register environment
gym.register(
    id="MultiAgentDriving-v0",
    entry_point="envs.multiagent_driving:MultiAgentDrivingEnv",
)

env = gym.make("MultiAgentDriving-v0")
state, _ = env.reset()
print("Custom Multi-Agent Driving Env is working!")
env.render()