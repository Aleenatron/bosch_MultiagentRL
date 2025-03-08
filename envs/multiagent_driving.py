import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class MultiAgentDrivingEnv(gym.Env):
    """
    A simple 2D multi-agent driving environment.
    """
    metadata = {"render_modes": ["console"], "render_fps": 30} # Fixed render mode key

    def __init__(self, num_agents=2):
        super(MultiAgentDrivingEnv, self).__init__()

        self.num_agents = num_agents
        self.action_space = spaces.Discrete(3)  # Actions: Left, Right, Stay
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_agents, 4), dtype=np.float32)

        self.state = np.zeros((self.num_agents, 4), dtype=np.float32)  # Initialize state
        self.fig, self.ax = plt.subplots()

    def render(self, mode="human"):
        self.ax.clear()
        self.ax.set_xlim(0, 1)  # Track width
        self.ax.set_ylim(0, 1)  # Track height
        
        for agent in self.state:
            self.ax.scatter(agent[0], agent[1], c='red', s=100)  # Draw cars
        
        plt.pause(0.1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(self.num_agents, 4).astype(np.float32)  # Ensure dtype float32
        return self.state, {}  # Gymnasium requires returning (obs, info)

    def step(self, actions):
    # Convert actions into a NumPy array and ensure it's the right shape
        actions = np.array(actions).flatten()

        if actions.shape[0] == 1:
            actions = np.repeat(actions, self.num_agents)

        if actions.shape[0] != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions but got {actions.shape[0]}: {actions}")

        # Initialize rewards
        rewards = np.zeros(self.num_agents)
        done = False

        # Update state based on actions
        for i in range(self.num_agents):
            if actions[i] == 0:  # Move left
                self.state[i][0] -= 0.05
            elif actions[i] == 2:  # Move right
                self.state[i][0] += 0.05

            # Reward based on distance to center (0.5 is optimal)
            rewards[i] = 1.0 - abs(self.state[i][0] - 0.5)

            # Check if agent is out of bounds
            if self.state[i][0] < 0 or self.state[i][0] > 1:
                done = True

        # 🛠️ Convert rewards array to a scalar sum
        total_reward = np.sum(rewards)

        return self.state, total_reward, done, False, {}




    def render(self, mode="console"):
        if mode == "console":
            print(f"Agent States: {self.state}")

    def close(self):
        pass
