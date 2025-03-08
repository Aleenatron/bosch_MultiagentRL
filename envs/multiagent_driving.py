import gymnasium as gym
from gymnasium import spaces
import numpy as np

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(self.num_agents, 4).astype(np.float32)  # Ensure dtype float32
        return self.state, {}  # Gymnasium requires returning (obs, info)

    def step(self, actions):
        """
        Moves agents based on actions (simplified update logic).
        """
        actions = np.clip(actions, 0, 2)  # Ensure actions are within valid range

        # Example: Modify the first element of each state row based on action
        for i in range(self.num_agents):
            if actions[i] == 0:  # Move Left
                self.state[i, 0] = max(0, self.state[i, 0] - 0.05)
            elif actions[i] == 1:  # Move Right
                self.state[i, 0] = min(1, self.state[i, 0] + 0.05)
            # Stay (action == 2) does nothing

        rewards = np.random.rand(self.num_agents)  # Dummy rewards for now
        done = False  # Episode termination logic not implemented
        return self.state, rewards, done, False, {}  # Correct return format

    def render(self, mode="console"):
        if mode == "console":
            print(f"Agent States: {self.state}")

    def close(self):
        pass
