"""additional wrapper for the minigrid env"""

import gym

class NoImageWrapper(gym.ObservationWrapper):
    """Ensure the env is not interpreted as image env. Which we dont need."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.observation_space['image'] = gym.spaces.Box(
            low=0,
            high=15,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )

    def observation(self, observation):
        return observation
