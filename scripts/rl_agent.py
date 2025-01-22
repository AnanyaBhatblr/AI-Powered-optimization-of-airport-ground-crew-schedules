# scripts/rl_agent.py
import gym
import numpy as np
from stable_baselines3 import DQN

class CrewTaskEnv(gym.Env):
    def __init__(self):
        super(CrewTaskEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        reward = np.random.rand()
        return self.observation_space.sample(), reward, False, {}

    def reset(self):
        return self.observation_space.sample()

if __name__ == "__main__":
    env = CrewTaskEnv()
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("../models/rl_agent")

    print("Reinforcement Learning Agent trained and saved!")
