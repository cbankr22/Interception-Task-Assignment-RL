import gym
import random
from only_ones_that_matter import map


class envtest(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100),
             gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100),
             gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100)])
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        x, mountains = map.createMap(5)
        x, yx, types = map.addMissiles(x, random.randint(1, 6))
        self.x = x
        self.mountains = mountains
        self.yx = yx
        self.types = types

    def step(self, action):

        missiles = (12 - self.state.count(0)) / 2
        reward = -1 * abs(action - missiles)


        done = True
        info = []

        return self.state, reward, done, info

    def reset(self, x, mountains, yx, types):

        self.mountains = mountains
        self.x = x
        self.yx = yx
        self.types = types
        self.state = yx

        return self.state

