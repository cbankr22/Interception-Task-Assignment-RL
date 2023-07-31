import math
import gym
import random

# super basic environment where given the location of the asset the missile is attacking, it picks which silo to launch out of
class choosewhere(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(100), gym.spaces.Discrete(100)])
        self.state = [0, 0]

    def step(self, action):

        silos = [(25, 5), (50, 5), (75, 5)]
        # reward is calculated given how far the silo chosen is from the asset the missile is attacking
        dist = math.hypot(silos[action][1] - self.state[1], silos[action][0] - self.state[0])
        reward = -dist

        info = []
        done = True
        return self.state, reward, done, info

    def reset(self):
        self.state = [random.randint(1, 100), random.randint(0, 1)]
        return self.state



