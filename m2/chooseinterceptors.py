import gym
import numpy as np
from gym import spaces
import random





class envtest(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(5), gym.spaces.Discrete(5), gym.spaces.Discrete(5), gym.spaces.Discrete(5)])
        self.state = [0, 0, 0, 0]

    def step(self, action):
        t1, t2, t3, left = self.state
        done = False
        if left <= 0:
            done = True
        left -= 1
        if action == 0:
            if t1 == 1:
                reward = 1
            else:
                reward = 0
        if action == 1:
            if t2 == 1:
                reward = 1
            else:
                reward = 0
        if action == 2:
            if t3 == 1:
                reward = 1
            else:
                reward = 0
        info = []

        m = [0, 0, 0]
        m[random.randint(0, 2)] = 1

        self.state = m[0], m[1], m[2], left
        return self.state, reward, done, info


    def reset(self):
        m = [0, 0, 0]
        m[random.randint(0, 2)] = 1

        self.state = m[0], m[1], m[2], random.randint(1, 10)
        print(self.state)
        return self.state

