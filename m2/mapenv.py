import gym
import random
from only_ones_that_matter import map


class envtest(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(100), gym.spaces.Discrete(100)])
        self.state = [0, 0]
        x, mountains = map.createMap(5)
        x, yx, types = map.addMissiles(x, random.randint(1, 6))
        self.x = x
        self.mountains = mountains
        self.yx = yx

    def step(self, action):
        m1y, m1x = self.state
        done = False
        missile = [map.Missile(action, 3)]
        yx = [m1y, m1x]
        print(yx)
        count = map.attackAsset(self.x, missile, yx)
        if count <= 10:
            print('---------------------------------------------------------------------------------------------------------------------------------------------------------------')
        self.yx.pop(0)
        self.yx.pop(0)
        self.yx.append(0)
        self.yx.append(0)
        reward = -1 * count
        if self.yx[0] == 0:
            done = True

        info = []
        self.state = self.yx[0], self.yx[1]
        return self.state, reward, done, info

    def reset(self):
        p = []
        x, mountains = map.createMap(5)
        x, yx, types = map.addMissiles(x, random.randint(1, 6))
        self.x = x
        self.yx = yx
        for i in range(2):
            p.append(self.yx[i])
        self.state = p
        print(self.state)
        print(self.yx)
        return self.state

