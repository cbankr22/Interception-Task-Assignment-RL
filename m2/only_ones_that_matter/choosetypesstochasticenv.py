import gym
import numpy as np
import random
from only_ones_that_matter import Radar, map

# class that learns how to choose types given radar output
class choosetypes(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100)])
        # state is probabilities for each missile type
        self.state = [0, 0, 0]
        x, mountains = map.createMap(0)
        x, yx, types = map.addMissiles(x, random.randint(1, 6))
        missiles = []
        assetschosen = []
        self.x = x
        self.mountains = mountains
        self.yx = yx
        self.types = types
        self.num = (12 - self.yx.count(0)) / 2
        self.missiles = missiles
        self.assetschosen = assetschosen

    def step(self, action):

        done = False
        reward = 0
        # randomly sample given probabilities in state
        sample = random.choices([0, 1, 2], self.state)
        if action == sample[0]:
            reward = 1
        if self.missiles:
            self.state = Radar.probs(self.missiles[0].freq, self.missiles[0])
            self.missiles.pop(0)
        else:
            done = True
        # given the probabilities, agent learns to pick which interceptor, i think we can expand this, as it is quite basic
        info = []
        return self.state, reward, done, info

    def reset(self, x, mountains, yx, lassets, rassets, assetschosen, types):
        missiles = []
        self.mountains = mountains
        self.x = x
        self.yx = yx
        self.types = types
        self.state = yx
        self.num = int((12 - self.yx.count(0)) / 2)
        if self.num == 0:
            self.num += 1
        self.assetschosen = assetschosen
        for i in range(self.num):
            missiles.append(
                map.Missile(self.yx[i * 2], self.yx[(i * 2) + 1], self.assetschosen[i], [], [], i, 0, self.types[i], False))
            missiles[i].freq = np.round(Radar.samplefreq(missiles[i]))
        self.missiles = missiles
        self.state = Radar.probs(self.missiles[0].freq, self.missiles[0])
        self.missiles.pop(0)
        return self.state

