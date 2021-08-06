import math

import gym
import numpy as np
import random

from only_ones_that_matter import Radar, map


class choosewhen2(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100),
             gym.spaces.Discrete(100), gym.spaces.Discrete(100),
             gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100),
             gym.spaces.Discrete(100), gym.spaces.Discrete(100),
             gym.spaces.Discrete(100)])
        # state is x, y of each missile, 0s if there isnt a missile, 12th index is the number of times the waiting step has been activated
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # x is the 100x100 grid, mountains is the top left xy coordinate of each mountain, yx is the yx values for each missiles, types is the missile types
        x, mountains = map.createMap(0)
        x, yx, types = map.addMissiles(x, 1)
        missiles = []
        assetschosen = []
        count = 0
        self.x = x
        self.mountains = mountains
        self.yx = yx
        self.types = types
        # number of missiles chosen
        self.num = (12 - self.yx.count(0)) / 2
        self.missiles = missiles
        self.assetschosen = assetschosen
        self.count = count
        self.interceptors = []

    def step(self, action):

        done = False
        reward = 0
        # action 0 is waiting action, action 1 is deploy interceptors action
        for i in range(action):
            map.updateXY(self.missiles)

            done = True


        for i in range(self.num):
            p1, p2, p3 = Radar.probs(self.missiles[i].freq, self.missiles[i])
            reward -= 10 * (100 - ((p1 * (p1 * 100)) + (p2 * (p2 * 100)) + (p3 * (p3 * 100))))

            count = 0
            # sees if deployment was too late, if interceptor couldn't intercept, reward is -100,000
            if self.missiles[i].path:
                for path in self.missiles[i].path:
                    count += 1
                    if path[1] == self.interceptors[i].x:
                        dist = math.hypot(self.interceptors[i].x - path[1], self.interceptors[i].y - path[0])
                        if dist > count:
                            reward = -100_000
                    self.missiles[i].path.pop(0)

        # sees if missile reached its target, if it did, reward is -100,000
        for i in range(self.num):
            self.yx[i * 2], self.yx[(i * 2) + 1] = self.missiles[i].y, self.missiles[i].x
            if [self.missiles[i].y, self.missiles[i].x] == self.missiles[i].target:
                reward = -100_000

                done = True

        info = []
        self.count += 1
        self.state = self.yx
        self.state.pop(12)
        self.state.append(self.count)

        return self.state, reward, done, info

    def reset(self, x, mountains, yx, lassets, rassets, assetschosen, types):
        # reset method for environment
        missiles = []
        self.mountains = mountains
        self.x = x

        self.yx = yx
        self.types = types
        self.state = yx
        self.num = int((12 - self.yx.count(0)) / 2)
        self.assetschosen = assetschosen


        for i in range(self.num):
            missiles.append(
                map.Missile(self.yx[i * 2], self.yx[(i * 2) + 1], self.assetschosen[i], [], [], i, 0, self.types[i],
                            False))
            missiles[i].path = map.pathfind(x, (missiles[i].y, missiles[i].x),
                                            (missiles[i].target[0], missiles[i].target[1]))
            missiles[i].freq = np.round(Radar.samplefreq(missiles[i]))
            self.interceptors.append(map.Interceptors(missiles[i].y, random.randint(4, 8), [], [], [], i, 0, i))
        self.missiles = missiles
        # we are assuming the choosewhere network is trained, and this is the optimal solution for that network (network would've picked these silos anyways, but this is faster)
        for i in range(self.num):
            if self.missiles[i].target[0] <= 37:
                self.interceptors[i].y, self.interceptors[i].x = 25, 5
            elif self.missiles[i].target[0] <= 63:
                self.interceptors[i].y, self.interceptors[i].x = 50, 5
            else:
                self.interceptors[i].y, self.interceptors[i].x = 75, 5

        self.state.append(self.count)
        return self.state
# avg reward ~ -4000
# avg waiting steps ~ 36
