import gym
import random
from only_ones_that_matter import map


class choosewhen(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(99)
        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(100), gym.spaces.Discrete(100), gym.spaces.Discrete(100)])
        self.state = [0, 0, 0]
        x, mountains = map.createMap(5)
        x, yx, types = map.addMissiles(x, 1)
        lassets, rassets, assetschosen = map.getAssets(x, yx)
        missiles = []
        interceptors = []
        locations = [[], [], [], [], [], []]
        self.x = x
        self.mountains = mountains
        self.yx = yx
        self.types = types
        self.lassets = lassets
        self.rassets = rassets
        self.assetschosen = assetschosen
        self.missiles = missiles
        self.locations = locations
        self.num = (12 - self.yx.count(0)) / 2
        self.interceptors = interceptors

    def step(self, action):
        action += 1
        reward = 0
        done = True
        info = []
        for i in range(5):

            map.attackAsset(x, self.missiles[i], self.interceptors[i])
        reward = -len(self.interceptors[0].path)

        return self.state, reward, done, info

    def reset(self, x, mountains, yx, types, lassets, rassets, assetschosen):
        missiles = []
        interceptors = []
        self.mountains = mountains
        self.x = x
        self.yx = yx
        self.types = types
        self.lassets = lassets
        self.rassets = rassets
        self.assetschosen = assetschosen
        self.num = (12 - self.yx.count(0)) / 2
        for i in range(int(self.num)):
            missiles.append(map.Missile(self.yx[i * 2], self.yx[(i * 2) + 1], self.assetschosen[i], [], i))
            interceptors.append(map.Interceptors(random.randint(1, 99), 4, [], [], i))
        self.missiles = missiles
        self.missiles = map.getPath(self.x, self.missiles)
        self.locations = [[self.yx[0], self.yx[1]], [self.yx[2], self.yx[3]], [self.yx[4], self.yx[5]], [self.yx[6], self.yx[7]], [self.yx[8], self.yx[9]], [self.yx[10], self.yx[11]]]

        self.state = self.missiles[0].y, self.missiles[0].target[0], self.missiles[0].target[1]
        self.interceptors = interceptors
        return self.state

e = choosewhen()
x, mountains = map.createMap(5)
x, yx, types = map.addMissiles(x, 5)
lassets, rassets, assetschosen = map.getAssets(x, yx)
observation = e.reset(x, mountains, yx, types, lassets, rassets, assetschosen)

e.step(50)
map.showMap(x)
