import numpy as np
import random
import sys
from only_ones_that_matter import pathfinding, Radar

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

# map module, mainly used for the visualization, each method is self explanatory in the name

# attack asset method is not worth looking at, but the intercept method is, it provides the optimal point of interception
class Missile():
    def __init__(self, y, x, target, path, step_path, num, freq, type, dead):
        self.y = y
        self.x = x
        self.target = target
        self.path = path
        self.step_path = step_path
        self.num = num
        self.freq = freq
        self.type = type
        self.dead = dead


class Interceptors():
    def __init__(self, y, x, target, path, step_path, num, type, missile):
        self.y = y
        self.x = x
        self.target = target
        self.path = path
        self.step_path = step_path
        self.num = num
        self.type = type
        self.missile = missile


class Map():
    def __init__(self, x, mountains, yx, types, count):
        self.x = x
        self.mountains = mountains
        self.yx = yx
        self.types = types
        self.count = count

    def getCount(self):
        return self.count


def updateXY(missiles):
    for i in range(len(missiles)):
        if len(missiles[i].path) > 1:
            missiles[i].y, missiles[i].x = missiles[i].path[1]

            missiles[i].path.pop(0)
            missiles[i].freq = np.round(Radar.samplefreq(missiles[i]))


def createMap(mountain):
    x = np.chararray((100, 100), unicode=True)
    x[:] = '#'
    mountains = []
    for h in range(mountain):
        mounx = random.randint(20, 70)
        mouny = random.randint(20, 70)
        mountains.append([mounx, mouny])
        for i in range(30):
            for j in range(30):
                x[mouny + i][mounx + j] = '-'

    for i in range(100):
        if random.randint(1, 4) == 1:
            x[i, 0] = 'A'
        if random.randint(1, 4) == 1:
            x[i, 1] = 'A'
        x[i, 2] = '|'
        x[i, 97] = '|'
        if random.randint(1, 4) == 1:
            x[i, 98] = 'A'
        if random.randint(1, 4) == 1:
            x[i, 99] = 'A'

    return x, mountains


def addMissiles(x, missiles):
    yx = []
    types = []

    while missiles != 0:
        y1, x1 = random.randint(5, 95), random.randint(80, 90)
        if x[y1, x1] == '#':
            missiles -= 1
            yx.append(y1)
            yx.append(x1)
            types.append(random.randint(1, 3))
    for i in range(12 - len(yx)):
        yx.append(0)

    return x, yx, types


def getAssets(x, yx):
    leftAssets = []
    rightAssets = []
    p = []
    list = yx[:]

    for i in range(100):
        if x[i, 0] == 'A':
            leftAssets.append([i, 0])
        if x[i, 1] == 'A':
            leftAssets.append([i, 1])
    for i in range(100):
        if x[i, 98] == 'A':
            rightAssets.append([i, 98])
        if x[i, 99] == 'A':
            rightAssets.append([i, 99])

    while list[0] != 0 and list[1] != 0:
        p.append(random.choice(leftAssets))
        list.pop(0)
        list.pop(0)
        list.append(0)
        list.append(0)

    return leftAssets, rightAssets, p


def pathfind(x, start, end):
    y = stateSpace(x)
    path = pathfinding.astar(y, start, end)

    return path


def stateSpace(x):
    y = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            if x[i, j] == '#' or x[i, j] == '|' or x[i, j] == 'A' or x[i, j] == '>' or x[i, j] == '<' or x[i, j] == '1' or x[i, j] == '2' or x[i, j] == '3' or x[i, j] == '0':
                y[i, j] = 0
            else:
                y[i, j] = 1
    return y


def attackAsset(x, missile, interceptor, step):
    missile.path = pathfind(x, (missile.y, missile.x), (missile.target[0], missile.target[1]))
    interceptor_path = intercept(x, missile, interceptor)

    missed = False
    if missile.type != interceptor.type:

        missed = True
    else:
        missile.dead = True

    step_missile_path = []
    step_interceptor_path = []

    if not interceptor_path:
        missile.dead = False
        missile.step_path = pathfind(x, (missile.y, missile.x), (missile.target[0], missile.target[1]))

        for step in missile.step_path:

            if step[1] == interceptor.x:
                interceptor.step_path = pathfind(x, (interceptor.y, interceptor.x), step)


        return False

    copy = missile.path[:]
    for i in range(len(interceptor_path)):
        if not step:
            x[missile.path[i]] = '>'
            x[interceptor_path[i]] = '<'
        missile.y, missile.x = missile.path[i]
        interceptor.y, interceptor.x = interceptor_path[i]
        interceptor.path.append(interceptor_path[i])
        step_interceptor_path.append(interceptor_path[i])
        step_missile_path.append(missile.path[i])

    if (missile.x != interceptor.x) or (missile.y != interceptor.y):
        interceptor_path = missile.path[:]
        interceptor_path.reverse()
        interceptor_index = interceptor_path.index((interceptor.y, interceptor.x))
        missile_index = missile.path.index((missile.y, missile.x))
        interceptor_path = interceptor_path[interceptor_index:]
        missile.path = missile.path[missile_index:]

        for i in range(1, len(missile.path)):

            if (missile.path[i] == interceptor_path[i]) or (missile.path[i - 1] == interceptor_path[i]) or (missile.path[i] == interceptor_path[i - 1]):

                missile.y, missile.x = missile.path[i]
                interceptor.y, interceptor.x = interceptor_path[i]
                interceptor.path.append(interceptor_path[i])
                step_interceptor_path.append(interceptor_path[i])
                step_missile_path.append(missile.path[i])
                break
            missile.y, missile.x = missile.path[i]
            interceptor.y, interceptor.x = interceptor_path[i]
            interceptor.path.append(interceptor_path[i])
            step_interceptor_path.append(interceptor_path[i])
            step_missile_path.append(missile.path[i])
    interceptor.step_path = step_interceptor_path
    if not missed:

        missile.step_path = step_missile_path
    else:
        missile.step_path = copy

    return True


def getPath(x, missiles):
    for missile in missiles:
        if missile.x == 0 and missile.y == 0:
            break
        coords = pathfind(x, (missile.y, missile.x), (missile.target[0], missile.target[1]))
        missile.path = coords
    return missiles


def intercept(x, missile, interceptor):
    done = False
    mid = int((len(missile.path) / 2))

    while not done:
        if mid >= len(missile.path):
            return False
        interceptor.target = missile.path[mid]
        interceptor_path = pathfind(x, (interceptor.y, interceptor.x), (interceptor.target[0], interceptor.target[1]))
        index = missile.path.index(interceptor.target)
        mpath = missile.path[:index]
        if not interceptor_path:
            return False
        else:
            if len(interceptor_path) < len(mpath):
                done = True

            else:

                mid += 1

    if abs(len(interceptor_path) - len(mpath)) > 2:
        while abs(len(interceptor_path) - len(mpath)) > 2:
            interceptor.target = missile.path[mid]
            interceptor_path = pathfind(x, (interceptor.y, interceptor.x),
                                        (interceptor.target[0], interceptor.target[1]))
            index = missile.path.index(interceptor.target)
            mpath = missile.path[:index]
            mid -= 1

    return interceptor_path

    if not interceptor_path:
        missile.step_path = missile.step_path + pathfind(x, (missile.x, missile.y),
                                                         (missile.target[0], missile.target[1]))
    for step in missile.step_path:

        if step[0] == interceptor.x:
            interceptor.step_path = pathfind(x, (interceptor.x, interceptor.y), step)
    return False

def best_missile(missiles, interceptors):
    for i in range(len(missiles)):
        for j in range(len(interceptors)):
            print(missiles[i].target[0])

    return missiles, interceptors
