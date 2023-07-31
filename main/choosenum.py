from choosenumenv import envtest
from DQNNetwork import Agent
import numpy as np
from only_ones_that_matter import map
from choosetypesenv import typeenv
from choosewhenenv import choosewhen

env = envtest()
env2 = typeenv()
env3 = choosewhen()

x, mountains = map.createMap(0)
x, yx, types = map.addMissiles(x, 1)
observation = env.reset(x, mountains, yx, types)
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0,
              input_dims=[12], lr=0.001)
agent2 = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=5, eps_end=0,
              input_dims=[5], lr=0.001)
agent3 = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=99, eps_end=0.01,
              input_dims=[3], lr=0.001)
rewards = []
done = False
test = 0
act = [0, 0, 0, 0, 0]
r = []
for i in range(100000):
    print(i)
    x, mountains = map.createMap(0)
    x, yx, types = map.addMissiles(x, 1)
    lassets, rassets, assetschosen = map.getAssets(x, yx)
    observation = env.reset(x, mountains, yx, types)
    observation2 = env2.reset(x, mountains, yx, types)
    observation3 = env3.reset(x, mountains, yx, types, lassets, rassets, assetschosen)
    m = map.Map(x, mountains, yx, types, 0)
    done = False

    action = agent.choose_action(observation)

    observation_, reward, done, info = env.step(action)
    m.count = action
    act = [0, 0, 0, 0, 0]
    for i in range(m.count):

        done = False
        action2 = agent2.choose_action(observation2)
        observation2_, reward2, done2, info2 = env2.step(action2)
        act[action2] += 1
        agent2.store_transition(observation2, action2, reward2,
                                   observation2_, done2)
        rewards.append(reward2)
        agent2.learn()
        observation2 = observation2_
        if i == m.count - 1:
            done2 = True
        if done2:
            print('types ', env2.totaltypes)
            print('avg reward ', np.mean(rewards[-100:]))
            print('actions taken ', act)

    for i in range(1):
        action3 = agent3.choose_action(observation3)

        observation3_, reward3, done3, info3 = env3.step(action3)

        agent3.store_transition(observation3, action3, reward3,
                               observation3_, done3)

        agent3.learn()
        observation3 = observation3_
        r.append(reward3)

        if i == m.count - 1:
            done3 = True
        if done3:
            print('action taken', action3)
            print('env 3 obs ', observation3_)
            print('env3 avg reward ', np.mean(r[-100:]))
    print(observation)
    print(action)

    agent.store_transition(observation, action, reward,
                               observation_, done)
    rewards.append(reward)
    agent.learn()
    observation = observation_
    print(np.mean(rewards))


#-2.17
