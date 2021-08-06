from only_ones_that_matter.choosetypesstochasticenv import envtest
import random
from DQNNetwork import Agent
import numpy as np
from only_ones_that_matter import map

env = envtest()


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3, eps_end=0,
              input_dims=[3], lr=0.001)
rewards = []
done = False
x, mountains = map.createMap(0)
x, yx, freqs = map.addMissiles(x, random.randint(1, 6))
lassets, rassets, assetschosen = map.getAssets(x, yx)
observation = env.reset(x, mountains, yx, lassets, rassets, assetschosen, freqs)
for i in range(50000):
        done = False

        action = agent.choose_action(observation)
        o = observation
        observation_, reward, done, info = env.step(action)

        agent.store_transition(observation, action, reward,
                                   observation_, done)
        rewards.append(reward)
        agent.learn()
        observation = observation_
        if done:

            print('avg reward ', np.mean(rewards[-1000:]))
            if random.randint(1, 100) == 5:
                print('state ', o)
                print('action taken ', action)
            x, mountains = map.createMap(0)
            x, yx, freqs = map.addMissiles(x, random.randint(1, 6))
            lassets, rassets, assetschosen = map.getAssets(x, yx)
            observation = env.reset(x, mountains, yx, lassets, rassets, assetschosen, freqs)

#-2.17
