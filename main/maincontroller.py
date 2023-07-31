import os
import random
from only_ones_that_matter.choosewhen2env import choosewhen2
from only_ones_that_matter import choosetypesstochasticenv, map

import numpy as np
from only_ones_that_matter.PPO import PPO


def train():
    env_name1 = "choosewhen2"
    env_name2 = "choosetypes"
    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 250  # max timesteps in one episode

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    update_timestep = max_ep_len  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    print("training environment name : maincontroller")

    choosewhen = choosewhen2()
    choosetypes = choosetypesstochasticenv.choosetypes()

    state_dim_choosewhen = len(choosewhen.observation_space.sample())
    state_dim_choosetypes = len(choosetypes.observation_space.sample())

    action_dim_choosewhen = choosewhen.action_space.n
    action_dim_choosetypes = choosetypes.action_space.n

    run_num_pretrained = 0

    directory1 = "PPO_preTrained"
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    directory2 = "PPO_preTrained"
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    directory1 = directory1 + '/' + env_name1 + '/'
    directory2 = directory2 + '/' + env_name2 + '/'

    if not os.path.exists(directory1):
        os.makedirs(directory1)
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    checkpoint_path_choosewhen = directory1 + "PPO_{}_{}_{}.pth".format(env_name1, random_seed, run_num_pretrained)
    checkpoint_path_choosetypes = directory2 + "PPO_{}_{}_{}.pth".format(env_name2, random_seed, run_num_pretrained)

    # initialize a PPO agent
    ppo_agent_when = PPO(state_dim_choosewhen, action_dim_choosewhen, lr_actor / 10, lr_critic / 10, gamma, K_epochs, eps_clip,
                         has_continuous_action_space,
                         action_std)
    ppo_agent_types = PPO(state_dim_choosetypes, action_dim_choosetypes, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                          has_continuous_action_space,
                          action_std)


    episodes = 50000
    # training loop
    rewards_choosewhen = []
    rewards_choosetypes = []
    r = 0
    time_step = 0
    actionstaken = 0
    correctactions = 0
    while r < episodes:
        wait = 0
        x, mountains = map.createMap(0)
        x, yx, freqs = map.addMissiles(x, random.randint(1, 6))
        lassets, rassets, assetschosen = map.getAssets(x, yx)
        observation_choosewhen = choosewhen.reset(x, mountains, yx, lassets, rassets, assetschosen, freqs)
        for t in range(1, max_ep_len + 1):


            action_choosewhen = ppo_agent_when.select_action(observation_choosewhen)
            observation_choosewhen, reward_choosewhen, done_choosewhen, _ = choosewhen.step(action_choosewhen)
            if action_choosewhen != 1:
                wait += 1
            if action_choosewhen == 1:
                yx = choosewhen.yx

                observation_choosetypes = choosetypes.reset(x, mountains, yx, lassets, rassets, assetschosen, freqs)
                print(observation_choosetypes)
                done_choosetypes = False
                while not done_choosetypes:
                    action_choosetypes = ppo_agent_types.select_action(observation_choosetypes)

                    observation_choosetypes, reward_choosetypes, done_choosetypes, _ = choosetypes.step(action_choosetypes)

                    actionstaken += 1
                    if reward_choosetypes == 1:
                        correctactions += 1

                    rewards_choosetypes.append(reward_choosetypes)
                    ppo_agent_types.buffer.rewards.append(reward_choosetypes)
                    ppo_agent_types.buffer.is_terminals.append(done_choosetypes)
                if random.randint(1, 1000) == 5:
                    ppo_agent_types.save(checkpoint_path_choosetypes)
                if time_step % update_timestep == 0:
                    ppo_agent_types.update()



            ppo_agent_when.buffer.rewards.append(reward_choosewhen)
            ppo_agent_when.buffer.is_terminals.append(done_choosewhen)

            time_step += 1
            rewards_choosewhen.append(reward_choosewhen)
            actionstaken = 0
            correctactions = 0
            if time_step % update_timestep == 0:
                 ppo_agent_when.update()

            if done_choosewhen:
                print('avg reward ', np.mean(rewards_choosewhen[-10000:]))
                print('waiting steps ', wait)
                print('avg reward for choosetypes ', np.mean(rewards_choosetypes[-10000:]))
                if random.randint(1, 1000) == 5:
                    ppo_agent_when.save(checkpoint_path_choosewhen)

                r += 1
                break

    choosewhen.close()


if __name__ == '__main__':
    train()
