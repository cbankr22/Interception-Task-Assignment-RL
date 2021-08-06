import os
import random
from only_ones_that_matter.choosewhen2env import choosewhen2
import numpy as np

from only_ones_that_matter import map
from only_ones_that_matter.PPO import PPO


def train():
    env_name = "choosewhen2"
    has_continuous_action_space = False  # continuous action space; else discrete
    max_ep_len = 250  # max timesteps in one episode
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    update_timestep = max_ep_len  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0000003
    lr_critic = 0.000001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    print("training environment name : " + env_name)

    env = choosewhen2()
    state_dim = len(env.observation_space.sample())

    action_dim = env.action_space.n

    run_num_pretrained = 0

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    env_name1 = "choosewhen2"
    directory1 = "PPO" + '/' + "PPO_preTrained" + '/' + env_name1 + '/'
    checkpoint_path1 = directory1 + "PPO_{}_{}_{}.pth".format(env_name1, 0, 0)
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    ppo_agent.load(checkpoint_path1)

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    episodes = 50000
    r = 0
    # training loop
    rewards = []
    avg_rewards = []
    act = []
    me = []
    while r < episodes:
        wait = 0
        x, mountains = map.createMap(0)
        x, yx, freqs = map.addMissiles(x, random.randint(1, 6))
        lassets, rassets, assetschosen = map.getAssets(x, yx)
        observation = env.reset(x, mountains, yx, lassets, rassets, assetschosen, freqs)
        current_ep_reward = 0
        re = 0
        for t in range(1, max_ep_len + 1):

            action = ppo_agent.select_action(observation)


            observation, reward, done, _ = env.step(action)
            if action != 1:
                wait += 1
            else:
                re = reward
                me.append(wait)
            act.append(action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward
            rewards.append(re)
            # 5.44 avg waiting steps
            if done:
                print(np.mean(rewards))
                print(np.mean(me))
                if random.randint(1, 100) == 5:
                    ppo_agent.save(checkpoint_path)

            if time_step % update_timestep == 0:
                ppo_agent.update()

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    env.close()


if __name__ == '__main__':
    train()



# method that visualization module uses to choose number of waiting steps
def choosewhen(x, mountains, yx, types, lassets, rassets, assetschosen):
    env_name = "choosewhen2"
    directory = "PPO" + '/' + "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, 0, 0)
    env = choosewhen2()
    ppo_agent = PPO(13, env.action_space.n, 0.00003, 0.0001, .99, 80, .2, False, .1)
    ppo_agent.load(checkpoint_path)
    waiting_steps = 0
    done = False
    observation = env.reset(x, mountains, yx, lassets, rassets, assetschosen, types)

    while not done:
        action = ppo_agent.select_action(observation)
        observation, reward, done, _ = env.step(action)
        waiting_steps += 1


    return waiting_steps




