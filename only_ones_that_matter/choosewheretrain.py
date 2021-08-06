import os
import random
from only_ones_that_matter.choosewhere import choosewhere
import numpy as np

from only_ones_that_matter.PPO import PPO

def train():
    # simple PPO training function, all 3 neural networks are similar
    env_name = "choosewhere"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 250  # max timesteps in one episode
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    update_timestep = max_ep_len  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network
    random_seed = 0  # set random seed if required (0 = no random seed)

    print("training environment name : " + env_name)

    env = choosewhere()
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


    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    episodes = 500000
    r = 0
    # training loop
    rewards = []
    act = []
    while r < episodes:

        observation = env.reset()
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):



            action = ppo_agent.select_action(observation)
            observation, reward, done, _ = env.step(action)
            act.append(action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            #print(n_uavs)
            time_step += 1
            current_ep_reward += reward
            rewards.append(reward)
            if done:
                print(r)
                print('avg reward ', np.mean(rewards))

                observation = env.reset()
                if random.randint(1, 1000) == 5:
                    ppo_agent.save(checkpoint_path)

            r += 1
                #print(r)
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()


            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1


if __name__ == '__main__':
    train()
