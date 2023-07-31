import numpy as np
from only_ones_that_matter.choosewhere import choosewhere
# import pybullet_envs
from only_ones_that_matter.PPO import PPO

#################################### Testing ###################################

# testing method mostly used for debugging, visualization module used the choosewhere method to pick silos to laucnh out of
def test():

    print("============================================================================================")


    env_name = "choosewhere"
    has_continuous_action_space = False
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving


    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2            # clip parameter for PPO
    gamma = 0.99             # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    c = choosewhere()

    # state space dimension
    state_dim = 2

    # action space dimension
    action_dim = c.action_space.n


    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num


    directory = "PPO" + '/' + "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")



    test_running_reward = 0
    episodes = 10000
    i = 0
    reward = 0
    rewards = []

    for ep in range(i, episodes):

        observation = c.reset()

        for t in range(1, max_ep_len+1):

            action = ppo_agent.select_action(observation)
            print(observation)
            print(action)
            print(reward)
            observation, reward, done, _ = c.step(action)
            rewards.append(reward)
            if done:

                observation = c.reset()

                break

        # clear buffer
        ppo_agent.buffer.clear()


        print('Episode: {} \t\t Avg Reward: {}'.format(ep, round(np.mean(rewards), 3)))


    c.close()


    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : ", np.mean(rewards))

    print("============================================================================================")




if __name__ == '__main__':

    test()


def choosewhe(observation):
    env_name = "choosewhere"
    directory = "PPO" + '/' + "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, 0, 0)
    c = choosewhere()

    ppo_agent = PPO(2, c.action_space.n, 0.0003, 0.001, .99, 80, .2, False, .1)
    ppo_agent.load(checkpoint_path)

    action = ppo_agent.select_action(observation)

    return action