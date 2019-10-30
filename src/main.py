"""
File for running training and testing simulations.  Various parameters such 
as number of episodes run, hyper params, which tests to utilize can all be 
controlled below.
"""
import gym, agent
import matplotlib.pyplot as plt
import numpy as np
import random as rand


# global params specified in project outline
NUM_RWD = 100  # Number of rewards to average over
CONVERGE = 200  # Reward convergance target


def param_test(env, episodes, agent, params, verbose=False):
    """
    Runs simulation under given conditions and tests specified range of 
    hyperparameters, tracks total reward from each episode and returns
    list of those values.
    """
    feat_count  = env.env.observation_space.shape[0]
    act_count = env.env.action_space.n

    # track each params rewards and process a new learner per param
    param_rewards = []
    for param_value in params:

        if verbose:
            print('Processing param: ' + str(param_value))

        # build the learner to utilize in the simulation, update the param value
        # being adjust in the simulation here
        learner = agent(num_feat=feat_count, num_acts=act_count, 
                        alpha=param_value, gamma=0.99, 
                        epsilon=0.99, epsilon_decay=0.99)

        # run simulation and track the returned rewards
        rewards = run_simulation(env, episodes, learner, verbose=verbose)
        
        # only track every 10th reward for cleaner plots
        param_rewards.append(rewards[::10] + [rewards[-1]])
    
    return param_rewards


def run_simulation(env, episodes, learner, render=False, verbose=False):
    """
    Runs simulation under given conditions, tracks total reward from
    each episode and returns list of those values.
    """
    reward_totals = []

    # run max specified number of episodes
    for i_episode in range(episodes):
        observation = env.reset()
        step, reward_total = 0, 0

        # grab random action using the environment, set the initial state
        rand_act = env.action_space.sample()
        action = learner.querysetstate(observation, rand_act)
        while True:
            step += 1

            # if rendering requested runs render engine
            if render:
                env.render()
            
            # take the next step specified by the learner
            observation, reward, done, info = env.step(action)
            reward_total += reward

            # check if simuluation is complete
            if done:
                learner.terminate(observation, reward)
                break

            # retrieve random action, pass on new params to learner
            rand_act = env.action_space.sample()
            action = learner.query(observation, reward, rand_act)

        # track final rewards
        reward_totals.append(reward_total)

        # if there are enough data points check for convergance
        avg_msg = ''
        if len(reward_totals) > NUM_RWD:
            moving_avg = sum(reward_totals[-NUM_RWD:]) / float(NUM_RWD)

            if verbose:
                avg_msg = ' | Rwd avg: {0:<9.3f}'.format(moving_avg)

            # average the last specified number of rewards for convergance
            if moving_avg > CONVERGE:
                break

        # log results after each episode if verbose requested
        if verbose:
            msg = 'Epsd: {0:<4d} | steps: {1:<4d} | rwd: {2:<9.3f}' + avg_msg
            print(msg.format(i_episode, step, reward_total))

    # return the rewards total gathered from each episode
    return reward_totals


if __name__ == '__main__':
    # create the Lunar Lander environment to simulate and test
    env = gym.make('LunarLander-v2')

    # # hyper param values tested
    # alphas = [.3, .1, .01, .001, .0001]
    # epsilons = [.2, .99]

    # # train the learner on the given number of episodes
    # print('\nTesting params...')
    # episodes = 2000
    # rwd_totals = param_test(env, episodes, agent.QLearner, alphas, verbose=True)
    # print('Tests complete.')

    # # plot rewards from param testing
    # for idx, y_vals in enumerate(rwd_totals):
    #     x_vals = [x for x in range(len(y_vals))]
    #     plt.plot(x_vals, y_vals, label=" = " + str(alphas[idx]))

    # # build rest of param plot
    # plt.ylim(-1000, 300)
    # plt.title('Hyper Param Reward Per Episode\nBased on different ')
    # plt.ylabel('reward')
    # plt.xlabel('episode')
    # plt.legend()
    # plt.show()

    # build the learner to utilize in the simulation
    feat_count  = env.env.observation_space.shape[0]
    act_count = env.env.action_space.n
    learner = agent.QLearner(num_feat=feat_count, num_acts=act_count, 
                             alpha=0.0001, gamma=0.99, 
                             epsilon=0.99, epsilon_decay=0.999)

    # train the learner on the given number of episodes
    print('\nTraining learner...')
    episodes = 10000
    rewards = run_simulation(env, episodes, learner, verbose=True)
    print('Training complete.')

    # plot rewards from training
    y_vals = rewards[::10] + [rewards[-1]]
    x_vals = [x for x in range(len(y_vals))]
    plt.plot(x_vals, y_vals)
    plt.ylim(-1000, 300)
    plt.title('Reward Per Training Episode \nUntil Convergance')
    plt.ylabel('reward') 
    plt.xlabel('episode')
    plt.show()

    # test the trained leaner on the specified number of episodes
    print('\nTesting learner...')
    learner.train = False
    episodes = 100
    rewards = run_simulation(env, episodes, learner, render=False)
    print('Testing complete.')

    # plot rewards from testing
    plt.plot(rewards)
    plt.ylim(-1000, 300)
    plt.title('Reward for 100 Trials with Trainer Learner')
    plt.ylabel('reward') 
    plt.xlabel('episode')
    plt.show()

    # stop the environment and end the simulation
    env.close()   
    print('\nSimulation complete.')
