import gym
import matplotlib.pyplot as plt

import agent
from simulation import run_simulation


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


if __name__ == '__main__':
    # create the Lunar Lander environment to simulate and test
    env = gym.make('LunarLander-v2')

    # hyper param values tested
    alphas = [.3, .1, .01, .001, .0001]
    # epsilons = [.2, .99]

    # train the learner on the given number of episodes
    print('\nTesting params...')
    episodes = 2000
    rwd_totals = param_test(env, episodes, agent.QLearner, alphas, verbose=True)
    print('Tests complete.')

    # plot rewards from param testing
    for idx, y_vals in enumerate(rwd_totals):
        x_vals = [x for x in range(len(y_vals))]
        plt.plot(x_vals, y_vals, label=" = " + str(alphas[idx]))

    # build rest of param plot
    plt.ylim(-1000, 300)
    plt.title('Hyper Param Reward Per Episode\nBased on different ')
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.legend()
    plt.show()

    # stop the environment and end the simulation
    env.close()
    print('\nSimulation complete.')
