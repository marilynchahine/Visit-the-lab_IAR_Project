import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import time
import sys
from tqdm import tqdm

from environment_generation import Lab_structure
from E_envs import Lab_environment, Gridworld, Gridworld_one_distance, GoToHumanVision
from E_envs import Lab_environment_HRI, Human, Lab_HRI_evaluation
from I_play_function import play
from agents import Epsilon_greedy_MF, Rmax, Epsilon_MB_horizon
from agents import Epsilon_greedy_MB

from visualization import save_gif, save_mp4
from agents import MFLearnerOnMB
from nav_interaction import NavigationInteraction
from graphics import get_max_Q_values_and_policy, plot_2D
np.random.seed(18)


# MFlearneronMB
"""
nav_env = Gridworld()
MB_agent = Epsilon_greedy_MB(nav_env, gamma=0.9, epsilon=0.05,
                             max_iterations=1, step_update=1000)
nav_agent_1 = MFLearnerOnMB(MB_agent, gamma=0.9, alpha=0.5)
nav_agent_2 = Epsilon_greedy_MF(nav_env, gamma=0.9, alpha=0.5, epsilon=0.05)

number_of_trials = 200
number_of_steps = 20
pictures = []

nav_rewards_1_1 = play(nav_env, nav_agent_1,
                              number_of_trials, number_of_steps)

nav_rewards_2_1 = play(nav_env, nav_agent_2,
                              number_of_trials, number_of_steps)

plt.plot(nav_rewards_1_1)

"""

"""

np.save("starting_Q_table_nav.npy", nav_agent_1.Q_MF)
starting_Q = np.load("starting_Q_table_nav.npy")

nav_agent_3 = Epsilon_greedy_MF(nav_env, gamma=0.9,
                                       alpha=0.2, epsilon=0.05, initial_Q=starting_Q)

number_of_trials = 10000
number_of_steps = 20
pictures = []

nav_rewards_1_2 = play(nav_env, nav_agent_1,
                              number_of_trials, number_of_steps, pictures)


nav_rewards_2_2 = play(nav_env, nav_agent_2,
                              number_of_trials, number_of_steps, pictures)

nav_rewards_3_2 = play(nav_env, nav_agent_3,
                              number_of_trials, number_of_steps, pictures)

nav_rewards_3_1 = [0]*len(nav_rewards_3_2)

nav_rewards_1 = nav_rewards_1_1 + nav_rewards_1_2
nav_rewards_2 = nav_rewards_2_1 + nav_rewards_2_2
nav_rewards_3 = nav_rewards_3_1 + nav_rewards_3_2


# plot_rewards_agents(rewards, rewards2, nav_agent, agent2, task = 'nav')

plot_1D(nav_rewards_1, nav_agent_1, task='nav')
plot_1D(nav_rewards_2, nav_agent_2, task='nav')
plot_1D(nav_rewards_3, nav_agent_3, task='nav')

plot_rewards_agents(nav_rewards_1,
                    nav_rewards_2,
                    nav_rewards_3)


for i in range(int(nav_env.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        nav_agent_1.Q[i*nav_env.number_nav_states:(i+1)*nav_env.number_nav_states])
    plot_2D(q_values, nav_env)


for i in range(int(nav_env.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        nav_agent_3.Q[i*nav_env.number_nav_states:(i+1)*nav_env.number_nav_states])
    plot_2D(q_values, nav_env)


for i in range(int(nav_env.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        nav_agent_2.Q[i*nav_env.number_nav_states:(i+1)*nav_env.number_nav_states])
    plot_2D(q_values, nav_env)

for i in range(int(nav_env.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        nav_agent_1.Q_MF[i*nav_env.number_nav_states:(i+1)*nav_env.number_nav_states])
    plot_2D(q_values, nav_env)

"""
# MFLearneronMB, social_task


"""
nav_env = Gridworld()
human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0.5, losing_attention=0.1,
              orientation_change_rate=0.1, random_movement=0.1)  # medium human


social_environment = Lab_environment_HRI(nav_env, human)

MB_agent = Epsilon_greedy_MB(social_environment, gamma=0.9,
                             epsilon=0.05, max_iterations=50, step_update=10000)
social_agent_1 = MFLearnerOnMB(MB_agent, gamma=0.9, alpha=0.2)
social_agent_2 = Epsilon_greedy_MF(social_environment, gamma=0.9, alpha=0.2, epsilon=0.05)


# social_agent = VI_softmax(social_environment, gamma=0.9,
# tau=0.1, max_iterations=50, step_update=10000)

# social_visualization = Visualization(social_environment)
# social_visualization2 = Visualization(social_environment2)
# social_visualization3 = Visualization(social_environment3)

number_of_trials = 40000
number_of_steps = 20
pictures = []

social_rewards_1_1 = play(social_environment, social_agent_1,
                          number_of_trials, number_of_steps, pictures)


social_rewards_2_1 = play(social_environment, social_agent_2,
                          number_of_trials, number_of_steps, pictures)


np.save("starting_Q_table.npy", social_agent_1.Q_MF)
starting_Q = np.load("starting_Q_table.npy")

social_agent_3 = Epsilon_greedy_MF(social_environment, gamma=0.9,
                                   alpha=0.2, epsilon=0.05, initial_Q=starting_Q)

number_of_trials = 40000
number_of_steps = 20
pictures = []

social_rewards_1_2 = play(social_environment, social_agent_1,
                          number_of_trials, number_of_steps, pictures)


social_rewards_2_2 = play(social_environment, social_agent_2,
                          number_of_trials, number_of_steps, pictures)

social_rewards_3_2 = play(social_environment, social_agent_3,
                          number_of_trials, number_of_steps, pictures)

social_rewards_3_1 = [0]*len(social_rewards_3_2)

social_rewards_1 = social_rewards_1_1 + social_rewards_1_2
social_rewards_2 = social_rewards_2_1 + social_rewards_2_2
social_rewards_3 = social_rewards_3_1 + social_rewards_3_2

# np.save("rewards_1.npy", social_rewards_1)
# np.save("rewards_2.npy", social_rewards_2)
# np.save("rewards_3.npy", social_rewards_3)


# social_rewards3 = play(social_environment3, social_agent3, social_visualization3,
# number_of_trials, number_of_steps, pictures)

plot_rewards_agents(social_rewards_1, social_rewards_2, social_rewards_3)

basic_plot(social_rewards_1)
basic_plot(social_rewards_2)
basic_plot(social_rewards_3)

"""

"""
# Experimental Scenario

experimental_transitions = ExperimentalTransitions()
nav_env = ExperimentalScenario(experimental_transitions)
nav_agent = VI_softmax(nav_env, gamma=0.9,
                              tau=0.1, max_iterations=50, step_update=10000)

number_of_trials = 10000
number_of_steps = 20
pictures = []

rewards = play(nav_env, nav_agent,
               number_of_trials, number_of_steps, pictures)

plot_1D(rewards, nav_agent, task='nav in the Experimental Scenario')


q_table_nav = nav_agent.Q

# np.save("experiment/qvalues/q_table_nav", q_table_nav)

# for i in range(int(nav_env.max_label)+1):
#     q_values = nav_agent.Q[i*nav_env.number_nav_states:(
#         i+1)*nav_env.number_nav_states]
#     np.save("experiment/qvalues/Q_values_label_" + str(i), q_values)
# plot_2D(q_values, nav_env)

"""

# DEFS


basic_human = Human(speeds=[1, 0, 0], failing_rate=0, pointing_need=0, losing_attention=0,
                    orientation_change_rate=0.1, random_movement=0.1)
hard_human = Human(speeds=[0.33, 0.33, 0.34], failing_rate=0.5, pointing_need=0.5,
                   losing_attention=0.2, orientation_change_rate=0.3, random_movement=0.2)

fast_human = Human(speeds=[0, 0.5, 0.5], failing_rate=0.05, pointing_need=0, losing_attention=0.05,
                   orientation_change_rate=0.15, random_movement=0.05)

slow_human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0, losing_attention=0.05,
                   orientation_change_rate=0.15, random_movement=0.05)

pointing_need_human = Human(speeds=[0, 0.5, 0.5], failing_rate=0.05, pointing_need=1,
                            losing_attention=0.05, orientation_change_rate=0.15,
                            random_movement=0.05)


env_names = ['basic_gridworld', 
             'lab_env', 
             'social_basic', 
             'social_hard', 
             'social_avg',
             'go_to_h', 
             'social_slow', 
             'social_no_pointing']

environments = {'basic_gridworld': Gridworld,
                'lab_env': Lab_environment,
                'social_basic': Lab_environment_HRI,
                'social_hard': Lab_environment_HRI,
                'social_avg': Lab_environment_HRI,
                'social_no_pointing': Lab_environment_HRI,
                'social_slow': Lab_environment_HRI,
                'go_to_h': GoToHumanVision,
                }

environment_parameters = {
    'basic_gridworld': {},
    'lab_env': {'structure': Lab_structure()},
    'social_basic': {'nav_env': Gridworld(), 'human': basic_human},
    'social_hard': {'nav_env': Gridworld(), 'human': hard_human},
    'social_avg': {'nav_env': Gridworld(), 'human': fast_human},
    'social_slow': {'nav_env': Gridworld(), 'human': slow_human},
    'social_no_pointing': {'nav_env': Gridworld(), 'human': pointing_need_human},
    'go_to_h': {'nav_env': Gridworld()},
}

agent_names = ['e_greedy_MB', 'e_greedy_MF', 'Rmax_MB_nav', 'Rmax_MB_soc']

agents = {'e_greedy_MB': Epsilon_greedy_MB,
          'e_greedy_MF': Epsilon_greedy_MF,
          'Rmax_MB_nav': Rmax,
          'Rmax_MB_soc': Rmax}

agent_parameters = {
    'e_greedy_MB': {'gamma': 0.9, 'epsilon': 0.05, 'max_iterations': 1, 'step_update': 1000},
    'e_greedy_MF': {'gamma': 0.9, 'epsilon': 0.05, 'alpha': 0.5},
    'Rmax_MB_nav': {'gamma': 0.9, 'Rmax': 1, 'm': 1, 'max_iterations': 1, 'step_update': 1000},
    'Rmax_MB_soc': {'gamma': 0.9, 'Rmax': 1, 'm': 5, 'max_iterations': 1, 'step_update': 1000}}




def get_mean_and_std(dictionary, condition='agent'):
    """Compute the mean and the standard error of the mean of a dictionnary of results."""
    keys = list(dictionary.keys())
    nb_iters = max(map(lambda x: x[2], keys))  # gets the max of the iterations
    # gets all the environments
    all_env = list(np.unique(list(map(lambda x: x[0], keys))))
    # gets all the agents
    all_agents = list(np.unique(list(map(lambda x: x[1], keys))))

    if condition == 'agent':
        mean = {name_agent: np.mean([dictionary[env, name_agent, i]
                                     for i in range(nb_iters) for env in all_env], axis=0)
                for name_agent in all_agents}
        std = {name_agent: np.std([dictionary[env, name_agent, i]
                                   for i in range(nb_iters) for env in all_env], axis=0)
               for name_agent in all_agents}
    else:
        mean = {name_env: np.mean([dictionary[name_env, name_agent, i]
                                   for i in range(nb_iters) for name_agent in all_agents], axis=0)
                for name_env in all_env}
        std = {name_env: np.std([dictionary[name_env, name_agent, i]
                                 for i in range(nb_iters) for name_agent in all_agents], axis=0)
               for name_env in all_env}
    return (mean, std)


def reducing_with_batches(array, batch_numbers):
    if len(array) % batch_numbers != 0:
        print("Batches are not be equal.")
    batch_size = len(array) // batch_numbers
    return np.mean(array[:batch_size*batch_numbers].reshape(-1, batch_size), axis=1)


def extracting_results(dic_of_rewards, batches=100, condition='agent'):
    mean, std = get_mean_and_std(dic_of_rewards, condition=condition)
    reduced_mean = {name: reducing_with_batches(
        table, batch_numbers=batches) for name, table in mean.items()}
    reduced_std = {name: reducing_with_batches(
        table, batch_numbers=batches) for name, table in std.items()}

    return reduced_mean, reduced_std


def plot_curves(means, stds, all_agent_steps, condition='agent'):
    """Plot the results"""

    colors = {'e_greedy_MB': 'tab:blue',
              'softmax_MB': "#ffac1e",
              'softmax_MF': "#ff7763",
              'e_greedy_MF': 'tab:green',
              'Rmax_MB_nav': 'tab:orange',
              'Rmax_MB_soc': 'tab:orange',
              'MF_on_MB': 'tab:gray',
              'social_basic': 'tab:blue',
              'social_avg': 'tab:orange',
              'social_hard': 'tab:green',
              'social_slow': 'tab:blue',
              'social_no_pointing': 'tab_green'}

    fig = plt.figure(dpi=300)
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(linestyle='--')

    condition_tested = list(means.keys())
    # length_trial = len(means[agents_tested[0]])
    # x_range = [i * total_steps/length_trial for i in range(length_trial)]
    for idx_cond, name_cond in enumerate(condition_tested):

        length_trial = len(means[name_cond])
        x_range = [(i+1/2) * all_agent_steps[name_cond] /
                   length_trial for i in range(length_trial)]
        if name_cond == 'MF_on_MB':
            shift = 20000*20
            x_range = np.array(x_range)+shift
            plt.axvline(shift, color=colors[name_cond], linestyle='--')
        yerr0 = means[name_cond] - stds[name_cond]
        yerr1 = means[name_cond] + stds[name_cond]

        plt.fill_between(x_range, yerr0, yerr1,
                         color=colors[name_cond], alpha=0.25)

        plt.plot(x_range, means[name_cond],
                 color=colors[name_cond], label=name_cond)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.savefig('all_plots/' + str(time.time())+' .pdf', bbox_inches='tight')


# nav task


# agent_to_test = ['Rmax_MB_nav', 'e_greedy_MB', 'e_greedy_MF']
# play_parameters = {'trials': 15000, 'max_step': 20}
# environment_to_test = ['basic_gridworld']
# nb_tests = 1
# rewards = main_function(agent_to_test, environment_to_test, nb_tests, play_parameters, 1)

# mean, std = extracting_results(rewards)

# total_steps = {agent_name: play_parameters['trials'] *
#                play_parameters['max_step'] for agent_name in agent_to_test}

# plot_curves(mean, std, total_steps)
# Social task
"""
agent_to_test = ['Rmax_MB_soc', 'e_greedy_MB', 'e_greedy_MF']
play_parameters = {'trials': 100000, 'max_step': 20}
environment_to_test = ['social_basic']
nb_tests = 10
rewards = main_function(agent_to_test, environment_to_test, nb_tests, play_parameters, 1)

mean, std = extracting_results(rewards)

total_steps = {agent_name: play_parameters['trials'] *
               play_parameters['max_step'] for agent_name in agent_to_test}

plot_curves(mean, std, total_steps)

"""

# Social task only MF
"""
agent_to_test = ['e_greedy_MF']
play_parameters = {'trials': 200000, 'max_step': 20}
environment_to_test = ['social_env_basic']
nb_tests = 10
rewards_MF = main_function(agent_to_test, environment_to_test, nb_tests, play_parameters, 1)

mean, std = extracting_results(rewards_MF)

total_steps = {agent_name: play_parameters['trials'] *
               play_parameters['max_step'] for agent_name in agent_to_test}

plot_curves(mean, std, total_steps)
"""
# Go to Human vision
"""
agent_to_test = ['e_greedy_MB', 'e_greedy_MF', 'Rmax_MB_soc']
play_parameters = {'trials': 6000, 'max_step': 20}
environment_to_test = ['go_to_h']
nb_tests = 10
rewards = main_function(agent_to_test, environment_to_test, nb_tests, play_parameters, 1)
mean, std = extracting_results(rewards)

total_steps = {agent_name: play_parameters['trials'] *
               play_parameters['max_step'] for agent_name in agent_to_test}

plot_curves(mean, std, total_steps)
"""

# different_humans
"""
agent_to_test = ['e_greedy_MB']
environment_to_test = ['social_basic', 'social_avg', 'social_hard']
play_parameters = {'trials': 20000, 'max_step': 20}
nb_tests = 10
# rewards = main_function(agent_to_test, environment_to_test, nb_tests, play_parameters, 1)
rewards = np.load(
    "/home/augustin/Bureau/HRI-Lab-experiment/all_rewards/three_humans.npy", allow_pickle=True).item()
mean, std = extracting_results(rewards, condition='social')

total_steps = {name_cond: play_parameters['trials'] *
               play_parameters['max_step'] for name_cond in environment_to_test}

plot_curves(mean, std, total_steps)

"""

# nav Gridworld

"""
nav_env = Gridworld()
nav_agent = Rmax(nav_env, gamma=0.9, m=1, Rmax=1,
                        max_iterations=1, step_update=1000)

number_of_trials = 20000
number_of_steps = 20
pictures = []

rewards = play(nav_env, nav_agent,
               number_of_trials, number_of_steps, pictures, "nav")

nav_Q = nav_agent.Q


np.save("data/article/nav_Q_Rmax.npy", nav_Q)

for i in range(int(nav_env.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        nav_agent.Q[i*nav_env.number_nav_states:(i+1)*nav_env.number_nav_states])
    plot_2D(q_values, nav_env)


# plotting q tables

# for i in range(int(nav_env.max_label)+1):
# q_values, best_actions = get_max_Q_values_and_policy(
# nav_agent.Q[i*nav_env.number_nav_states:(i+1)*nav_env.number_nav_states])
# plot_2D(q_values, nav_env)


"""
# Interaction

"""
fast_human_2 = Human(speeds=[0, 0, 1], failing_rate=0, pointing_need=0, losing_attention=0,
                     orientation_change_rate=0.1, random_movement=0.1)

np.random.seed(12)
nav_env = Gridworld(size=60)
human = basic_human
social_environment = Lab_environment_HRI(nav_env, human)
social_agent = Epsilon_greedy_MB(social_environment, epsilon=0.05,
                                 gamma=0.9, max_iterations=1, step_update=1000)

number_of_trials = 20000
number_of_steps = 15
pictures = []

social_rewards = play(social_environment, social_agent,
                      number_of_trials, number_of_steps, pictures, 'social')
plt.plot(social_rewards)


# save_gif()
# save_mp4("visualization.gif")


np.save("data/article/speed_1/soc_Q.npy", social_agent.Q)
np.save("data/article/speed_1/soc_R.npy", social_agent.R)
np.save("data/article/speed_1/soc_T.npy", social_agent.tSAS)
np.save("data/article/speed_1/soc_nSA.npy", social_agent.nSA)
np.save("data/article/speed_1/soc_nSAS.npy", social_agent.nSAS)
np.save("data/article/speed_1/soc_Rsum.npy", social_agent.Rsum)

"""

# np.save("data/article/horizon/nSA_horizon.npy", social_agent.nSA_horizon)
# np.save("data/article/horizon/R_horizon.npy", social_agent.R_horizon)


"""

# print(q_table_social)
# print(q_table_social.ndim)
# print(len(q_table_social))


# Go to Human
nav_env = Gridworld()
go_to_human_environment = GoToHumanVision(nav_env)
go_to_human_agent = Rmax(go_to_human_environment, m=5, Rmax=1,
                         gamma=0.9, max_iterations=1, step_update=1000)

number_of_trials = 5000
number_of_steps = 20
pictures = []
go_to_human_rewards = play(go_to_human_environment, go_to_human_agent,
                           number_of_trials, number_of_steps, pictures)

# plot_1D_avg(go_to_human_rewards, go_to_human_agent)


q_table_go_to_human = go_to_human_agent.Q

np.save("data/article/go_Q_Rmax.npy", q_table_go_to_human)

"""

# Final evaluation

"""
np.random.seed(12)
q_table_nav = np.load("data/article/nav_Q_Rmax.npy")
q_table_social = np.load("data/article/soc_Q.npy")
q_table_go_to_human = np.load("data/article/go_Q.npy")

social_tSAS = np.load("data/article/soc_T.npy")
social_R = np.load("data/article/soc_R.npy")
social_nSA = np.load("data/article/soc_nSA.npy")
social_nSAS = np.load("data/article/soc_nSAS.npy")
social_Rsum = np.load("data/article/soc_Rsum.npy")

all_rewards = [[], [], []]
for idx, human in enumerate([human1, human2, human3]):
    np.random.seed(13)
    nav_env = Gridworld()
    social_environment = Lab_environment_HRI(nav_env, human)
    environment = Lab_HRI_evaluation(nav_env, human)

    learning_agent = Epsilon_greedy_MB(social_environment, gamma=0.9, epsilon=0.05,
                                       max_iterations=1, step_update=1000)

    agent = navInteraction(environment,
                                  q_table_nav,
                                  q_table_social,
                                  q_table_go_to_human,
                                  learning_agent,
                                  social_R,
                                  social_tSAS,
                                  social_nSA,
                                  social_nSAS,
                                  social_Rsum)
    number_of_trials = 300
    number_of_steps = 50

    pictures = []
    rewards = play(environment, agent, number_of_trials, number_of_steps, pictures)
    all_rewards[idx].append(rewards)
    # save_gif(gif_name="visualisation"+str(idx)+".gif")


"""


# Comparing three humans


def get_moving_average(rewards, avg=20):
    index_avg = [i * len(rewards) // avg for i in range(avg)]
    moving_average_rewards = [np.mean(rewards[index_avg[index]:index_avg[index + 1]])
                              for index in range(len(index_avg) - 1)]
    return index_avg, moving_average_rewards


def plot_different_humans(all_rewards):

    plt.figure(dpi=300)
    colors = {1: 'tab:blue', 0: 'tab:orange', 2: 'tab:gray'}
    # colors = {0: 'tab:blue', 1: 'tab:green', 2: 'tab:red'}
    labels = {0: 'Speed 3', 2: 'Speed 1',
              1: 'Speed 2'}
    for idx_reward in range(len(all_rewards)):
        reward = all_rewards[idx_reward]
        mean_reward, std_reward = np.mean(
            reward, axis=0), np.std(reward, axis=0)
        index_avg, avg_rewards = get_moving_average(mean_reward, avg=50)
        _, avg_std = get_moving_average(std_reward, avg=50)

        plt.plot(index_avg[:-1], avg_rewards,
                 color=colors[idx_reward], label=labels[idx_reward])

        plt.fill_between(index_avg[:-1], np.array(avg_rewards) - np.array(avg_std),
                         np.array(avg_rewards) + np.array(avg_std),
                         color=colors[idx_reward],
                         alpha=0.2)
    plt.ylabel('#Goals reached', fontsize=12)
    plt.xlabel('#Trials (Steps x100)', fontsize=12)
    # plt.ylim(-0.5, 13.9)
    plt.legend(loc='upper left')
    plt.grid(linestyle='--')
    plt.savefig('all_plots/'+str(time.time())+'.pdf', bbox_inches='tight')
    plt.show()


np.random.seed(12)
q_table_nav = np.load("data/article/original/nav_Q_Rmax.npy")
q_table_go_to_human = np.load("data/article/original/go_Q.npy")


q_table_social = np.load("data/article/speed_1/soc_Q.npy")

# q_table_social_basic = np.load("data/article/horizon/soc_Q.npy")
# print(sum(np.argmax(q_table_social, axis=1) == np.argmax(q_table_social_horizon, axis=1)))

# social_R = np.load("data/article/horizon/soc_R.npy")
# social_tSAS = np.load("data/article/horizon/soc_T.npy")
# social_nSA = np.load("data/article/horizon/soc_nSA.npy")
# social_nSAS = np.load("data/article/horizon/soc_nSAS.npy")
# social_Rsum = np.load("data/article/horizon/soc_Rsum.npy")
# social_nSA_h = np.load("data/article/horizon/nSA_horizon.npy")
# social_R_h = np.load("data/article/horizon/R_horizon.npy")

social_R = np.load("data/article/speed_1/soc_R.npy")
social_tSAS = np.load("data/article/speed_1/soc_T.npy")
social_nSA = np.load("data/article/speed_1/soc_nSA.npy")
social_nSAS = np.load("data/article/speed_1/soc_nSAS.npy")
social_Rsum = np.load("data/article/speed_1/soc_Rsum.npy")


fast_human_2 = Human(speeds=[0, 0, 1], failing_rate=0, pointing_need=0, losing_attention=0,
                     orientation_change_rate=0.1, random_movement=0.1)
mid_human_2 = Human(speeds=[0, 1, 0], failing_rate=0, pointing_need=1, losing_attention=0,
                    orientation_change_rate=0.1, random_movement=0.1)


all_rewards = [[], [], []]
for i in tqdm(range(10)):
    # [fast_human, basic_human, pointing_need_human]
    for idx, human in enumerate([fast_human_2, mid_human_2, basic_human]):
        nav_env = Gridworld()
        social_environment = Lab_environment_HRI(nav_env, human)
        environment = Lab_HRI_evaluation(nav_env, human)

        learning_agent = Epsilon_greedy_MB(social_environment, gamma=0.9,
                                           epsilon=0.05, max_iterations=1, step_update=1000)

        agent = NavigationInteraction(environment,
                                      q_table_nav,
                                      q_table_social,
                                      q_table_go_to_human,
                                      learning_agent,
                                      social_R,
                                      social_tSAS,
                                      social_nSA,
                                      social_nSAS,
                                      social_Rsum)
        number_of_trials = 500
        number_of_steps = 100

        pictures = []
        rewards = play(environment, agent, number_of_trials,
                       number_of_steps, pictures)
        all_rewards[idx].append(rewards)
np.save("all_rewards/rewards_3_agents_test" +
        str(time.time())+".npy", all_rewards)
plot_different_humans(all_rewards)


"""
human = basic_human

environment_n = 'social_basic'
agent_n = 'MF_on_MB'
play_parameters = {'trials': 80000, 'max_step': 20}
nb_tests = 10


def get_q_table_MF():
    rewards = {}
    for i in range(nb_tests):
        nav_env = Gridworld()
        environment = Lab_environment_HRI(nav_env, human)
        MB_agent = Epsilon_greedy_MB(environment, gamma=0.9,
                                     epsilon=0.05, max_iterations=1, step_update=1000)
        agent = MFLearnerOnMB(MB_agent, gamma=0.9, alpha=0.5)
        reward = play(environment, agent, **play_parameters)
        np.save('data/article/agent.Q_MF_'+str(i)+'_.npy', agent.Q_MF)

# get_q_table_MF()


def get_MF_performance_with_q_tables():
    rewards = {}
    for i in range(nb_tests):
        nav_env = Gridworld()
        environment = Lab_environment_HRI(nav_env, human)
        init_Q = np.load('data/article/original/agent.Q_MF_'+str(i)+'_.npy')
        agent = Epsilon_greedy_MF(environment, gamma=0.9,
                                  epsilon=0.05, alpha=0.5, initial_Q=init_Q)
        reward = play(environment, agent, **play_parameters)
        trial_name = (environment_n, agent_n, i)
        rewards[(trial_name)] = reward
    np.save('all_rewards/reward_MF_on_MB'+str(time.time())+'.npy', rewards)


# get_MF_performance_with_q_tables()
"""

# re-use savved data
"""
agent_to_test = ['MF_on_MB', 'e_greedy_MB', 'e_greedy_MF']
play_parameters = {'trials': 100000, 'max_step': 20}
environment_to_test = ['social_basic']
nb_tests = 10
total_steps = {agent_name: play_parameters['trials'] *
               play_parameters['max_step'] for agent_name in agent_to_test}

total_steps['MF_on_MB'] = 80000*20

mean, std = extracting_results(all_rewards, batches=100)


plot_curves(mean, std, total_steps)
"""

"""


def main():
    agent_names = ['e_greedy_MB', 'softmax_MB', 'softmax_MF',
                   'e_greedy_MF', 'Rmax_MB_nav', 'Rmax_MB_soc']
    env_names = ['basic_gridworld', 'lab_env', 'social_env_basic', 'social_env_hard', 'go_to_h']

    # nav task
    play_parameters = {'trials': 100, 'max_step': 20}

    agent_to_test = ['e_greedy_MB', 'e_greedy_MF', 'Rmax_MB_nav']
    # agent_to_test = ['e_greedy_MB']

    environment_to_test = ['basic_gridworld']
    nb_tests = 10
    rewards = main_function(agent_to_test, environment_to_test, nb_tests, play_parameters, 1)
    mean, std = extracting_results(rewards)
    total_steps = play_parameters['trials']*play_parameters['max_step']
    plot_curves(mean, std, total_steps)


if __name__ == '__main__':
    sys.exit(main())
"""
