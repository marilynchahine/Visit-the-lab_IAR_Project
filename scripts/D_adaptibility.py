from tqdm import tqdm
from E_envs import Gridworld, Human, Lab_env_HRI, SocialGridworld, GoToHumanVision
from E_envs import Lab_HRI_evaluation
from H_nav_interaction import NavigationInteraction
from G_agents import Rmax, Epsilon_greedy_MB
from I_play_function import play
import J_constants as const
import K_variables as var
import numpy as np
import time
from M_graphics import plot_different_humans

import os

# ---------------------------------------------------------------------------- #
# Saving Q-Tables for checking adaptability to different humans
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Navigation
# ---------------------------------------------------------------------------- #

# np.random.seed(1)
# nav_env = Gridworld(size=24)
# nav_agent = Rmax(nav_env, **const.Rmax_MB_nav)
# number_of_trials = 25000
# number_of_steps = 20

# rewards = play(nav_env,
#                nav_agent,
#                number_of_trials,
#                number_of_steps)

# np.save("../data/q_table/nav_Q_Rmax.npy", nav_agent.Q)

# ---------------------------------------------------------------------------- #
# Go to human
# ---------------------------------------------------------------------------- #
# np.random.seed(2)
# nav_env = Gridworld()
# go_to_human_environment = GoToHumanVision(nav_env)
# go_to_human_agent = Epsilon_greedy_MB(go_to_human_environment,
#                                       **const.e_greedy_MB_param)

# number_of_trials = 10000
# number_of_steps = 20

# go_to_human_rewards = play(go_to_human_environment,
#                            go_to_human_agent,
#                            number_of_trials,
#                            number_of_steps)


# np.save("../data/q_table/go_Q.npy", go_to_human_agent.Q)

# ---------------------------------------------------------------------------- #
# Function to measure adaptibility to human variability
# ---------------------------------------------------------------------------- #

def run_and_plot_human_variability(nb_iters,
                                   nb_trials,
                                   nb_steps,
                                   human_names,
                                   init_nav,
                                   init_go,
                                   init_soc,
                                   title_plot):
    all_rewards = {human_name: [] for human_name in human_names}
    for i in range(nb_iters):
        for human_name in human_names:
            nav_env = Gridworld()
            human = Human(**var.human_parameters[human_name])
            social_environment = Lab_env_HRI(nav_env, human)
            environment = Lab_HRI_evaluation(nav_env, human)

            navigation_agent = Epsilon_greedy_MB(
                social_environment,
                **const.e_greedy_MB_no_explo_param)

            go_to_human_agent = Epsilon_greedy_MB(
                social_environment,
                **const.e_greedy_MB_no_explo_param)

            e_greedy_MB_param = {'gamma': 0.9,
                                 'epsilon': 0.05,
                                 'max_iterations': 1,
                                 'step_update': 100}

            social_agent = Epsilon_greedy_MB(
                social_environment,
                **e_greedy_MB_param)

            agent = NavigationInteraction(
                environment,
                navigation_agent,
                social_agent,
                go_to_human_agent,
                init_nav,
                init_soc,
                init_go
            )
            number_of_trials = nb_trials
            number_of_steps = nb_steps

            visual = {'render': False,
                      'trial': [],
                      'task_type': "HRI"}

            rewards = play(environment,
                           agent,
                           number_of_trials,
                           number_of_steps,
                           visual)
            all_rewards[human_name].append(rewards)
        
    # ADDED create directory to save results in
    save_dir = '../data/all_rewards/'
    os.makedirs(save_dir, exist_ok=True) 

    np.save("../data/all_rewards/" + str(time.time())+".npy",
            all_rewards)

    plot_different_humans(all_rewards, title_plot)


# ---------------------------------------------------------------------------- #
# Save Interaction Table
# ---------------------------------------------------------------------------- #


def save_interaction(random_seed,
                     name_human,
                     number_of_trials=20000,
                     number_of_steps=20):

    np.random.seed(random_seed)

    human = Human(**var.human_parameters[name_human])

    nav_env = SocialGridworld(size=120)

    social_environment = Lab_env_HRI(nav_env, human)
    social_agent = Epsilon_greedy_MB(social_environment,
                                     **const.e_greedy_MB_param)

    social_rewards = play(social_environment,
                          social_agent,
                          number_of_trials,
                          number_of_steps)

    # ADDED create directory to save results in
    save_dir = '../data/q_table/'
    os.makedirs(save_dir, exist_ok=True) 

    np.save("../data/q_table/"+name_human+"/soc_Q.npy", social_agent.Q)
    np.save("../data/q_table/"+name_human+"/soc_R.npy", social_agent.R)
    np.save("../data/q_table/"+name_human+"/soc_T.npy", social_agent.tSAS)
    np.save("../data/q_table/"+name_human+"/soc_nSA.npy", social_agent.nSA)
    np.save("../data/q_table/"+name_human+"/soc_nSAS.npy", social_agent.nSAS)
    np.save("../data/q_table/"+name_human+"/soc_Rsum.npy", social_agent.Rsum)
    # np.save("data/article/horizon/nSA_horizon.npy", social_agent.nSA_horizon)
    # np.save("data/article/horizon/R_horizon.npy", social_agent.R_horizon)


# ---------------------------------------------------------------------------- #
# Evaluate Interaction Table
# ---------------------------------------------------------------------------- #


def evaluate_human(random_seed,
                   name_human,
                   humans_to_test,
                   nb_trials=300,
                   nb_steps=100,
                   nb_iters=10):

    np.random.seed(random_seed)
    q_table_nav = np.load("../data/q_table/nav_Q_Rmax.npy")
    q_table_go_to_human = np.load("../data/q_table/go_Q.npy")
    q_table_social = np.load("../data/q_table/"+name_human+"/soc_Q.npy")

    social_R = np.load("../data/q_table/"+name_human+"/soc_R.npy")
    social_tSAS = np.load("../data/q_table/"+name_human+"/soc_T.npy")
    social_nSA = np.load("../data/q_table/"+name_human+"/soc_nSA.npy")
    social_nSAS = np.load("../data/q_table/"+name_human+"/soc_nSAS.npy")
    social_Rsum = np.load("../data/q_table/"+name_human+"/soc_Rsum.npy")

    init_nav = {'Q': q_table_nav}
    init_go = {'Q': q_table_go_to_human}
    init_soc = {'Q': q_table_social,
                'R': social_R,
                'tSAS': social_tSAS,
                'nSA': social_nSA,
                'nSAS': social_nSAS,
                'Rsum': social_Rsum}

    title = 'Trained with '+name_human
    run_and_plot_human_variability(nb_iters,
                                   nb_trials,
                                   nb_steps,
                                   humans_to_test,
                                   init_nav,
                                   init_go,
                                   init_soc, title_plot=title)


# ---------------------------------------------------------------------------- #
# Save Fast Human
# ---------------------------------------------------------------------------- #

seed = 3
name_human = 'fast_human'
number_of_trials = 25000
number_of_steps = 20
save_interaction(seed,
                 name_human,
                 number_of_trials,
                 number_of_steps)

# ---------------------------------------------------------------------------- #
# Save Basic Human
# ---------------------------------------------------------------------------- #

seed = 4
name_human = 'basic_human'
number_of_trials = 25000
number_of_steps = 20
save_interaction(seed,
                 name_human,
                 number_of_trials,
                 number_of_steps)

# ---------------------------------------------------------------------------- #
# Save Basic Human speed 3
# ---------------------------------------------------------------------------- #

seed = 5
name_human = 'basic_human_speed_3'
number_of_trials = 25000
number_of_steps = 20
save_interaction(seed,
                 name_human,
                 number_of_trials,
                 number_of_steps)

# ---------------------------------------------------------------------------- #
# Save Basic Human speed random
# ---------------------------------------------------------------------------- #

seed = 6
name_human = 'basic_human_speed_random'
number_of_trials = 25000
number_of_steps = 20
save_interaction(seed,
                 name_human,
                 number_of_trials,
                 number_of_steps)


# ---------------------------------------------------------------------------- #
# Fast Human - Fast, Basic, pointing_need
# ---------------------------------------------------------------------------- #
seed = 7
name_human = 'fast_human'
humans_to_test = ['basic_human', 'pointing_need_human', 'fast_human']
nb_trials = 500
nb_steps = 100
nb_iters = 10

evaluate_human(seed,
               name_human,
               humans_to_test,
               nb_trials,
               nb_steps,
               nb_iters)

# ---------------------------------------------------------------------------- #
# Basic Human - Basic_1, Basic_2, Basic_3
# ---------------------------------------------------------------------------- #

seed = 8
name_human = 'basic_human'
humans_to_test = ['basic_human', 'basic_human_speed_2', 'basic_human_speed_3']
nb_trials = 1000
nb_steps = 100
nb_iters = 10

evaluate_human(seed,
               name_human,
               humans_to_test,
               nb_trials,
               nb_steps,
               nb_iters)

# ---------------------------------------------------------------------------- #
# Basic_3 Human - Basic_1, Basic_2, Basic_3
# ---------------------------------------------------------------------------- #

seed = 9
name_human = 'basic_human_speed_3'
humans_to_test = ['basic_human', 'basic_human_speed_2', 'basic_human_speed_3']
nb_trials = 1000
nb_steps = 100
nb_iters = 10

evaluate_human(seed,
               name_human,
               humans_to_test,
               nb_trials,
               nb_steps,
               nb_iters)


# ---------------------------------------------------------------------------- #
# Random Speed Basic Human -  Basic_1, Basic_2, Basic_3
# ---------------------------------------------------------------------------- #

seed = 10
name_human = 'basic_human_speed_random'
humans_to_test = ['basic_human', 'basic_human_speed_2', 'basic_human_speed_3']
nb_trials = 1000
nb_steps = 100
nb_iters = 10

evaluate_human(seed,
               name_human,
               humans_to_test,
               nb_trials,
               nb_steps,
               nb_iters)
