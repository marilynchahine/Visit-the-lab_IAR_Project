from multiprocessing import Pool, freeze_support, cpu_count
import os
import numpy as np

if __name__ == "__main__":
    freeze_support()
        
    from B_main import launch_and_plot
    from N_MF_MB_plot import main_MF_MB, main_MF_MB_plot
    from C_MFvsMB import evaluate_agents, plot_distances, plot_Q_curves, plot_2D_maps, plot_all_rewards_curves, generate_MF_tables
    from E_envs import Gridworld, Human, Lab_env_HRI
    from G_agents import Rmax, Epsilon_greedy_MB, Epsilon_greedy_MF
    from M_graphics import get_max_Q_values_and_policy, plot_2D, plot_1D
    from I_play_function import play


    """

    PART 1: Reproducing Figures 6, 7 and 8

    PART 2: Exploring MF bootstrapping parameters that could improve performance

    PART 3: Exploring multi-model options for human variability

    """



# -------------------------------------------------------------------------------------------------- #
#                               PART 1: Reproducing Figures 6, 7, and 8
# -------------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Figure 6 - Learning of the three modules by reinforcement learning agents 
# ---------------------------------------------------------------------------- #
    """
# -------------------------------- #
# Navigation Task
# -------------------------------- #

    agent = ['Rmax_MB_nav', 'e_greedy_MB', 'e_greedy_MF']
    env = ['gridworld']
    nb_iters = 10
    play_params = {'trials': 25000, 'max_step': 20}
    start_seed = 1
    proc = 2
    cond = 'agent'

    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)

    
# -------------------------------- #
# 'Go to human vision' Task
# -------------------------------- #

    agent = ['Rmax_MB_soc', 'e_greedy_MB', 'e_greedy_MF']
    env = ['go_to_h']
    nb_iters = 10
    play_params = {'trials': 10000, 'max_step': 20}
    start_seed = 3
    proc = 2
    cond = 'agent'

    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)
    

# -------------------------------- #
# Social Task
# -------------------------------- #

    agent = ['Rmax_MB_soc', 'e_greedy_MB', 'e_greedy_MF']
    env = ['social_basic']
    nb_iters = 10
    play_params = {'trials': 100000, 'max_step': 20}
    start_seed = 2
    proc = 2
    cond = 'agent'

    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)
    

# ----------------------------------------------------------------- #
# Social Task with Model-Free bootstrapped on e-greedy Model-Based
# ----------------------------------------------------------------- #

    # in C_MF_MB_plot, change MB model to bootstrap MF on before running

    nb_iters = 10
    main_MF_MB(nb_iters)
    main_MF_MB_plot(nb_iters)
    """




    # ---------------------------------------------------------------------------- #
    #   Figure 7: Q-values learned by a model-free agent on the navigation task
    # ---------------------------------------------------------------------------- #
    
    starting_seed = 1
    nb_iters = 10

    # Get V*(s) for navigation task using Rmax -- run once -- done

    """
    nb_trials_navigation = 20000
    nb_steps_navigation = 20
    navigation_environment = Gridworld()
    navigation_agent = Rmax(navigation_environment, gamma=0.9,
                            m=1, Rmax=1, max_iterations=1, step_update=1000)
    navigation_rewards = play(navigation_environment, navigation_agent,
                            nb_trials_navigation, nb_steps_navigation)
    best_q_values_navigation, _ = get_max_Q_values_and_policy(navigation_agent.Q)
    save_dir = "some results/opt_q_values/navigation"
    os.makedirs(save_dir, exist_ok=True) 
    np.save("some results/opt_q_values/navigation/opt_V_navigation_Rmax.npy", best_q_values_navigation)
    """

    # Get V*(s) for navigation task using e-greedy MB -- run once -- done

    """
    nb_trials_navigation = 20000
    nb_steps_navigation = 20
    navigation_environment = Gridworld()
    navigation_agent = Epsilon_greedy_MB(navigation_environment, gamma=0.9,
                                         epsilon = 0.05, max_iterations=1, step_update=1000)
    navigation_rewards = play(navigation_environment, navigation_agent,
                            nb_trials_navigation, nb_steps_navigation)
    best_q_values_navigation, _ = get_max_Q_values_and_policy(navigation_agent.Q)
    save_dir = "some results/opt_q_values/navigation"
    os.makedirs(save_dir, exist_ok=True) 
    np.save("some results/opt_q_values/navigation/opt_V_navigation_Epsilon_greedy_MB.npy", best_q_values_navigation)
    """

    # Launch MFonMB agents and generate plots

    environments = ["navigation"]
    play_parameters = {"navigation": {'trials': 20000, 'max_step': 20}}

    agents = ['Rmax', 'Epsilon_greedy_MB', 'Epsilon_greedy_MF']

    agent_parameters = {
        'Epsilon_greedy_MB': {'gamma': 0.9, 'epsilon': 0.05, 'max_iterations': 1, 'step_update': 1000},
        'Rmax': {'gamma': 0.9, 'm': 1, 'Rmax': 1, 'max_iterations': 1, 'step_update': 1000},
        'Epsilon_greedy_MF': {'gamma': 0.9, 'epsilon': 0.05, 'alpha': 0.5}}
    
    agent_names = {'Rmax': Rmax,
               'Epsilon_greedy_MB': Epsilon_greedy_MB,
               'Epsilon_greedy_MF': Epsilon_greedy_MF}

    rewards, times = evaluate_agents(environments, agents, nb_iters, play_parameters, agent_parameters, starting_seed, agent_names)

    agents_with_MF = ['Rmax', 'Epsilon_greedy_MB']

    navigation_environment = Gridworld()
    environments_name = {"navigation": navigation_environment}
           
    generate_MF_tables(agents_with_MF, environments_name, play_parameters, starting_seed, nb_iters)
    plot_all_rewards_curves(agents_with_MF, environments, starting_seed, nb_iters)
    plot_Q_curves(0, agents_with_MF, starting_seed, nb_iters)
    plot_2D_maps(0, agents_with_MF, starting_seed, nb_iters)
    plot_distances(0, agents_with_MF, starting_seed, nb_iters)




# ---------------------------------------------------------------------------- #
# Figure 8 - Learning and adaptability to different human behaviors
# ---------------------------------------------------------------------------- #

# -------------------------------- #
# Social Task with 3 Humans
# -------------------------------- #

# -------------------------------- #
# Visit the Lab Task
# -------------------------------- #


# -------------------------------------------------------------------------------------------------- #
#            PART 2: Exploring MF bootstrapping parameters that could improve performance
# -------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------- #
#                    PART 3: Exploring multi-model options for human variability
# -------------------------------------------------------------------------------------------------- #






    # -------------------------------- #
    # TO DEBUG - FIG 7 ON Social Task
    # -------------------------------- #

    # Get V*(s) for social task using Rmax -- run once

    """ TO DEBUG
    nb_trials_social = 200000
    nb_steps_social = 20

    navigation_environment = Gridworld()
    human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0.5, losing_attention=0.05,
                orientation_change_rate=0.2, random_movement=0.1)
    social_environment = Lab_env_HRI(navigation_environment, human)

    social_agent = Rmax(social_environment, gamma=0.9,
                            m=1, Rmax=1, max_iterations=1, step_update=1000)
    
    social_rewards = play(social_environment, social_agent, nb_trials_social, nb_steps_social)

    best_q_values_social, _ = get_max_Q_values_and_policy(social_agent.Q)

    save_dir = "some results/opt_q_values/social"
    os.makedirs(save_dir, exist_ok=True) 
    np.save("some results/opt_q_values/social/opt_V_social_Rmax.npy", best_q_values_social)


    # Get V*(s) for social task using e-greedy MB -- run once

    nb_trials_social = 200000
    nb_steps_social = 20

    navigation_environment = Gridworld()
    human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0.5, losing_attention=0.05,
                orientation_change_rate=0.2, random_movement=0.1)
    social_environment = Lab_env_HRI(navigation_environment, human)

    social_agent = Epsilon_greedy_MB(social_environment, gamma=0.9,
                                    epsilon=0.1, max_iterations=1, step_update=1000)
    
    social_rewards = play(social_environment, social_agent, nb_trials_social, nb_steps_social)

    best_q_values_social, _ = get_max_Q_values_and_policy(social_agent.Q)
    save_dir = "some results/opt_q_values/social"
    os.makedirs(save_dir, exist_ok=True) 
    np.save("some results/opt_q_values/social/opt_V_social_Epsilon_greedy_MB.npy", best_q_values_social)

    
    # evaluate agents and produce plots

    environments = ["social"]

    # agents = ['Rmax', 'Epsilon_greedy_MB', 'Epsilon_greedy_MF']
    agents = ['Rmax']

    play_parameters = {"social": {'trials': 40000, 'max_step': 20}}

    agent_parameters = {
        'Epsilon_greedy_MB': {'gamma': 0.9, 'epsilon': 0.05, 'max_iterations': 1, 'step_update': 1000},
        'Rmax': {'gamma': 0.9, 'm': 1, 'Rmax': 1, 'max_iterations': 1, 'step_update': 1000},
        'Epsilon_greedy_MF': {'gamma': 0.9, 'epsilon': 0.05, 'alpha': 0.5}}

    
    agent_names = {'Rmax': Rmax,
               'Epsilon_greedy_MB': Epsilon_greedy_MB,
               'Epsilon_greedy_MF': Epsilon_greedy_MF}

    # rewards, times = evaluate_agents(environments, agents, nb_iters, play_parameters, agent_parameters, starting_seed, agent_names)

    # agents_with_MF = ['Rmax', 'Epsilon_greedy_MB']
    agents_with_MF = ['Rmax']

    human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0.5, losing_attention=0.05,
                orientation_change_rate=0.2, random_movement=0.1)
    social_environment = Lab_env_HRI(navigation_environment, human)
    environments_name = {"social": social_environment}
                     
    generate_MF_tables(agents_with_MF, environments_name, play_parameters, starting_seed, nb_iters)
    plot_all_rewards_curves(agents_with_MF, environments, starting_seed, nb_iters)
    # requires debuging V*(s) above
    plot_Q_curves(1, agents_with_MF, starting_seed, nb_iters)
    plot_2D_maps(1, agents_with_MF, starting_seed, nb_iters)
    plot_distances(1, agents_with_MF, starting_seed, nb_iters)
    """