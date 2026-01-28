from multiprocessing import Pool, freeze_support, cpu_count
import os
import numpy as np

if __name__ == "__main__":
    freeze_support()
        
    from B_main import launch_and_plot
    from N_MF_MB_plot import main_MF_MB, main_MF_MB_plot
    from C_MFvsMB import evaluate_agents, plot_distances, plot_Q_curves, plot_2D_maps, plot_all_rewards_curves, generate_MF_tables
    from E_envs import Gridworld, Human, Lab_env_HRI, GoToHumanVision
    from G_agents import Rmax, Epsilon_greedy_MB, Epsilon_greedy_MF
    from M_graphics import get_max_Q_values_and_policy, plot_2D, plot_1D
    from I_play_function import play
    from D_adaptibility import save_interaction, evaluate_human
    import J_constants as const


    """

    Reproducing Figures 6, 7 and 8

    """

# ---------------------------------------------------------------------------- #
# Figure 6 - Learning of the three modules by reinforcement learning agents 
# ---------------------------------------------------------------------------- #
    
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

    # 'all_data/all_imgs/1D-plots/' + str(time.time()) +'.pdf'
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

    # 'all_data/all_imgs/1D-plots/' + str(time.time()) +'.pdf'
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

    # 'all_data/all_imgs/1D-plots/' + str(time.time()) +'.pdf'
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
    


    
# ---------------------------------------------------------------------------- #
#   Figure 7: Q-values learned by a model-free agent on the navigation task
# ---------------------------------------------------------------------------- #
    
    starting_seed = 1
    nb_iters = 10

    # -------------------------------- #
    #  Get V*(s) -- run once -- done
    # -------------------------------- #


    #  for navigation task using Rmax

    nb_trials_navigation = 20000
    nb_steps_navigation = 20
    navigation_environment = Gridworld()
    navigation_agent = Rmax(navigation_environment, gamma=0.9,
                            m=1, Rmax=1, max_iterations=1, step_update=1000)
    navigation_rewards = play(navigation_environment, navigation_agent,
                            nb_trials_navigation, nb_steps_navigation)
    best_q_values_navigation, _ = get_max_Q_values_and_policy(navigation_agent.Q)
    save_dir = "all_data/some results/opt_q_values/navigation"
    os.makedirs(save_dir, exist_ok=True) 
    np.save("all_data/some results/opt_q_values/navigation/opt_V_navigation_Rmax.npy", best_q_values_navigation)
    

    # for navigation task using e-greedy MB 

    nb_trials_navigation = 20000
    nb_steps_navigation = 20
    navigation_environment = Gridworld()
    navigation_agent = Epsilon_greedy_MB(navigation_environment, gamma=0.9,
                                         epsilon = 0.05, max_iterations=1, step_update=1000)
    navigation_rewards = play(navigation_environment, navigation_agent,
                            nb_trials_navigation, nb_steps_navigation)
    best_q_values_navigation, _ = get_max_Q_values_and_policy(navigation_agent.Q)
    save_dir = "all_data/some results/opt_q_values/navigation"
    os.makedirs(save_dir, exist_ok=True) 
    np.save("all_data/some results/opt_q_values/navigation/opt_V_navigation_Epsilon_greedy_MB.npy", best_q_values_navigation)
    

    # -------------------------------------------------------- #
    #  Launch MB, MF, and MF on MB agents on navigation task
    # -------------------------------------------------------- #
    
    environments = ["navigation"]
    play_parameters = {"navigation": {'trials': 20000, 'nb_steps': 20}}

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

    play_parameters = {"navigation": {'trials': 20000, 'max_step': 20}}
    

    # -------------------------------- #
    #          Plot results
    # -------------------------------- #

    # all_data/data_MFMB/agentname_envname_i/rewards_MBMF_agent.npy
    # all_data/data_MFMB/agentname_envname_i/final_Q_MBMF_agent.npy
    generate_MF_tables(agents_with_MF, environments_name, play_parameters, starting_seed, nb_iters)
    # all_data/data_MFMB/agentname_envname_rewards_comparison.png
    plot_all_rewards_curves(agents_with_MF, environments, starting_seed, nb_iters)
    # all_data/data_MFMB/agentname_q_comparison_halfway.png
    # all_data/data_MFMB/agent_name_q_comparison.png
    plot_Q_curves(0, agents_with_MF, starting_seed, nb_iters)
    # all_data/data_MFMB/agentname_envname_i/tmp1/...
    plot_2D_maps(0, agents_with_MF, starting_seed, nb_iters)

    # all_data/data_MFMB/article/agent_name_distance_comparison_halfway.pdf
    # DEBUG
    # plot_distances(0, agents_with_MF, starting_seed, nb_iters)
    
    


# ---------------------------------------------------------------------------- #
# Figure 8 - Learning and adaptability to different human behaviors
# ---------------------------------------------------------------------------- #

    seed = 42

    # ---------------------------------------------------------------------------- #
    # Saving Q-Tables for checking adaptability to different humans
    # ---------------------------------------------------------------------------- #

    # navigation task

    np.random.seed(seed)
    nav_env = Gridworld()
    # take model that performed best
    nav_agent = Rmax(nav_env, **const.Rmax_MB_nav)
    number_of_trials = 25000
    number_of_steps = 20

    rewards = play(nav_env,
                nav_agent,
                number_of_trials,
                number_of_steps)

    save_dir = 'all_data/data/q_table'
    os.makedirs(save_dir, exist_ok=True) 
    np.save("all_data/data/q_table/nav_Q_Rmax.npy", nav_agent.Q)

    # go to human vision task
    np.random.seed(seed)
    nav_env = Gridworld()
    go_to_human_environment = GoToHumanVision(nav_env)
    # take model that performed best
    go_to_human_agent = Epsilon_greedy_MB(go_to_human_environment,
                                        **const.e_greedy_MB_param)

    number_of_trials = 10000
    number_of_steps = 20

    go_to_human_rewards = play(go_to_human_environment,
                            go_to_human_agent,
                            number_of_trials,
                            number_of_steps)

    save_dir = 'all_data/data/q_table'
    os.makedirs(save_dir, exist_ok=True) 
    np.save("all_data/data/q_table/go_Q.npy", go_to_human_agent.Q)
    

    # ---------------------------------------------------------------------------- #
    # Saving interactions with the types of human to train on
    # ---------------------------------------------------------------------------- #
    
    # Fast human -> Human 2
    name_human = 'fast_human'
    number_of_trials = 25000
    number_of_steps = 20
    # 'all_data/data/q_table/fast_human'
    save_interaction(seed,
                    name_human,
                    number_of_trials,
                    number_of_steps)
    
    # ---------------------------------------------------------------- #
    # Training on Human 2 (fast human).
    # Testing on basic human (1), fast human (2), and hard human (3).
    # ---------------------------------------------------------------- #

    name_human = 'fast_human'
    humans_to_test = ['basic_human', 'fast_human', 'hard_human']
    nb_trials = 500
    nb_steps = 100
    nb_iters = 10

    # "all_data/all_imgs/1D-plots/three_humans"+str(time.time())+".pdf"
    # to get the rewards per step plot, follow the comments in:
    # - run_and_plot_human_variability (D_adaptability.py)
    # - get_mean_and_std (M_graphics.py)
    evaluate_human(seed,
                name_human,
                humans_to_test,
                nb_trials,
                nb_steps,
                nb_iters)

    # -------------------------------------------------------------------------------- #
    # Training on Human 2 (fast human).
    # Testing on fast human (2), fast human w/ pointing needed, and basic human (1)
    # -------------------------------------------------------------------------------- #

    name_human = 'fast_human'
    humans_to_test = ['basic_human', 'pointing_need_human', 'fast_human']
    nb_trials = 1000
    nb_steps = 100
    nb_iters = 10

    # "all_data/all_imgs/1D-plots/three_humans"+str(time.time())+".pdf"
    # to get the rewards per step plot, follow the comments in:
    # - run_and_plot_human_variability (D_adaptability.py)
    # - get_mean_and_std (M_graphics.py)
    evaluate_human(seed,
                name_human,
                humans_to_test,
                nb_trials,
                nb_steps,
                nb_iters)
                