from multiprocessing import Pool, freeze_support, cpu_count
import os
import numpy as np
import time

from D_adaptibility import save_interaction, evaluate_human
from E_envs import Gridworld, GoToHumanVision, Lab_HRI_evaluation, Lab_env_HRI, Human
from H_nav_interaction import NavigationInteraction
from G_agents import Rmax, Epsilon_greedy_MB
from I_play_function import play
from M_graphics import plot_different_humans
import J_constants as const
import K_variables as var


# NOTE, for the best performances as seen in Fig. 6 reproduction:
# Use Rmax for Navigation
# Use e-greedy MB for Go to Human Vision and Social Task


# ----------------------------------------------------------------------- #
#      Checking if it is feasible to use a model trained on one human 
#                   type to use on all types of humans
# ----------------------------------------------------------------------- #
# Training each type of human then testing each model on all human types.
# ----------------------------------------------------------------------- #

""" DONE
seed = 42
train_on = ['basic_human', 'fast_human', 'hard_human']
humans_to_test = ['basic_human', 'fast_human', 'hard_human']
nb_trials = 500
nb_steps = 100
nb_iters = 10

# "all_data/all_imgs/1D-plots/three_humans"+str(time.time())+".pdf" 
# to get the rewards per step plot, follow the comments in:
# - run_and_plot_human_variability (D_adaptability.py)
# - get_mean_and_std (M_graphics.py)
for human in train_on:
    evaluate_human(seed,
                human,
                humans_to_test,
                nb_trials,
                nb_steps,
                nb_iters)
"""
        

# -----------------------------------------Main Goal--------------------------------------- #
#   Constructing a multi-model agent that changes model according to the 
#   type of human it is faced with.
# ----------------------------------------------------------------------------------------- #


# ---------------------------------------Sub-Goal 1----------------------------------------- #
#   Start with an easy version:
#   The type of human the robot is dealing with is given
# ------------------------------------------------------------------------------------------ #

# ---------------------------------------Sub-Goal 2----------------------------------------- #
#   Start with an easy version:
#   The parameters of the human the robot is dealing with is given
#   -> Specify conditions that define which parameters will lead to which model being used
# ------------------------------------------------------------------------------------------ #

# ---------------------------------------Final Goal----------------------------------------- #
#   The human instance is given but the parameters are unknown
#   The robot has to infer which model to use based on observed behavior
# ------------------------------------------------------------------------------------------ #


class MultiModelAgent_Sub():

    def __init__(self):
        self.nav = None
        self.goToH = None
        self.H1 = None
        self.H2 = None
        self.H3 = None


    # loading pre-trained models for navigation, go to human vision and social task on H1, H2, and H3
    def load_models(self):

        # navigation, same qtable for all humans
        q_table_nav = np.load("all_data/data/q_table/nav_Q_Rmax.npy")
        q_table_go_to_human = np.load("all_data/data/q_table/go_Q.npy")

        self.nav = {'Q': q_table_nav}
        self.goToH = {'Q': q_table_go_to_human}


        # H1

        # save H1 qtable
        if not os.path.isdir('all_data/data/q_table/basic_human'):
            name_human = 'basic_human'
            number_of_trials = 25000
            number_of_steps = 20
            # 'all_data/data/q_table/basic_human'
            save_interaction(seed,
                            name_human,
                            number_of_trials,
                            number_of_steps)
        
        # load H1 qtable into class instance
        q_table_social = np.load("all_data/data/q_table/basic_human/soc_Q.npy")
        social_R = np.load("all_data/data/q_table/basic_human/soc_R.npy")
        social_tSAS = np.load("all_data/data/q_table/basic_human/soc_T.npy")
        social_nSA = np.load("all_data/data/q_table/basic_human/soc_nSA.npy")
        social_nSAS = np.load("all_data/data/q_table/basic_human/soc_nSAS.npy")
        social_Rsum = np.load("all_data/data/q_table/basic_human/soc_Rsum.npy")


        self.H1 = {'Q': q_table_social,
                    'R': social_R,
                    'tSAS': social_tSAS,
                    'nSA': social_nSA,
                    'nSAS': social_nSAS,
                    'Rsum': social_Rsum}

        # H2

        # save H2 qtable
        if not os.path.isdir('all_data/data/q_table/fast_human'):
            name_human = 'fast_human'
            number_of_trials = 25000
            number_of_steps = 20
            # 'all_data/data/q_table/basic_human'
            save_interaction(seed,
                            name_human,
                            number_of_trials,
                            number_of_steps)
            
        # load H2 qtable into class instance
        q_table_social = np.load("all_data/data/q_table/fast_human/soc_Q.npy")
        social_R = np.load("all_data/data/q_table/fast_human/soc_R.npy")
        social_tSAS = np.load("all_data/data/q_table/fast_human/soc_T.npy")
        social_nSA = np.load("all_data/data/q_table/fast_human/soc_nSA.npy")
        social_nSAS = np.load("all_data/data/q_table/fast_human/soc_nSAS.npy")
        social_Rsum = np.load("all_data/data/q_table/fast_human/soc_Rsum.npy")

        self.H2 = {'Q': q_table_social,
                    'R': social_R,
                    'tSAS': social_tSAS,
                    'nSA': social_nSA,
                    'nSAS': social_nSAS,
                    'Rsum': social_Rsum}
        
        # H3

        # save H3 qtable
        if not os.path.isdir('all_data/data/q_table/hard_human'):
            name_human = 'hard_human'
            number_of_trials = 25000
            number_of_steps = 20
            # 'all_data/data/q_table/basic_human'
            save_interaction(seed,
                            name_human,
                            number_of_trials,
                            number_of_steps)
            
        # load H3 qtable into class instance
        q_table_social = np.load("all_data/data/q_table/hard_human/soc_Q.npy")
        social_R = np.load("all_data/data/q_table/hard_human/soc_R.npy")
        social_tSAS = np.load("all_data/data/q_table/hard_human/soc_T.npy")
        social_nSA = np.load("all_data/data/q_table/hard_human/soc_nSA.npy")
        social_nSAS = np.load("all_data/data/q_table/hard_human/soc_nSAS.npy")
        social_Rsum = np.load("all_data/data/q_table/hard_human/soc_Rsum.npy")

        self.H3 = {'Q': q_table_social,
                    'R': social_R,
                    'tSAS': social_tSAS,
                    'nSA': social_nSA,
                    'nSAS': social_nSAS,
                    'Rsum': social_Rsum}
        
        print("DONE LOADING MODELS")
    
    # sub-goal 1: human class is given
    # loads the corresponding model based on the human type we're testing on
    # takes in a list of human names to test on
    def evaluate_1(self, seed, human_names, nb_iters, nb_trials, nb_steps):

        """
        seed: random seed for reproducibility
        human_names: list of human names to test on
        nb_iters: number of iterations to average over
        nb_trials: number of trials per iteration
        nb_steps: number of steps per trial
        """

        np.random.seed(seed)

        all_rewards = {human_name: [] for human_name in human_names}

        # loop through the humans we want to test on
        for human_name in human_names:

            print("TESTING SUBGOAL 1 ON HUMAN: ", human_name)

            if human_name == 'basic_human':
                params = self.H1
            elif human_name == 'fast_human':
                params = self.H2
            elif human_name == 'hard_human':
                params = self.H3
            
            for _ in range(nb_iters):
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
                    self.nav,
                    params,     # this makes that the model is trained on the given human
                    self.goToH
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

        save_dir = 'all_data/data/all_rewards/'
        os.makedirs(save_dir, exist_ok=True) 

        np.save("all_data/data/all_rewards/" + str(time.time())+".npy", all_rewards)

        plot_different_humans(all_rewards, "Multi-Model Sub-goal 1")


    # sub-goal 2: human parameters are given
    # loads the corresponding model based on the human parameters we're testing on
    def evaluate_2(self, seed, human_params_all, nb_iters, nb_trials, nb_steps):
        """
        human_params_all: list of human parameter dictionaries to test on
        """
        
        np.random.seed(seed)

        all_rewards = {"Human params " + str(i): [] for i in range(len(human_params_all))}

        # loop through the humans we want to test on
        for i in range(len(human_params_all)):

            human_params = human_params_all[i]

            print("TESTING SUBGOAL 2 ON HUMAN PARAMS: ", human_params)
            
            speed = human_params['speeds']
            failing_rate = human_params['failing_rate']
            pointing_need = human_params['pointing_need']
            losing_attention = human_params['losing_attention']
            orientation_change_rate = human_params['orientation_change_rate']
            random_movement = human_params['random_movement']


            # ----------------HOW TO PICK THE MODEL BASED ON PARAMETERS----------------- #
            #
            #                    ! TO ADAPT TO IMPROVE PERFORMANCE !
            #
            #  To make it simple for now, we ignore speed:
            #
            #  1. if any of the probabilities is high ( >= 0.35), use H3 model
            #  2. if any of the probabilities is medium (0.1 < p < 0.35), use H2 model
            #  3.if all prbabilities are easy (p <= 0.1), use H1 model

            if (failing_rate >= 0.35 or pointing_need >= 0.35 or
                        losing_attention >= 0.35 or orientation_change_rate >= 0.35 or
                        random_movement >= 0.35):
                print("Picked model H3")
                params = self.H3
            elif (0.1 < failing_rate < 0.35 or 0.1 < pointing_need < 0.35 or
                        0.1 < losing_attention < 0.35 or 0.1 < orientation_change_rate < 0.35 or
                        0.1 < random_movement < 0.35):
                print("Picked model H2")
                params = self.H2
            else:
                print("Picked model H1")
                params = self.H1
            # --------------------------- MODEL PICKED -------------------------------- #


            for _ in range(nb_iters):
                nav_env = Gridworld()
                human = Human(**human_params)
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
                    self.nav,
                    params,     # this makes that the model is trained on the given human
                    self.goToH
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

                all_rewards["Human params " + str(i)].append(rewards)

        save_dir = 'all_data/data/all_rewards/'
        os.makedirs(save_dir, exist_ok=True) 

        np.save("all_data/data/all_rewards/" + str(time.time())+".npy", all_rewards)

        """ 
        Plot colors correspond to:
        'Human params 0': 'tab:blue',
        'Human params 1': 'tab:green',
        'Human params 2': 'tab:orange'
        """
        plot_different_humans(all_rewards, "Multi-Model Sub-goal 2")


    # main goal:
    # give a human instance with unknown parameters
    # determine which model to use based on observed behavior
    def evaluate_3(self, seed, humans, nb_iters, nb_trials, nb_steps):
        """
        humans: list of Human class instances to test on
        """
        
        np.random.seed(seed)

        all_rewards = {"Human " + str(i): [] for i in range(len(humans))}

        # loop through the humans we want to test on
        for i in range(len(humans)):

            human = humans[i]

            print("TESTING MAIN GOAL ON HUMAN " + str(i))

            # ----------------------HOW TO PICK THE MODEL BASED ON OBSERVATIONS----------------------- #
            # 1. have the human have interactions w/ no robot interference for a set number of steps
            # 2. count the number of times each type of behavior is observed
            # 3. calculate a probability for each behavior type
            # 4. pick the model based on the probabilities calculated ( same as in sub-goal 2 )
            # ---------------------------------------------------------------------------------------- #

            steps_to_observe = 10000

            # check human class to find how each is coded
            speed_1_count = 0
            speed_2_count = 0 
            speed_3_count = 0
            failing_hellos = 0
            pointing_needs = 0
            attention_losses = 0
            orientation_changes = 0
            random_movements = 0

            nav_env = Gridworld()
            social_environment = Lab_env_HRI(nav_env, human)

            # ----------------------------- COUNTING SPEEDS ----------------------------- #

            social_environment.new_episode()

            for _ in range(steps_to_observe):
                # set the human state such that it is moving towards the robot
                social_environment.human.human_state = 3
                social_environment.H_sees_R = True
                social_environment.R_looks_at_H = True
                social_environment.H_looks_at_R = True

                # observe its moving speed

                social_environment.human.human_pos = 0
                social_environment.pos = 5

                old_distance = social_environment.distance

                social_environment.make_step(24)   # stay action

                curr_distance = social_environment.distance

                movement_size = abs(curr_distance - old_distance)

                if movement_size == 1:
                    speed_1_count += 1
                elif movement_size == 2:
                    speed_2_count += 1
                elif movement_size == 3:
                    speed_3_count += 1

            # ----------------------------- COUNTING FAILED HELLOS ----------------------------- #

            social_environment.new_episode()

            for _ in range(steps_to_observe):
                # saving whether robot can attempt a hello as next action
                # appropriate_vision = social_environment.H_sees_R and social_environment.R_looks_at_H
                # hello_needed = appropriate_vision and not social_environment.H_looks_at_R

                social_environment.human.human_state = 0  # human not attentive

                # good interaction distance 
                social_environment.human_pos = 0
                social_environment.pos = 5

                # making sure a hello is needed
                social_environment.H_sees_R = True
                social_environment.R_looks_at_H = True
                social_environment.H_looks_at_R = False

                # print(social_environment.appropriate_distance)
                social_environment.make_step(26)   # hello action

                # did hello work? aka did it change the human's attention
                if social_environment.human.human_state == 0:
                    failing_hellos += 1


            # ----------------------------- COUNTING FAILED COME ACTIONS ----------------------------- #

            social_environment.new_episode()

            for _ in range(steps_to_observe):
                
                # good interaction distance
                social_environment.human_pos = 0
                social_environment.pos = 5

                social_environment.human.human_state = 1  # human attentive
                social_environment.make_step(27)   # come action

                # if point need -> human state goes to 2
                # else it goes to 3 directly and human follows robot
                if social_environment.human.human_state == 2:
                    pointing_needs += 1


            # ---------------------- COUNTING ATTENTION LOSSES ---------------------- #

            social_environment.new_episode()

            for _ in range(steps_to_observe):
                
                # human attentive
                social_environment.human.human_state = 1

                # good interaction distance
                social_environment.human_pos = 0
                social_environment.pos = 5

                social_environment.make_step(24)   # stay action

                # did human lose attention
                if social_environment.human.human_state == 0:
                    attention_losses += 1


            # ----------------------------- COUNTING ----------------------------- #
            # ----------------------- ORIENTATION CHANGES ------------------------ #
            # ------------------------- RANDOM MOVEMENTS ------------------------- #

            social_environment.new_episode()

            for _ in range(steps_to_observe):

                old_orrientation = social_environment.human_orientation
                old_position = social_environment.human_pos

                # making sure interaction doesn't impact movement/orientation
                social_environment.human_pos = 0
                social_environment.pos = 50

                social_environment.make_step(24)   # stay action

                curr_orrientation = social_environment.human_orientation
                curr_position = social_environment.human_pos

                # did human randomly change orientation or position?
                if old_orrientation != curr_orrientation:
                    orientation_changes += 1

                if old_position != curr_position:
                    random_movements += 1


            # approximating human behavior parameters
            speeds = [speed_1_count/steps_to_observe, speed_2_count/steps_to_observe, speed_3_count/steps_to_observe]
            failing_rate = failing_hellos/steps_to_observe
            pointing_need = pointing_needs/steps_to_observe
            losing_attention = attention_losses/steps_to_observe
            orientation_change_rate = orientation_changes/steps_to_observe
            random_movement = random_movements/steps_to_observe


            print("------------------ OBSERVED HUMAN PARAMETERS ------------------")
            print("Real Human Parameters:")
            print("Speeds: ", human.speeds)
            print("Failing Rate: ", human.failing_rate) 
            print("Pointing Need: ", human.pointing_need)
            print("Losing Attention: ", human.losing_attention)
            print("Orientation Change Rate: ", human.orientation_change_rate)
            print("Random Movement: ", human.random_movement)

            print("Approximated Human Parameters:")
            print("Speeds: ", speeds)
            print("Failing Rate: ", failing_rate)
            print("Pointing Need: ", pointing_need)
            print("Losing Attention: ", losing_attention)
            print("Orientation Change Rate: ", orientation_change_rate)
            print("Random Movement: ", random_movement)


            # picking model based on approximated parameters
            if (failing_rate >= 0.35 or pointing_need >= 0.35 or
                        losing_attention >= 0.35 or orientation_change_rate >= 0.35 or
                        random_movement >= 0.35):
                print("Picked model H3")
                params = self.H3
            elif (0.1 < failing_rate < 0.35 or 0.1 < pointing_need < 0.35 or
                        0.1 < losing_attention < 0.35 or 0.1 < orientation_change_rate < 0.35 or
                        0.1 < random_movement < 0.35):
                print("Picked model H2")
                params = self.H2
            else:
                print("Picked model H1")
                params = self.H1

            # run interaction with inputted human instance using picked model
            for _ in range(nb_iters):
                
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
                    self.nav,
                    params,     # this makes that the model is trained on the given human
                    self.goToH
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

                all_rewards["Human " + str(i)].append(rewards)

        save_dir = 'all_data/data/all_rewards/'
        os.makedirs(save_dir, exist_ok=True) 

        np.save("all_data/data/all_rewards/" + str(time.time())+".npy", all_rewards)
        
        # Plot colors correspond to:
        # 'Human 0': 'tab:blue',
        # 'Human 1': 'tab:green',
        # 'Human 2': 'tab:orange'
        plot_different_humans(all_rewards, "Multi-Model Agent")





# TEST PARAMS

# load models
multi_model_agent = MultiModelAgent_Sub()
multi_model_agent.load_models()

seed = 42

nb_iters = 10
nb_trials = 500
nb_steps = 100


# ---------------------------------------Sub-Goal 1 TEST----------------------------------------- #

# evaluate on all humans
humans_to_test = ['basic_human', 'fast_human', 'hard_human']


# figure will be saved in: all_data/all_imgs/1D-plots/three_humans"+str(time.time())+".pdf"
# rename figure after run
# multi_model_agent.evaluate_1(seed, humans_to_test, nb_iters, nb_trials, nb_steps)


# ---------------------------------------Sub-Goal 2 TEST----------------------------------------- #

# evaluate on human parameters that each should map to one of the three models

human_params_to_test = [
    {
        'speeds': [0, 1 , 0],
        'failing_rate': 0.05,
        'pointing_need': 0.05,
        'losing_attention': 0.05,
        'orientation_change_rate': 0.05,
        'random_movement': 0.05
    },
    {
        'speeds': [0, 1 , 0],
        'failing_rate': 0.2,
        'pointing_need': 0.2,
        'losing_attention': 0.2,
        'orientation_change_rate': 0.2,
        'random_movement': 0.2
    },
    {
        'speeds': [0, 1, 0],
        'failing_rate': 0.3,
        'pointing_need': 0.3,
        'losing_attention': 0.3,
        'orientation_change_rate': 0.3,
        'random_movement': 0.3
    }
]


# figure will be saved in: all_data/all_imgs/1D-plots/three_humans"+str(time.time())+".pdf"
# rename figure after run
multi_model_agent.evaluate_2(seed, human_params_to_test, nb_iters, nb_trials, nb_steps)


# ---------------------------------------Sub-Goal 3 TEST----------------------------------------- #

# using the same human parameters as sub-goal 2 but already fed into human instances
# so if we get the same or similar results, then the model succeeded at inferring the human behavior parameters

# create human instances from test 2 parameters
humans = []

for i in range(len(human_params_to_test)):
    humans.append(Human(**human_params_to_test[i]))


# figure will be saved in: all_data/all_imgs/1D-plots/three_humans"+str(time.time())+".pdf"
# rename figure after run
multi_model_agent.evaluate_3(seed, humans, nb_iters, nb_trials, nb_steps)