import numpy as np


class NavigationInteraction:
    def __init__(self,
                 environment,
                 nav_agent,
                 social_agent,
                 go_agent,
                 init_nav={},
                 init_soc={},
                 init_go={}):

        self.environment = environment
        self.nav_agent = nav_agent
        self.social_agent = social_agent
        self.go_agent = go_agent

        self.init_values(nav_agent, init_nav)
        self.init_values(social_agent, init_soc)
        self.init_values(go_agent, init_go)

        self.required_direction = np.random.randint(8, dtype=np.int32)

        self.update_dir_and_state()

    def init_values(self, agent, all_values):
        for name, value in all_values.items():
            setattr(agent, name, value)

    def learn(self, old_state, reward, new_state, action):

        # modifies the direction of the new state stored in self.social_state
        self.update_dir_and_state()

        # if the human is in the highest state of interaction and did an
        # appropriate action
        social_reward = self.environment.get_reward()

        # checks whether to update action 29 or not.
        if not self.action_29:
            self.social_agent.learn(old_state, social_reward,
                             self.social_state, action)

        else:
            self.social_agent.learn(old_state, social_reward,
                             self.social_state, 29)

    def update_required_direction(self, human_pos, rewarded_room):
        """This method takes the q-values from the navigation map, looks at the
        value of the Q-table corresponding to the human position and goal room,
        and changes in which direction the human should go. The case in which
        there is no best direction and the human should not move is not 
        considered here. Further modifications could take it into account.
        """
        human_pos = self.environment.nav_state_to_state(human_pos,
                                                        rewarded_room)
        q_values = self.nav_agent.Q[human_pos].copy()

        speeds = [0.333, 0.333, 0.334]
        coeff_q_val = np.array(([speeds[0]]*4+[speeds[1]]*4+[speeds[2]]*4)*2, dtype=np.float32)
        probas = coeff_q_val * q_values[:24]
        prefered_directions = np.zeros(8, dtype=np.float32)
        for i in range(len(probas)):
            direction = self.environment.action_to_direction[i]
            prefered_directions[direction] += probas[i]

        current_dir_value = prefered_directions[self.required_direction]
        best_dir_value = np.max(prefered_directions)
        if current_dir_value < best_dir_value:
            all_best_dirs = np.flatnonzero(
                prefered_directions == np.max(prefered_directions))
            self.required_direction = np.random.choice(all_best_dirs)

        self.environment.new_required_direction(self.required_direction)

    def update_state_new_dir(self, state, new_direction):
        '''Take the required direction and change the environment state 
        according to this new direction.'''
        multiD_state = self.environment.oneD_to_multiD(
            self.environment.cardinalities,
            state)
        multiD_state[2] = new_direction
        oneD_state = self.environment.multiD_to_1D(
            self.environment.cardinalities,
            multiD_state)
        return oneD_state

    def choose_action_go(self):
        cardinalities = [8,8,5]
        multiD_state_go_to_human = np.array(
            [self.environment.position_angle,
             self.environment.human_orientation,
             self.environment.distance_for_the_robot], dtype=np.float32)

        state_go_to_human = self.environment.multiD_to_1D(
            cardinalities,
            multiD_state_go_to_human)

        action = self.go_agent.choose_action(state_go_to_human)
        return action

    def choose_action(self, state):
        """Update the necessary direction towards which the human should go 
        and modifies the social state accordingly. 
        """

        action = self.social_agent.choose_action(self.social_state)

        self.action_29 = (action == 29)
        if action == 29:
            action = self.choose_action_go()

        # The returned action here is always inferior or equal to 28.
        return action

    def update_dir_and_state(self):
        self.update_required_direction(self.environment.human_pos,
                                       self.environment.rewarded_room)

        self.social_state = self.update_state_new_dir(
                self.environment.agent_state,
                self.required_direction)
