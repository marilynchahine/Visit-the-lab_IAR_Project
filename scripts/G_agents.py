
import numpy as np

# ---------Model-free agents--------------


class Epsilon_greedy_MF:

    def __init__(
            self,
            environment,
            gamma=0.95,
            alpha=0.7,
            epsilon=0.05,
            initial_Q = None):

        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        if initial_Q is not None:
            self.Q = initial_Q
        else:
            self.Q = np.zeros((self.size_environment, self.size_actions), np.float32)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.randint(self.size_actions, dtype=np.int32)
        else:
            q_values = self.Q[state]
            max_indexes = np.flatnonzero(q_values == q_values.max())
            action = np.random.choice(max_indexes)
        return action

    def learn(self, old_state, reward, new_state, action):
        max_q = np.max(self.Q[new_state])
        delta = reward + self.gamma * max_q - self.Q[old_state][action]
        self.Q[old_state][action] = self.Q[old_state][action]+self.alpha*delta

# --------- End of Model-free agents--------------

# --------Model-based agents-------


class Basic_MB:

    def __init__(self,
                 environment,
                 gamma=0.95,
                 max_iterations=10000,
                 step_update=1):

        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)

        self.gamma = gamma
        self.max_iterations = max_iterations
        self.step_update = step_update

        self.shape_SA = (self.size_environment, self.size_actions)
        self.shape_SAS = (self.size_environment,
                          self.size_actions, self.size_environment)

        self.R = np.zeros(self.shape_SA, dtype=np.float32)
        self.Rsum = np.zeros(self.shape_SA, dtype=np.float32)
        self.R_VI = np.zeros(self.shape_SA, dtype=np.float32)  # Reward for value iteration

        self.nSA = np.zeros(self.shape_SA, dtype=np.float32)
        self.nSAS = np.zeros(self.shape_SAS, dtype=np.float32)

        self.tSAS = np.ones(self.shape_SAS, dtype=np.float32) / self.size_environment
        self.Q = np.zeros(self.shape_SA, dtype=np.float32)

        self.counter = 0

    def choose_action(self, state):
        q_values = self.Q[state]
        max_indexes = np.flatnonzero(q_values == q_values.max())
        return np.random.choice(max_indexes)

    def learn_the_model(self, old_state, reward, new_state, action):
        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1
        self.Rsum[old_state][action] += reward
        new_reward = self.Rsum[old_state][action] / self.nSA[old_state][action]
        self.R[old_state][action] = new_reward

    def learn(self, old_state, reward, new_state, action):

        self.learn_the_model(old_state, reward, new_state, action)
        self.compute_reward_VI(old_state, action)
        self.compute_transitions(old_state, action)
        self.counter += 1
        self.value_iteration()

    def compute_transitions(self, old_state, action):
        transitions = self.nSAS[old_state][action] / \
            self.nSA[old_state][action]
        self.tSAS[old_state][action] = transitions

    def compute_reward_VI(self, old_state, action):
        self.R_VI[old_state][action] = self.R[old_state][action]

    def value_iteration(self):
        if self.counter % self.step_update == 0:
            threshold = 1e-3
            converged = False
            nb_iters = 0
            while (not converged and nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)

                diff = np.abs(self.Q - new_Q)
                self.Q = new_Q
                if np.max(diff) < threshold:
                    converged = True


class Epsilon_greedy_MB(Basic_MB):

    def __init__(self,
                 environment,
                 gamma,
                 epsilon,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, max_iterations, step_update)
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.randint(self.size_actions, dtype=np.int32)
        else:
            q_values = self.Q[int(state)]
            action = np.random.choice(
                np.flatnonzero(q_values == q_values.max()))
        return action


class Rmax(Basic_MB):

    def __init__(self,
                 environment,
                 gamma,
                 Rmax,
                 m,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, max_iterations, step_update)

        self.Rmax = Rmax
        self.m = m
        self.R_VI = np.ones(self.shape_SA, dtype=np.float32) * self.Rmax
        self.Q = np.ones(self.shape_SA, dtype=np.float32) * self.Rmax / (1 - self.gamma)
        self.known_states = np.zeros(self.shape_SA, dtype=np.float32)
        self.update_Q = False

    def compute_reward_VI(self, old_state, action):
        '''Use frequentist reward after m passages'''
        if self.nSA[old_state][action] == self.m:
            self.R_VI[old_state][action] = self.R[old_state][action]

    def compute_transitions(self, old_state, action):
        if self.nSA[old_state][action] == self.m:
            self.tSAS[old_state][action] = self.nSAS[old_state][action] / \
                self.nSA[old_state][action]
            self.known_states[old_state][action] = 1
            self.update_Q = True

    def value_iteration(self):
        if self.update_Q and self.counter % self.step_update == 0:
            threshold = 1e-3
            converged = False
            nb_iters = 0
            while not converged and (nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)

                diff = np.abs(self.Q[self.known_states > 0] -
                              new_Q[self.known_states > 0])
                self.Q[self.known_states > 0] = new_Q[self.known_states > 0]
                if np.max(diff) < threshold:
                    converged = True
            self.Q_value_changed = False

# -------------End of MB agents-----------

# ------------MF on MB agents----------


class MFLearnerOnMB:
    def __init__(self, MB_agent, gamma=0.95, alpha=0.3):
        self.environment = MB_agent.environment
        self.MB_agent = MB_agent
        self.gamma = gamma
        self.alpha = alpha
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        self.Q_MF = np.zeros(self.shape_SA, dtype=np.float32) / (1 - self.gamma)
        self.Q = self.MB_agent.Q

    def learn(self, old_state, reward, new_state, action):
        '''Learning of the MB and of the MF agent'''
        self.MB_agent.learn(old_state, reward, new_state, action)

        self.learn_MF(old_state, reward, new_state, action)

        self.Q = self.MB_agent.Q

    def learn_MF(self, old_state, reward, new_state, action):
        '''Model-free update of one Q-value of the MF agent'''
        max_q = np.max(self.Q_MF[new_state])
        delta = reward + self.gamma * max_q - self.Q_MF[old_state][action]
        self.Q_MF[old_state][action] = self.Q_MF[old_state][action] + \
            self.alpha*delta

    def choose_action(self, state):
        '''The action is chosen by the MB agent'''
        return self.MB_agent.choose_action(state)

# ------------ End of MF on MB agents ----------


# ------------ Finite Horizon MB agents ----------

class FiniteHorizonMB(Basic_MB):

    def __init__(self,
                 environment,
                 gamma=0.95,
                 horizon=10,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, max_iterations, step_update)

        self.horizon = horizon

        # horizon tables
        self.shape_SAH = (self.size_environment,
                          self.size_actions, self.horizon)
        self.R_horizon = np.zeros(self.shape_SAH, dtype=np.float32)
        self.nSA_horizon = np.zeros(self.shape_SAH, dtype='int')

    def learn_the_model(self, old_state, reward, new_state, action):
        '''Learns the model with respect to the horizon'''
        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1
        self.Rsum[old_state][action] += reward

        index_update = int(self.nSA[old_state][action] % self.horizon)

        if self.nSA[old_state][action] > self.horizon:
            forget_state = self.nSA_horizon[old_state][action][index_update]
            forget_reward = self.R_horizon[old_state][action][index_update]

            self.Rsum[old_state][action] -= forget_reward
            self.nSAS[old_state][action][forget_state] -= 1

        self.nSA_horizon[old_state][action][index_update] = new_state
        self.R_horizon[old_state][action][index_update] = reward

        self.norm_factor = min(self.nSA[old_state][action], self.horizon)

        new_reward = self.Rsum[old_state][action] / self.norm_factor
        self.R[old_state][action] = new_reward

    def compute_transitions(self, old_state, action):
        transitions = self.nSAS[old_state][action] / self.norm_factor
        self.tSAS[old_state][action] = transitions


class Epsilon_MB_horizon(FiniteHorizonMB):
    def __init__(self,
                 environment,
                 gamma,
                 horizon,
                 epsilon,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, 
                         gamma, 
                         horizon, 
                         max_iterations, 
                         step_update)

        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.randint(self.size_actions, dtype=np.int32)
        else:
            q_values = self.Q[state]
            max_indexes = np.flatnonzero(q_values == q_values.max())
            action = np.random.choice(max_indexes)
        return action


# ------------ End of Finite Horizon MB agents ----------
