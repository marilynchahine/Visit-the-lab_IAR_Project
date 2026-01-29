# non_stationary_train_test.py
import os
import time
import numpy as np
import gc
from E_envs import SocialGridworld, Lab_env_HRI, Human
from G_agents import Epsilon_greedy_MB, Epsilon_greedy_MF
import J_constants as const   
from K_variables import envs, env_subparams, env_subclasses, agents, agent_params

# -----------------------
# Agent registry
# -----------------------
AGENTS = {
    "e_greedy_MB": Epsilon_greedy_MB,
    "e_greedy_MF": Epsilon_greedy_MF,
}

from multiprocessing import Pool, freeze_support

def one_job(job):
    agent_name, run_id, seed = job

    # ---- Humans ----
    env_before = "social_basic"
    human_after = const.basic_human_speed_3_param

    env_test = "social_basic_speed_2"
    human_test_after = const.basic_human_speed_3_param

    # ---- Params ----
    nav_size = 12
    max_step = 20
    train_trials = 300000
    test_trials = 300000
    switch_train = 130000
    switch_test = 130000

    # TRAIN (A->B)
    train_rewards, trained_agent = run_non_stationary_train(
        agent_name=agent_name,
        env_name=env_before,
        human_params_after=human_after,
        trials=train_trials,
        max_step=max_step,
        switch_trial=switch_train,
        nav_size=nav_size,
        seed=seed,
    )

    # TEST (C->D) continue learning
    test_rewards, _ = run_non_stationary_train(
        agent_name=agent_name,
        env_name=env_test,
        human_params_after=human_test_after,
        trials=test_trials,
        max_step=max_step,
        switch_trial=switch_test,
        nav_size=nav_size,
        seed=seed + 999,
        init_agent=trained_agent,
    )

    full_curve = np.concatenate([train_rewards, test_rewards])
    del train_rewards
    del test_rewards
    gc.collect() 

    return agent_name, run_id, full_curve


def build_environment(env_name: str, nav_size: int):
    """
    Build an environment exactly like play_function.one_parameter_play_function,
    but with an override for nav_env size to keep memory under control.
    """
    sub_class_env = env_subclasses[env_name]
    sub_param_env = env_subparams[env_name]

    environment_parameters = {}
    for key in sub_class_env.keys():
        params = dict(sub_param_env.get(key, {}))
        if key == "nav_env":
            # Critical for memory: avoid huge state spaces
            params["size"] = nav_size
        environment_parameters[key] = sub_class_env[key](**params)

    environment = envs[env_name](**environment_parameters)
    return environment
# -----------------------
# One episode = max_step steps
# -----------------------
def run_one_episode(env, agent, max_step):
    env.new_episode()
    state = env.agent_state
    total_reward = 0

    for _ in range(max_step):
        action = agent.choose_action(state)
        reward, new_state = env.make_step(action)
        agent.learn(state, reward, new_state, action)
        total_reward += reward
        state = new_state

    return total_reward


# -----------------------
# Non-stationary run
# -----------------------
def run_non_stationary1(
    agent_name,
    human_before,
    human_after,
    trials,
    max_step,
    switch_trial,
    nav_size,
    seed,
    init_agent=None,
):
    np.random.seed(seed)

    # --- nav env
    nav_env = SocialGridworld(size=nav_size)

    # --- social env
    env = Lab_env_HRI(
        nav_env=nav_env,
        human=human_before,
        random_human_pos=False,
        deterministic=True,
    )

    # --- agent
    if init_agent is None:
        agent = AGENTS[agent_name](
            environment=env,
            gamma=0.95,
            epsilon=0.05,
        )
    else:
        agent = init_agent

    rewards = []

    for t in range(trials):
        if t == switch_trial:
            env.human = human_after   # non-stationary human here

        ep_reward = run_one_episode(env, agent, max_step)
        rewards.append(ep_reward)

    return np.array(rewards), agent



def run_non_stationary_train(
    agent_name: str,
    env_name: str,
    human_params_after: dict,
    trials: int,
    max_step: int,
    switch_trial: int,
    nav_size: int,
    seed: int,
    init_agent=None,
):
    np.random.seed(seed)

    env = build_environment(env_name, nav_size=nav_size)

    if init_agent is None:
        agent = agents[agent_name](env, **agent_params[agent_name])
    else:
        agent = init_agent
        agent.environment = env  # 让 agent 指向新的 env（很重要）

    rewards = []

    for trial in range(trials):
        if trial == switch_trial:
            env.human = Human(**human_params_after)
            env.new_episode()

        cumulative = 0.0
        env.new_episode()

        for _ in range(max_step):
            s = env.agent_state
            a = agent.choose_action(s)
            r, s2 = env.make_step(a)
            agent.learn(s, r, s2, a)
            cumulative += r

        rewards.append(cumulative)

    return np.array(rewards, dtype=np.float32), agent




# -----------------------
# Main: Train then Test
# -----------------------
def main():
    agent_list = ["e_greedy_MB", "e_greedy_MF"]
    n_runs = 2
    seed0 = 123
    proc = 2  

    jobs = []
    for agent_name in agent_list:
        for run_id in range(n_runs):
            jobs.append((agent_name, run_id, seed0 + run_id))

    with Pool(processes=proc) as pool:
        out = pool.map(one_job, jobs)

    results = {a: [] for a in agent_list}
    for agent_name, run_id, full_curve in out:
        results[agent_name].append(full_curve)

    out_path = f"../data/non_stationary_train_test_{time.time()}.npy"
    os.makedirs("../data", exist_ok=True)
    np.save(out_path, results)
    print("Saved:", out_path)

if __name__ == "__main__":
    freeze_support()
    main()
    
