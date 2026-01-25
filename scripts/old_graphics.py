# import pygame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def get_max_Q_values_and_policy(table):
    best_values = np.max(table, axis=1)
    random_noise = 1e-5 * np.random.random(table.shape)
    best_actions = np.argmax(table + random_noise, axis=1)
    return best_values, best_actions


def plot_2D(table, environment, path=''):
    plt.figure(dpi=300)
    table = np.reshape(table, (environment.height, environment.width))
    sns.heatmap(table, cmap='crest', cbar=False, annot=table, fmt='.1f',
                annot_kws={"size": 35 / (np.sqrt(len(table)) + 2.5)})
    plt.xticks([])
    plt.yticks([])
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close()


# dont worry augustin: i swear i will clean this shitty code!

def get_moving_average(rewards, avg=100):
    index_avg = [i * len(rewards) // avg for i in range(avg)]
    moving_average_rewards = [np.mean(rewards[index_avg[index]:index_avg[index + 1]])
                              for index in range(len(index_avg) - 1)]
    return index_avg, moving_average_rewards


def plot_rewards_agents(rewards_MB, rewards_MF, rewards_MBMF, path='', title=''):

    index_avg_MB, moving_avg_rewards_MB = get_moving_average(rewards_MB)
    index_avg_MF, moving_avg_rewards_MF = get_moving_average(rewards_MF)
    index_avg_MBMF, moving_avg_rewards_MBMF = get_moving_average(rewards_MBMF)

    plt.figure(dpi=300)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    # error_average_rewards1 = [np.std(rewards1[index_avg1[index]:index_avg1[index + 1]])
    #                           for index in range(len(index_avg1) - 1)]

    # error_average_rewards2 = [np.std(rewards2[index_avg2[index]:index_avg2[index + 1]])
    #                           for index in range(len(index_avg2) - 1)]

    # error_average_rewards3 = [np.std(rewards3[index_avg3[index]:index_avg3[index + 1]])
    #                           for index in range(len(index_avg3) - 1)]

    # plt.fill_between(index_avg1[:-1], np.array(moving_average_rewards1) - np.array(error_average_rewards1),
    #                  np.array(moving_average_rewards1) + np.array(error_average_rewards1), color='tab:blue', alpha=0.2)
    # plt.fill_between(index_avg2[:-1], np.array(moving_average_rewards2) - np.array(error_average_rewards2),
    #                  np.array(moving_average_rewards2) + np.array(error_average_rewards2), color='tab:green', alpha=0.2)
    # plt.fill_between(index_avg3[:-1], np.array(moving_average_rewards3) - np.array(error_average_rewards3),
    #                  np.array(moving_average_rewards3) + np.array(error_average_rewards3), color='tab:red', alpha=0.2)

    plt.plot(index_avg_MB[:-1], moving_avg_rewards_MB,
             color='tab:blue', linewidth=3, label="MB")
    plt.plot(index_avg_MF[:-1], moving_avg_rewards_MF,
             color='tab:green', linewidth=3, label="MF")
    plt.plot(np.array(index_avg_MBMF[:-1])+max(index_avg_MBMF),
             moving_avg_rewards_MBMF,
             color='tab:red', linewidth=3, label="MF on MB")

    plt.axvline(max(index_avg_MBMF), color='red', linestyle='--')

    plt.ylabel('Mean Reward', fontsize=12)
    plt.xlabel('Trial (x20 for steps)', fontsize=12)
    plt.legend(loc='upper left')
    if title != '':
        plt.title(title)
    if path != '':
        plt.savefig(path)
    plt.show()


def get_mean_and_std(rewards):
    return np.mean(rewards, axis=0), np.std(rewards, axis=0)


def plot_vs_Q(q_table_MB, q_table_MF, q_table_MFMB, task_name, best_q_per_task, name_agent, title='', path=''):

    best_q = best_q_per_task
    normalized_q_MF = q_table_MF / best_q
    normalized_q_MFMB = q_table_MFMB / best_q
    normalized_q_MB = q_table_MB / best_q
    idx = np.argsort(best_q)

    list_splits = np.round(list(set(list(best_q))), 2)

    best_q = np.round(best_q, 2)
    packs_by_value = [np.where(best_q == list_splits[i])[0]
                      for i in range(len(list_splits))]

    norm_sort_MF = [np.array(normalized_q_MF)[idx_value] for idx_value in packs_by_value]
    norm_sort_MB = [np.array(normalized_q_MB)[idx_value] for idx_value in packs_by_value]
    norm_sort_MFMB = [np.array(normalized_q_MFMB)[idx_value] for idx_value in packs_by_value]

    plt.figure(figsize=(18, 6), dpi=300)
    titles = {0: name_agent, 1: "MF", 2: "MF on " + name_agent}
    data = {0: norm_sort_MB, 1: norm_sort_MF, 2: norm_sort_MFMB}
    for i in range(3):
        plt.subplot(1, 3, i+1)
        sns.set(style="whitegrid")

        sns.boxplot(data=data[i])

        plt.title(titles[i])
        plt.xticks(range(len(list_splits)), list_splits)
        plt.xlabel('V*(s)', fontsize=12)
        plt.ylabel('V(s)/V*(s)', fontsize=12)
        plt.ylim(-0.05, 1.05)
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_vs_distance(q_table_MB, q_table_MF, q_table_MFMB, task_name,
                     best_q_per_task, distances, name_agent, title='', path=''):

    best_q = best_q_per_task
    normalized_q_MF = q_table_MF / best_q
    normalized_q_MFMB = q_table_MFMB / best_q
    normalized_q_MB = q_table_MB / best_q

    max_distance = np.max(distances)

    list_splits = np.arange(1, max_distance+1)
    print(list_splits)

    best_q = np.round(best_q, 2)
    packs_by_value = [np.where(distances == list_splits[i])[0]
                      for i in range(len(list_splits))]

    norm_sort_MF = [np.array(normalized_q_MF)[idx_value] for idx_value in packs_by_value]
    norm_sort_MB = [np.array(normalized_q_MB)[idx_value] for idx_value in packs_by_value]
    norm_sort_MFMB = [np.array(normalized_q_MFMB)[idx_value] for idx_value in packs_by_value]

    plt.figure(figsize=(18, 6), dpi=300)
    titles = {0: name_agent, 1: "MF", 2: "MF on " + name_agent}
    data = {0: norm_sort_MB, 1: norm_sort_MF, 2: norm_sort_MFMB}
    for i in range(3):
        plt.subplot(1, 3, i+1)
        sns.set(style="whitegrid")

        sns.boxplot(data=data[i])

        plt.title(titles[i])
        plt.xticks(range(len(list_splits)), list_splits)
        plt.xlabel('Number of 1-step action to get a reward', fontsize=12)
        plt.ylabel('V(s)/V*(s)', fontsize=12)
        plt.ylim(-0.05, 1.05)
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_with_std(list_rewards_MB, list_rewards_MF, list_rewards_MBMF, path='', title=''):

    mean_MB, std_MB = get_mean_and_std(list_rewards_MB)
    mean_MF, std_MF = get_mean_and_std(list_rewards_MF)
    mean_MBMF, std_MBMF = get_mean_and_std(list_rewards_MBMF)
    print(len(mean_MB))
    index_MB, mean_moving_MB = get_moving_average(mean_MB)
    _, std_moving_MB = get_moving_average(std_MB)

    index_MF, mean_moving_MF = get_moving_average(mean_MF)
    _, std_moving_MF = get_moving_average(std_MF)

    index_MBMF, mean_moving_MBMF = get_moving_average(mean_MBMF)
    _, std_moving_MBMF = get_moving_average(std_MBMF)

    plt.figure(dpi=300)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    colors = {0: '#1E88E5', 1: '#FFC107', 2: '#D81B60'}

    plt.plot(index_MB[: -1], mean_moving_MB,
             color=colors[0], linewidth=3, label="MB")
    plt.plot(index_MF[:-1], mean_moving_MF,
             color=colors[1], linewidth=3, label="MF")
    plt.plot(np.array(index_MBMF[:-1])+max(index_MBMF),
             mean_moving_MBMF,
             color=colors[2], linewidth=3, label="MF on MB")

    plt.axvline(max(index_MBMF), color='red', linestyle='--')

    plt.fill_between(index_MB[:-1], np.array(mean_moving_MB) - np.array(std_moving_MB),
                     np.array(mean_moving_MB) + np.array(std_moving_MB),
                     color=colors[0],
                     alpha=0.2)
    plt.fill_between(index_MF[:-1], np.array(mean_moving_MF) - np.array(std_moving_MF),
                     np.array(mean_moving_MF) + np.array(std_moving_MF),
                     color=colors[1],
                     alpha=0.2)

    plt.fill_between(np.array(index_MBMF[:-1])+max(index_MBMF), np.array(mean_moving_MBMF) - np.array(std_moving_MBMF),
                     np.array(mean_moving_MBMF) + np.array(std_moving_MBMF),
                     color=colors[2],
                     alpha=0.2)

    plt.ylabel('Mean Reward', fontsize=12)
    plt.xlabel('Trial (x20 for steps)', fontsize=12)
    plt.legend(loc='upper left')
    if title != '':
        plt.title(title)
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_1D(rewards, path=''):
    # plt.figure(dpi=300)
    plt.rc('axes', axisbelow=True)
    plt.grid()
    index_avg = [i * len(rewards) // 20 for i in range(20)]
    moving_median = np.array([np.mean(rewards[index_avg[index]:index_avg[index + 1]])
                              for index in range(len(index_avg) - 1)])
    quartile = np.array([0.6745 * np.std(rewards[index_avg[index]:index_avg[index + 1]])
                         for index in range(len(index_avg) - 1)])
    plt.fill_between(index_avg[:-1], moving_median - quartile,
                     moving_median + quartile, color='tab:blue', alpha=0.2)
    plt.plot(index_avg[:-1], moving_median, 'tab:blue',
             linewidth=3)
    # plt.scatter(np.arange(len(rewards)), rewards, color='tab:blue', marker='+', s=4, alpha=0.4)
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Trial', fontsize=12)
    # plt.legend(loc='lower right')
    if path != '':
        plt.savefig(path)
    plt.show()


def basic_plot(rewards):
    index_avg = [i * len(rewards) // 20 for i in range(20)]
    moving_average_rewards = [np.mean(rewards[index_avg[index]:index_avg[index + 1]])
                              for index in range(len(index_avg) - 1)]
    plt.grid()
    plt.plot(index_avg[: -1], moving_average_rewards,
             color='tab:blue', linewidth=3, label="Easy Human")
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Trial', fontsize=12)
    plt.show()


def plot_different_humans(all_rewards):

    plt.figure(dpi=300)
    colors = {0: 'tab:orange', 1: 'tab:blue', 2: 'tab:green'}
    # colors = {0: 'tab:blue', 1: 'tab:green', 2: 'tab:red'}
    labels = {0: 'Same human', 1: 'Two times slower human',
              2: 'Same human with pointing need'}
    for idx_reward in range(len(all_rewards)):
        reward = all_rewards[idx_reward]
        mean_reward, std_reward = np.mean(reward, axis=0), np.std(reward, axis=0)
        index_avg, avg_rewards = get_moving_average(mean_reward, avg=20)
        _, avg_std = get_moving_average(std_reward, avg=20)

        plt.plot(index_avg[:-1], avg_rewards,
                 color=colors[idx_reward], label=labels[idx_reward])

        plt.fill_between(index_avg[:-1], np.array(avg_rewards) - np.array(avg_std),
                         np.array(avg_rewards) + np.array(avg_std),
                         color=colors[idx_reward],
                         alpha=0.2)
    plt.ylabel('#Goals reached', fontsize=12)
    plt.xlabel('#Trials (Steps x100)', fontsize=12)
    plt.savefig("all_plots/comparison between 3 humans_500.pdf")
    # plt.ylim(-0.5, 13.9)
    plt.legend(loc='upper left')
    plt.xlim(0, 300)
    plt.savefig('all_plots/'+str(time.time())+'.pdf')
    plt.grid(linestyle='--')
    plt.show()
