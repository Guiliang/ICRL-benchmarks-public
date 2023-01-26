import copy
import os

import numpy as np

from interface.plot_results.plot_results_dirs import get_plot_results_dir
from utils.data_utils import read_running_logs, compute_moving_average, mean_std_plot_results, \
    mean_std_plot_valid_rewards, mean_std_test_results
from utils.plot_utils import plot_curve, plot_shadow_curve


def plot_results(mean_results_moving_avg_dict,
                 std_results_moving_avg_dict,
                 episode_plots,
                 ylim, label, method_names,
                 save_label,
                 legend_dict=None,
                 linestyle_dict=None,
                 legend_size=20,
                 axis_size=None,
                 img_size=None,
                 title=None):
    plot_mean_y_dict = {}
    plot_std_y_dict = {}
    plot_x_dict = {}
    for method_name in method_names:
        # plot_x_dict.update({method_name: [i for i in range(len(mean_results_moving_avg_dict[method_name]))]})
        plot_x_dict.update({method_name: episode_plots[method_name]})
        plot_mean_y_dict.update({method_name: mean_results_moving_avg_dict[method_name]})
        plot_std_y_dict.update({method_name: std_results_moving_avg_dict[method_name]})
    if save_label is not None:
        plot_name = './plot_results/{0}'.format(save_label)
    else:
        plot_name = None
    plot_shadow_curve(draw_keys=method_names,
                      x_dict_mean=plot_x_dict,
                      y_dict_mean=plot_mean_y_dict,
                      x_dict_std=plot_x_dict,
                      y_dict_std=plot_std_y_dict,
                      img_size=img_size if img_size is not None else (6, 5.8),
                      ylim=ylim,
                      title=title,
                      xlabel='Episode',
                      ylabel=label,
                      legend_dict=legend_dict,
                      legend_size=legend_size,
                      linestyle_dict=linestyle_dict,
                      axis_size=axis_size if axis_size is not None else 18,
                      title_size=20,
                      plot_name=plot_name, )


def generate_plots():
    axis_size = None
    save_msg = ''
    modes = ['train']
    plot_mode = 'all-methods'
    last_num = 100

    env_id = 'HCWithPos-v0'
    max_episodes = 6000
    average_num = 100
    max_reward = 10000
    min_reward = 0
    plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    label_key = [None, None, None, None]
    img_size = None
    save = False
    title = 'Blocked Half-Cheetah'
    constraint_keys = ['constraint']
    plot_y_lim_dict = {'reward': (0, 7000),
                       'reward_nc': (0, 5000),
                       'constraint': (0, 1.1),
                       'reward_valid': (0, 5000),
                       }
    method_names_labels_dict = {
        "GAIL_HCWithPos-v0_with-action": 'GACL',  # 'GAIL',
        "Binary_HCWithPos-v0_with-action": 'BC2L',  # 'Binary',
        "ICRL_Pos_with-action": 'MECL',  # 'ICRL',
        "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3": "VICRL",
        # "PPO_Pos": 'PPO',
        "PPO_lag_Pos": 'PPO_lag',
    }

    if plot_mode == 'part':
        for method_name in method_names_labels_dict.copy().keys():
            if 'PPO' not in method_names_labels_dict[method_name]:
                del method_names_labels_dict[method_name]
    else:
        method_names_labels_dict = method_names_labels_dict

    linestyle_all = {
        "PPO": '-' if plot_mode == 'part' else '--',
        "PPO_lag": '-' if plot_mode == 'part' else '--',
        "PI-Lag": '-' if plot_mode == 'part' else '--',
        'PPO_lag1': '-',
        'PPO_lag2': '-',
        'Bound': '-',
        "GACL": ':',  # 'GAIL',
        "GACL-0.01": ":",
        "GACL-0.1": ":",
        "GACL-0.3": "--",
        "GACL-0.5": "-.",
        "GACL-Full": "-",
        "BC2L": '--',  # 'Binary',
        "BC2L-0.01": ":",
        "BC2L-0.1": ":",
        "BC2L-0.3": "--",
        "BC2L-0.5": "-.",
        "BC2L-Full": "-",
        "MECL": '-.',  # 'ICRL',
        "MECL-0.01": ":",
        "MECL-0.1": ":",
        "MECL-0.3": "--",
        "MECL-0.5": "-.",
        "MECL-Full": "-",
        "VICRL-VaR": "-",
        "VICRL-SR": "-",
        "VICRL": "-",
        "VICRL1": "-",
        "VICRL2": "-",
        "VICRL3": "-",
        "VICRL4": "-",
        "VICRL5": "-",
        "VICRL6": "-",
        "VICRL7": "-",
        "VICRL8": "-",
        "VICRL-0.01": ":",
        "VICRL-0.1": ":",
        "VICRL-0.3": "--",
        "VICRL-0.5": "-.",
        "VICRL-Full": "-",
        "Ram": "-",
        "Ram-1": "-",
        "Ram-0.8": "-",
        "Ram-0.5": "-",
        "Ram-0.2": "-",
        "Ram-0": "-",
        "VICRL_Hard": "-",
    }

    linestyle_dict = {}
    for method_name in method_names_labels_dict.keys():
        for linestyle_key in linestyle_all.keys():
            if method_names_labels_dict[method_name] == linestyle_key:
                linestyle_dict.update({method_name: linestyle_all[linestyle_key]})

    for mode in modes:
        # plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out']
        log_path_dict = get_plot_results_dir(env_id)

        all_mean_dict = {}
        all_std_dict = {}
        all_episodes_dict = {}
        for method_name in method_names_labels_dict.keys():
            all_results = []
            # all_valid_rewards = []
            # all_valid_episodes = []
            if method_name == 'Bound':
                results = {}
                for key in bound_results:
                    results.update({key: [bound_results[key] for item in range(max_episodes + 1000)]})
                all_results.append(results)
            else:
                for log_path in log_path_dict[method_name]:
                    monitor_path_all = []
                    if mode == 'train':
                        run_files = os.listdir(log_path)
                        for file in run_files:
                            if 'monitor' in file:
                                monitor_path_all.append(log_path + file)
                    else:
                        monitor_path_all.append(log_path + 'test/test.monitor.csv')
                    if (method_names_labels_dict[method_name] == "PPO" or
                        method_names_labels_dict[method_name] == "PPO_lag" or
                        method_names_labels_dict[method_name] == "PI-Lag") and plot_mode != "part":
                        if 'reward_nc' in plot_key:
                            plot_key[plot_key.index('reward_nc')] = 'reward'
                    # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
                    results, valid_rewards, valid_episodes = read_running_logs(monitor_path_all=monitor_path_all,
                                                                               read_keys=plot_key,
                                                                               max_reward=max_reward,
                                                                               min_reward=min_reward,
                                                                               max_episodes=max_episodes + float(
                                                                                   max_episodes / 5),
                                                                               constraint_keys=constraint_keys)
                    if (method_names_labels_dict[method_name] == "PPO" or
                        method_names_labels_dict[method_name] == "PPO_lag" or
                        method_names_labels_dict[method_name] == "PI-Lag") and plot_mode != "part":
                        results_copy_ = copy.copy(results)
                        for key in results.keys():
                            fill_value = np.mean(results_copy_[key][-last_num:])
                            results[key] = [fill_value for item in range(max_episodes + 1000)]
                    # all_valid_rewards.append(valid_rewards)
                    # all_valid_episodes.append(valid_episodes)
                    all_results.append(results)
            if mode == 'test':
                mean_std_test_results(all_results, method_name)

            mean_dict, std_dict, episodes = mean_std_plot_results(all_results)
            # mean_valid_rewards, std_valid_rewards, valid_episodes = \
            #     mean_std_plot_valid_rewards(all_valid_rewards, all_valid_episodes)
            # mean_dict.update({'reward_valid': mean_valid_rewards})
            # std_dict.update({'reward_valid': std_valid_rewards})
            # episodes.update({'reward_valid': valid_episodes})
            all_mean_dict.update({method_name: {}})
            all_std_dict.update({method_name: {}})
            all_episodes_dict.update({method_name: {}})

            if not os.path.exists(os.path.join('./plot_results/', env_id)):
                os.mkdir(os.path.join('./plot_results/', env_id))
            if not os.path.exists(os.path.join('./plot_results/', env_id, method_name)):
                os.mkdir(os.path.join('./plot_results/', env_id, method_name))

            for idx in range(len(plot_key)):
                print(method_name, plot_key[idx])
                if method_name == 'Bound' and (plot_key[idx] == 'avg_distance' or plot_key[idx] == 'avg_velocity'):
                    continue
                mean_results_moving_average = compute_moving_average(result_all=mean_dict[plot_key[idx]],
                                                                     average_num=average_num)
                std_results_moving_average = compute_moving_average(result_all=std_dict[plot_key[idx]],
                                                                    average_num=average_num)
                episode_plot = episodes[plot_key[idx]][:len(mean_results_moving_average)]
                if max_episodes:
                    mean_results_moving_average = mean_results_moving_average[:max_episodes]
                    std_results_moving_average = std_results_moving_average[:max_episodes]
                    episode_plot = episode_plot[:max_episodes]
                all_mean_dict[method_name].update({plot_key[idx]: mean_results_moving_average})
                if (method_names_labels_dict[method_name] == "PPO" or
                    method_names_labels_dict[method_name] == "PPO_lag") and plot_mode != "part":
                    all_std_dict[method_name].update({plot_key[idx]: np.zeros(std_results_moving_average.shape)})
                else:
                    all_std_dict[method_name].update({plot_key[idx]: std_results_moving_average / 2})
                all_episodes_dict[method_name].update({plot_key[idx]: episode_plot})
                plot_results(mean_results_moving_avg_dict={method_name: mean_results_moving_average},
                             std_results_moving_avg_dict={method_name: std_results_moving_average},
                             episode_plots={method_name: episode_plot},
                             label=plot_key[idx],
                             method_names=[method_name],
                             ylim=plot_y_lim_dict[plot_key[idx]],
                             save_label=os.path.join(env_id, method_name, plot_key[idx] + '_' + mode),
                             title=title,
                             axis_size=axis_size,
                             img_size=img_size,
                             linestyle_dict=linestyle_dict,
                             )
        for idx in range(len(plot_key)):
            mean_results_moving_avg_dict = {}
            std_results_moving_avg_dict = {}
            espisode_dict = {}
            plot_method_names = list(method_names_labels_dict.keys())
            for method_name in method_names_labels_dict.keys():
                if method_name == 'Bound' and (plot_key[idx] == 'avg_distance' or plot_key[idx] == 'avg_velocity'):
                    plot_method_names.remove('Bound')
                    continue
                mean_results_moving_avg_dict.update({method_name: all_mean_dict[method_name][plot_key[idx]]})
                std_results_moving_avg_dict.update({method_name: all_std_dict[method_name][plot_key[idx]]})
                espisode_dict.update({method_name: all_episodes_dict[method_name][plot_key[idx]]})
                # if (plot_key[idx] == 'reward_valid' or plot_key[idx] == 'constraint') and mode == 'test':
                #     print(method_name, plot_key[idx],
                #           all_mean_dict[method_name][plot_key[idx]][-1],
                #           all_std_dict[method_name][plot_key[idx]][-1])
                print(plot_key[idx],
                      method_name,
                      np.mean(mean_results_moving_avg_dict[method_name][-100:]),
                      np.mean(std_results_moving_avg_dict[method_name][-100:]))
            if save:
                save_label = os.path.join(env_id,
                                          plot_key[idx] + '_' + mode + save_msg + '_' + env_id + '_' + plot_mode)
            else:
                save_label = None

            plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                         std_results_moving_avg_dict=std_results_moving_avg_dict,
                         episode_plots=espisode_dict,
                         label=label_key[idx],
                         method_names=plot_method_names,
                         ylim=plot_y_lim_dict[plot_key[idx]],
                         save_label=save_label,
                         # legend_size=18,
                         legend_dict=method_names_labels_dict,
                         title=title,
                         axis_size=axis_size,
                         img_size=img_size,
                         linestyle_dict=linestyle_dict,
                         )


if __name__ == "__main__":
    generate_plots()
