import os

from utils.data_utils import read_running_logs, compute_moving_average, mean_std_plot_results, \
    mean_std_plot_valid_rewards
from utils.plot_utils import plot_curve, plot_shadow_curve


def plot_results(mean_results_moving_avg_dict,
                 std_results_moving_avg_dict,
                 episode_plots,
                 ylim, label, method_names,
                 save_label,
                 legend_dict=None,
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
    plot_shadow_curve(draw_keys=method_names,
                      x_dict_mean=plot_x_dict,
                      y_dict_mean=plot_mean_y_dict,
                      x_dict_std=plot_x_dict,
                      y_dict_std=plot_std_y_dict,
                      img_size=img_size if img_size is not None else (5.7, 5.6),
                      ylim=ylim,
                      title=title,
                      xlabel='Episode',
                      # ylabel=label,
                      legend_dict=legend_dict,
                      legend_size=legend_size,
                      axis_size=axis_size if axis_size is not None else 18,
                      title_size=20,
                      plot_name='./plot_results/{0}'.format(save_label), )


def generate_plots():
    # env_id = 'HCWithPos-v0'
    # method_names_labels_dict = {
    #     # "PPO_Pos": 'PPO',
    #     # "PPO_lag_Pos": 'PPO_lag',
    #     "GAIL_HCWithPos-v0_with-action": 'GACL',  # 'GAIL',
    #     "Binary_HCWithPos-v0_with-action": 'BC2L',  # 'Binary',
    #     "ICRL_Pos_with-action": 'MECL',  # 'ICRL',
    #     "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3": "VICRL",
    # }
    # env_id = 'AntWall-V0'
    # method_names_labels_dict = {
    #     # "PPO-AntWall": 'PPO',
    #     # "PPO-Lag-AntWall": 'PPO_lag',
    #     "GAIL_AntWall-v0_with-action": 'GACL',  # 'GAIL',
    #     "Binary_AntWall-v0_with-action_nit-50": 'BC2L',  # 'Binary',
    #     "ICRL_AntWall_with-action_nit-50": 'MECL',  # 'ICRL',
    #     "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1": "VICRL",
    #     # VICRL_AntWall-v0_with-action_no_is_nit-50_p-9-1, VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1, VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2
    # }
    env_id = 'highD_velocity_constraint'
    method_names_labels_dict = {
        # "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": 'PPO',
        # "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": 'PPO_lag',
        "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'GACL',
        "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'BC2L',
        "ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'MECL',
        "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": "VICRL",
    }
    # env_id = 'highD_velocity_constraint_dim2'
    # method_names_labels_dict = {
    #     # "PPO_highD_no-velocity": 'PPO',
    #     # "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": 'PPO_lag',
    #     "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'GACL',
    #     "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": 'BC2L',
    #     "ICRL_highD_velocity_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": 'MECL',
    #     "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": "VICRL",
    # }

    # env_id = 'highD_velocity_constraint_vm-30'
    # method_names_labels_dict = {
    #     "ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30": 'PPO',
    #     "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30": 'PPO_lag',
    # }

    # env_id = 'highD_velocity_constraint_vm-35'
    # method_names_labels_dict = {
    #     "ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35": 'PPO',
    #     "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35": 'PPO_lag',
    # }

    # env_id = 'highD_distance_constraint'
    # method_names_labels_dict = {
    #     # "ppo_highD_no_slo_distance_dm-20": 'PPO',
    #     # "ppo_lag_highD_no_slo_distance_dm-20": 'PPO_lag',
    #     'GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20': 'GACL',
    #     'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20': 'BC2L',
    #     'ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20': 'MECL',
    #     'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20': 'VICRL',
    # }

    # env_id = 'highD_distance_constraint_dim6'
    # method_names_labels_dict = {
    #     # "ppo_highD_no_slo_distance_dm-20": 'PPO',
    #     # "ppo_lag_highD_no_slo_distance_dm-20": 'PPO_lag',
    #     "GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20_dim-6": 'GACL',
    #     "Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": 'BC2L',
    #     "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": 'MECL',
    #     "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": 'VICRL',
    # }

    # env_id = 'highD_distance_constraint_dm-40'
    # method_names_labels_dict = {
    #     "ppo_highD_no_slo_distance_dm-40": 'PPO',
    #     "ppo_lag_highD_no_slo_distance_dm-40": 'PPO_lag',
    # }

    # env_id = 'highD_distance_constraint_dm-60'
    # method_names_labels_dict = {
    #     "ppo_highD_no_slo_distance_dm-60": 'PPO',
    #     "ppo_lag_highD_no_slo_distance_dm-60": 'PPO_lag',
    #     # 'GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-60': 'GACL',
    #     # 'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-60': 'BC2L',
    #     # 'ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60': 'MECL',
    #     # 'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60': 'VICRL',
    # }

    # env_id = 'InvertedPendulumWall-v0'
    # method_names_labels_dict = {
    #     # "PPO_Pendulum": 'PPO',
    #     # "PPO_lag_Pendulum": 'PPO_lag',
    #     "GAIL_PendulumWall": 'GACL',  # 'GAIL',
    #     "Binary_PendulumWall": 'BC2L',  # 'Binary',
    #     "ICRL_Pendulum": 'MECL',  # 'ICRL',
    #     "VICRL_PendulumWall": 'VICRL',
    # }
    # env_id = 'WalkerWithPos-v0'
    # method_names_labels_dict = {
    #     # "PPO_Walker": 'PPO',
    #     # "PPO_lag_Walker": 'PPO_lag',
    #     "GAIL_Walker": 'GACL',  # 'GACL'
    #     "Binary_Walker": 'BC2L',  # 'Binary
    #     "ICRL_Walker": 'MECL',  # 'ICRL',
    #     # "VICRL_Walker-v0_p-9e-3-1e-3": 'VICRL',
    #     "VICRL_Walker-v0_p-9e-3-1e-3_cl-64-64": 'VICRL',
    # }
    # env_id = 'SwimmerWithPos-v0'
    # method_names_labels_dict = {
    #     # "ppo_SwmWithPos-v0_update_b-5e-1": 'PPO',
    #     # "ppo_lag_SwmWithPos-v0_update_b-5e-1": 'PPO_lag',
    #     "GAIL_SwmWithPos-v0": 'GACL',
    #     "Binary_SwmWithPos-v0_update_b-5e-1": 'BC2L',
    #     "ICRL_SwmWithPos-v0_update_b-5e-1": 'MECL',
    #     "VICRL_SwmWithPos-v0_update_b-5e-1_piv-5": 'VICRL',
    # }
    modes = ['train']
    plot_mode = 'all'
    img_size = None
    axis_size = None
    if plot_mode == 'part':
        for method_name in method_names_labels_dict.copy().keys():
            if method_names_labels_dict[method_name] != 'PPO' and method_names_labels_dict[method_name] != 'PPO_lag':
                del method_names_labels_dict[method_name]
    else:
        method_names_labels_dict = method_names_labels_dict
    for mode in modes:
        # plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out']
        if env_id == 'highD_velocity_constraint':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8, 6.5]
            title = 'HighD Velocity Constraint'
            constraint_key = 'is_over_speed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'is_over_speed': None}
            log_path_dict = {
                "PPO_highD_velocity": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_velocity_penalty-multi_env-Mar-20-2022-10:21-seed_123/',
                    '../save_model/PPO-highD-velocity/train_ppo_highD_velocity_penalty-multi_env-Mar-21-2022-05:29-seed_123/',
                ],
                "PPO_highD_no-velocity": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty-multi_env-Mar-20-2022-10:18-seed_123/',
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty-multi_env-Mar-21-2022-05:30-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10-multi_env-Apr-02-2022-01:16-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": [
                    # '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-06-2022-11:28-seed_123/',
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-May-07-2022-02:01-seed_123/',
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-May-07-2022-03:35-seed_321/',
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-May-07-2022-07:05-seed_666/',
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-45": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-45-multi_env-Apr-05-2022-09:47-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-50": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50-multi_env-Apr-05-2022-09:45-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm--45": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm--45-multi_env-Apr-03-2022-04:07-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm--50": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm--50-multi_env-Apr-03-2022-04:02-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_vm-45": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_vm-45-multi_env-Apr-03-2022-04:10-seed_123/'
                ],
                'PPO_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40': [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-06-2022-11:29-seed_123/'
                ],
                'PPO_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-45': [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-45-multi_env-Apr-04-2022-01:46-seed_123/'
                ],
                'PPO_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50': [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50-multi_env-Apr-04-2022-01:47-seed_123/'
                ],
                'PPO_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-5e-1': [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-5e-1-multi_env-Apr-06-2022-06:10-seed_123/'
                ],
                'PPO_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-7e-1': [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-7e-1-multi_env-Apr-06-2022-06:07-seed_123/'
                ],
                'PPO_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-9e-1': [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-9e-1-multi_env-Apr-06-2022-06:11-seed_123/'
                ],
                'PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40': [
                    # '../save_model/PPO-Lag-highD/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-10-2022-12:45-seed_123/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-25-2022-13:34-seed_123/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-25-2022-13:35-seed_321/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-25-2022-13:46-seed_666/',
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40': [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-13-2022-12:42-seed_123/',
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40': [
                    # '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-19-2022-07:07-seed_123/',
                    # '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-19-2022-07:07-seed_321/',
                    # '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-19-2022-07:07-seed_666/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-28-2022-06:42-seed_123/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:50-seed_321/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:50-seed_666/',
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-14-2022-07:10-seed_123/'
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    # '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-18-2022-13:04-seed_123/',
                    # '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-18-2022-13:04-seed_321/',
                    # '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-18-2022-13:04-seed_666/',
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:48-seed_123/',
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:49-seed_321/',
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:49-seed_666/',

                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-15-2022-04:42-seed_123/',
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-15-2022-04:43-seed_123/',
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_cnl-64-64_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_cnl-64-64_vm-40-multi_env-Apr-15-2022-04:43-seed_123/'
                ],
                "VICRL_highD_velocity_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-17-2022-11:33-seed_123/'
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e2_fs-5k_nee-10_lr-5e-4_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e2_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-17-2022-10:06-seed_123/'
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e2_fs-5k_nee-10_lr-5e-4_cnl-64-64_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e2_fs-5k_nee-10_lr-5e-4_cnl-64-64_vm-40-multi_env-Apr-17-2022-10:03-seed_123/'
                ],
                "VICRL_highD_velocity_constraint_p-9e-2-1e-2_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-2-1e-2_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-17-2022-11:32-seed_123/'
                ],
                "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:48-seed_123/',
                    '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:48-seed_321/',
                    '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:48-seed_666/',
                ],
                "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:47-seed_123/',
                    '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:49-seed_321/',
                    '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:58-seed_666/',
                    # '../save_model/GAIL-highD/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-May-03-2022-06:44-seed_123/',
                ],
            }
        elif env_id == 'highD_velocity_constraint_dim2':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8.4, 6.5]
            title = 'Simplified HighD Velocity Constraint'
            constraint_key = 'is_over_speed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate']
            # plot_y_lim_dict = {'reward': (-50, 50),
            #                    'reward_nc': (0, 50),
            #                    'is_collision': (0, 1),
            #                    'is_off_road': (0, 1),
            #                    'is_goal_reached': (0, 1),
            #                    'is_time_out': (0, 1),
            #                    'avg_velocity': (20, 50),
            #                    'is_over_speed': (0, 1)}
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'is_over_speed': None}
            log_path_dict = {
                'PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40': [
                    # '../save_model/PPO-Lag-highD/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-10-2022-12:45-seed_123/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-25-2022-13:34-seed_123/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-25-2022-13:35-seed_321/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-25-2022-13:46-seed_666/',
                ],
                "ICRL_highD_velocity-dim2": [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-26-2022-00:37/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-26-2022-08:02/'
                ],
                "ICRL_highD_velocity-dim2-buff": [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-28-2022-06:56/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-28-2022-09:44/'
                ],
                "ICRL_highD_velocity-dim3-buff": [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_dim-3-multi_env-Mar-30-2022-06:05/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_dim-3-multi_env-Mar-31-2022-00:47-seed_321//'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dim-2': [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dim-2-multi_env-Apr-07-2022-23:34-seed_123/'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-08-2022-01:12-seed_123/'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-08-2022-01:12-seed_123/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-11-2022-00:31-seed_123/'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-11-2022-00:30-seed_123/',
                ],
                'ICRL_highD_velocity_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    # '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-13-2022-10:20-seed_123/',
                    # '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-01-2022-12:59-seed_123/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-01-2022-12:59-seed_321/',
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-01-2022-12:59-seed_666/',
                ],
                'ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-11-2022-04:12-seed_123/',
                ],
                "VICRL_highD_velocity-dim2-buff": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_no_is_p-1-1_dim-2-multi_env-Mar-31-2022-06:36-seed_123/'
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": [
                    # '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-14-2022-07:09-seed_123/',
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-01-2022-06:55-seed_123/',
                    # '../save_model/VICRL-highD/-velocitytrain_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-01-2022-07:04-seed_321/',
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-01-2022-07:23-seed_666/',
                ],
                "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": [
                    '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-14-2022-07:02-seed_123/'
                ],
                "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": [
                    '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-02-2022-20:10-seed_123/',
                    '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-02-2022-20:10-seed_321/',
                    '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-May-02-2022-20:10-seed_666/',
                ],
                "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                    '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:47-seed_123/',
                    '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:49-seed_321/',
                    '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:58-seed_666/',
                    # '../save_model/GAIL-highD/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-May-03-2022-06:44-seed_123/',
                ],

            }
        elif env_id == 'highD_velocity_constraint_vm-30':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8.4, 6.5]
            title = 'HighD Velocity Constraint (Velocity<30m/s)'
            constraint_key = 'is_over_speed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'is_over_speed': None}
            log_path_dict = {
                "ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30-multi_env-Jun-05-2022-12:46-seed_123/',
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30-multi_env-Jun-05-2022-13:01-seed_321/',
                ],
                'PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30': [
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30-multi_env-Jun-05-2022-13:01-seed_123/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30-multi_env-Jun-05-2022-16:37-seed_321/',
                ],

            }
        elif env_id == 'highD_velocity_constraint_vm-35':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8.4, 6.5]
            title = 'HighD Velocity Constraint (Velocity<35m/s)'
            constraint_key = 'is_over_speed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'is_over_speed': None}
            log_path_dict = {
                "ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35": [
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35-multi_env-Jun-05-2022-13:01-seed_123/',
                    '../save_model/PPO-highD-velocity/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35-multi_env-Jun-05-2022-13:01-seed_321/',
                ],
                'PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35': [
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35-multi_env-Jun-05-2022-15:29-seed_123/',
                    '../save_model/PPO-Lag-highD-velocity/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35-multi_env-Jun-05-2022-17:06-seed_321/',
                ],

            }
        elif env_id == 'highD_distance_constraint':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8, 6.5]
            title = 'HighD Distance Constraint'
            constraint_key = 'is_too_closed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
                        'is_too_closed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
                         'Over Closed Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'avg_distance': None,
                               'is_over_speed': None,
                               'is_too_closed': None}
            # plot_y_lim_dict = {'reward': (-50, 50),
            #                    'reward_nc': (0, 50),
            #                    'is_collision': (0, 1),
            #                    'is_off_road': (0, 1),
            #                    'is_goal_reached': (0, 1),
            #                    'is_time_out': (0, 1),
            #                    'avg_velocity': (20, 50),
            #                    'is_over_speed': (0, 1),
            #                    'avg_distance': (50, 100),
            #                    'is_too_closed': (0, 0.5)}
            log_path_dict = {
                "ppo_highD_no_slo_distance_dm-5": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-24-2022-00:31-seed_123/',
                    # '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-23-2022-04:26-seed_123/',
                ],
                "ppo_highD_no_slo_distance_dm-10": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-24-2022-00:31-seed_123/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:58-seed_321/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:58-seed_666/',
                ],
                "ppo_highD_no_slo_distance_dm-20": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-24-2022-00:52-seed_123/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:58-seed_321/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:58-seed_666/',
                ],
                "ppo_highD_no_slo_distance_dm-40": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-40-multi_env-Jun-03-2022-11:13-seed_123/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-40-multi_env-Jun-03-2022-11:13-seed_321/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-5": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-24-2022-00:53-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-26-2022-00:49-seed_321/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-26-2022-00:49-seed_666/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-10": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-24-2022-00:53-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:49-seed_321/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:49-seed_666/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-20": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-24-2022-00:53-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:54-seed_321/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:54-seed_666/',
                ],
                "ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-5": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-5-multi_env-May-25-2022-10:17-seed_123/',
                ],
                "ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-10": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-10-multi_env-May-25-2022-10:17-seed_123/',
                ],
                "ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-25-2022-10:17-seed_123/',
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-27-2022-08:59-seed_123/',
                ],
                "ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20-multi_env-May-27-2022-08:59-seed_123/',
                ],
                "ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20-multi_env-May-27-2022-08:59-seed_123/',
                ],
                "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-28-2022-09:58-seed_123/',
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-29-2022-01:50-seed_321/',
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-29-2022-01:50-seed_666/',
                ],
                "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20-multi_env-May-28-2022-09:58-seed_123/',
                ],
                "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20-multi_env-May-28-2022-09:58-seed_123/',
                ],
                "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20": [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20-multi_env-May-28-2022-10:19-seed_123/',
                ],
                "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20": [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-28-2022-10:19-seed_123/',
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-29-2022-01:50-seed_321/',
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-29-2022-01:50-seed_666/',
                ],
                "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20": [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20-multi_env-May-28-2022-10:19-seed_123/',
                ],
                "VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20": [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:39-seed_123/',
                ],
                "Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20": [
                    '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-30-2022-02:11-seed_123/',
                    '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:38-seed_321/',
                    '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:38-seed_666/',
                ],
                "GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20": [
                    '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20-multi_env-May-30-2022-02:11-seed_123/',
                    '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:39-seed_321/',
                    '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:40-seed_666/'
                ]
            }
        elif env_id == 'highD_distance_constraint_dim6':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8, 6.5]
            title = 'Simplified HighD Distance Constraint'
            constraint_key = 'is_too_closed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
                        'is_too_closed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
                         'Over Closed Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'avg_distance': None,
                               'is_over_speed': None,
                               'is_too_closed': None}
            # plot_y_lim_dict = {'reward': (-50, 50),
            #                    'reward_nc': (0, 50),
            #                    'is_collision': (0, 1),
            #                    'is_off_road': (0, 1),
            #                    'is_goal_reached': (0, 1),
            #                    'is_time_out': (0, 1),
            #                    'avg_velocity': (20, 50),
            #                    'is_over_speed': (0, 1),
            #                    'avg_distance': (50, 100),
            #                    'is_too_closed': (0, 0.5)}
            log_path_dict = {
                "ppo_highD_no_slo_distance_dm-5": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-24-2022-00:31-seed_123/',
                    # '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-23-2022-04:26-seed_123/',
                ],
                "ppo_highD_no_slo_distance_dm-10": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-24-2022-00:31-seed_123/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:58-seed_321/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:58-seed_666/',
                ],
                "ppo_highD_no_slo_distance_dm-20": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-24-2022-00:52-seed_123/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:58-seed_321/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:58-seed_666/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-5": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-24-2022-00:53-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-26-2022-00:49-seed_321/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-5-multi_env-May-26-2022-00:49-seed_666/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-10": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-24-2022-00:53-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:49-seed_321/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-10-multi_env-May-26-2022-00:49-seed_666/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-20": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-24-2022-00:53-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:54-seed_321/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-20-multi_env-May-26-2022-00:54-seed_666/',
                ],
                "GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20_dim-6": [
                    '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-02-2022-11:21-seed_123/',
                    '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-02-2022-11:21-seed_321/',
                    '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-02-2022-11:21-seed_666/',
                ],
                "Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": [
                    '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-02-2022-09:35-seed_123/',
                    '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-02-2022-09:35-seed_321/',
                    '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-02-2022-09:35-seed_666/',
                ],
                "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-May-30-2022-05:11-seed_123/',
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-01-2022-01:26-seed_321/',
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-01-2022-11:51-seed_666/',
                ],
                "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-May-30-2022-05:11-seed_123/',
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-01-2022-01:27-seed_321/',
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-01-2022-02:12-seed_666/',
                ],
                "VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-May-31-2022-04:39-seed_123/',
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-01-2022-01:27-seed_321/',
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6-multi_env-Jun-01-2022-02:12-seed_666/',
                ],
            }
        elif env_id == 'highD_distance_constraint_dm-40':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8, 6.5]
            title = 'HighD Distance Constraint (Distance>40m)'
            constraint_key = 'is_too_closed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
                        'is_too_closed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
                         'Over Closed Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'avg_distance': None,
                               'is_over_speed': None,
                               'is_too_closed': None}
            # plot_y_lim_dict = {'reward': (-50, 50),
            #                    'reward_nc': (0, 50),
            #                    'is_collision': (0, 1),
            #                    'is_off_road': (0, 1),
            #                    'is_goal_reached': (0, 1),
            #                    'is_time_out': (0, 1),
            #                    'avg_velocity': (20, 50),
            #                    'is_over_speed': (0, 1),
            #                    'avg_distance': (50, 100),
            #                    'is_too_closed': (0, 0.5)}
            log_path_dict = {
                "ppo_highD_no_slo_distance_dm-40": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-40-multi_env-Jun-03-2022-11:13-seed_123/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-40-multi_env-Jun-03-2022-11:13-seed_321/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-40": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-40-multi_env-Jun-03-2022-11:14-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-40-multi_env-Jun-03-2022-11:14-seed_321/',
                ],
            }
        elif env_id == 'highD_distance_constraint_dm-60':
            max_episodes = 5000
            average_num = 200
            max_reward = 50
            min_reward = -50
            axis_size = 20
            img_size = [8, 6.5]
            title = 'HighD Distance Constraint (Distance>60m)'
            constraint_key = 'is_too_closed'
            plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
                        'is_too_closed']
            label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                         'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
                         'Over Closed Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'reward_valid': None,
                               'is_collision': None,
                               'is_off_road': None,
                               'is_goal_reached': None,
                               'is_time_out': None,
                               'avg_velocity': None,
                               'avg_distance': None,
                               'is_over_speed': None,
                               'is_too_closed': None}
            # plot_y_lim_dict = {'reward': (-50, 50),
            #                    'reward_nc': (0, 50),
            #                    'is_collision': (0, 1),
            #                    'is_off_road': (0, 1),
            #                    'is_goal_reached': (0, 1),
            #                    'is_time_out': (0, 1),
            #                    'avg_velocity': (20, 50),
            #                    'is_over_speed': (0, 1),
            #                    'avg_distance': (50, 100),
            #                    'is_too_closed': (0, 0.5)}
            log_path_dict = {
                "ppo_highD_no_slo_distance_dm-60": [
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-60-multi_env-Jun-05-2022-13:01-seed_123/',
                    '../save_model/PPO-highD-distance/train_ppo_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-60-multi_env-Jun-05-2022-13:01-seed_321/',
                ],
                "ppo_lag_highD_no_slo_distance_dm-60": [
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-60-multi_env-Jun-05-2022-18:24-seed_123/',
                    '../save_model/PPO-Lag-highD-distance/train_ppo_lag_highD_no_slo_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_dm-60-multi_env-Jun-05-2022-19:16-seed_321/',
                ],
                'GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-60': [
                    '../save_model/GAIL-highD-distance/'
                ],
                'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-60': [
                    '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-60-multi_env-Jun-04-2022-11:34-seed_123/'
                ],
                'ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60': [
                    '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60-multi_env-Jun-04-2022-11:31-seed_123/'
                ],
                'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60': [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60-multi_env-Jun-04-2022-11:31-seed_123/'
                ],
                'VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60': [
                    '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-2-1e-2_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60-multi_env-Jun-04-2022-11:31-seed_123/'
                ]
            }
        elif env_id == 'HCWithPos-v0':
            max_episodes = 6000
            average_num = 100
            max_reward = 10000
            min_reward = -10000
            gap = 1
            plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
            label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
            # plot_key = ['reward', 'constraint']
            # label_key = ['reward', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': (0, 7000),
                               'reward_nc': (0, 6000),
                               'constraint': (0, 1.1),
                               'reward_valid': (0, 6000),
                               }
            title = 'Blocked Half-Cheetah'
            constraint_key = 'constraint'
            log_path_dict = {
                "PPO_Pos": [
                    '../save_model/PPO-HC/train_ppo_HCWithPos-v0-multi_env-Apr-06-2022-05:18-seed_123/',
                    '../save_model/PPO-HC/train_ppo_HCWithPos-v0-multi_env-Apr-07-2022-10:23-seed_321/',
                    '../save_model/PPO-HC/train_ppo_HCWithPos-v0-multi_env-Apr-07-2022-05:13-seed_666/'
                ],
                "PPO_lag_Pos": [
                    # '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-11-2022-07:16-seed_123/',
                    # '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-11-2022-07:18-seed_321/',
                    # '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-11-2022-07:18-seed_666/'
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-21-2022-04:49-seed_123/',
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-21-2022-06:27-seed_321/',
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-21-2022-08:05-seed_456/',
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-21-2022-09:42-seed_654/',
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-21-2022-11:18-seed_666/'
                ],
                "ICRL_Pos": [
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-06-2022-10:56-seed_123/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-06-2022-10:58-seed_321/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-06-2022-10:59-seed_666/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-07:36-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-05:43-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-07:16-seed_666/'
                ],
                "ICRL_Pos_crl-5e-3": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_crl-5e-3-multi_env-Apr-12-2022-12:46-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_crl-5e-3-multi_env-Apr-12-2022-12:49-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_crl-5e-3-multi_env-Apr-12-2022-13:01-seed_666/',
                ],
                "ICRL_Pos_with-buffer": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-12:19-seed_321/',
                ],
                "ICRL_Pos_with-buffer_with-action": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer-Apr-06-2022-10:56-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer-Apr-06-2022-10:58-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer-Apr-06-2022-10:59-seed_666/',
                ],
                "ICRL_Pos_with-buffer_with-action_crl-5e-3": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer_crl-5e-3-multi_env-Apr-12-2022-12:43-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer_crl-5e-3-multi_env-Apr-12-2022-12:49-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer_crl-5e-3-multi_env-Apr-12-2022-13:02-seed_666/',
                ],
                "ICRL_Pos_with-buffer-100k_with-action": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-23:50-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-23:50-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-23:51-seed_666/',
                ],
                "ICRL_Pos_with-action": [
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-Apr-06-2022-10:56-seed_123/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-Apr-06-2022-10:58-seed_321/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-Apr-06-2022-10:59-seed_666/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-08:54-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-10:43-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-12:29-seed_456/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-14:17-seed_654/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-16:04-seed_666/',
                ],
                "ICRL_Pos_with-action_crl-5e-3": [
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-12-2022-12:43-seed_123/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-12-2022-12:49-seed_321/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-12-2022-13:02-seed_666/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-21-2022-04:49-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-21-2022-06:42-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-21-2022-08:34-seed_456/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-21-2022-10:26-seed_654/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-21-2022-12:18-seed_666/'
                ],
                "VICRL_Pos": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_p-1-1_no_is-multi_env-Apr-07-2022-10:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_p-1-1_no_is-multi_env-Apr-07-2022-10:25-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_p-1-1_no_is-multi_env-Apr-07-2022-10:27-seed_666/',
                ],
                "VICRL_Pos_with-action": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_p-1-1_no_is-multi_env-Apr-07-2022-10:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_p-1-1_no_is-multi_env-Apr-07-2022-10:26-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_p-1-1_no_is-multi_env-Apr-07-2022-10:27-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-1_no_is-multi_env-Apr-07-2022-10:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-1_no_is-multi_env-Apr-07-2022-10:26-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-1_no_is-multi_env-Apr-07-2022-10:27-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action_p-1-9": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-9_no_is-multi_env-Apr-08-2022-01:00-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-9_no_is-multi_env-Apr-08-2022-01:01-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-9_no_is-multi_env-Apr-08-2022-01:06-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action_p-1e-1-9e-1": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1e-1-9e-1_no_is-multi_env-Apr-08-2022-01:00-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1e-1-9e-1_no_is-multi_env-Apr-08-2022-01:01-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1e-1-9e-1_no_is-multi_env-Apr-08-2022-01:06-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action_p-9-1": [
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-08-2022-04:32-seed_123/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-08-2022-04:32-seed_321/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-08-2022-04:33-seed_666/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-07:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-07:22-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-06:54-seed_666/'
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1": [
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-08-2022-04:32-seed_123/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-08-2022-04:33-seed_321/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-08-2022-04:33-seed_666/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-11-2022-06:56-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-11-2022-07:00-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-11-2022-07:01-seed_666/'
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-2-1e-2": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_no_is-multi_env-Apr-11-2022-11:17-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_no_is-multi_env-Apr-11-2022-11:18-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_no_is-multi_env-Apr-11-2022-11:21-seed_666/'
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-21-2022-05:00-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-21-2022-06:42-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-21-2022-08:25-seed_456/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-21-2022-10:08-seed_654/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-21-2022-11:52-seed_666/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-11-2022-11:22-seed_123/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-11-2022-11:18-seed_321/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-11-2022-11:21-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3_bs-64-1e3": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_bs-64-1e3_no_is-multi_env-Apr-11-2022-11:17-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_bs-64-1e3_no_is-multi_env-Apr-11-2022-11:18-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_bs-64-1e3_no_is-multi_env-Apr-11-2022-11:21-seed_666/'
                ],
                "Binary_HCWithPos-v0_with-action": [
                    '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-04:49-seed_123/',
                    '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-06:27-seed_321/',
                    '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-08:05-seed_456/',
                    '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-09:43-seed_654/',
                    '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-11:20-seed_666/',
                    # '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-20-2022-11:19-seed_123/',
                    # '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-20-2022-12:56-seed_321/',
                    # '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-20-2022-14:35-seed_456/',
                    # '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-20-2022-14:35-seed_456/',
                    # '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-20-2022-17:55-seed_666/',
                ],
                "GAIL_HCWithPos-v0_with-action": [
                    '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-05:55-seed_123/',
                    '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-07:32-seed_321/',
                    '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-09:18-seed_456/',
                    '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-11:02-seed_654/',
                    '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-12:45-seed_666/'
                    # '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-20-2022-07:27-seed_123/',
                    # '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-20-2022-08:16-seed_321/',
                    # '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-20-2022-09:03-seed_456/',
                    # '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-20-2022-09:50-seed_654/',
                    # '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-20-2022-10:37-seed_666/'
                ],
            }
        elif env_id == 'LGW-v0':
            max_episodes = 3000
            average_num = 100
            gap = 1
            max_reward = float('inf')
            min_reward = -float('inf')
            plot_key = ['reward', 'reward_nc', 'constraint']
            plot_y_lim_dict = {'reward': (0, 60),
                               'constraint': (0, 1)}
            log_path_dict = {
                'PPO_lag_LapGrid': [
                    '../save_model/PPO-Lag-LapGrid/train_ppo_lag_LGW-v0-multi_env-Apr-13-2022-13:24-seed_123/',
                    '../save_model/PPO-Lag-LapGrid/train_ppo_lag_LGW-v0-multi_env-Apr-13-2022-13:37-seed_321/',
                    '../save_model/PPO-Lag-LapGrid/train_ppo_lag_LGW-v0-multi_env-Apr-13-2022-13:50-seed_666/'
                ],
                'ICRL_LapGrid_with-action': [
                    '../save_model/ICRL-LapGrid/train_ICRL_LGW-v0_with-action-multi_env-Apr-14-2022-05:18-seed_123/',
                    '../save_model/ICRL-LapGrid/train_ICRL_LGW-v0_with-action-multi_env-Apr-14-2022-05:32-seed_321/',
                    '../save_model/ICRL-LapGrid/train_ICRL_LGW-v0_with-action-multi_env-Apr-14-2022-05:41-seed_666/'
                ],
                'VICRL-LapGrid_with-action': [
                    '../save_model/VICRL-LapGrid/train_VICRL_LGW-v0_with-action-multi_env-Apr-14-2022-05:18-seed_123/',
                    '../save_model/VICRL-LapGrid/train_VICRL_LGW-v0_with-action-multi_env-Apr-14-2022-05:31-seed_321/',
                    '../save_model/VICRL-LapGrid/train_VICRL_LGW-v0_with-action-multi_env-Apr-14-2022-05:40-seed_666/'
                ],
            }
        elif env_id == 'AntWall-V0':
            max_episodes = 15000
            average_num = 300
            title = 'Blocked Ant'
            max_reward = float('inf')
            min_reward = -float('inf')
            plot_key = ['reward', 'constraint', 'reward_valid']
            label_key = ['reward', 'Constraint Violation Rate', 'reward_valid']
            img_size = (6.7, 5.6)
            # plot_key = ['reward', 'constraint']
            # label_key = ['reward', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': (0, 0.5),
                               'reward_valid': None}
            # plot_y_lim_dict = {'reward': None,
            #                    'reward_nc': None,
            #                    'constraint': None,
            #                    'reward_valid': None}
            constraint_key = 'constraint'
            log_path_dict = {
                'PPO-AntWall': [
                    '../save_model/PPO-AntWall/train_ppo_AntWall-v0_nit-50-multi_env-Apr-19-2022-04:34-seed_123/',
                    '../save_model/PPO-AntWall/train_ppo_AntWall-v0_nit-50-multi_env-Apr-19-2022-08:24-seed_321/',
                    '../save_model/PPO-AntWall/train_ppo_AntWall-v0_nit-50-multi_env-Apr-19-2022-12:14-seed_456/',
                    '../save_model/PPO-AntWall/train_ppo_AntWall-v0_nit-50-multi_env-Apr-19-2022-16:02-seed_654/',
                    '../save_model/PPO-AntWall/train_ppo_AntWall-v0_nit-50-multi_env-Apr-19-2022-19:52-seed_666/',
                ],
                'PPO-Lag-AntWall': [
                    '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-v0_nit-50-multi_env-Apr-24-2022-12:21-seed_123/',
                    '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-v0_nit-50-multi_env-Apr-24-2022-17:08-seed_321/',
                    '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-v0_nit-50-multi_env-Apr-24-2022-21:50-seed_456/',
                    '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-v0_nit-50-multi_env-Apr-26-2022-06:13-seed_654/',
                    '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-v0_nit-50-multi_env-Apr-26-2022-16:18-seed_666/',
                    # '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-multi_env-Apr-15-2022-00:48-seed_123/',
                    # '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-multi_env-Apr-15-2022-02:57-seed_321/',
                    # '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-multi_env-Apr-15-2022-05:05-seed_456/',
                    # '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-multi_env-Apr-15-2022-07:11-seed_654/',
                    # '../save_model/PPO-Lag-AntWall/train_ppo_lag_AntWall-multi_env-Apr-15-2022-09:15-seed_666/',
                ],
                'ICRL_AntWall-v0_with-action_no_is_nit-100': [
                    '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action_nit-100-multi_env-Apr-16-2022-00:51-seed_123/'
                ],
                'ICRL_AntWall_with-action': [
                    '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-11:36-seed_123/',
                    '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-14:25-seed_321/',
                    '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-16:53-seed_456/',
                    '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-19:26-seed_654/',
                    '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-21:58-seed_666/',
                ],
                'ICRL_AntWall_with-action_nit-50': [
                    '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0_with-action_nit-50-multi_env-Apr-25-2022-03:01-seed_123/',
                    '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0_with-action_nit-50-multi_env-Apr-25-2022-08:21-seed_321/',
                    '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0_with-action_nit-50-multi_env-Apr-25-2022-13:49-seed_456/',
                    '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0_with-action_nit-50-multi_env-Apr-26-2022-05:16-seed_654/',
                    '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0_with-action_nit-50-multi_env-Apr-26-2022-14:43-seed_666/',
                    # '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action_nit-50-multi_env-Apr-18-2022-00:12-seed_123/',
                    # '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action_nit-50-multi_env-Apr-18-2022-06:46-seed_321/',
                    # '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action_nit-50-multi_env-Apr-18-2022-12:27-seed_456/',
                    # '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action_nit-50-multi_env-Apr-18-2022-19:00-seed_654/',
                    # '../save_model/ICRL-AntWall/train_ICRL_lag_AntWall-v0_with-action_nit-50-multi_env-Apr-17-2022-09:53-seed_123/',
                ],
                'VICRL_AntWall-v0_with-action_no_is_nit-100': [
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-100-multi_env-Apr-16-2022-00:52-seed_123/'
                ],
                # 'VICRL_AntWall_with-action': [
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-11:36-seed_123/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-14:27-seed_321/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-14:36-seed_456/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-15:04-seed_654/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action-multi_env-Apr-15-2022-15:14-seed_666/',
                # ],
                # 'VICRL_AntWall_with-action_no_is': [
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is-multi_env-Apr-16-2022-00:49-seed_123/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is-multi_env-Apr-16-2022-03:25-seed_321/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is-multi_env-Apr-16-2022-05:57-seed_456/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is-multi_env-Apr-16-2022-08:30-seed_654/',
                #     '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is-multi_env-Apr-16-2022-11:07-seed_666/',
                # ],
                'VICRL_AntWall-v0_with-action_no_is_nit-50_p-1e-1-9e-1': [
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1e-9-1e-1-multi_env-Apr-17-2022-09:53-seed_123/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1e-9-1e-1-multi_env-Apr-17-2022-15:24-seed_321/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1e-9-1e-1-multi_env-Apr-17-2022-20:47-seed_456/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1e-9-1e-1-multi_env-Apr-18-2022-02:19-seed_654/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1e-9-1e-1-multi_env-Apr-18-2022-08:05-seed_666/',
                ],
                'VICRL_AntWall-v0_with-action_no_is_nit-50_p-1e-2-9e-2': [
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-1e-2-9e-2-multi_env-Apr-24-2022-12:21-seed_123/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-1e-2-9e-2-multi_env-Apr-24-2022-17:28-seed_321/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-1e-2-9e-2-multi_env-Apr-26-2022-05:52-seed_456/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-1e-2-9e-2-multi_env-Apr-26-2022-15:55-seed_654/',
                ],
                'VICRL_AntWall-v0_with-action_no_is_nit-50_p-1-9': [
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1-9-multi_env-Apr-18-2022-00:12-seed_123/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1-9-multi_env-Apr-18-2022-06:06-seed_321/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1-9-multi_env-Apr-18-2022-11:07-seed_456/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1-9-multi_env-Apr-18-2022-16:44-seed_654/',
                    '../save_model/VICRL-AntWall/train_VICRL_lag_AntWall-v0_with-action_no_is_nit-50_p-1-9-multi_env-Apr-18-2022-22:34-seed_666/',
                ],
                'VICRL_AntWall-v0_with-action_no_is_nit-50_p-9-1': [
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9-1-multi_env-May-15-2022-05:23-seed_123/',
                ],
                'VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1': [
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1-multi_env-May-15-2022-05:08-seed_123/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1-multi_env-May-15-2022-17:19-seed_321/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1-multi_env-May-15-2022-22:34-seed_456/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1-multi_env-May-16-2022-03:43-seed_654/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1-multi_env-May-16-2022-08:48-seed_666/',
                ],
                'VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2': [
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2-multi_env-May-15-2022-04:43-seed_123/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2-multi_env-May-15-2022-10:40-seed_123/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2-multi_env-May-15-2022-15:46-seed_321/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2-multi_env-May-15-2022-20:50-seed_456/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2-multi_env-May-16-2022-01:51-seed_654/',
                    '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2-multi_env-May-16-2022-06:57-seed_666/',
                ],
                'GAIL_AntWall-v0_with-action': [
                    '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_with-action-multi_env-Apr-25-2022-19:22-seed_123/',
                    '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_with-action-multi_env-Apr-25-2022-11:40-seed_321/',
                    '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_with-action-multi_env-Apr-26-2022-06:12-seed_456/',
                    '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_with-action-multi_env-Apr-26-2022-15:59-seed_654/',
                    '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_with-action-multi_env-Apr-26-2022-21:51-seed_666/',
                ],
                'Binary_AntWall-v0_with-action_nit-50': [
                    # '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-25-2022-05:37-seed_123/',
                    # '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-26-2022-05:52-seed_321/',
                    # '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-26-2022-16:17-seed_456/',
                    # '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-26-2022-21:21-seed_654/',
                    # '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-27-2022-02:33-seed_666/',
                    '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-27-2022-12:09-seed_123/',
                    '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-27-2022-17:30-seed_321/',
                    '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-27-2022-22:42-seed_456/',
                    '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-28-2022-03:58-seed_654/',
                    # '../save_model/Binary-AntWall/train_Binary_AntWall-v0_with-action_nit-50-multi_env-Apr-28-2022-09:09-seed_666/',
                ],
            }
        elif env_id == 'InvertedPendulumWall-v0':
            max_episodes = 80000
            average_num = 2000
            title = 'Biased Pendulumn'
            max_reward = float('inf')
            min_reward = -float('inf')
            plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
            label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
            # plot_y_lim_dict = {'reward': (0, 100),
            #                    'reward_nc': (0, 100),
            #                    'constraint': (0, 1.1)}
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None,
                               'reward_valid': None}
            constraint_key = 'constraint'
            log_path_dict = {
                'PPO_Pendulum': [
                    # '../save_model/PPO-InvertedPendulumWall/train_ppo_InvertedPendulumWall-v0-multi_env-Apr-28-2022-13:01-seed_123/',
                    # '../save_model/PPO-InvertedPendulumWall/train_ppo_InvertedPendulumWall-v0-multi_env-Apr-30-2022-09:19-seed_123/',
                    '../save_model/PPO-InvertedPendulumWall/train_ppo_InvertedPendulumWall-v0-multi_env-May-06-2022-05:17-seed_123/',
                    '../save_model/PPO-InvertedPendulumWall/train_ppo_InvertedPendulumWall-v0-multi_env-May-06-2022-06:30-seed_321/',
                    '../save_model/PPO-InvertedPendulumWall/train_ppo_InvertedPendulumWall-v0-multi_env-May-06-2022-10:54-seed_456/',
                    '../save_model/PPO-InvertedPendulumWall/train_ppo_InvertedPendulumWall-v0-multi_env-May-06-2022-13:39-seed_654/',
                    '../save_model/PPO-InvertedPendulumWall/train_ppo_InvertedPendulumWall-v0-multi_env-May-06-2022-18:04-seed_666/',
                ],
                'PPO_lag_Pendulum': [
                    # '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-multi_env-Apr-30-2022-09:20-seed_123/',
                    '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-multi_env-May-05-2022-07:18-seed_123/',
                    '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-multi_env-May-05-2022-12:02-seed_321/',
                    '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-multi_env-May-05-2022-16:52-seed_456/',
                    '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-multi_env-May-05-2022-20:44-seed_654/',
                    '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-multi_env-May-06-2022-01:27-seed_666/',
                    # '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-multi_env-Apr-29-2022-05:41-seed_123/',
                ],
                'ICRL_Pendulum': [
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-10-2022-09:17-seed_123/',
                    '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-10-2022-14:51-seed_321/',
                    '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-10-2022-19:19-seed_456/',
                    '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-10-2022-23:49-seed_654/',
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-11-2022-04:20-seed_666/',
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0-multi_env-May-01-2022-06:09-seed_123/',
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2-multi_env-May-04-2022-05:57-seed_123/',
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2-multi_env-May-09-2022-14:46-seed_456/',
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2-multi_env-May-09-2022-12:17-seed_456/',
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2-multi_env-May-09-2022-17:48-seed_654/',
                    # '../save_model/ICRL-InvertedPendulumWall/train_ICRL_InvertedPendulumWall-v0_prl-1e-2-multi_env-May-09-2022-23:05-seed_666/'
                ],
                'VICRL_PendulumWall': [
                    # '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-1e-2-9e-2-multi_env-May-07-2022-01:40-seed_123/',
                    # '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-1e-2-9e-2-multi_env-May-09-2022-12:09-seed_123/',
                    # '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-1e-2-9e-2-multi_env-May-09-2022-16:26-seed_321/',
                    '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-9e-2-1e-2-multi_env-May-15-2022-05:27-seed_123/',
                    '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-9e-2-1e-2-multi_env-May-15-2022-11:00-seed_123/',
                    '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-9e-2-1e-2-multi_env-May-15-2022-14:58-seed_321/',
                    # '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-9e-2-1e-2-multi_env-May-15-2022-18:46-seed_456/',
                    '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-9e-2-1e-2-multi_env-May-15-2022-23:11-seed_654/',
                    '../save_model/VICRL-InvertedPendulumWall/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5_p-9e-2-1e-2-multi_env-May-16-2022-03:23-seed_666/',
                ],
                'Binary_PendulumWall': [
                    # '../save_model/Binary-InvertedPendulumWall/train_Binary_InvertedPendulumWall-v0_prl-1e-2-multi_env-May-09-2022-12:25-seed_123/',
                    # '../save_model/Binary-InvertedPendulumWall/train_Binary_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-10-2022-10:43-seed_123/',
                    '../save_model/Binary-InvertedPendulumWall/train_Binary_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-10-2022-17:34-seed_321/',
                    '../save_model/Binary-InvertedPendulumWall/train_Binary_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-10-2022-21:52-seed_456/',
                    '../save_model/Binary-InvertedPendulumWall/train_Binary_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-11-2022-02:09-seed_654/',
                    '../save_model/Binary-InvertedPendulumWall/train_Binary_InvertedPendulumWall-v0_prl-1e-2_lr-3e-5-multi_env-May-11-2022-07:52-seed_666/',
                ],
                'GAIL_PendulumWall': [
                    '../save_model/GAIL-InvertedPendulumWall/train_GAIL_disclr-1e-5_InvertedPendulumWall-v0-multi_env-May-10-2022-12:55-seed_123/',
                    '../save_model/GAIL-InvertedPendulumWall/train_GAIL_disclr-1e-5_InvertedPendulumWall-v0-multi_env-May-23-2022-13:26-seed_123/',
                    '../save_model/GAIL-InvertedPendulumWall/train_GAIL_disclr-1e-5_InvertedPendulumWall-v0-multi_env-May-23-2022-19:43-seed_321/',
                    '../save_model/GAIL-InvertedPendulumWall/train_GAIL_disclr-1e-5_InvertedPendulumWall-v0-multi_env-May-24-2022-02:06-seed_456/',
                ]
            }
        elif env_id == 'WalkerWithPos-v0':
            max_episodes = 40000
            average_num = 2000
            title = 'Blocked Walker'
            max_reward = float('inf')
            min_reward = -float('inf')
            plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
            label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
            # plot_y_lim_dict = {'reward': (0, 700),
            #                    'reward_nc': (0, 700),
            #                    'constraint': (0, 1)}
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None,
                               'reward_valid': None}
            constraint_key = 'constraint'
            log_path_dict = {
                'PPO_Walker': [
                    '../save_model/PPO-Walker/train_ppo_WalkerWithPos-v0-multi_env-May-06-2022-06:56-seed_123/',
                    '../save_model/PPO-Walker/train_ppo_WalkerWithPos-v0-multi_env-May-06-2022-10:37-seed_321/',
                    '../save_model/PPO-Walker/train_ppo_WalkerWithPos-v0-multi_env-May-06-2022-14:35-seed_456/',
                    '../save_model/PPO-Walker/train_ppo_WalkerWithPos-v0-multi_env-May-06-2022-18:18-seed_654/',
                    '../save_model/PPO-Walker/train_ppo_WalkerWithPos-v0-multi_env-May-06-2022-19:37-seed_666/',
                ],
                'PPO_lag_Walker': [
                    # '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-multi_env-May-04-2022-12:26-seed_123/',
                    '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-multi_env-May-07-2022-01:19-seed_123/',
                    '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-multi_env-May-07-2022-06:01-seed_321/',
                    '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-multi_env-May-07-2022-10:43-seed_456/',
                    '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-multi_env-May-07-2022-15:29-seed_654/',
                    '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-multi_env-May-07-2022-20:05-seed_666/',
                ],
                'Binary_Walker': [
                    '../save_model/Binary-WalkerWithPos/train_Binary_WalkerWithPos-v0-multi_env-May-24-2022-01:09-seed_123/',
                    '../save_model/Binary-WalkerWithPos/train_Binary_WalkerWithPos-v0-multi_env-May-24-2022-06:33-seed_123/',
                    '../save_model/Binary-WalkerWithPos/train_Binary_WalkerWithPos-v0-multi_env-May-24-2022-11:45-seed_321/',
                    '../save_model/Binary-WalkerWithPos/train_Binary_WalkerWithPos-v0-multi_env-May-24-2022-16:56-seed_456/',
                    '../save_model/Binary-WalkerWithPos/train_Binary_WalkerWithPos-v0-multi_env-May-24-2022-22:07-seed_654/',
                ],
                'GAIL_Walker': [
                    '../save_model/GAIL-WalkerWithPos/train_GAIL_WalkerWithPos-v0-multi_env-May-23-2022-14:30-seed_123/',
                ],
                'ICRL_Walker': [
                    '../save_model/ICRL-WalkerWithPos/train_ICRL_WalkerWithPos-v0-multi_env-May-18-2022-10:41-seed_123/',
                    '../save_model/ICRL-WalkerWithPos/train_ICRL_WalkerWithPos-v0-multi_env-May-18-2022-21:04-seed_321/',
                    '../save_model/ICRL-WalkerWithPos/train_ICRL_WalkerWithPos-v0-multi_env-May-19-2022-02:11-seed_456/',
                    '../save_model/ICRL-WalkerWithPos/train_ICRL_WalkerWithPos-v0-multi_env-May-19-2022-07:27-seed_654/',
                    '../save_model/ICRL-WalkerWithPos/train_ICRL_WalkerWithPos-v0-multi_env-May-19-2022-12:49-seed_666/',
                ],
                'VICRL_Walker-v0_p-9e-3-1e-3': [
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3-multi_env-May-18-2022-11:38-seed_123/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3-multi_env-May-18-2022-17:41-seed_321/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3-multi_env-May-18-2022-23:57-seed_456/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3-multi_env-May-19-2022-06:14-seed_654/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3-multi_env-May-19-2022-12:32-seed_666/',
                ],
                'VICRL_Walker-v0_p-9e-3-1e-3_cl-64-64': [
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-2-1e-2_cl-64-64-multi_env-May-17-2022-01:39-seed_123/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3_cl-64-64-multi_env-May-18-2022-10:42-seed_321/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3_cl-64-64-multi_env-May-18-2022-17:19-seed_456/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3_cl-64-64-multi_env-May-18-2022-23:31-seed_654/',
                    '../save_model/VICRL-WalkerWithPos/train_VICRL_WalkerWithPos-v0_p-9e-3-1e-3_cl-64-64-multi_env-May-19-2022-05:32-seed_666/',
                ],
            }
        elif env_id == 'SwimmerWithPos-v0':
            max_episodes = 10000
            average_num = 200
            title = 'Blocked Swimmer'
            max_reward = float('inf')
            min_reward = -float('inf')
            plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
            label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None,
                               'reward_valid': None}
            constraint_key = 'constraint'
            log_path_dict = {
                # 'train_ppo_SwmWithPos-v0': [
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0-multi_env-May-26-2022-00:09-seed_123/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0-multi_env-May-26-2022-05:15-seed_321/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0-multi_env-May-26-2022-09:50-seed_456/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0-multi_env-May-26-2022-14:51-seed_654/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0-multi_env-May-26-2022-20:08-seed_666/',
                # ],
                # 'train_ppo_SwmWithPos-v0_b--1': [
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_b--1-multi_env-May-27-2022-00:05-seed_123/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_b--1-multi_env-May-27-2022-00:29-seed_321/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_b--1-multi_env-May-27-2022-00:53-seed_456/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_b--1-multi_env-May-27-2022-01:17-seed_654/',
                #     '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_b--1-multi_env-May-27-2022-01:40-seed_666/',
                # ],
                'ppo_SwmWithPos-v0_update_b-5e-1': [
                    '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-10:10-seed_123/',
                    '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-11:28-seed_321/',
                    '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-12:46-seed_456/',
                    '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-14:14-seed_654/',
                    '../save_model/PPO-Swm/train_ppo_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-15:39-seed_666/',
                ],
                # 'ppo_lag_SwmWithPos-v0': [
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0-multi_env-May-26-2022-00:09-seed_123/',
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0-multi_env-May-26-2022-08:53-seed_321/',
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0-multi_env-May-26-2022-18:01-seed_456/',
                # ],
                # 'ppo_lag_SwmWithPos-v0_b--1': [
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_b--1-multi_env-May-27-2022-00:05-seed_123/',
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_b--1-multi_env-May-27-2022-00:32-seed_321/',
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_b--1-multi_env-May-27-2022-01:00-seed_456/',
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_b--1-multi_env-May-27-2022-01:27-seed_654/',
                #     '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_b--1-multi_env-May-27-2022-01:53-seed_666/',
                # ],
                'ppo_lag_SwmWithPos-v0_update_b-5e-1': [
                    '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-10:10-seed_123/',
                    '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-11:15-seed_321/',
                    '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-12:21-seed_456/',
                    '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-13:26-seed_654/',
                    '../save_model/PPO-Lag-Swm/train_ppo_lag_SwmWithPos-v0_update_b-5e-1-multi_env-May-29-2022-14:34-seed_666/',
                ],
                "GAIL_SwmWithPos-v0": [
                    '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0-multi_env-Jun-01-2022-00:12-seed_123/',
                    '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0-multi_env-Jun-01-2022-02:08-seed_321/',
                    '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0-multi_env-Jun-01-2022-04:00-seed_456/',
                    '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0-multi_env-Jun-01-2022-05:51-seed_654/',
                    '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0-multi_env-Jun-01-2022-07:44-seed_666/',
                ],
                "Binary_SwmWithPos-v0_update_b-5e-1": [
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1-multi_env-Jun-01-2022-00:11-seed_123/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1-multi_env-Jun-01-2022-02:23-seed_321/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1-multi_env-Jun-01-2022-04:33-seed_456/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1-multi_env-Jun-01-2022-06:41-seed_654/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1-multi_env-Jun-01-2022-08:50-seed_666/',
                ],
                "Binary_SwmWithPos-v0_update_b-5e-1_piv-5": [
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-00:12-seed_123/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-02:17-seed_321/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-04:22-seed_456/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-06:28-seed_654/',
                    '../save_model/Binary-Swm/train_Binary_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-08:34-seed_666/',
                ],
                'ICRL_SwmWithPos-v0_update_b-5e-1': [
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-01:45-seed_123/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-03:52-seed_321/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-06:01-seed_456/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-08:08-seed_654/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-10:15-seed_666/',
                ],
                'ICRL_SwmWithPos-v0_update_b-5e-1_piv-5': [
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-00:12-seed_321/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-00:12-seed_321/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-02:19-seed_456/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-04:27-seed_654/',
                    '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-06:35-seed_666/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1': [
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-05:08-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-07:23-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-09:40-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1-multi_env-May-30-2022-11:54-seed_654/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1_cl-64-64': [
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_cl-64-64-multi_env-May-30-2022-14:35-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_cl-64-64-multi_env-May-30-2022-16:48-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_cl-64-64-multi_env-May-30-2022-19:00-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_cl-64-64-multi_env-May-30-2022-21:10-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_cl-64-64-multi_env-May-30-2022-23:22-seed_666/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1_plr-1': [
                    # '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_plr-1-multi_env-May-30-2022-14:36-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_plr-1-multi_env-May-30-2022-14:36-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_plr-1-multi_env-May-30-2022-16:46-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_plr-1-multi_env-May-30-2022-18:54-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_plr-1-multi_env-May-30-2022-21:02-seed_666/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-2-1e-2': [
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-2-1e-2-multi_env-May-30-2022-14:36-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-2-1e-2-multi_env-May-30-2022-16:45-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-2-1e-2-multi_env-May-30-2022-18:57-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-2-1e-2-multi_env-May-30-2022-21:07-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-2-1e-2-multi_env-May-30-2022-23:18-seed_666/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1_p-9-1': [
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9-1-multi_env-May-31-2022-01:58-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9-1-multi_env-May-31-2022-06:02-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9-1-multi_env-May-31-2022-08:27-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9-1-multi_env-May-31-2022-10:46-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9-1-multi_env-May-31-2022-13:04-seed_666/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-3-1e-3': [
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-3-1e-3-multi_env-May-31-2022-03:37-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-3-1e-3-multi_env-May-31-2022-06:00-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-3-1e-3-multi_env-May-31-2022-08:26-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-3-1e-3-multi_env-May-31-2022-10:47-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-9e-3-1e-3-multi_env-May-31-2022-13:06-seed_666/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1_piv-5': [
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-Jun-01-2022-00:12-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-May-31-2022-01:58-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-May-31-2022-04:19-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-May-31-2022-06:40-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-5-multi_env-May-31-2022-09:03-seed_666/',
                ],
                'VICRL_SwmWithPos-v0_update_b-5e-1_piv-10': [
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-10-multi_env-May-31-2022-05:01-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-10-multi_env-May-31-2022-07:21-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-10-multi_env-May-31-2022-09:34-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-10-multi_env-May-31-2022-11:46-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-10-multi_env-May-31-2022-14:07-seed_666/',
                ],
                "VICRL_SwmWithPos-v0_update_b-5e-1_p-90-10": [
                    # '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-90-10-multi_env-Jun-01-2022-00:12-seed_123/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-90-10-multi_env-Jun-01-2022-00:12-seed_321/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-90-10-multi_env-Jun-01-2022-02:32-seed_456/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-90-10-multi_env-Jun-01-2022-05:00-seed_654/',
                    '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_update_b-5e-1_p-90-10-multi_env-Jun-01-2022-07:31-seed_666/',
                ]
            }
        else:
            raise ValueError("Unknown env id {0}".format(env_id))

        all_mean_dict = {}
        all_std_dict = {}
        all_episodes_dict = {}
        for method_name in method_names_labels_dict.keys():
            all_results = []
            all_valid_rewards = []
            all_valid_episodes = []
            for log_path in log_path_dict[method_name]:
                monitor_path_all = []
                if mode == 'train':
                    run_files = os.listdir(log_path)
                    for file in run_files:
                        if 'monitor' in file:
                            monitor_path_all.append(log_path + file)
                else:
                    monitor_path_all.append(log_path + 'test/test.monitor.csv')

                # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
                results, valid_rewards, valid_episodes = read_running_logs(monitor_path_all=monitor_path_all,
                                                                           read_keys=plot_key,
                                                                           max_reward=max_reward, min_reward=min_reward,
                                                                           max_episodes=max_episodes + float(
                                                                               max_episodes / 5),
                                                                           constraint_key=constraint_key)
                all_valid_rewards.append(valid_rewards)
                all_valid_episodes.append(valid_episodes)
                all_results.append(results)

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
                all_std_dict[method_name].update({plot_key[idx]: std_results_moving_average})
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
                             )
        for idx in range(len(plot_key)):
            mean_results_moving_avg_dict = {}
            std_results_moving_avg_dict = {}
            espisode_dict = {}
            for method_name in method_names_labels_dict.keys():
                mean_results_moving_avg_dict.update({method_name: all_mean_dict[method_name][plot_key[idx]]})
                std_results_moving_avg_dict.update({method_name: all_std_dict[method_name][plot_key[idx]]})
                espisode_dict.update({method_name: all_episodes_dict[method_name][plot_key[idx]]})
                if (plot_key[idx] == 'reward_nc' or plot_key[idx] == 'constraint') and mode == 'test':
                    print(method_name, plot_key[idx],
                          all_mean_dict[method_name][plot_key[idx]][-1],
                          all_std_dict[method_name][plot_key[idx]][-1])
            plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                         std_results_moving_avg_dict=std_results_moving_avg_dict,
                         episode_plots=espisode_dict,
                         label=label_key[idx],
                         method_names=list(method_names_labels_dict.keys()),
                         ylim=plot_y_lim_dict[plot_key[idx]],
                         save_label=os.path.join(env_id, plot_key[idx] + '_' + mode + '_' + env_id + '_' + plot_mode),
                         # legend_size=18,
                         legend_dict=method_names_labels_dict,
                         title=title,
                         axis_size=axis_size,
                         img_size=img_size,
                         )


if __name__ == "__main__":
    generate_plots()
