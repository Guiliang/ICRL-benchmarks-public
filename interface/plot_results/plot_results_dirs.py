def get_plot_results_dir(env_id):
    if env_id == 'highD_velocity_constraint':
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
                # '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-28-2022-06:42-seed_123/',
                # '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:50-seed_321/',
                # '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:50-seed_666/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Sep-20-2022-09:11-seed_123/'
            ],
            "ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40": [
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-26-2022-07:33-seed_123/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-26-2022-13:00-seed_321/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-26-2022-13:06-seed_666/',
            ],
            "ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40": [
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-06-2023-16:00-seed_123/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-07-2023-10:22-seed_321/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-08-2023-02:03-seed_666/',
            ],
            "ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-2": [
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-2-multi_env-Jan-09-2023-17:24-seed_123/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-2-multi_env-Jan-09-2023-19:45-seed_321/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-2-multi_env-Jan-09-2023-22:06-seed_666/',
            ],
            "ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1": [
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-06-2023-15:58-seed_123/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-06-2023-21:11-seed_321/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-07-2023-02:15-seed_666/',
            ],
            "ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1": [
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-06-2023-15:58-seed_123/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-07-2023-00:11-seed_321/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-07-2023-07:23-seed_666/',
            ],
            "ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1": [
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-06-2023-15:58-seed_123/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-07-2023-03:41-seed_321/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-07-2023-13:50-seed_666/',
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
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40-multi_env-Sep-18-2022-12:15-seed_123/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-18-2022-12:15-seed_123/',
                # '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-22-2022-06:33-seed_321/',
                # '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-22-2022-06:33-seed_666/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-24-2022-05:28-seed_321/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-24-2022-05:28-seed_666/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40-multi_env-Sep-25-2022-03:21-seed_123/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40-multi_env-Sep-25-2022-03:51-seed_321/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40-multi_env-Sep-25-2022-03:55-seed_666/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40-multi_env-Jan-04-2023-19:40-seed_123/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40-multi_env-Jan-05-2023-06:24-seed_321/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40-multi_env-Jan-05-2023-16:39-seed_666/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-1e-1": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-04-2023-19:41-seed_123/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-04-2023-22:34-seed_321/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-05-2023-01:24-seed_666/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-3e-1": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-04-2023-19:41-seed_123/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-05-2023-00:35-seed_321/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-05-2023-05:09-seed_666/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-5e-1": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-04-2023-19:42-seed_123/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-05-2023-01:50-seed_321/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-05-2023-08:05-seed_666/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4-plr-5e-2_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4-plr-5e-2_no-buffer_vm-40-multi_env-Sep-18-2022-12:15-seed_123/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4-plr-5e-3_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4-plr-5e-3_no-buffer_vm-40-multi_env-Sep-18-2022-12:16-seed_123/',
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
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-1e-1": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-1e-1-multi_env-Sep-26-2022-12:42-seed_123/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-5e-1": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-5e-1-multi_env-Sep-26-2022-12:42-seed_123/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-7e-1": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-7e-1-multi_env-Sep-26-2022-12:42-seed_123/',
            ],
            "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-9e-1": [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-9e-1-multi_env-Sep-26-2022-12:42-seed_123/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-9e-1-multi_env-Sep-27-2022-02:10-seed_321/',
                # '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-9e-1-multi_env-Sep-27-2022-02:10-seed_666/',
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-Sep-24-2022-12:56-seed_321/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_plr-5e-4_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_plr-5e-4_no-buffer_vm-40-Sep-24-2022-13:03-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-5_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-5_no-buffer_vm-40-Sep-24-2022-12:57-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-5_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-5_no-buffer_vm-40-Sep-24-2022-17:05-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-6_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-6_no-buffer_vm-40-Sep-24-2022-17:06-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-5_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-5_no-buffer_vm-40-Sep-24-2022-17:04-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-2-1e-2_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-2-1e-2_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-Sep-24-2022-20:48-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_plr-1e-2_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_plr-1e-2_no-buffer_vm-40-Sep-24-2022-20:48-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_cl-64-64_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_cl-64-64_no-buffer_vm-40-Sep-25-2022-00:29-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-7e-1_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-7e-1_no-buffer_vm-40-Sep-25-2022-00:10-seed_123/'
            ],
            "debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40": [
                '../save_model/VICRL-highD-velocity/debug-part-train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40-Sep-25-2022-00:06-seed_123/'
            ],
            "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:48-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:48-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-29-2022-04:48-seed_666/',
            ],
            "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40": [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-26-2022-12:42-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-26-2022-12:59-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40-multi_env-Sep-26-2022-13:00-seed_666/',
            ],
            "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40": [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-06-2023-15:52-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-07-2023-09:57-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-08-2023-03:10-seed_666/',
            ],
            "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1": [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-06-2023-15:52-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-07-2023-04:10-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-07-2023-13:36-seed_666/',
            ],
            "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1": [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-06-2023-15:52-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-07-2023-00:24-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-07-2023-08:12-seed_666/',
            ],
            "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1": [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-06-2023-15:52-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-06-2023-21:01-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-07-2023-01:55-seed_666/',
            ],
            "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-2e-1": [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-2-multi_env-Jan-09-2023-17:24-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-2-multi_env-Jan-09-2023-19:53-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_vm-40_data-1e-2-multi_env-Jan-09-2023-22:15-seed_666/',
            ],
            "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": [
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:47-seed_123/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:49-seed_321/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-Apr-30-2022-13:58-seed_666/',
                # '../save_model/GAIL-highD/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40-multi_env-May-03-2022-06:44-seed_123/',
            ],
            "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40": [
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-06-2023-15:56-seed_123/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-06-2023-22:55-seed_321/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40-multi_env-Jan-07-2023-06:16-seed_666/',
            ],
            "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-1e-1": [
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-06-2023-15:57-seed_123/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-06-2023-22:54-seed_321/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-1e-1-multi_env-Jan-07-2023-05:51-seed_666/',
            ],
            "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-3e-1": [
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-06-2023-15:57-seed_123/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-06-2023-22:14-seed_321/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-3e-1-multi_env-Jan-07-2023-05:24-seed_666/',
            ],
            "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-5e-1": [
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-06-2023-15:57-seed_123/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-06-2023-23:18-seed_321/',
                '../save_model/GAIL-highD-velocity/train_GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_mspe-5e1_no-buffer_vm-40_data-5e-1-multi_env-Jan-07-2023-06:40-seed_666/',
            ],
        }
    elif env_id == 'highD_distance_constraint':
        max_episodes = 5000
        average_num = 200
        max_reward = 50
        min_reward = -50
        axis_size = 20
        img_size = [8.5, 6.5]
        title = 'HighD Distance Constraint'
        constraint_key = 'is_too_closed'
        plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                    'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
                    'is_too_closed']
        label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                     'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
                     'Distance Constraint Violation Rate']
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
            "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20": [
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-23:25-seed_123/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-23:25-seed_321/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-23:25-seed_666/',
            ],
            "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20": [
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_cl-64-64_no-buffer_dm-20-multi_env-May-28-2022-09:58-seed_123/',
            ],
            "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20": [
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_plr-1e-3_no-buffer_dm-20-multi_env-May-28-2022-09:58-seed_123/',
            ],
            "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1": [
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-16-2023-16:04-seed_123/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-17-2023-01:53-seed_321/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-17-2023-11:28-seed_666/',
            ],
            "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2": [
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-16-2023-16:04-seed_123/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-17-2023-00:26-seed_321/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-17-2023-08:36-seed_666/',
            ],
            "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-3e-1": [
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-3e-1-multi_env-Jan-16-2023-16:04-seed_123/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-3e-1-multi_env-Jan-17-2023-05:55-seed_321/',
            ],
            "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-5e-1": [
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-5e-1-multi_env-Jan-16-2023-16:05-seed_123/',
                '../save_model/ICRL-highD-distance/train_ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-5e-1-multi_env-Jan-17-2023-12:54-seed_321/',
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
            "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20": [
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-26-2022-12:36-seed_123/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-13:05-seed_321/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-13:06-seed_666/',
            ],
            "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-7e-1_no-buffer_dm-20": [
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-7e-1_no-buffer_dm-20-multi_env-Sep-26-2022-12:36-seed_123/',
            ],
            "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_dm-20": [
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_dm-20-multi_env-Sep-26-2022-12:56-seed_123/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_dm-20-multi_env-Sep-27-2022-23:20-seed_321/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_dm-20-multi_env-Sep-27-2022-23:21-seed_666/',
            ],
            'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-1': [
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-16-2023-17:41-seed_123/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-17-2023-03:41-seed_321/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-17-2023-13:48-seed_666/',
            ],
            'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-2': [
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-16-2023-17:41-seed_123/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-17-2023-01:51-seed_321/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-17-2023-10:10-seed_666/',
            ],
            'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-3e-1': [
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-3e-1-multi_env-Jan-16-2023-17:41-seed_123/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-3e-1-multi_env-Jan-17-2023-07:03-seed_321/',
            ],
            'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-5e-1': [
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-5e-1-multi_env-Jan-16-2023-17:41-seed_123/',
                '../save_model/VICRL-highD-distance/train_VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-1e-4_acbf-5e-1_mspe-5e1_no-buffer_dm-20_data-5e-1-multi_env-Jan-17-2023-10:49-seed_321/',
            ],
            'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1':[
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-16-2023-16:03-seed_123/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-17-2023-02:20-seed_321/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-1-multi_env-Jan-17-2023-12:21-seed_666/',
            ],
            'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2': [
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-16-2023-16:03-seed_123/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-17-2023-00:37-seed_321/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-1e-2-multi_env-Jan-17-2023-09:19-seed_666/',
            ],
            'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-3e-1': [
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-3e-1-multi_env-Jan-16-2023-16:03-seed_123/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-3e-1-multi_env-Jan-17-2023-05:26-seed_321/',
            ],
            'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-5e-1':[
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-5e-1-multi_env-Jan-16-2023-16:03-seed_123/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_mspe-5e1_no-buffer_dm-20_data-5e-1-multi_env-Jan-17-2023-08:54-seed_321/',
            ],
            "Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20": [
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-30-2022-02:11-seed_123/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:38-seed_321/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:38-seed_666/',
            ],
            "Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_no-buffer_dm-20": [
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-23:27-seed_123/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-23:27-seed_321/',
                '../save_model/Binary-highD-distance/train_Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-1e-4_no-buffer_dm-20-multi_env-Sep-27-2022-23:27-seed_666/',
            ],
            "GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20": [
                '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20-multi_env-May-30-2022-02:11-seed_123/',
                '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:39-seed_321/',
                '../save_model/GAIL-highD-distance/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20-multi_env-May-31-2022-04:40-seed_666/'
            ]
        }
    elif env_id == 'highD_velocity_distance_constraint':
        log_path_dict = {
            "ppo_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20": [
                '../save_model/PPO-highD-velocity-distance/train_ppo_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20-multi_env-Aug-23-2022-05:41-seed_123/',
                '../save_model/PPO-highD-velocity-distance/train_ppo_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20-multi_env-Aug-23-2022-05:41-seed_321/',
                '../save_model/PPO-highD-velocity-distance/train_ppo_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20-multi_env-Aug-23-2022-05:41-seed_666/',
            ],
            "ppo_lag_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20": [
                '../save_model/PPO-Lag-highD-velocity-distance/train_ppo_lag_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20-multi_env-Aug-14-2022-13:27-seed_123/',
                # '../save_model/PPO-Lag-highD-velocity-distance/train_ppo_lag_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20-multi_env-Aug-15-2022-07:55-seed_123/',
                '../save_model/PPO-Lag-highD-velocity-distance/train_ppo_lag_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20-multi_env-Aug-23-2022-05:41-seed_321/',
                '../save_model/PPO-Lag-highD-velocity-distance/train_ppo_lag_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20-multi_env-Aug-15-2022-07:57-seed_666/',
            ],
            "GAIL_velocity_distance_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20": [
                '../save_model/GAIL-highD-velocity-distance/train_GAIL_velocity_distance_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-29-2022-05:12-seed_123/',
                '../save_model/GAIL-highD-velocity-distance/train_GAIL_velocity_distance_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-30-2022-17:43-seed_321/',
                '../save_model/GAIL-highD-velocity-distance/train_GAIL_velocity_distance_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-30-2022-17:49-seed_666/'
            ],
            "Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20": [
                '../save_model/Binary-highD-velocity-distance/train_Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-29-2022-05:12-seed_123/',
                '../save_model/Binary-highD-velocity-distance/train_Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-30-2022-07:47-seed_321/',
                '../save_model/Binary-highD-velocity-distance/train_Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-30-2022-08:29-seed_666/'
            ],
            "Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20": [
                '../save_model/Binary-highD-velocity-distance/train_Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Sep-28-2022-14:14-seed_123/',
                '../save_model/Binary-highD-velocity-distance/train_Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Sep-28-2022-14:14-seed_321/',
                '../save_model/Binary-highD-velocity-distance/train_Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Sep-28-2022-14:14-seed_666/'
            ],
            "ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20": [
                '../save_model/ICRL-highD-velocity-distance/train_ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-25-2022-08:19-seed_123/',
                '../save_model/ICRL-highD-velocity-distance/train_ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-25-2022-08:19-seed_321/',
                '../save_model/ICRL-highD-velocity-distance/train_ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-25-2022-08:19-seed_666/',
            ],
            "ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20": [
                # '../save_model/ICRL-highD-velocity-distance/train_ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Sep-28-2022-14:14-seed_123/',
                '../save_model/ICRL-highD-velocity-distance/train_ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Sep-28-2022-14:14-seed_321/',
                '../save_model/ICRL-highD-velocity-distance/train_ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Sep-28-2022-14:14-seed_666/',
            ],
            "VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40_dm-20": [
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40_dm-20-multi_env-Aug-28-2022-00:05-seed_123/',
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40_dm-20-multi_env-Aug-29-2022-05:15-seed_123/',
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40_dm-20-multi_env-Aug-30-2022-05:28-seed_666/'
            ],
            "VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20": [
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Aug-28-2022-00:05-seed_123/',
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Aug-29-2022-05:49-seed_321/',
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20-multi_env-Aug-29-2022-05:49-seed_666/'
            ],
            "VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20": [
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20-multi_env-Aug-27-2022-08:05-seed_123/',
            ],
            "VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_clay-64-64_no-buffer_vm-40_dm-20": [
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_clay-64-64_no-buffer_vm-40_dm-20-multi_env-Sep-15-2022-12:48-seed_123/',
            ],
            "VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_clay-64-64_no-buffer_vm-40_dm-20": [
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_clay-64-64_no-buffer_vm-40_dm-20-multi_env-Sep-17-2022-00:23-seed_123/',
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_clay-64-64_no-buffer_vm-40_dm-20-multi_env-Sep-15-2022-12:49-seed_321/',
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_clay-64-64_no-buffer_vm-40_dm-20-multi_env-Sep-17-2022-00:23-seed_666/',
            ],
            "VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_clay-64-64-64_no-buffer_vm-40_dm-20": [
                '../save_model/VICRL-highD-velocity-distance/train_VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_clay-64-64-64_no-buffer_vm-40_dm-20-multi_env-Sep-15-2022-12:49-seed_321/',
            ],
        }
    elif env_id == 'WGW-v0':
        log_path_dict = {
            'PI-Lag-setting1': [
                '../save_model/PI-Lag-WallGrid/train_ppo_lag_WGW-v0_max-nu-1-setting1-Dec-29-2022-19:07-seed_123/',
            ],
            'PI-Lag-setting2': [
                '../save_model/PI-Lag-WallGrid/train_ppo_lag_WGW-v0_max-nu-1-setting2-Dec-21-2022-17:08-seed_123/',
            ],
            'PI-Lag-setting3': [
                '../save_model/PI-Lag-WallGrid/train_ppo_lag_WGW-v0_max-nu-1-setting3-Dec-23-2022-17:37-seed_123/',
            ],
            'PI-Lag-setting4': [
                '../save_model/PI-Lag-WallGrid/train_ppo_lag_WGW-v0_max-nu-1-setting4-Dec-26-2022-19:19-seed_123/',
            ],
            "ICRL_without-action_by_games_max-nu-1_with-buffer-setting1": [
                '../save_model/ICRL-WallGrid/train_ICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting1-Dec-23-2022-17:22-seed_123/',
            ],
            "ICRL_without-action_by_games_max-nu-1_with-buffer-setting2": [
                '../save_model/ICRL-WallGrid/train_ICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting2-Dec-23-2022-17:22-seed_123/',
            ],
            "ICRL_without-action_by_games_max-nu-1_with-buffer-setting3": [
                '../save_model/ICRL-WallGrid/train_ICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting3-Dec-23-2022-17:49-seed_123/',
            ],
            "ICRL_without-action_by_games_max-nu-1_with-buffer-setting4": [
                '../save_model/ICRL-WallGrid/train_ICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting1-Dec-23-2022-17:22-seed_123/',
            ],
            "VICRL_without-action_by_games_max-nu-1_with-buffer-setting1": [
                '../save_model/VICRL-WallGrid/train_VICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_p-9e-2-1e-2_with-buffer-setting1_mean-Dec-29-2022-17:40-seed_123/',
            ],
            "VICRL_without-action_by_games_max-nu-1_with-buffer-setting2": [
                '../save_model/VICRL-WallGrid/train_VICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_p-9e-2-1e-2_with-buffer-setting2_mean-Dec-29-2022-18:05-seed_123/',
            ],
            "VICRL_without-action_by_games_max-nu-1_with-buffer-setting3": [
                '../save_model/VICRL-WallGrid/train_VICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_p-9e-2-1e-2_with-buffer-setting3_mean-Dec-29-2022-18:05-seed_123/',
            ],
            "VICRL_without-action_by_games_max-nu-1_with-buffer-setting4": [
                '../save_model/VICRL-WallGrid/train_VICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_p-9e-2-1e-2_with-buffer-setting4_mean-Dec-29-2022-18:05-seed_123/',
            ],
            "Binary_without-action_by_games_max-nu-1_with-buffer-setting1": [
                '../save_model/Binary-WallGrid/train_Binary_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting1-Dec-23-2022-17:56-seed_123/',
            ],
            "Binary_without-action_by_games_max-nu-1_with-buffer-setting2": [
                '../save_model/Binary-WallGrid/train_Binary_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting2-Dec-23-2022-17:56-seed_123/',
            ],
            "Binary_without-action_by_games_max-nu-1_with-buffer-setting3": [
                '../save_model/Binary-WallGrid/train_Binary_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting3-Dec-23-2022-18:17-seed_123/',
            ],
            "Binary_without-action_by_games_max-nu-1_with-buffer-setting4": [
                '../save_model/Binary-WallGrid/train_Binary_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting4-Dec-26-2022-21:14-seed_123/',
            ],
            "GAIL_without-action_by_games_with-buffer-setting1": [
                '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting1-Jan-20-2023-17:28-seed_123/',
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting1-Jan-19-2023-17:22-seed_123/',
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting1-Jan-19-2023-19:37-seed_123/',
            ],
            "GAIL_without-action_by_games_with-buffer-setting2": [
                '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting2-Jan-20-2023-17:28-seed_123/',
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting2-Jan-19-2023-17:31-seed_123/',
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting2-Jan-20-2023-10:34-seed_123/',
            ],
            "GAIL_without-action_by_games_with-buffer-setting3": [
                '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting3-Jan-20-2023-17:28-seed_123/',
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting3-Jan-19-2023-17:44-seed_123/'
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting3-Jan-20-2023-10:37-seed_123/',
            ],
            "GAIL_without-action_by_games_with-buffer-setting4": [
                '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting4-Jan-20-2023-17:20-seed_123/',
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting4-Jan-19-2023-17:52-seed_123/',
                # '../save_model/GAIL-WallGrid/train_GAIL_WGW-v0_without-action_by_games_with-buffer-setting4-Jan-20-2023-10:41-seed_123/',
            ],
        }
    elif env_id == 'HCWithPos-v0':
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
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-08:54-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-10:43-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-12:29-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-14:17-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-multi_env-Apr-22-2022-16:04-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_noise-5e-3": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-11-2022-20:48-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-12-2022-01:01-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-12-2022-03:56-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-3": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-3-multi_env-Nov-11-2022-16:39-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-3-multi_env-Nov-11-2022-18:47-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-3-multi_env-Nov-11-2022-21:03-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_noise-5e-2": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-11-2022-20:48-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-12-2022-00:59-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-12-2022-03:52-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-2": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-2-multi_env-Nov-11-2022-16:42-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-2-multi_env-Nov-11-2022-18:50-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-5e-2-multi_env-Nov-11-2022-21:05-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_noise-1e-3": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-11-2022-20:48-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-11-2022-23:43-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-12-2022-02:26-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-3": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-3-multi_env-Nov-11-2022-16:34-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-3-multi_env-Nov-11-2022-18:08-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-3-multi_env-Nov-11-2022-19:44-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_noise-1e-2": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-11-2022-20:48-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-11-2022-23:11-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-12-2022-01:15-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-2": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-2-multi_env-Nov-11-2022-16:34-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-2-multi_env-Nov-11-2022-18:07-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-2-multi_env-Nov-11-2022-19:44-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_noise-1e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-11-2022-20:48-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-11-2022-23:14-seed_321/',
            ],
            "ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-1-multi_env-Nov-11-2022-16:34-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_piv-1e1_noise-1e-1-multi_env-Nov-11-2022-18:05-seed_321/',
            ],
            "ICRL_HCWithPos-v0_with-action_random-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-13:13-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-15:30-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-17:39-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-19:57-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-22:15-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_random-8e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-13:13-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-15:30-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-17:39-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-19:56-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-22:13-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_random-5e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-13:12-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-15:30-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-17:39-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-20:00-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-22:19-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_random-2e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-13:12-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-15:29-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-17:36-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-19:54-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-22:12-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_data-1e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-17:50-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-19:36-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-21:22-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-23:06-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-28-2022-00:49-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_data-3e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-17:51-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-19:08-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-20:26-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-21:43-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-22:59-seed_666/',
            ],
            "ICRL_HCWithPos-v0_with-action_data-5e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-17:51-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-19:54-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-21:41-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-23:28-seed_654/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-28-2022-01:13-seed_666/',
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
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-07:23-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-07:22-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-06:54-seed_666/'
            ],
            "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1": [
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
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1_no_is": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1_no_is-multi_env-Aug-10-2022-10:46-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1_no_is-multi_env-Aug-11-2022-04:54-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1_no_is-multi_env-Aug-11-2022-06:39-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1_no_is-multi_env-Aug-11-2022-08:26-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1_no_is-multi_env-Aug-11-2022-10:16-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1_no_is": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1_no_is-multi_env-Aug-10-2022-10:48-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1_no_is-multi_env-Aug-11-2022-04:54-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1_no_is-multi_env-Aug-11-2022-06:49-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1_no_is-multi_env-Aug-11-2022-08:42-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1_no_is-multi_env-Aug-11-2022-10:33-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1_no_is": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1_no_is-multi_env-Aug-10-2022-10:46-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1_no_is-multi_env-Aug-11-2022-04:55-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1_no_is-multi_env-Aug-11-2022-06:44-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1_no_is-multi_env-Aug-11-2022-08:28-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1_no_is-multi_env-Aug-11-2022-10:15-seed_666/'
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1-b_no_is": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1-b_no_is-multi_env-Aug-11-2022-13:30-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1-b_no_is-multi_env-Aug-11-2022-15:26-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1-b_no_is-multi_env-Aug-11-2022-17:23-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1-b_no_is-multi_env-Aug-11-2022-19:20-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1-b_no_is-multi_env-Aug-11-2022-21:16-seed_666/'
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1-b_no_is": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1-b_no_is-multi_env-Aug-11-2022-13:31-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1-b_no_is-multi_env-Aug-11-2022-15:18-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1-b_no_is-multi_env-Aug-11-2022-17:14-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1-b_no_is-multi_env-Aug-11-2022-19:14-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1-b_no_is-multi_env-Aug-11-2022-21:15-seed_666/'
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1-b_no_is": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1-b_no_is-multi_env-Aug-11-2022-15:23-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1-b_no_is-multi_env-Aug-11-2022-17:17-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1-b_no_is-multi_env-Aug-11-2022-19:08-seed_666/'
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-Aug-12-2022-01:54-seed_123/',
            ],
            "VICRL_HCWithPos-v0_with_action_p-1-1_no_is_hard": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_p-1-1_no_is_hard-multi_env-Aug-10-2022-09:31-seed_123/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_hard": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_hard-multi_env-Aug-10-2022-09:31-seed_123/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-1e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-1e-1-multi_env-Sep-17-2022-19:32-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-1e-1-multi_env-Sep-17-2022-21:45-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-1e-1-multi_env-Sep-18-2022-02:10-seed_654/',
                # ''
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-1e-1-multi_env-Sep-18-2022-00:44-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-5e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-5e-1-multi_env-Sep-17-2022-19:32-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-5e-1-multi_env-Sep-17-2022-21:45-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-5e-1-multi_env-Sep-18-2022-02:10-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-5e-1-multi_env-Sep-18-2022-02:10-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-5e-1-multi_env-Sep-18-2022-00:44-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-9e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-9e-1-multi_env-Sep-17-2022-19:32-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-9e-1-multi_env-Sep-17-2022-21:45-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-9e-1-multi_env-Sep-18-2022-02:10-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-9e-1-multi_env-Sep-18-2022-02:10-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-9e-1-multi_env-Sep-18-2022-00:44-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-04:49-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-06:27-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-08:05-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-09:43-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-11:20-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_noise-1e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-11-2022-21:11-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-12-2022-01:25-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-12-2022-04:12-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_piv-1e1_noise-1e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_piv-1e1_noise-1e-1-multi_env-Nov-12-2022-21:53-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_piv-1e1_noise-1e-1-multi_env-Nov-12-2022-23:05-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_piv-1e1_noise-1e-1-multi_env-Nov-13-2022-00:18-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_noise-1e-2": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-11-2022-21:11-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-11-2022-23:40-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-12-2022-01:39-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_noise-1e-3": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-11-2022-21:11-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-11-2022-23:33-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-12-2022-01:36-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_noise-5e-2": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-11-2022-21:11-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-12-2022-00:13-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-12-2022-02:54-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_noise-5e-3": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-11-2022-21:11-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-12-2022-00:12-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-12-2022-02:53-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_random-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-13:14-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-15:30-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-17:37-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-19:57-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-22:16-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_random-8e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-13:14-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-15:31-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-17:36-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-19:54-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-22:12-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_random-5e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-13:14-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-15:31-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-17:38-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-19:57-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-22:16-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_random-2e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-13:13-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-15:30-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-17:35-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-19:55-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-22:12-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_data-1e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-17:51-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-19:51-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-21:35-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-27-2022-23:19-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-1e-1-multi_env-Dec-28-2022-01:02-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_data-3e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-17:51-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-19:42-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-21:32-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-23:22-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-28-2022-01:12-seed_666/',
            ],
            "Binary_HCWithPos-v0_with-action_data-5e-1": [
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-17:51-seed_123/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-19:40-seed_321/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-21:26-seed_456/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-23:13-seed_654/',
                '../save_model/Binary-HC/train_Binary_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-28-2022-01:00-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-05:55-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-07:32-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-09:18-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-11:02-seed_654/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action-multi_env-Apr-21-2022-12:45-seed_666/'
            ],
            "GAIL_HCWithPos-v0_with-action_noise-1e-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-11-2022-21:12-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-12-2022-01:35-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-1-multi_env-Nov-12-2022-04:24-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_noise-1e-2": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-11-2022-21:12-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-11-2022-23:24-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-2-multi_env-Nov-12-2022-01:12-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_noise-1e-3": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-11-2022-21:12-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-11-2022-23:17-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-1e-3-multi_env-Nov-12-2022-01:06-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_noise-5e-2": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-11-2022-21:13-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-12-2022-00:01-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-5e-2-multi_env-Nov-12-2022-02:30-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_noise-5e-3": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-11-2022-21:13-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-11-2022-23:48-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_noise-5e-3-multi_env-Nov-12-2022-02:09-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_random-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-16:35-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-18:49-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-21:04-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-1-multi_env-Dec-29-2022-23:18-seed_654/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-1-multi_env-Dec-30-2022-01:15-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_random-8e-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-16:35-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-18:50-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-21:01-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-29-2022-23:15-seed_654/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-8e-1-multi_env-Dec-30-2022-01:12-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_random-5e-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-16:34-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-18:49-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-21:01-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-29-2022-23:13-seed_654/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-5e-1-multi_env-Dec-30-2022-01:11-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_random-2e-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-16:34-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-18:48-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-20:57-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-29-2022-23:08-seed_654/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_random-2e-1-multi_env-Dec-30-2022-01:04-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_data-3e-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-17:46-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-19:27-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-21:10-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-27-2022-22:53-seed_654/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-3e-1-multi_env-Dec-28-2022-00:37-seed_666/',
            ],
            "GAIL_HCWithPos-v0_with-action_data-5e-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-17:46-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-19:22-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-21:00-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-27-2022-22:37-seed_654/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_with-action_data-5e-1-multi_env-Dec-28-2022-00:16-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-1-multi_env-Dec-29-2022-13:13-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-1-multi_env-Dec-29-2022-15:56-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-1-multi_env-Dec-29-2022-18:35-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-1-multi_env-Dec-29-2022-21:24-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-1-multi_env-Dec-30-2022-00:12-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-2e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-2e-1-multi_env-Dec-29-2022-13:13-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-2e-1-multi_env-Dec-29-2022-15:53-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-2e-1-multi_env-Dec-29-2022-18:33-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1-multi_env-Dec-29-2022-21:21-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1-multi_env-Dec-30-2022-00:08-seed_666/'
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-5e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-5e-1-multi_env-Dec-29-2022-13:13-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-5e-1-multi_env-Dec-29-2022-15:54-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-5e-1-multi_env-Dec-29-2022-18:33-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-5e-1-multi_env-Dec-29-2022-21:20-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-5e-1-multi_env-Dec-30-2022-00:09-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1-multi_env-Dec-29-2022-13:13-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1-multi_env-Dec-29-2022-15:55-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1-multi_env-Dec-29-2022-18:34-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1-multi_env-Dec-29-2022-21:21-seed_654/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1-multi_env-Dec-30-2022-00:08-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_CVAR-6e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_CVAR-0-multi_env-Sep-17-2022-11:48-seed_123/'
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-2": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-2-multi_env-Nov-11-2022-16:44-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-2-multi_env-Nov-11-2022-19:18-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-2-multi_env-Nov-11-2022-22:57-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-3": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-3-multi_env-Nov-11-2022-16:47-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-3-multi_env-Nov-11-2022-19:22-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-5e-3-multi_env-Nov-11-2022-23:04-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-1-multi_env-Nov-11-2022-16:44-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-1-multi_env-Nov-11-2022-18:31-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-1-multi_env-Nov-11-2022-20:11-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-11:16-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-13:35-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-16:00-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_piv-1e1_noise-1e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-11:45-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-14:28-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-16:43-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_piv-1e1_noise-1e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-11:45-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-14:29-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-1-multi_env-Nov-12-2022-16:49-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-2": [
                # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-2-multi_env-Nov-11-2022-16:44-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-2-multi_env-Nov-11-2022-18:34-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-2-multi_env-Nov-11-2022-20:21-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-2": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-12-2022-11:16-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-12-2022-12:45-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-12-2022-14:15-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-2": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-13-2022-02:35-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-13-2022-03:59-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-13-2022-05:22-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-2": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-12-2022-22:02-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-12-2022-23:33-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-2-multi_env-Nov-13-2022-01:05-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-3": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-3-multi_env-Nov-11-2022-16:44-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-3-multi_env-Nov-11-2022-18:33-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_piv-1e1_noise-1e-3-multi_env-Nov-11-2022-20:22-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-3": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-10-2022-14:30-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-10-2022-16:38-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-12-2022-11:16-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-12-2022-12:45-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-12-2022-14:14-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-3": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-13-2022-02:21-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-13-2022-03:49-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e0-1e0_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-13-2022-05:19-seed_666/',
            ],
            "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-3": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-13-2022-06:46-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-13-2022-08:10-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_clr-5e-3_no_is_noise-1e-3-multi_env-Nov-13-2022-09:32-seed_666/',
            ],
        }
    else:
        raise ValueError("Unknown env id {0}".format(env_id))
    return log_path_dict
