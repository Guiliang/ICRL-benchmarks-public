import json
import os
import sys
import time
import gym
import numpy as np
import datetime
import yaml

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from utils.env_utils import check_if_duplicate_seed
from common.cns_env import make_train_env, make_eval_env, sync_envs_normalization_ppo
from utils.plot_utils import plot_curve
from exploration.exploration import ExplorationRewardCallback
from stable_baselines3 import PPO, PPOLagrangian
from stable_baselines3.common import logger
from common.cns_evaluation import evaluate_icrl_policy
from stable_baselines3.common.vec_env import VecNormalize

from utils.data_utils import ProgressBarManager, del_and_make, read_args, load_config, process_memory
from utils.model_utils import load_ppo_config


def train(args):
    config, debug_mode, log_file_path, partial_data, num_threads, seed = load_config(args)

    if num_threads > 1:
        multi_env = True
        config.update({'multi_env': True})
    else:
        multi_env = False
        config.update({'multi_env': False})

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['PPO']['forward_timesteps'] = 2000  # 2000
        config['PPO']['n_steps'] = 200
        config['running']['n_eval_episodes'] = 10
        config['running']['save_every'] = 1
        debug_msg = 'debug-'
        partial_data = True
    if partial_data:
        debug_msg += 'part-'

    if num_threads is not None:
        config['env']['num_threads'] = num_threads

    print(json.dumps(config, indent=4), file=log_file, flush=True)
    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')
    # today = datetime.date.today()
    # currentTime = today.strftime("%b-%d-%Y-%h-%m")
    save_model_mother_dir = '{0}/{1}/{5}{2}{3}-{4}-seed_{6}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        '-multi_env' if multi_env else '',
        current_time_date,
        debug_msg,
        seed
    )

    skip_running = check_if_duplicate_seed(seed=seed,
                                           config=config,
                                           current_time_date=current_time_date,
                                           save_model_mother_dir=save_model_mother_dir,
                                           log_file=log_file)
    if skip_running:
        return

    if not os.path.exists('{0}/{1}/'.format(config['env']['save_dir'], config['task'])):
        os.mkdir('{0}/{1}/'.format(config['env']['save_dir'], config['task']))
    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)
    print("Saving to the file: {0}".format(save_model_mother_dir), file=log_file, flush=True)

    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    mem_prev = process_memory()
    time_prev = time.time()
    # Create the vectorized environments
    train_env, env_configs = make_train_env(env_id=config['env']['train_env_id'],
                                            config_path=config['env']['config_path'],
                                            save_dir=save_model_mother_dir,
                                            base_seed=seed,
                                            group=config['group'],
                                            num_threads=num_threads,
                                            use_cost=config['env']['use_cost'],
                                            normalize_obs=not config['env']['dont_normalize_obs'],
                                            normalize_reward=not config['env']['dont_normalize_reward'],
                                            normalize_cost=not config['env']['dont_normalize_cost'],
                                            cost_info_str=config['env']['cost_info_str'],
                                            reward_gamma=config['env']['reward_gamma'],
                                            cost_gamma=config['env']['cost_gamma'],
                                            log_file=log_file,
                                            part_data=partial_data,
                                            multi_env=multi_env,
                                            )

    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)

    eval_env, env_configs = make_eval_env(env_id=config['env']['eval_env_id'],
                                          config_path=config['env']['config_path'],
                                          save_dir=save_test_mother_dir,
                                          group=config['group'],
                                          use_cost=config['env']['use_cost'],
                                          normalize_obs=not config['env']['dont_normalize_obs'],
                                          cost_info_str=config['env']['cost_info_str'],
                                          log_file=log_file,
                                          part_data=partial_data)

    mem_loading_environment = process_memory()
    time_loading_environment = time.time()
    print("Loading environment consumed memory: {0:.2f}/{1:.2f} and time {2:.2f}:".format(
        float(mem_loading_environment - mem_prev) / 1000000,
        float(mem_loading_environment) / 1000000,
        time_loading_environment - time_prev),
        file=log_file, flush=True)
    mem_prev = mem_loading_environment
    time_prev = time_loading_environment

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    # print('is_discrete', is_discrete, file=log_file, flush=True)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    # Logger
    if log_file is None:
        ppo_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        ppo_logger = logger.HumanOutputFormat(log_file)

    ppo_parameters = load_ppo_config(config, train_env, seed, log_file)
    if config['group'] == 'PPO':
        create_ppo_agent = lambda: PPO(**ppo_parameters)
    elif config['group'] == 'PPO-Lag':
        create_ppo_agent = lambda: PPOLagrangian(**ppo_parameters)
    else:
        raise ValueError("Unknown ppo group: {0}".format(config['group']))
    ppo_agent = create_ppo_agent()

    # Callbacks
    all_callbacks = []
    if config['PPO']['use_curiosity_driven_exploration']:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(explorationCallback)

    timesteps = 0.
    mem_before_training = process_memory()
    time_before_training = time.time()
    print("Setting model consumed memory: {0:.2f}/{1:.2f} and time: {2:.2f}".format(
        float(mem_before_training - mem_prev) / 1000000,
        float(mem_before_training) / 1000000,
        time_before_training - time_prev),
        file=log_file, flush=True)
    mem_prev = mem_before_training
    time_prev = time_before_training

    # Train
    start_time = time.time()
    # print(colorize("\nBeginning training", color="green", bold=True), file=log_file, flush=True)
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward = -np.inf
    for itr in range(config['running']['n_iters']):
        if config['PPO']['reset_policy'] and itr != 0:
            # print(colorize("Resetting agent", color="green", bold=True), file=log_file, flush=True)
            print("Resetting agent", file=log_file, flush=True)
            ppo_agent = create_ppo_agent()

        # Update agent
        with ProgressBarManager(config['PPO']['forward_timesteps']) as callback:
            if config['group'] == 'PPO':
                ppo_agent.learn(total_timesteps=config['PPO']['forward_timesteps'],
                                callback=callback)
            elif config['group'] == 'PPO-Lag':
                ppo_agent.learn(
                    total_timesteps=config['PPO']['forward_timesteps'],
                    cost_function=config['env']['cost_info_str'],  # Cost should come from cost wrapper
                    callback=[callback] + all_callbacks
                )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += ppo_agent.num_timesteps

        mem_during_training = process_memory()
        time_during_training = time.time()
        print("Itr: {3}, Training consumed memory: {0:.2f}/{1:.2f} and time {2:.2f}".format(
            float(mem_during_training - mem_prev) / 1000000,
            float(mem_during_training) / 1000000,
            time_during_training - time_prev,
            itr), file=log_file, flush=True)
        mem_prev = mem_during_training
        time_prev = time_during_training

        # Evaluate:
        # reward on true environment
        sync_envs_normalization_ppo(train_env, eval_env)
        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs = \
            evaluate_icrl_policy(model=ppo_agent,
                                 env=eval_env,
                                 record_info_names=config['env']["record_info_names"],
                                 n_eval_episodes=config['running']['n_eval_episodes'],
                                 deterministic=False)

        # Save
        if itr % config['running']['save_every'] == 0:
            path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
            del_and_make(path)
            ppo_agent.save(os.path.join(path, "nominal_agent"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, "train_env_stats.pkl"))
            if costs is not None:
                for record_info_name in config['env']["record_info_names"]:
                    plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
                    plot_curve(draw_keys=[record_info_name],
                               x_dict={record_info_name: plot_record_infos},
                               y_dict={record_info_name: plot_costs},
                               xlabel=record_info_name,
                               ylabel='cost',
                               save_name=os.path.join(path, "{0}".format(record_info_name)),
                               apply_scatter=True
                               )
            # env_tmp = train_env
            # while isinstance(env_tmp, VecEnvWrapper):
            #     env_tmp = env_tmp.venv
            #     if isinstance(env_tmp, VecCostWrapper) or isinstance(env_tmp, InternalVecCostWrapper):
            #         env_tmp = env_tmp.venv
            # for i in len(env_tmp.envs):
            #     env_tmp.envs[i].info_saving_file = open(os.path.join(path, 'info_saving_{0}.txt'.format(i)), 'w')
            #     env_tmp.envs[i].info_saving_items = ['ego_velocity', 'cost']

        # (2) best
        if mean_nc_reward > best_true_reward:
            # print(colorize("Saving new best model", color="green", bold=True), flush=True, file=log_file)
            print("Saving new best model", flush=True, file=log_file)
            ppo_agent.save(os.path.join(save_model_mother_dir, "best_nominal_model"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))

        # Update best metrics
        if mean_nc_reward > best_true_reward:
            best_true_reward = mean_nc_reward

        # Collect metrics
        metrics = {
            "time(m)": (time.time() - start_time) / 60,
            "run_iter": itr,
            "timesteps": timesteps,
            "true/mean_nc_reward": mean_nc_reward,
            "true/std_nc_reward": std_nc_reward,
            "true/mean_reward": mean_reward,
            "true/std_reward": std_reward,
            "best_true/best_reward": best_true_reward
        }

        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})

        # Log
        if config['verbose'] > 0:
            ppo_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)

        mem_during_testing = process_memory()
        time_during_testing = time.time()
        print("Itr: {3}, Validating consumed memory: {0:.2f}/{1:.2f} and time {2:.2f}".format(
            float(mem_during_testing - mem_prev) / 1000000,
            float(mem_during_testing) / 1000000,
            time_during_testing - time_prev,
            itr), file=log_file, flush=True)
        mem_prev = mem_during_testing
        time_prev = time_during_testing


if __name__ == "__main__":
    args = read_args()
    train(args)
