import json
import os
import sys
import time
import gym
import numpy as np
import datetime
import yaml
from matplotlib import pyplot as plt

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from common.cns_sampler import ConstrainedRLSampler
from common.cns_visualization import traj_visualization_2d, constraint_visualization_2d
from utils.true_constraint_functions import get_true_cost_function
from stable_baselines3.iteration import PolicyIterationLagrange
from utils.env_utils import check_if_duplicate_seed
from common.cns_env import make_train_env, make_eval_env, sync_envs_normalization_ppo
from utils.plot_utils import plot_curve
from exploration.exploration import ExplorationRewardCallback
from stable_baselines3 import PPO, PPOLagrangian
from stable_baselines3.common import logger
from common.cns_evaluation import evaluate_icrl_policy
from stable_baselines3.common.vec_env import VecNormalize

from utils.data_utils import ProgressBarManager, del_and_make, read_args, load_config, process_memory, print_resource
from utils.model_utils import load_ppo_config, load_policy_iteration_config
import warnings

warnings.filterwarnings("ignore")


def train(args):
    # load config
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
        if 'PPO' in config.keys():
            config['PPO']['forward_timesteps'] = 2000  # 2000
            config['PPO']['n_steps'] = 200
        else:
            config['iteration']['max_iter'] = 2
        config['running']['n_eval_episodes'] = 10
        config['running']['save_every'] = 1
        debug_msg = 'debug-'
    if num_threads is not None:
        config['env']['num_threads'] = num_threads

    # print the current config
    print(json.dumps(config, indent=4), file=log_file, flush=True)

    # init saving dir for the running models
    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')
    save_model_mother_dir = '{0}/{1}/{5}{2}{3}-{4}-seed_{6}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        '-multi_env' if multi_env else '',
        current_time_date,
        debug_msg,
        seed
    )
    if not os.path.exists('{0}/{1}/'.format(config['env']['save_dir'], config['task'])):
        os.mkdir('{0}/{1}/'.format(config['env']['save_dir'], config['task']))
    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)
    print("Saving to the file: {0}".format(save_model_mother_dir), file=log_file, flush=True)

    # save the running config
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
                                            circle_info=config['env']['circle_info'] if 'Circle' in config[
                                                'env']['train_env_id'] else None,
                                            max_scene_per_env=config['env']['max_scene_per_env']
                                            if 'max_scene_per_env' in config['env'].keys() else None
                                            )

    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)

    sampling_env, env_configs = make_eval_env(env_id=config['env']['eval_env_id'],
                                              config_path=config['env']['config_path'],
                                              save_dir=save_test_mother_dir,
                                              group=config['group'],
                                              use_cost=config['env']['use_cost'],
                                              normalize_obs=not config['env']['dont_normalize_obs'],
                                              cost_info_str=config['env']['cost_info_str'],
                                              log_file=log_file,
                                              part_data=partial_data,
                                              circle_info=config['env']['circle_info'] if 'Circle' in config[
                                                  'env']['train_env_id'] else None,
                                              max_scene_per_env=config['env']['max_scene_per_env']
                                              if 'max_scene_per_env' in config['env'].keys() else None
                                              )
    # init sampler
    if 'WGW' in config['env']['train_env_id']:
        sampler = ConstrainedRLSampler(rollouts=10,
                                       store_by_game=True,  # I move the step out
                                       cost_info_str=None,
                                       sample_multi_env=False,
                                       env_id=config['env']['eval_env_id'],
                                       env=sampling_env)

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Loading environment', log_file=log_file)
    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    # Logger
    if log_file is None:
        ppo_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        ppo_logger = logger.HumanOutputFormat(log_file)
    # visualize the cost function for gridworld
    if 'WGW' in config['env']['train_env_id']:
        ture_cost_function = get_true_cost_function(env_id=config['env']['train_env_id'],
                                                    env_configs=env_configs)
        constraint_visualization_2d(cost_function=ture_cost_function,
                                    feature_range=config['env']["visualize_info_ranges"],
                                    select_dims=config['env']["record_info_input_dims"],
                                    num_points_per_feature=env_configs['map_height'],
                                    obs_dim=2,
                                    acs_dim=1,
                                    save_path=save_model_mother_dir
                                    )

    if config['group'] == 'PPO':
        ppo_parameters = load_ppo_config(config=config,
                                         train_env=train_env,
                                         seed=seed,
                                         log_file=log_file)
        forward_timesteps = config['PPO']['forward_timesteps']
        create_policy_agent = lambda: PPO(**ppo_parameters)
    elif config['group'] == 'PPO-Lag':
        ppo_parameters = load_ppo_config(config=config,
                                         train_env=train_env,
                                         seed=seed,
                                         log_file=log_file)
        forward_timesteps = config['PPO']['forward_timesteps']
        create_policy_agent = lambda: PPOLagrangian(**ppo_parameters)
    elif config['group'] == 'PI-Lag':
        iteration_parameters = load_policy_iteration_config(config=config,
                                                            env_configs=env_configs,
                                                            train_env=train_env,
                                                            seed=seed,
                                                            log_file=log_file)
        forward_timesteps = config['iteration']['max_iter']
        create_policy_agent = lambda: PolicyIterationLagrange(**iteration_parameters)
    else:
        raise ValueError("Unknown ppo group: {0}".format(config['group']))
    policy_agent = create_policy_agent()

    # Callbacks
    all_callbacks = []
    if 'PPO' in config['group'] and config['PPO']['use_curiosity_driven_exploration']:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(explorationCallback)

    timesteps = 0.

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Setting model', log_file=log_file)

    # Train
    start_time = time.time()
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward = -np.inf
    for itr in range(config['running']['n_iters']):

        # Update agent
        with ProgressBarManager(forward_timesteps) as callback:
            if config['group'] == 'PPO':
                policy_agent.learn(total_timesteps=forward_timesteps,
                                   callback=callback)
            else:
                policy_agent.learn(
                    total_timesteps=forward_timesteps,
                    cost_function=config['env']['cost_info_str'],  # Cost should come from cost wrapper
                    callback=[callback] + all_callbacks
                )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += policy_agent.num_timesteps

        # monitor the memory and running time
        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Training PPO model',
                                             log_file=log_file)

        # Evaluate:
        # reward on true environment
        save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
        if itr % config['running']['save_every'] == 0:
            del_and_make(save_path)
        else:
            save_path = None
        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs = \
            evaluate_icrl_policy(model=policy_agent,
                                 env=sampling_env,
                                 render=True if 'Circle' in config['env']['train_env_id'] else False,
                                 record_info_names=config['env']["record_info_names"],
                                 n_eval_episodes=config['running']['n_eval_episodes'],
                                 deterministic=False,
                                 cost_info_str=config['env']['cost_info_str'],
                                 save_path=save_path, )

        # visualize the trajectory for the grid world env
        if 'WGW' in config['env']['train_env_id'] and itr % config['running']['save_every'] == 0:
            orig_observations, observations, actions, rewards, sum_rewards, lengths = sampler.sample_from_agent(
                policy_agent=policy_agent,
                new_env=sampling_env,
            )
            traj_visualization_2d(config=config,
                                  observations=orig_observations,
                                  save_path=save_path, )

        # Save
        if itr % config['running']['save_every'] == 0:
            # path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
            # del_and_make(path)
            policy_agent.save(os.path.join(save_path, "nominal_agent"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_path, "train_env_stats.pkl"))
            if costs is not None:
                for record_info_name in config['env']["record_info_names"]:
                    plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
                    plot_curve(draw_keys=[record_info_name],
                               x_dict={record_info_name: plot_record_infos},
                               y_dict={record_info_name: plot_costs},
                               xlabel=record_info_name,
                               ylabel='cost',
                               save_name=os.path.join(save_path, "{0}".format(record_info_name)),
                               apply_scatter=True
                               )

        # (2) best
        if mean_nc_reward > best_true_reward:
            # print(colorize("Saving new best model", color="green", bold=True), flush=True, file=log_file)
            print("Saving new best model", flush=True, file=log_file)
            policy_agent.save(os.path.join(save_model_mother_dir, "best_nominal_model"))
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

        # monitor the memory and running time
        mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                             process_name='Evaluation', log_file=log_file)


if __name__ == "__main__":
    args = read_args()
    train(args)
