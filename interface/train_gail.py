import datetime
import json
import os
import sys
import time
import warnings

import gym
import numpy as np
import yaml
from matplotlib import pyplot as plt

from common.cns_sampler import ConstrainedRLSampler

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from utils.env_utils import check_if_duplicate_seed
from stable_baselines3.iteration.policy_interation_gail import PolicyIterationGail
from common.cns_env import make_train_env, make_eval_env, SaveEnvStatsCallback
from common.cns_evaluation import CNSEvalCallback
from common.cns_save_callbacks import CNSCheckpointCallback
from common.cns_visualization import PlotCallback, constraint_visualization_2d, traj_visualization_2d
from constraint_models.constraint_net.gail_net import GailDiscriminator, GailCallback
from exploration.exploration import CostShapingCallback
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from utils.data_utils import read_args, load_config, process_memory, load_expert_data, print_resource, del_and_make
from utils.model_utils import load_ppo_config, load_policy_iteration_config
from utils.true_constraint_functions import get_true_cost_function
import stable_baselines3.common.callbacks as callbacks

warnings.filterwarnings("ignore")


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
        config['device'] = 'cpu'
        debug_msg = 'debug-'
        partial_data = True
        # debug_msg += 'part-'
        if 'iteration' in config.keys():
            config['iteration']['max_iter'] = 2
        else:
            config['running']['save_every'] = 2048
            config['running']['eval_every'] = 1024
    if partial_data:
        debug_msg += 'part-'

    if num_threads is not None:
        config['env']['num_threads'] = int(num_threads)

    print(json.dumps(config, indent=4), file=log_file, flush=True)
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

    # skip_running = check_if_duplicate_seed(seed=seed,
    #                                        config=config,
    #                                        current_time_date=current_time_date,
    #                                        save_model_mother_dir=save_model_mother_dir,
    #                                        log_file=log_file)
    # if skip_running:
    #     return

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
                                            group=config['group'],
                                            save_dir=save_model_mother_dir,
                                            use_cost=False,
                                            base_seed=seed,
                                            num_threads=num_threads,
                                            normalize_obs=not config['env']['dont_normalize_obs'],
                                            normalize_reward=not config['env']['dont_normalize_reward'],
                                            normalize_cost=False,
                                            reward_gamma=config['env']['reward_gamma'],
                                            multi_env=multi_env,
                                            part_data=partial_data,
                                            log_file=log_file,
                                            noise_mean=config['env']['noise_mean'] if 'Noise' in config['env'][
                                                'train_env_id'] else None,
                                            noise_std=config['env']['noise_std'] if 'Noise' in config['env'][
                                                'train_env_id'] else None,
                                            max_scene_per_env=config['env']['max_scene_per_env']
                                            if 'max_scene_per_env' in config['env'].keys() else None,
                                            )
    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)
    eval_env, env_configs = make_eval_env(env_id=config['env']['eval_env_id'],
                                          config_path=config['env']['config_path'],
                                          save_dir=save_test_mother_dir,
                                          group=config['group'],
                                          num_threads=1,
                                          mode='test',
                                          use_cost=False,
                                          normalize_obs=not config['env']['dont_normalize_obs'],
                                          cost_info_str=config['env']['cost_info_str'],
                                          part_data=partial_data,
                                          multi_env=False,
                                          log_file=log_file,
                                          noise_mean=config['env']['noise_mean'] if 'Noise' in config['env'][
                                              'train_env_id'] else None,
                                          noise_std=config['env']['noise_std'] if 'Noise' in config['env'][
                                              'train_env_id'] else None,
                                          max_scene_per_env=config['env']['max_scene_per_env']
                                          if 'max_scene_per_env' in config['env'].keys() else None,
                                          )

    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Loading environment', log_file=log_file)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    recon_obs = config['DISC']['recon_obs'] if 'recon_obs' in config['DISC'].keys() else False
    if recon_obs:
        obs_dim = env_configs['map_height'] * env_configs['map_width']
    else:
        obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(train_env.action_space, gym.spaces.Box):
        action_low, action_high = train_env.action_space.low, train_env.action_space.high

    # Load expert data
    expert_path = config['running']['expert_path']
    if debug_mode:
        expert_path = expert_path.replace('expert_data/', 'expert_data/debug_')
    if 'expert_rollouts' in config['running'].keys():
        expert_rollouts = config['running']['expert_rollouts']
    else:
        expert_rollouts = None
    (expert_obs_games, expert_acs_games, expert_rs_games), expert_mean_reward = load_expert_data(
        expert_path=expert_path,
        # use_pickle5=is_mujoco(config['env']['train_env_id']),  # True for the Mujoco envs
        num_rollouts=expert_rollouts,
        add_next_step=False,
        log_file=log_file
    )
    if 'store_by_game' in config['running'].keys() and config['running']['store_by_game']:
        expert_obs = expert_obs_games
        expert_acs = expert_acs_games
        expert_rs = expert_rs_games
    else:
        expert_obs = np.concatenate(expert_obs_games, axis=0)
        expert_acs = np.concatenate(expert_acs_games, axis=0)
        expert_rs = np.concatenate(expert_rs_games, axis=0)

    # Logger
    if log_file is None:
        gail_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        gail_logger = logger.HumanOutputFormat(log_file)

    discriminator = GailDiscriminator(
        obs_dim,
        acs_dim,
        config['DISC']['disc_layers'],
        config['DISC']['disc_batch_size'],
        get_schedule_fn(config['DISC']['disc_learning_rate']),
        expert_obs,
        expert_acs,
        is_discrete,
        obs_select_dim=config['DISC']['disc_obs_select_dim'],
        acs_select_dim=config['DISC']['disc_acs_select_dim'],
        clip_obs=config['DISC']['clip_obs'],
        initial_obs_mean=None,
        initial_obs_var=None,
        action_low=action_low,
        action_high=action_high,
        num_spurious_features=config['DISC']['num_spurious_features'],
        freeze_weights=config['DISC']['freeze_gail_weights'],
        eps=float(config['DISC']['disc_eps']),
        device=config['device'],
        recon_obs=recon_obs,
        env_configs=env_configs,
    )
    # TODO: add more config
    true_cost_function = get_true_cost_function(env_id=config['env']['eval_env_id'],
                                                env_configs=env_configs)

    if config['DISC']['use_cost_shaping_callback']:
        costShapingCallback = CostShapingCallback(true_cost_function,
                                                  obs_dim,
                                                  acs_dim,
                                                  use_nn_for_shaping=config['DISC']['use_cost_net'])
        all_callbacks = [costShapingCallback]
    else:
        gail_update = GailCallback(discriminator=discriminator,
                                   learn_cost=config['DISC']['learn_cost'],
                                   true_cost_function=true_cost_function,
                                   save_dir=save_model_mother_dir,
                                   plot_disc=False)
        all_callbacks = [gail_update]

    # Define and train model
    if 'PPO' in config.keys():
        ppo_parameters = load_ppo_config(config=config,
                                         train_env=train_env,
                                         seed=seed,
                                         log_file=log_file)
        create_nominal_agent = lambda: PPO(**ppo_parameters)
        reset_policy = None
        reset_every = None
        # forward_timesteps = None
    elif 'iteration' in config.keys():
        if "planning" in config['running'].keys() and config['running']['planning']:
            planning_config = config['Plan']
            config['Plan']['top_candidates'] = int(config['running']['sample_rollouts'])
        else:
            planning_config = None
        sampler = ConstrainedRLSampler(rollouts=int(config['running']['sample_rollouts']),
                                       store_by_game=True,  # I move the step out
                                       cost_info_str=config['env']['cost_info_str'],
                                       sample_multi_env=False,
                                       env_id=config['env']['eval_env_id'],
                                       env=eval_env,
                                       planning_config=planning_config)
        iteration_parameters = load_policy_iteration_config(config=config,
                                                            env_configs=env_configs,
                                                            train_env=train_env,
                                                            seed=seed,
                                                            log_file=log_file)
        create_nominal_agent = lambda: PolicyIterationGail(discriminator=discriminator,
                                                           **iteration_parameters)
        reset_policy = config['iteration']['reset_policy']
        reset_every = config['iteration']['reset_every']
        # forward_timesteps = config['iteration']['forward_timesteps']
    else:
        raise ValueError("Unknown model {0}.".format(config['group']))
    model = create_nominal_agent()

    # All callbacks
    save_periodically = CNSCheckpointCallback(
        env=train_env,
        save_freq=config['running']['save_every'],
        save_path=save_model_mother_dir,
        verbose=0)
    save_env_stats = SaveEnvStatsCallback(train_env, save_model_mother_dir)
    save_best = CNSEvalCallback(
        eval_env=eval_env,
        eval_freq=config['running']['eval_every'],
        deterministic=False,
        best_model_save_path=save_model_mother_dir,
        verbose=0,
        callback_on_new_best=save_env_stats)
    # plot_callback = PlotCallback(
    #     train_env_id=config['env']['train_env_id'],
    #     plot_freq=config['running']['save_every'],
    #     plot_save_dir=save_model_mother_dir
    # )
    # Organize all callbacks in list
    all_callbacks.extend([save_periodically, save_best])

    # Train
    if 'PPO' in config.keys():
        model.learn(total_timesteps=int(config['PPO']['timesteps']),
                    callback=all_callbacks)
    elif 'iteration' in config.keys():
        timesteps = 0.
        print("\nBeginning training", file=log_file, flush=True)
        for itr in range(config['running']['n_iters']):
            if reset_policy and itr % reset_every == 0:
                print("\nResetting agent", file=log_file, flush=True)
                model = create_nominal_agent()
            model.learn(
                iteration=config['iteration']['max_iter'],
                cost_function=config['env']['cost_info_str'],
                callback=all_callbacks
            )
            timesteps += model.num_timesteps
            # monitor the memory and running time
            mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                                 time_prev=time_prev,
                                                 process_name='Training PolicyIterationLagrange model',
                                                 log_file=log_file)

            # Evaluate:
            save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
            if itr % config['running']['save_every'] == 0:
                del_and_make(save_path)
            else:
                save_path = None
            sync_envs_normalization(train_env, eval_env)
            orig_observations, observations, actions, rewards, sum_rewards, lengths = sampler.sample_from_agent(
                policy_agent=model,
                new_env=eval_env,
            )
            # visualize the trajectories for gridworld
            if 'WGW' in config['env']['train_env_id'] and itr % config['running']['save_every'] == 0:
                traj_visualization_2d(config=config,
                                      observations=orig_observations,
                                      save_path=save_path,
                                      model_name=args.config_file.split('/')[-1].split('.')[0],
                                      title='Iteration-{0}'.format(itr),
                                      )

            # Save
            if itr % config['running']['save_every'] == 0:
                model.save(os.path.join(save_path, "nominal_agent"))
                if isinstance(train_env, VecNormalize):
                    train_env.save(os.path.join(save_path, "train_env_stats.pkl"))

                # visualize the cost function
                if 'WGW' in config['env']['train_env_id'] and itr % config['running']['save_every'] == 0:
                    constraint_visualization_2d(cost_function=discriminator.cost_function,
                                                feature_range=config['env']["visualize_info_ranges"],
                                                select_dims=config['env']["record_info_input_dims"],
                                                obs_dim=train_env.observation_space.shape[0],
                                                acs_dim=1 if is_discrete else train_env.action_space.shape[0],
                                                save_path=save_path,
                                                model_name=args.config_file.split('/')[-1].split('.')[0],
                                                title='Iteration-{0}'.format(itr),
                                                )

    else:
        raise ValueError("Unknown model {0}.".format(config['group']))

    # Save final discriminator
    if not config['DISC']['freeze_gail_weights']:
        discriminator.save(os.path.join(save_model_mother_dir, "gail_discriminator.pt"))

    # Save normalization stats
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))


if __name__ == "__main__":
    args = read_args()
    train(args)
