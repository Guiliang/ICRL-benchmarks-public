import datetime
import json
import os
import sys
import time

import gym
import yaml

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from utils.env_utils import check_if_duplicate_seed
from common.cns_env import make_train_env, make_eval_env, SaveEnvStatsCallback
from common.cns_evaluation import CNSEvalCallback
from common.cns_save_callbacks import CNSCheckpointCallback
from common.cns_visualization import PlotCallback
from constraint_models.constraint_net.gail_net import GailDiscriminator, GailCallback
from exploration.exploration import CostShapingCallback
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecNormalize
from utils.data_utils import read_args, load_config, process_memory, load_expert_data, print_resource
from utils.model_utils import load_ppo_config
from utils.true_constraint_functions import get_true_cost_function
import stable_baselines3.common.callbacks as callbacks


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
        # config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['running']['save_every'] = 2048
        config['running']['eval_every'] = 1024
        debug_msg = 'debug-'
        partial_data = True
        # debug_msg += 'part-'
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
                                          log_file=log_file)

    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Loading environment', log_file=log_file)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
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
    (expert_obs, expert_acs, expert_rs), expert_mean_reward = load_expert_data(
        expert_path=expert_path,
        # use_pickle5=is_mujoco(config['env']['train_env_id']),  # True for the Mujoco envs
        num_rollouts=expert_rollouts,
        store_by_game=False,
        add_next_step=False,
        log_file=log_file
    )

    # Logger
    if log_file is None:
        gail_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        gail_logger = logger.HumanOutputFormat(log_file)

    # Do we want to restore gail from a saved model?
    if config['DISC']['gail_path'] is not None:
        discriminator = GailDiscriminator.load(
            config.gail_path,
            obs_dim=obs_dim,
            acs_dim=acs_dim,
            is_discrete=is_discrete,
            expert_obs=expert_obs,
            expert_acs=expert_acs,
            obs_select_dim=config['DISC']['disc_obs_select_dim'],
            acs_select_dim=config['DISC']['disc_acs_select_dim'],
            clip_obs=None,
            obs_mean=None,
            obs_var=None,
            action_low=action_low,
            action_high=action_high,
            device=config['device'],
        )
        discriminator.freeze_weights = config['DISC']['freeze_gail_weights']
    else:  # Initialize GAIL and setup its callback
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
            device=config['device']
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
    ppo_parameters = load_ppo_config(config=config, train_env=train_env, seed=seed, log_file=log_file)
    model = PPO(**ppo_parameters)

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
    model.learn(total_timesteps=int(config['PPO']['timesteps']),
                callback=all_callbacks)

    # Save final discriminator
    if not config['DISC']['freeze_gail_weights']:
        discriminator.save(os.path.join(save_model_mother_dir, "gail_discriminator.pt"))

    # Save normalization stats
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))


if __name__ == "__main__":
    args = read_args()
    train(args)
