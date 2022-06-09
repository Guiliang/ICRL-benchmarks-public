import datetime
import importlib
import json
import os
import pickle
import sys
import time
import random

import gym
import numpy as np
import yaml

from constraint_models.constraint_net.constraint_net import ConstraintNet
from exploration.exploration import ExplorationRewardCallback
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from tqdm import tqdm

import commonroad_environment.commonroad_rl.gym_commonroad  # this line must be included
from utils.data_utils import read_args, load_config, ProgressBarManager, del_and_make
from utils.env_utils import make_train_env, make_eval_env, multi_threads_sample_from_agent
from utils.model_utils import get_net_arch


def null_cost(x, *args):
    # Zero cost everywhere
    return np.zeros(x.shape[:1])


def load_expert_data(expert_path, num_rollouts, log_file):
    file_names = os.listdir(expert_path)
    # file_names = [i for i in range(29)]
    sample_names = random.sample(file_names, num_rollouts)
    # TODO: this function must be rewritten
    expert_mean_reward = []
    for i in range(num_rollouts):
        file_name = sample_names[i]
        with open(os.path.join(expert_path, file_name), "rb") as f:
            data = pickle.load(f)
        if i == 0:
            expert_obs = data['observations']
            expert_acs = data['actions']
        else:
            expert_obs = np.concatenate([expert_obs, data['observations']], axis=0)
            expert_acs = np.concatenate([expert_acs, data['actions']], axis=0)
        expert_mean_reward.append(data['reward_sum'])

    expert_mean_reward = np.mean(expert_mean_reward)
    expert_mean_length = expert_obs.shape[0] / num_rollouts

    print('Expert_mean_reward: {0} and Expert_mean_length: {1}.'.format(expert_mean_reward, expert_mean_length),
          file=log_file,
          flush=True)

    return (expert_obs, expert_acs), expert_mean_reward


def train(config):
    config, debug_mode, log_file_path = load_config(args)

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:
        config['env']['num_threads'] = 1
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['PPO']['forward_timesteps'] = 2000
        config['running']['n_eval_episodes'] = 10
        config['running']['save_every'] = 1
        debug_msg = 'debug-'

    print(json.dumps(config, indent=4), file=log_file, flush=True)

    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')
    # today = datetime.date.today()
    # currentTime = today.strftime("%b-%d-%Y-%h-%m")

    save_model_mother_dir = '{0}/{1}/{4}{2}-{3}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        current_time_date,
        debug_msg
    )

    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)

    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    # Create the vectorized environments
    train_env = make_train_env(env_id=config['env']['train_env_id'],
                               config_path=config['env']['config_path'],
                               save_dir=save_model_mother_dir,
                               base_seed=config['seed'],
                               num_threads=config['env']['num_threads'],
                               use_cost=config['env']['use_cost'],
                               normalize_obs=not config['env']['dont_normalize_obs'],
                               normalize_reward=not config['env']['dont_normalize_reward'],
                               normalize_cost=not config['env']['dont_normalize_cost'],
                               cost_info_str=config['env']['cost_info_str'],
                               reward_gamma=config['env']['reward_gamma'],
                               cost_gamma=config['env']['cost_gamma'],
                               part_data=debug_mode, )

    # We don't need cost when taking samples
    save_valid_mother_dir = os.path.join(save_model_mother_dir, "valid/")
    if not os.path.exists(save_valid_mother_dir):
        os.mkdir(save_valid_mother_dir)
    sampling_env = make_eval_env(env_id=config['env']['eval_env_id'],
                                 config_path=config['env']['config_path'],
                                 save_dir=save_valid_mother_dir,
                                 mode='valid',
                                 use_cost=False,
                                 normalize_obs=not config['env']['dont_normalize_obs'],
                                 part_data=debug_mode,
                                 log_file=log_file)
    # We don't need cost when during evaluation
    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)
    eval_env = make_eval_env(env_id=config['env']['eval_env_id'],
                             config_path=config['env']['config_path'],
                             save_dir=save_test_mother_dir,
                             mode='valid',
                             use_cost=False,
                             normalize_obs=not config['env']['dont_normalize_obs'],
                             part_data=debug_mode,
                             log_file=log_file)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    print('is_discrete', is_discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(sampling_env.action_space, gym.spaces.Box):
        action_low, action_high = sampling_env.action_space.low, sampling_env.action_space.high

    # Load expert data
    (expert_obs, expert_acs), expert_mean_reward = load_expert_data(expert_path=config['running']['expert_path'],
                                                                    num_rollouts=config['running']['expert_rollouts'],
                                                                    log_file=log_file
                                                                    )
    # Logger
    icrl_logger = logger.HumanOutputFormat(sys.stdout)

    # Initialize constraint net, true constraint net
    cn_lr_schedule = lambda x: (config['CN']['anneal_clr_by_factor'] ** (config['running']['n_iters'] * (1 - x))) \
                               * config['CN']['cn_learning_rate']
    constraint_net = ConstraintNet(
        obs_dim=obs_dim,
        acs_dim=acs_dim,
        hidden_sizes=config['CN']['cn_layers'],
        batch_size=config['CN']['cn_batch_size'],
        lr_schedule=cn_lr_schedule,
        expert_obs=expert_obs,
        expert_acs=expert_acs,
        is_discrete=is_discrete,
        regularizer_coeff=config['CN']['cn_reg_coeff'],
        obs_select_dim=None if len(config['CN']['cn_obs_select_dim']) == 0 else config['CN']['cn_obs_select_dim'],
        acs_select_dim=None if len(config['CN']['cn_acs_select_dim']) == 0 else config['CN']['cn_acs_select_dim'],
        no_importance_sampling=config['CN']['no_importance_sampling'],
        per_step_importance_sampling=config['CN']['per_step_importance_sampling'],
        clip_obs=config['CN']['clip_obs'],
        initial_obs_mean=None if not config['CN']['cn_normalize'] else np.zeros(obs_dim),
        initial_obs_var=None if not config['CN']['cn_normalize'] else np.ones(obs_dim),
        action_low=action_low,
        action_high=action_high,
        target_kl_old_new=config['CN']['cn_target_kl_old_new'],
        target_kl_new_old=config['CN']['cn_target_kl_new_old'],
        train_gail_lambda=config['CN']['train_gail_lambda'],
        eps=config['CN']['cn_eps'],
        device=config['device']
    )

    # Pass constraint net cost function to cost wrapper (train env)
    train_env.set_cost_function(constraint_net.cost_function)

    # Initialize agent
    create_nominal_agent = lambda: PPOLagrangian(
        policy=config['PPO']['policy_name'],
        env=train_env,
        learning_rate=config['PPO']['learning_rate'],
        n_steps=config['PPO']['n_steps'],
        batch_size=config['PPO']['batch_size'],
        n_epochs=config['PPO']['n_epochs'],
        reward_gamma=config['PPO']['reward_gamma'],
        reward_gae_lambda=config['PPO']['reward_gae_lambda'],
        cost_gamma=config['PPO']['cost_gamma'],
        cost_gae_lambda=config['PPO']['cost_gae_lambda'],
        clip_range=config['PPO']['clip_range'],
        clip_range_reward_vf=None if not config['PPO']['clip_range_reward_vf'] else config['PPO']['clip_range_reward_vf'],
        clip_range_cost_vf=None if not config['PPO']['clip_range_cost_vf'] else config['PPO']['clip_range_cost_vf'],
        ent_coef=config['PPO']['ent_coef'],
        reward_vf_coef=config['PPO']['reward_vf_coef'],
        cost_vf_coef=config['PPO']['cost_vf_coef'],
        max_grad_norm=config['PPO']['max_grad_norm'],
        use_sde=config['PPO']['use_sde'],
        sde_sample_freq=config['PPO']['sde_sample_freq'],
        target_kl=config['PPO']['target_kl'],
        penalty_initial_value=config['PPO']['penalty_initial_value'],
        penalty_learning_rate=config['PPO']['penalty_learning_rate'],
        budget=config['PPO']['budget'],
        seed=config['seed'],
        device=config['device'],
        verbose=config['verbose'],
        pid_kwargs=dict(alpha=config['PPO']['budget'],
                        penalty_init=config['PPO']['penalty_initial_value'],
                        Kp=config['PPO']['proportional_control_coeff'],
                        Ki=config['PPO']['integral_control_coeff'],
                        Kd=config['PPO']['derivative_control_coeff'],
                        pid_delay=config['PPO']['pid_delay'],
                        delta_p_ema_alpha=config['PPO']['proportional_cost_ema_alpha'],
                        delta_d_ema_alpha=config['PPO']['derivative_cost_ema_alpha'], ),
        policy_kwargs=dict(net_arch=get_net_arch(config))
    )

    nominal_agent = create_nominal_agent()

    # Callbacks
    all_callbacks = []
    if config['PPO']['use_curiosity_driven_exploration']:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config['device'])
        all_callbacks.append(explorationCallback)

    # Warmup
    timesteps = 0.
    if config['PPO']['warmup_timesteps']:
        # print(colorize("\nWarming up", color="green", bold=True))
        print("\nWarming up", file=log_file, flush=True)
        with ProgressBarManager(config['PPO']['warmup_timesteps']) as callback:
            nominal_agent.learn(total_timesteps=config['PPO']['warmup_timesteps'],
                                cost_function=null_cost,  # During warmup we dont want to incur any cost
                                callback=callback)
            timesteps += nominal_agent.num_timesteps

    # Train
    start_time = time.time()
    # print(utils.colorize("\nBeginning training", color="green", bold=True), flush=True)
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward, best_true_cost, best_forward_kl, best_reverse_kl = -np.inf, np.inf, np.inf, np.inf
    for itr in range(config['running']['n_iters']):
        if config['PPO']['reset_policy'] and itr != 0:
            # print(utils.colorize("Resetting agent", color="green", bold=True), flush=True)
            print("\nResetting agent", file=log_file, flush=True)
            nominal_agent = create_nominal_agent()
        current_progress_remaining = 1 - float(itr) / float(config['running']['n_iters'])

        # Update agent
        with ProgressBarManager(config['PPO']['forward_timesteps']) as callback:
            nominal_agent.learn(
                total_timesteps=config['PPO']['forward_timesteps'],
                cost_function="",  # Cost should come from cost wrapper
                callback=[callback] + all_callbacks
            )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += nominal_agent.num_timesteps

        # Sample nominal trajectories
        sync_envs_normalization(train_env, sampling_env)
        orig_observations, observations, actions, rewards, lengths = multi_threads_sample_from_agent(
            agent=nominal_agent,
            env=sampling_env,
            rollouts=config['running']['expert_rollouts'])

        # Update constraint net
        mean, var = None, None
        if config['CN']['cn_normalize']:
            mean, var = sampling_env.obs_rms.mean, sampling_env.obs_rms.var

        backward_metrics = constraint_net.train_nn(iterations=config['CN']['backward_iters'],
                                                   nominal_obs=orig_observations,
                                                   nominal_acs=actions,
                                                   episode_lengths=lengths,
                                                   obs_mean=mean,
                                                   obs_var=var,
                                                   current_progress_remaining=current_progress_remaining)

        # Pass updated cost_function to cost wrapper (train_env)
        train_env.set_cost_function(constraint_net.cost_function)

        # Evaluate:
        # reward on true environment
        sync_envs_normalization(train_env, eval_env)
        average_true_reward, std_true_reward = evaluate_policy(nominal_agent, eval_env,
                                                               n_eval_episodes=10,
                                                               deterministic=False)

        # Save
        # (1) periodically
        if itr % config['running']['save_every'] == 0:
            path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
            del_and_make(path)
            nominal_agent.save(os.path.join(path, "nominal_agent"))
            constraint_net.save(os.path.join(path, "constraint_net"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, "train_env_stats.pkl"))

        # (2) best
        if average_true_reward > best_true_reward:
            # print(utils.colorize("Saving new best model", color="green", bold=True), flush=True)
            print("Saving new best model", file=log_file, flush=True)
            nominal_agent.save(os.path.join(save_model_mother_dir, "best_nominal_model"))
            constraint_net.save(os.path.join(save_model_mother_dir, "best_constraint_net_model"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))

        # Update best metrics
        if average_true_reward > best_true_reward:
            best_true_reward = average_true_reward

        # Collect metrics
        metrics = {
            "time(m)": (time.time() - start_time) / 60,
            "iteration": itr,
            "timesteps": timesteps,
            "true/reward": average_true_reward,
            "true/reward_std": std_true_reward,
            "best_true/best_reward": best_true_reward
        }

        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})
        metrics.update(backward_metrics)

        # Log
        if config['verbose'] > 0:
            icrl_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)


if __name__ == "__main__":
    args = read_args()
    train(args)
