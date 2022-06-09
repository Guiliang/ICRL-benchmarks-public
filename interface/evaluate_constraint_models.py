import json
import logging
import os
import pickle
import time
from typing import Union, Callable
from PIL import Image
import gym
import numpy as np
import yaml
from gym import Env

from common.cns_env import make_env
from common.cns_evaluation import evaluate_with_synthetic_data
from common.cns_visualization import plot_constraints
from constraint_models.constraint_net.variational_constraint_net import VariationalConstraintNet
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPOLagrangian
from constraint_models.constraint_net.constraint_net import ConstraintNet
from commonroad_environment.commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv


from utils.data_utils import load_config, read_args, save_game_record

# def make_env(env_id, seed,  , info_keywords=()):
#     log_dir = 'icrl/test_log'
#
#     logging_path = 'icrl/test_log'
#
#     if log_dir is not None:
#         os.makedirs(log_dir, exist_ok=True)
#
#     def _init():
#         env = gym.make(env_id, logging_path=logging_path, **env_kwargs)
#         rank = 0
#         env.seed(seed + rank)
#         log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
#         env = Monitor(env, log_file, info_keywords=info_keywords)
#         return env
#
#     return _init
from utils.env_utils import is_commonroad, is_mujoco, get_all_env_ids, get_benchmark_ids
from utils.plot_utils import pngs2gif


class CommonRoadVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.on_reset = None
        self.start_times = np.array([])

    def set_on_reset(self, on_reset_callback: Callable[[Env, float], None]):
        self.on_reset = on_reset_callback

    def reset(self):
        self.start_times = np.array([time.time()] * self.num_envs)
        return super().reset()

    def reset_benchmark(self, benchmark_ids):
        self.start_times = np.array([time.time()] * self.num_envs)
        return super().reset_benchmark(benchmark_ids)

    def step_wait(self):
        out_of_scenarios = False
        for env_idx in range(self.num_envs):
            (obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx],) = self.envs[env_idx].step(
                self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs

                # Callback
                # elapsed_time = time.time() - self.start_times[env_idx]
                # self.on_reset(self.envs[env_idx], elapsed_time)
                # self.start_times[env_idx] = time.time()

                # If one of the environments doesn't have anymore scenarios it will throw an Exception on reset()
                try:
                    obs = self.envs[env_idx].reset()
                except IndexError:
                    out_of_scenarios = True
            self._save_obs(env_idx, obs)
            self.buf_infos[env_idx]["out_of_scenarios"] = out_of_scenarios
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()


LOGGER = logging.getLogger(__name__)


def create_environments(env_id: str, viz_path: str, test_path: str, model_path: str, group: str, num_threads: int = 1,
                        normalize=True, env_kwargs=None, testing_env=False, part_data=False) -> CommonRoadVecEnv:
    """
    Create CommonRoad vectorized environment
    """
    if is_commonroad(env_id):
        if viz_path is not None:
            env_kwargs.update({"visualization_path": viz_path})
        if testing_env:
            env_kwargs.update({"play": False})
            env_kwargs["test_env"] = True
        multi_env = True if num_threads > 1 else False
        if multi_env and is_commonroad(env_id=env_id):
            env_kwargs['train_reset_config_path'] += '_split'
        if part_data and is_commonroad(env_id=env_id):
            env_kwargs['train_reset_config_path'] += '_debug'
            env_kwargs['test_reset_config_path'] += '_debug'
            env_kwargs['meta_scenario_path'] += '_debug'

    # Create environment
    envs = [make_env(env_id=env_id,
                     env_configs=env_kwargs,
                     rank=i,
                     log_dir=test_path,
                     multi_env=True if num_threads > 1 else False,
                     group=group,
                     seed=0)
            for i in range(num_threads)]
    env = CommonRoadVecEnv(envs)

    # def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
    #     # reset callback called before resetting the env
    #     if env.observation_dict["is_goal_reached"][-1]:
    #         LOGGER.info("Goal reached")
    #     else:
    #         LOGGER.info("Goal not reached")
    #     # env.render()
    #
    # env.set_on_reset(on_reset_callback)
    if normalize:
        LOGGER.info("Loading saved running average")
        vec_normalize_path = os.path.join(model_path, "train_env_stats.pkl")
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            print("Loading vecnormalize.pkl from {0}".format(model_path))
        else:
            raise FileNotFoundError("vecnormalize.pkl not found in {0}".format(model_path))
        # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    return env


def load_model(model_path: str, iter_msg: str, log_file, device: str, group: str):
    if iter_msg == 'best':
        ppo_model_path = os.path.join(model_path, "best_nominal_model")
        cns_model_path = os.path.join(model_path, "best_constraint_net_model")
    else:
        ppo_model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
        cns_model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'constraint_net')
    print('Loading ppo model from {0}'.format(ppo_model_path), flush=True, file=log_file)
    print('Loading cns model from {0}'.format(cns_model_path), flush=True, file=log_file)
    ppo_model = PPOLagrangian.load(ppo_model_path, device=device)
    if group == 'ICRL' or group == 'Binary':
        cns_model = ConstraintNet.load(cns_model_path, device=device)
    elif group == 'VICRL':
        cns_model = VariationalConstraintNet.load(cns_model_path, device=device)
    elif group == 'PPO' or group == 'PPO-Lag':
        cns_model = None
    else:
        raise ValueError("Unknown group: {0}".format(group))
    return ppo_model, cns_model


def evaluate():
    # config, debug_mode, log_file_path = load_config(args)

    # if log_file_path is not None:
    #     log_file = open(log_file_path, 'w')
    # else:
    debug_mode = True
    log_file = None
    num_threads = 1
    if_testing_env = False

    load_model_name = 'train_Binary_HCWithPos-v0_with-action-multi_env-Apr-21-2022-04:49-seed_123/'
    task_name = 'Binary-HC'
    iteration_msg = 25

    model_loading_path = os.path.join('../save_model', task_name, load_model_name)
    with open(os.path.join(model_loading_path, 'model_hyperparameters.yaml')) as reader:
        config = yaml.safe_load(reader)
    config["device"] = 'cpu'
    print(json.dumps(config, indent=4), file=log_file, flush=True)

    evaluation_path = os.path.join('../evaluate_model', config['task'], load_model_name)
    if not os.path.exists(os.path.join('../evaluate_model', config['task'])):
        os.mkdir(os.path.join('../evaluate_model', config['task']))
    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    viz_path = evaluation_path
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)

    if iteration_msg == 'best':
        env_stats_loading_path = model_loading_path
    else:
        env_stats_loading_path = os.path.join(model_loading_path, 'model_{0}_itrs'.format(iteration_msg))
    if config['env']['config_path'] is not None:
        with open(config['env']['config_path'], "r") as config_file:
            env_configs = yaml.safe_load(config_file)
    else:
        env_configs = {}
    env = create_environments(env_id=config['env']['train_env_id'],
                              viz_path=viz_path,
                              test_path=evaluation_path,
                              model_path=env_stats_loading_path,
                              group=config['group'],
                              num_threads=num_threads,
                              normalize=not config['env']['dont_normalize_obs'],
                              env_kwargs=env_configs,
                              testing_env=if_testing_env,
                              part_data=debug_mode)
    # is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    # env.reset()
    # img = env.render(mode="rgb_array")
    # from PIL import Image
    # im = Image.fromarray(img)
    # im.show()
    # im.save("tmp.jpeg")
    # mean, var = None, None
    # if config['CN']['cn_normalize']:
    #     mean, var = env.obs_rms.mean, env.obs_rms.var

    # TODO: this is for a quick check, maybe remove it in the future
    env.norm_reward = False
    ppo_model, cns_model = load_model(model_loading_path,
                                      iter_msg=iteration_msg,
                                      log_file=log_file,
                                      device=config["device"],
                                      group=config["group"])

    evaluate_with_synthetic_data(env_id=config['env']['train_env_id'],
                                 cns_model=cns_model,
                                 env_configs=env_configs,
                                 model_name=load_model_name,
                                 iteration_msg=iteration_msg)

    total_scenarios, benchmark_idx = 0, 0
    if is_commonroad(env_id=config['env']['train_env_id']):
        max_benchmark_num, env_ids, benchmark_total_nums = get_all_env_ids(num_threads, env)
        # num_collisions, num_off_road, num_goal_reaching, num_timeout = 0, 0, 0, 0
    elif is_mujoco(env_id=config['env']['train_env_id']):
        max_benchmark_num = 50 / num_threads  # max number of expert traj is 50 for mujoco
    else:
        raise ValueError("Unknown env_id: {0}".format(config['env']['train_env_id']))

    record_infos = {}
    for record_info_name in config['env']["record_info_names"]:
        record_infos.update({record_info_name: []})

    success = 0
    eval_obs_all = []
    eval_acs_all = []
    rewards_sum_all = []
    tmp = []
    if cns_model is not None:
        costs = []
    while benchmark_idx < max_benchmark_num:
        if is_commonroad(env_id=config['env']['train_env_id']):
            benchmark_ids = get_benchmark_ids(num_threads=num_threads, benchmark_idx=benchmark_idx,
                                              benchmark_total_nums=benchmark_total_nums, env_ids=env_ids)
            benchmark_num_per_step = len(benchmark_ids)
            obs = env.reset_benchmark(benchmark_ids=benchmark_ids)
            record_type = 'commonroad'
        elif is_mujoco(env_id=config['env']['train_env_id']):
            benchmark_ids = [str(i) for i in range((benchmark_idx)*num_threads, (benchmark_idx+1)*num_threads)]
            benchmark_num_per_step = num_threads
            obs = env.reset()
            record_type = 'mujoco'
        else:
            raise ValueError("Unknown env_id: {0}".format(config['env']['train_env_id']))
        # env.render()
        if not os.path.exists(os.path.join(viz_path, benchmark_ids[0])):
            os.mkdir(os.path.join(viz_path, benchmark_ids[0]))
        game_info_file = open(os.path.join(viz_path, benchmark_ids[0], 'info_record.txt'), 'w')
        # if is_commonroad(env_id=config['env']['train_env_id']):
        #     game_info_file.write(
        #         'current_step, velocity, cost, is_collision, is_off_road, is_goal_reached, is_time_out\n')
        # elif is_mujoco(env_id=config['env']['train_env_id']):
        #     game_info_file.write(
        #         'current_step, x_position, cost, is_break_constraint\n')
        reward_sum = 0
        running_step = 0
        done, state, info = False, None, None
        print("Running on the file {0}".format(benchmark_ids[0]), file=log_file, flush=True)
        # img = env.render(mode="rgb_array")
        # im = Image.fromarray(img)
        # im.show()
        tmp_flag = False
        while not done:
            action, state = ppo_model.predict(obs, state=state, deterministic=False)
            original_obs = env.get_original_obs() if isinstance(env, VecNormalize) else obs
            if info is not None:
                for record_info_name in config['env']["record_info_names"]:
                    if record_info_name == 'ego_velocity_x':
                        record_infos[record_info_name].append(np.mean(info[0]['ego_velocity'][0]))
                    elif record_info_name == 'ego_velocity_y':
                        record_infos[record_info_name].append(np.mean(info[0]['ego_velocity'][1]))
                    elif record_info_name == 'same_lane_leading_obstacle_distance':
                        record_infos[record_info_name].append(np.mean(info[0]['lanebase_relative_position'][0]))
                    else:
                        record_infos[record_info_name].append(np.mean(info[0][record_info_name]))
                if cns_model is not None:
                    cost = cns_model.cost_function(obs=original_obs, acs=action)
                    costs.append(cost)
                else:
                    cost = [None]
                if is_commonroad(env_id=config['env']['train_env_id']):
                    save_game_record(info=info[0],
                                     file=game_info_file,
                                     cost=cost[0],
                                     type=record_type)
                    env.render()
                eval_obs_all.append(obs[0])
                eval_acs_all.append(action[0])
            new_obs, reward, done, info = env.step(action)
            # print(obs[0][0])
            if obs[0][0] > 3:
                tmp_flag = True
            tmp.append(obs[0][0])
            # print(action, reward, new_obs)
            reward_sum += reward
            obs = new_obs
            running_step += 1
        game_info_file.close()
        print(reward_sum, '\n')
        print(tmp_flag)
        rewards_sum_all.append(reward_sum)
        # pngs2gif(png_dir=os.path.join(viz_path, benchmark_ids[0]))

        info = info[0]
        total_scenarios += 1
        termination_reasons = []
        if is_commonroad(env_id=config['env']['train_env_id']):
            if info["episode"].get("is_time_out", 0) == 1:
                termination_reasons.append("time_out")
            elif info["episode"].get("is_off_road", 0) == 1:
                termination_reasons.append("off_road")
            elif info["episode"].get("is_collision", 0) == 1:
                termination_reasons.append("collision")
            elif info["episode"].get("is_goal_reached", 0) == 1:
                termination_reasons.append("goal_reached")
            elif "is_over_speed" in info["episode"].keys() and info["episode"].get("is_over_speed", 0) == 1:
                termination_reasons.append("over_speed")
            elif "is_too_closed" in info["episode"].keys() and info["episode"].get("is_too_closed", 0) == 1:
                termination_reasons.append("too_closed")
            # if len(termination_reasons) == 0:
            #     termination_reasons = "other"
            # else:
            #     termination_reasons = ', '.join(termination_reasons)
        elif is_mujoco(env_id=config['env']['train_env_id']):
            if info["episode"].get('constraint', 0) == 1:
                termination_reasons.append("constraint")
            else:
                termination_reasons.append("game finished")

        if is_commonroad(config['env']['train_env_id']) and "goal_reached" in termination_reasons:
            print('{0}: goal reached'.format(benchmark_ids[0]), file=log_file, flush=True)
            success += 1
        elif is_mujoco(config['env']['train_env_id']) and "game finished" in termination_reasons:
            success += 1
        benchmark_idx += 1
    print('total', total_scenarios, 'success', success, file=log_file, flush=True)
    print(np.asarray(rewards_sum_all).mean(), np.asarray(rewards_sum_all).std())
    print(np.asarray(tmp).mean())
    # eval_obs_all = np.asarray(eval_obs_all)
    # eval_acs_all = np.asarray(eval_acs_all)
    # for record_info_idx in range(len(config['env']["record_info_names"])):
    #     record_info_name = config['env']["record_info_names"][record_info_idx]
    #     plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
    #     if len(eval_acs_all.shape) == 1:
    #         empirical_input_means = np.concatenate([eval_obs_all, np.expand_dims(eval_acs_all, 1)], axis=1).mean(0)
    #     else:
    #         empirical_input_means = np.concatenate([eval_obs_all, eval_acs_all], axis=1).mean(0)
    #     plot_constraints(cost_function=cns_model.cost_function,
    #                      feature_range=config['env']["visualize_info_ranges"][record_info_idx],
    #                      select_dim=config['env']["record_info_input_dims"][record_info_idx],
    #                      obs_dim=cns_model.obs_dim,
    #                      acs_dim=1 if is_discrete else cns_model.acs_dim,
    #                      device=cns_model.device,
    #                      save_name=os.path.join(evaluation_path, "{0}_visual.png".format(record_info_name)),
    #                      feature_data=plot_record_infos,
    #                      feature_cost=plot_costs,
    #                      feature_name=record_info_name,
    #                      empirical_input_means=empirical_input_means)


if __name__ == '__main__':
    # args = read_args()
    evaluate()
