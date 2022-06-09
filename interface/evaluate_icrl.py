import argparse
import csv
import glob
import logging
import os
import pickle
import re
import time
from typing import Union, Callable

import numpy as np
import yaml
import gym
from commonroad_rl.evaluate_model import LOGGER
from commonroad_rl.utils_run.vec_env import CommonRoadVecEnv
from gym import Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPOLagrangian

from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.train_model import LoggingMode
from commonroad_rl.utils_run.utils import ALGOS, get_wrapper_class

from utils.data_utils import load_config


def make_env(env_id, seed, env_kwargs, info_keywords=()):
    log_dir='icrl/test_log'

    logging_path='icrl/test_log'

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    def _init():
        env = gym.make(env_id, logging_path=logging_path, **env_kwargs)
        rank = 0
        env.seed(seed + rank)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        env = Monitor(env, log_file, info_keywords=info_keywords)
        return env

    return _init


# class CommonRoadVecEnv(DummyVecEnv):
#     def __init__(self, env_fns):
#         super().__init__(env_fns)
#         self.on_reset = None
#         self.start_times = np.array([])
#
#     def set_on_reset(self, on_reset_callback: Callable[[Env, float], None]):
#         self.on_reset = on_reset_callback
#
#     def reset(self):
#         self.start_times = np.array([time.time()] * self.num_envs)
#         return super().reset()
#
#     def step_wait(self):
#         out_of_scenarios = False
#         for env_idx in range(self.num_envs):
#             (obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx],) = self.envs[env_idx].step(
#                 self.actions[env_idx])
#             if self.buf_dones[env_idx]:
#                 # save final observation where user can get it, then reset
#                 self.buf_infos[env_idx]["terminal_observation"] = obs
#
#                 # Callback
#                 elapsed_time = time.time() - self.start_times[env_idx]
#                 self.on_reset(self.envs[env_idx], elapsed_time)
#                 self.start_times[env_idx] = time.time()
#
#                 # If one of the environments doesn't have anymore scenarios it will throw an Exception on reset()
#                 try:
#                     obs = self.envs[env_idx].reset()
#                 except IndexError:
#                     out_of_scenarios = True
#             self._save_obs(env_idx, obs)
#             self.buf_infos[env_idx]["out_of_scenarios"] = out_of_scenarios
#         return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()
# LOGGER = logging.getLogger(__name__)

def create_environments(env_id: str, viz_path: str, env_kwargs=None) -> CommonRoadVecEnv:
    """
    Create CommonRoad vectorized environment

    :param env_id: Environment gym id
    :param test_path: Path to the test files
    :param meta_path: Path to the meta-scenarios
    :param model_path: Path to the trained model
    :param viz_path: Output path for rendered images
    :param hyperparam_filename: The filename of the hyperparameters
    :param env_kwargs: Keyword arguments to be passed to the environment
    """
    env_kwargs.update({"visualization_path": viz_path,
                       "play": True})

    # Create environment
    # note that CommonRoadVecEnv is inherited from DummyVecEnv
    env = CommonRoadVecEnv([make_env(env_id, 0, env_kwargs=env_kwargs)])

    # env_fn = lambda: gym.make(env_id, play=True, **env_kwargs)
    # env = CommonRoadVecEnv([env_fn])

    def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
        # reset callback called before resetting the env
        if env.observation_dict["is_goal_reached"][-1]:
            LOGGER.info("Goal reached")
        else:
            LOGGER.info("Goal not reached")
        env.render()

    env.set_on_reset(on_reset_callback)
    normalize = True

    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    return env


def load_model(model_path: str):
    model_path = os.path.join(model_path, "best_nominal_model")
    model = PPOLagrangian.load(model_path)
    return model


def main(args):
    config, debug_mode, log_file_path = load_config(args)

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    with open(config['env']['config_path'], "r") as config_file:
        env_configs = yaml.safe_load(config_file)

    model_path = './evaluate_model/save'
    evaluation_path = os.path.join(model_path, 'evaluation')
    os.makedirs(evaluation_path, exist_ok=True)
    viz_path = os.path.join(evaluation_path, 'img')
    os.makedirs(viz_path, exist_ok=True)

    env = create_environments("commonroad-v1", viz_path, env_configs)
    model = load_model(model_path)

    num_collisions, num_off_road, num_goal_reaching, num_timeout, total_scenarios = 0, 0, 0, 0, 0
    num_scenarios = -1
    # In case there a no scenarios at all
    try:
        obs = env.reset()
    except IndexError:
        num_scenarios = 0

    count = 0
    success = 0
    while count != num_scenarios:
        done, state = False, None
        env.render()
        benchmark_id = env.venv.envs[0].benchmark_id
        print('senario id', benchmark_id)
        while not done:
            action, state = model.predict(obs, state=state, deterministic=False)
            obs, reward, done, info = env.step(action)
            env.render()

        # log collision rate, off-road rate, and goal-reaching rate
        info = info[0]
        total_scenarios += 1
        num_collisions += info["valid_collision"] if "valid_collision" in info else info["is_collision"]
        num_timeout += info.get("is_time_out", 0)
        num_off_road += info["valid_off_road"] if "valid_off_road" in info else info["is_off_road"]
        num_goal_reaching += info["is_goal_reached"]
        out_of_scenarios = info["out_of_scenarios"]

        termination_reason = "other"
        if info.get("is_time_out", 0) == 1:
            termination_reason = "time_out"
        elif info.get("is_off_road", 0) == 1:
            termination_reason = "off_road"
        elif info.get("is_collision", 0) == 1:
            termination_reason = "collision"
        elif info.get("is_goal_reached", 0) == 1:
            termination_reason = "goal_reached"

        if termination_reason == "goal_reached":
            print('goal reached')
            success += 1

        if out_of_scenarios:
            break
        count += 1


    print('total', count, 'success', success)

if __name__ == '__main__':
    main()