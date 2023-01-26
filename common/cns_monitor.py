import csv
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

import stable_baselines3.common.monitor
from utils.env_utils import is_commonroad, is_mujoco


class CNSMonitor(stable_baselines3.common.monitor.Monitor):
    def __init__(
            self,
            env: gym.Env,
            filename: Optional[str] = None,
            rank: Optional[int] = None,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
            track_keywords: Tuple[str, ...] = ()
    ):
        super(CNSMonitor, self).__init__(env=env,
                                         allow_early_resets=allow_early_resets,
                                         reset_keywords=reset_keywords,
                                         info_keywords=info_keywords,
                                         track_keywords=track_keywords,
                                         )
        if rank is not None:
            filename += 'r{0}_{1}'.format(rank, "monitor.csv")

        self.t_start = time.time()
        self.write_marker = False
        self.write_logger = True
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(self.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, self.EXT)
                else:
                    filename = filename + "." + self.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write("#%s\n" % json.dumps({"t_start": self.t_start, "env_id": env.spec and env.spec.id}))

            if is_mujoco(self.env.spec.id):
                self.logger = csv.DictWriter(self.file_handler,
                                             fieldnames=("reward", "reward_nc", "len",
                                                         "time", "constraint")
                                                        + reset_keywords + info_keywords + track_keywords,
                                             delimiter=",")
                self.event_dict = {
                    'is_constraint_break': 0
                }
            elif is_commonroad(self.env.spec.id):
                self.logger = csv.DictWriter(self.file_handler,
                                             fieldnames=("reward", "reward_nc", "len", "time",
                                                         "avg_velocity", "avg_distance",
                                                         "is_collision", "is_off_road",
                                                         "is_goal_reached", "is_time_out",
                                                         "is_over_speed", "is_too_closed", "env")
                                                        + reset_keywords + info_keywords + track_keywords,
                                             delimiter=",")
                self.event_dict = {
                    'is_collision': 0,
                    'is_off_road': 0,
                    'is_goal_reached': 0,
                    'is_time_out': 0,
                    'is_over_speed': 0,
                    'is_too_closed': 0
                }
            else:
                raise EnvironmentError("Unknown env_id {0}".format(self.env.spec.id))
            self.logger.writeheader()
            self.file_handler.flush()
            self.info_saving_file = None
            self.info_saving_items = []

    def get_logger(self):
        return self.logger

    def set_info_saving_file(self, info_saving_file, info_saving_items):
        if self.info_saving_file is not None:
            self.info_saving_file.close()
        self.info_saving_file = info_saving_file
        self.info_saving_items = info_saving_items

    def reset(self, **kwargs) -> np.ndarray:
        self.write_logger = True
        self.init_record_info()
        self.init_event_dict()
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError("Expected you to pass kwarg {} into reset".format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def reset_with_info(self, info) -> np.ndarray:
        self.write_logger = info['write_logger']
        self.init_record_info()
        self.init_event_dict()
        return self.env.reset_with_info(info)

    def init_record_info(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.needs_reset = False
        self.rewards = []
        self.rewards_not_constraint = []  # the rewards before breaking the constraint
        if is_commonroad(self.env.spec.id):
            self.ego_velocity_game = []
            self.lanebase_relative_position_game = []
        self.track = {key: [] for key in self.track_keywords}
        self.t_start = time.time()

    def init_event_dict(self):
        if is_mujoco(self.env.spec.id):
            self.event_dict = {
                'is_constraint_break': 0
            }
        elif is_commonroad(self.env.spec.id):
            self.event_dict = {
                'is_collision': 0,
                'is_off_road': 0,
                'is_goal_reached': 0,
                'is_time_out': 0,
                'is_over_speed': 0,
                'is_too_closed': 0
            }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:

        if self.write_marker and self.logger:
            marker_dict = {}
            for filename in self.logger.fieldnames:
                marker_dict.update({filename: '='})
            self.logger.writerow(marker_dict)
            self.file_handler.flush()
            self.write_marker = False

        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        if is_mujoco(self.env.spec.id):
            if info['lag_cost']:
                self.event_dict['is_constraint_break'] = 1
            # if self.env.spec.id == 'HCWithPos-v0' and info['xpos'] <= -3:
            #     self.event_dict['is_constraint_break'] = 1
            # if self.env.spec.id == 'LGW-v0' and action == 1:
            #     self.event_dict['is_constraint_break'] = 1
        elif is_commonroad(self.env.spec.id):
            if "ego_velocity" in info.keys():
                self.ego_velocity_game.append(info["ego_velocity"])
            if "lanebase_relative_position" in info.keys():
                self.lanebase_relative_position_game.append(info["lanebase_relative_position"])
            if info['is_collision']:
                self.event_dict['is_collision'] = 1
            if info['is_off_road']:
                self.event_dict['is_off_road'] = 1
            if info['is_goal_reached']:
                self.event_dict['is_goal_reached'] = 1
            if info['is_time_out']:
                self.event_dict['is_time_out'] = 1
            if 'is_over_speed' in info.keys() and info['is_over_speed']:
                self.event_dict['is_over_speed'] = 1
            if 'is_too_closed' in info.keys() and info['is_too_closed']:
                self.event_dict['is_too_closed'] = 1
        else:
            raise EnvironmentError("Unknown env_id {0}".format(self.env.spec.id))
        self.rewards.append(reward)
        if is_mujoco(self.env.spec.id):
            if not self.event_dict['is_constraint_break']:
                self.rewards_not_constraint.append(reward)
        elif is_commonroad(self.env.spec.id):
            if not self.event_dict['is_collision'] and not self.event_dict['is_off_road'] \
                    and not self.event_dict['is_time_out'] and not self.event_dict['is_over_speed'] \
                    and not self.event_dict['is_too_closed']:
                self.rewards_not_constraint.append(reward)
        else:
            raise EnvironmentError("Unknown env_id {0}".format(self.env.spec.id))

        for key in self.track_keywords:
            if key not in info:
                raise ValueError(f"Expected to find {key} in info dict")
            self.track[key].append(info[key])

        if self.info_saving_file is not None:
            info_saving_msg = ",".join([info[item] for item in self.info_saving_items])
            self.info_saving_file.write(info_saving_msg)

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_rew_nc = sum(self.rewards_not_constraint)
            assert len(self.rewards_not_constraint) <= len(self.rewards)
            if is_mujoco(self.env.spec.id):
                ep_info = {"reward": round(ep_rew, 2),
                           "reward_nc": round(ep_rew_nc, 2),
                           "len": ep_len,
                           "time": round(time.time() - self.t_start, 2),
                           'constraint': self.event_dict['is_constraint_break']}
            elif is_commonroad(self.env.spec.id):
                if len(self.ego_velocity_game) > 0:
                    ego_velocity_array = np.asarray(self.ego_velocity_game)
                    ego_velocity_game = np.sqrt(
                        np.square(ego_velocity_array[:, 0]) + np.square(ego_velocity_array[:, 1]))
                    ego_velocity_game_mean = float(ego_velocity_game.mean())
                else:
                    ego_velocity_game_mean = -1
                if len(self.lanebase_relative_position_game) > 0:
                    lanebase_relative_position_array = np.asarray(self.lanebase_relative_position_game)
                    lanebase_relative_position_game_mean = float(lanebase_relative_position_array.mean())
                else:
                    lanebase_relative_position_game_mean = -1
                # ego_velocity_tmp = np.sqrt(np.sum(np.square(ego_velocity_array), axis=1))
                ep_info = {
                    "reward": round(ep_rew, 2),
                    "reward_nc": round(ep_rew_nc, 2),
                    "len": ep_len,
                    "time": round(time.time() - self.t_start, 2),
                    "avg_velocity": round(ego_velocity_game_mean, 2),
                    "avg_distance": round(lanebase_relative_position_game_mean, 2),
                    "is_collision": self.event_dict['is_collision'],
                    "is_off_road": self.event_dict['is_off_road'],
                    "is_goal_reached": self.event_dict['is_goal_reached'],
                    "is_time_out": self.event_dict['is_time_out'],
                    'is_over_speed': self.event_dict['is_over_speed'] if 'is_over_speed' in info.keys() else -1,
                    'is_too_closed': self.event_dict['is_too_closed'] if 'is_too_closed' in info.keys() else -1,
                    "env": self.env.env.benchmark_id,
                }
                self.ego_velocity_game = []
                self.lanebase_relative_position_game = []
            else:
                raise EnvironmentError("Unknown env_id {0}".format(self.env.spec.id))
            for key in self.info_keywords:
                ep_info[key] = info[key]
            for key in self.track_keywords:
                ep_info[key] = sum(self.track[key])
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger and self.write_logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info["episode"] = ep_info
            self.init_event_dict()
        self.total_steps += 1
        return observation, reward, done, info
