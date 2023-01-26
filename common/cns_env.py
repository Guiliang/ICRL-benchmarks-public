import os
import pickle
from copy import copy, deepcopy

import numpy as np
import gym
import yaml
import stable_baselines3.common.vec_env as vec_env
from common.cns_monitor import CNSMonitor
from stable_baselines3.common import callbacks
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv, VecNormalize, VecCostWrapper
from utils.env_utils import CommonRoadExternalSignalsWrapper, MujocoExternalSignalWrapper, is_mujoco, is_commonroad


def make_env(env_id, env_configs, rank, log_dir, group, multi_env=False, seed=0):
    def _init():
        # import env
        if is_commonroad(env_id):
            # import commonroad_environment.commonroad_rl.gym_commonroad
            from commonroad_environment.commonroad_rl import gym_commonroad
        elif is_mujoco(env_id):
            # from mujuco_environment.custom_envs.envs import half_cheetah
            import mujuco_environment.custom_envs
        env_configs_copy = copy(env_configs)
        if multi_env and 'commonroad' in env_id:
            env_configs_copy.update(
                {'train_reset_config_path': env_configs['train_reset_config_path'] + '/{0}'.format(rank)}),
        if 'external_reward' in env_configs:
            del env_configs_copy['external_reward']
        env = gym.make(id=env_id, **env_configs_copy)
        env.seed(seed + rank)
        del env_configs_copy
        if is_commonroad(env_id) and 'external_reward' in env_configs:
            print("Using external signal for env: {0}.".format(env_id), flush=True)
            env = CommonRoadExternalSignalsWrapper(env=env,
                                                   group=group,
                                                   **env_configs)  # wrapper_config=env_configs['external_reward']
        elif is_mujoco(env_id):
            print("Using external signal for env: {0}.".format(env_id), flush=True)
            env = MujocoExternalSignalWrapper(env=env,
                                              group=group,
                                              **env_configs)
        monitor_rank = None
        if multi_env:
            monitor_rank = rank
        env = CNSMonitor(env=env, filename=log_dir, rank=monitor_rank)
        return env

    set_random_seed(seed)
    return _init


def make_train_env(env_id, config_path, save_dir, group='PPO', base_seed=0, num_threads=1,
                   use_cost=False, normalize_obs=True, normalize_reward=True, normalize_cost=True, multi_env=False,
                   part_data=False, **kwargs):
    if config_path is not None:
        with open(config_path, "r") as config_file:
            env_configs = yaml.safe_load(config_file)
            if is_commonroad(env_id):
                env_configs['max_scene_per_env'] = kwargs['max_scene_per_env']
                if multi_env:
                    env_configs['train_reset_config_path'] += '_split'
                if part_data:  # for debugging with only a partial dataset
                    env_configs['train_reset_config_path'] += '_debug'
                    env_configs['test_reset_config_path'] += '_debug'
                    env_configs['meta_scenario_path'] += '_debug'
    else:
        if 'Noise' in env_id:
            env_configs = {'noise_mean': kwargs['noise_mean'], 'noise_std': kwargs['noise_std']}
        else:
            env_configs = {}
    env = [make_env(env_id=env_id,
                    env_configs=env_configs,
                    rank=i,
                    log_dir=save_dir,
                    group=group,
                    multi_env=multi_env,
                    seed=base_seed)
           for i in range(num_threads)]

    env = vec_env.SubprocVecEnv(env)  # use multi-process running

    if use_cost:  # add external cost wrapper for reading cost info
        if group == 'PPO-Lag' or group == 'PI-Lag':
            env = InternalVecCostWrapper(env, kwargs['cost_info_str'])  # internal cost
        else:
            env = vec_env.VecCostWrapper(env, kwargs['cost_info_str'])  # external cost

    if group == 'PPO' or group == 'GAIL':  # without cost
        assert (all(key in kwargs for key in ['reward_gamma']))
        env = vec_env.VecNormalize(
            env,
            training=True,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            gamma=kwargs['reward_gamma'])
    else:  # Normalize cost returns
        assert (all(key in kwargs for key in ['cost_info_str', 'reward_gamma', 'cost_gamma']))
        env = vec_env.VecNormalizeWithCost(
            env, training=True,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            norm_cost=normalize_cost,
            cost_info_str=kwargs['cost_info_str'],
            reward_gamma=kwargs['reward_gamma'],
            cost_gamma=kwargs['cost_gamma'])
    return env, env_configs


def make_eval_env(env_id, config_path, save_dir, group='PPO', num_threads=1,
                  mode='test', use_cost=False, normalize_obs=True, cost_info_str='cost',
                  part_data=False, multi_env=False, **kwargs):
    if config_path is not None:
        with open(config_path, "r") as config_file:
            env_configs = yaml.safe_load(config_file)
            if is_commonroad(env_id):
                env_configs['max_scene_per_env'] = kwargs['max_scene_per_env']
                if multi_env:
                    env_configs['train_reset_config_path'] += '_split'
                if part_data:
                    env_configs['train_reset_config_path'] += '_debug'
                    env_configs['test_reset_config_path'] += '_debug'
                    env_configs['meta_scenario_path'] += '_debug'
        if is_commonroad(env_id) and mode == 'test':
            env_configs["test_env"] = True
    else:
        if 'Noise' in env_id:
            env_configs = {'noise_mean': kwargs['noise_mean'], 'noise_std': kwargs['noise_std']}
        else:
            env_configs = {}

    env = [make_env(env_id=env_id,
                    env_configs=env_configs,
                    rank=i,
                    group=group,
                    log_dir=os.path.join(save_dir, mode),
                    multi_env=multi_env)
           for i in range(num_threads)]

    env = vec_env.DummyVecEnv(env)  # use single-process running

    if use_cost:
        if group == 'PPO-Lag' or group == 'PI-Lag':
            env = InternalVecCostWrapper(env, cost_info_str)  # internal cost, use environment knowledge
        else:
            env = vec_env.VecCostWrapper(env, cost_info_str)  # external cost, must be learned
    if group == 'PPO' or group == 'GAIL':
        env = vec_env.VecNormalize(env, training=False, norm_obs=normalize_obs, norm_reward=False)
    else:
        env = vec_env.VecNormalizeWithCost(env, training=False, norm_obs=normalize_obs,
                                           norm_reward=False, norm_cost=False)
    return env, env_configs


class InternalVecCostWrapper(VecEnvWrapper):
    def \
            __init__(self, venv, cost_info_str='cost'):
        super().__init__(venv)
        self.cost_info_str = cost_info_str

    def step_async(self, actions: np.ndarray):
        self.actions = actions
        self.venv.step_async(actions)

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["cost_function"]
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        if infos is None:
            infos = {}
        # Cost depends on previous observation and current actions
        for i in range(len(infos)):
            infos[i][self.cost_info_str] = infos[i]['lag_cost']  # the pre-defined cost without learning
        self.previous_obs = obs.copy()
        return obs, rews, news, infos

    # def set_cost_function(self, cost_function):
    #     self.cost_function = cost_function

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.previous_obs = obs
        return obs

    def reset_with_values(self, info_dicts):
        """
        Reset all environments
        """
        obs = self.venv.reset_with_values(info_dicts)
        self.previous_obs = obs
        return obs

    @staticmethod
    def load(load_path: str, venv: VecEnv):
        """
        Loads a saved VecCostWrapper object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_cost_wrapper = pickle.load(file_handler)
        vec_cost_wrapper.set_venv(venv)
        return vec_cost_wrapper

    def save(self, save_path: str) -> None:
        """
        Save current VecCostWrapper object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)


# Define here to avoid circular import
def sync_envs_normalization_ppo(env: "GymEnv", eval_env: "GymEnv") -> None:
    """
    Sync eval env and train env when using VecNormalize

    :param env:
    :param eval_env:
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
        env_tmp = env_tmp.venv
        if isinstance(env_tmp, VecCostWrapper) or isinstance(env_tmp, InternalVecCostWrapper):
            env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv


class SaveEnvStatsCallback(callbacks.BaseCallback):
    def __init__(
            self,
            env,
            save_path
    ):
        super(SaveEnvStatsCallback, self).__init__()
        self.env = env
        self.save_path = save_path

    def _on_step(self):
        if isinstance(self.env, vec_env.VecNormalize):
            self.env.save(os.path.join(self.save_path, "train_env_stats.pkl"))
