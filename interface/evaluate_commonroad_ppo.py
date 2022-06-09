import json
import logging
import os
import pickle
import time
from typing import Union, Callable
import numpy as np
import yaml
from gym import Env

from common.cns_env import make_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from commonroad_environment.commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from utils.data_utils import load_config, read_args, save_game_record, load_ppo_model
from utils.env_utils import get_obs_feature_names
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

    def step_wait(self):
        out_of_scenarios = False
        for env_idx in range(self.num_envs):
            (obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx],) = self.envs[env_idx].step(
                self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs

                # Callback
                elapsed_time = time.time() - self.start_times[env_idx]
                self.on_reset(self.envs[env_idx], elapsed_time)
                self.start_times[env_idx] = time.time()

                # If one of the environments doesn't have anymore scenarios it will throw an Exception on reset()
                try:
                    obs = self.envs[env_idx].reset()
                except IndexError:
                    out_of_scenarios = True
            self._save_obs(env_idx, obs)
            self.buf_infos[env_idx]["out_of_scenarios"] = out_of_scenarios
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()


LOGGER = logging.getLogger(__name__)


def create_environments(env_id: str, viz_path: str, test_path: str, model_path: str,
                        normalize=True, env_kwargs=None, testing_env=False, debug_mode=False) -> CommonRoadVecEnv:

    """
    Create CommonRoad vectorized environment

    :param env_id: Environment gym id
    :param test_path: Path to the test files
    # :param meta_path: Path to the meta-scenarios
    :param model_path: Path to the trained model
    :param viz_path: Output path for rendered images
    # :param hyperparam_filename: The filename of the hyperparameters
    :param env_kwargs: Keyword arguments to be passed to the environment
    """
    env_kwargs.update({"visualization_path": viz_path,})
                       # "play": True})
    if testing_env:
        env_kwargs["test_env"] = True
    if debug_mode:
        env_kwargs['train_reset_config_path'] += '_debug'
        env_kwargs['test_reset_config_path'] += '_debug'
    # Create environment
    # note that CommonRoadVecEnv is inherited from DummyVecEnv
    env = CommonRoadVecEnv([make_env(env_id, env_kwargs,
                                     group=env_kwargs["env_kwargs"],
                                     rank=0,
                                     log_dir=test_path,
                                     seed=0)])

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
    if normalize:
        LOGGER.info("Loading saved running average")
        vec_normalize_path = os.path.join(model_path, "train_env_stats.pkl")
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
        else:
            raise FileNotFoundError("vecnormalize.pkl not found in {0}".format(model_path))
        # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    return env


def evaluate():
    # config, debug_mode, log_file_path = load_config(args)

    # if log_file_path is not None:
    #     log_file = open(log_file_path, 'w')
    # else:
    debug_mode = True
    log_file = None
    if_testing_env = False

    # load_model_name = 'train_ppo_highD_no_collision-multi_env-Mar-10-2022-00:18'
    load_model_name = 'train_ppo_highD-multi_env-Mar-10-2022-04:37'
    task_name = 'PPO-highD'
    iteration_msg = 'best'

    model_loading_path = os.path.join('../save_model', task_name, load_model_name)
    with open(os.path.join(model_loading_path, 'model_hyperparameters.yaml')) as reader:
        config = yaml.safe_load(reader)

    print(json.dumps(config, indent=4), file=log_file, flush=True)

    # if 'ppo' in config['env']['config_path']:
    #     config['env']['config_path'] = config['env']['config_path'].replace('_ppo', '')

    with open(config['env']['config_path'], "r") as config_file:
        env_configs = yaml.safe_load(config_file)

    evaluation_path = os.path.join('../evaluate_model',
                                   config['task'],
                                   load_model_name,
                                   iteration_msg+"-{0}".format('test' if if_testing_env else 'train'))
    if not os.path.exists(os.path.join('../evaluate_model', config['task'], load_model_name)):
        os.mkdir(os.path.join('../evaluate_model', config['task'], load_model_name))
    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    # viz_path = os.path.join(evaluation_path, 'img')
    viz_path = evaluation_path
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)

    # save_expert_data_path = os.path.join('../data/expert_data/', load_model_name)
    # if not os.path.exists(save_expert_data_path):
    #     os.mkdir(save_expert_data_path)
    if iteration_msg == 'best':
        env_stats_loading_path = model_loading_path
    else:
        env_stats_loading_path = os.path.join(model_loading_path, 'model_{0}_itrs'.format(iteration_msg))
    env = create_environments(env_id="commonroad-v1",
                              viz_path=viz_path,
                              test_path=evaluation_path,
                              model_path=env_stats_loading_path,
                              normalize=not config['env']['dont_normalize_obs'],
                              env_kwargs=env_configs,
                              testing_env=if_testing_env,
                              debug_mode=debug_mode)

    # TODO: this is for a quick check, maybe remove it in the future
    env.norm_reward = False

    feature_names = get_obs_feature_names(env)
    print("The observed features are: {0}".format(feature_names))

    model = load_ppo_model(model_loading_path, iter_msg=iteration_msg, log_file=log_file)
    num_collisions, num_off_road, num_goal_reaching, num_timeout, total_scenarios = 0, 0, 0, 0, 0
    num_scenarios = 200
    # In case there a no scenarios at all
    try:
        obs = env.reset()
    except IndexError:
        num_scenarios = 0

    count = 0
    success = 0
    benchmark_id_all = []
    while count != num_scenarios:
        done, state = False, None
        benchmark_id = env.venv.envs[0].benchmark_id
        if benchmark_id in benchmark_id_all:
        # if benchmark_id != 'DEU_LocationBUpper-3_13_T-1':
            print('skip game', benchmark_id, file=log_file, flush=True)
            obs = env.reset()
            continue
        else:
            benchmark_id_all.append(benchmark_id)
        print('senario id', benchmark_id, file=log_file, flush=True)
        env.render()
        game_info_file = open(os.path.join(viz_path, benchmark_id, 'info_record.txt'), 'w')
        game_info_file.write('current_step, velocity, is_collision, is_time_out, is_off_road, is_goal_reached\n')
        obs_all = []
        original_obs_all = []
        action_all = []
        reward_sum = 0
        running_step = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            new_obs, reward, done, info = env.step(action)
            reward_sum += reward
            obs_all.append(obs)
            original_obs = env.get_original_obs() if isinstance(env, VecNormalize) else obs
            original_obs_all.append(original_obs)
            action_all.append(action)
            save_game_record(info[0], game_info_file)
            env.render()
            obs = new_obs
            running_step += 1
        game_info_file.close()

        pngs2gif(png_dir=os.path.join(viz_path, benchmark_id))

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
            print('goal reached', file=log_file, flush=True)
            success += 1
            # print('saving exper data', file=log_file, flush=True)
            # saving_expert_data = {
            #     'observations': np.asarray(obs_all),
            #     'actions': np.asarray(action_all),
            #     'original_observations': np.asarray(original_obs_all),
            #     'reward_sum': reward_sum
            # }
        # break
        if out_of_scenarios:
            break
        count += 1

    print('total', count, 'success', success, file=log_file, flush=True)


if __name__ == '__main__':
    # args = read_args()
    evaluate()
