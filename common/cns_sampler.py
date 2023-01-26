import numpy as np

from planner.cross_entropy_method.cem import CEMAgent
from stable_baselines3.common import vec_env
from utils.env_utils import get_benchmark_ids, get_all_env_ids


class ConstrainedRLSampler:
    """
    Sampling based on the policy and planning
    """

    def __init__(self, rollouts, store_by_game, cost_info_str, env,
                 sample_multi_env, env_id, planning_config=None):
        self.rollouts = rollouts
        self.store_by_game = store_by_game
        self.planning_config = planning_config
        self.env = env
        self.cost_info_str = cost_info_str
        self.policy_agent = None
        self.sample_multi_env = sample_multi_env
        self.env_id = env_id
        self.sample_num_threads = self.env.num_envs
        if self.planning_config is not None:
            self.apply_planning = True
            self.planner = CEMAgent(config=self.planning_config,
                                    env=self.env,
                                    cost_info_str=cost_info_str,
                                    store_by_game=self.store_by_game)
        else:
            self.apply_planning = False

    def sample_from_agent(self, policy_agent, new_env):
        self.env = new_env
        self.policy_agent = policy_agent
        if self.apply_planning:
            self.planner.env = new_env

        if self.apply_planning:
            all_orig_obs, all_obs, all_acs, all_rs, all_sum_rewards, all_lengths = [], [], [], [], [], []
            for i in range(int(float(self.rollouts)/self.sample_num_threads)):
                if self.sample_multi_env:
                    origin_obs_games, obs_games, acs_games, rs_games, sum_rewards, lengths = \
                        self.planner.plan_multi_thread(previous_actions=[],
                                                       prior_policy=policy_agent)
                else:
                    origin_obs_games, obs_games, acs_games, rs_games, sum_rewards, lengths = \
                        self.planner.plan(previous_actions=[],
                                          prior_policy=policy_agent)
                for n in range(self.sample_num_threads):
                    all_orig_obs.append(origin_obs_games[n])
                    all_obs.append(obs_games[n])
                    all_acs.append(acs_games[n])
                    all_rs.append(rs_games[n])
                    all_sum_rewards.append(sum_rewards[n])
                    all_lengths.append(lengths[n])
            if self.store_by_game:
                return all_orig_obs, all_obs, all_acs, all_rs, all_sum_rewards, all_lengths
            else:
                all_orig_obs = np.concatenate(all_orig_obs, axis=0)
                all_obs = np.concatenate(all_obs)
                all_acs = np.concatenate(all_acs)
                all_rs = np.concatenate(all_rs)
            return all_orig_obs, all_obs, all_acs, all_rs, all_sum_rewards, all_lengths
        else:
            if self.sample_multi_env:
                return self.multi_threads_sample_with_policy()
            else:
                return self.single_thread_sample_with_policy()

    def single_thread_sample_with_policy(self):
        if isinstance(self.env, vec_env.VecEnv):
            assert self.env.num_envs == 1, "You must pass only one environment when using this function"
        all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
        sum_rewards, lengths = [], []
        for i in range(self.rollouts):
            # Avoid double reset, as VecEnv are reset automatically
            if i == 0:
                obs = self.env.reset()

            done, state = False, None
            episode_sum_reward = 0.0
            episode_length = 0

            origin_obs_game = []
            obs_game = []
            acs_game = []
            rs_game = []
            while not done:
                action, state = self.policy_agent.predict(obs, state=state, deterministic=False)
                origin_obs_game.append(self.env.get_original_obs())
                obs_game.append(obs)
                acs_game.append(action)
                if not self.store_by_game:
                    all_orig_obs.append(self.env.get_original_obs())
                    all_obs.append(obs)
                    all_acs.append(action)
                obs, reward, done, _info = self.env.step(action)
                if 'admissible_actions' in _info[0].keys():
                    self.policy_agent.admissible_actions = _info[0]['admissible_actions']
                rs_game.append(reward)
                if not self.store_by_game:
                    all_rs.append(reward)

                episode_sum_reward += reward
                episode_length += 1
            if self.store_by_game:
                origin_obs_game = np.squeeze(np.array(origin_obs_game), axis=1)
                obs_game = np.squeeze(np.array(obs_game), axis=1)
                acs_game = np.squeeze(np.array(acs_game), axis=1)
                rs_game = np.squeeze(np.asarray(rs_game))
                all_orig_obs.append(origin_obs_game)
                all_obs.append(obs_game)
                all_acs.append(acs_game)
                all_rs.append(rs_game)

            sum_rewards.append(episode_sum_reward)
            lengths.append(episode_length)

        # if self.store_by_game:
        return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths
        # else:
        #     all_orig_obs = np.squeeze(np.array(all_orig_obs), axis=1)
        #     all_obs = np.squeeze(np.array(all_obs), axis=1)
        #     all_acs = np.squeeze(np.array(all_acs), axis=1)
        #     all_rs = np.array(all_rs)
        #     sum_rewards = np.squeeze(np.array(sum_rewards), axis=1)
        #     lengths = np.array(lengths)
        #     return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths

    def multi_threads_sample_with_policy(self):
        # TODO: the current version is for commonroad RL, add support for mujoco
        rollouts = int(float(self.rollouts) / self.sample_num_threads)
        all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
        sum_rewards, all_lengths = [], []
        max_benchmark_num, env_ids, benchmark_total_nums = get_all_env_ids(self.sample_num_threads, self.env)
        assert rollouts <= min(benchmark_total_nums)
        for j in range(rollouts):
            benchmark_ids = get_benchmark_ids(num_threads=self.sample_num_threads, benchmark_idx=j,
                                              benchmark_total_nums=benchmark_total_nums, env_ids=env_ids)
            obs = self.env.reset_benchmark(benchmark_ids=benchmark_ids)  # force reset for all games
            multi_thread_already_dones = [False for i in range(self.sample_num_threads)]
            done, states = False, None
            episode_sum_rewards = [0 for i in range(self.sample_num_threads)]
            episode_lengths = [0 for i in range(self.sample_num_threads)]
            origin_obs_game = [[] for i in range(self.sample_num_threads)]
            obs_game = [[] for i in range(self.sample_num_threads)]
            acs_game = [[] for i in range(self.sample_num_threads)]
            rs_game = [[] for i in range(self.sample_num_threads)]
            while not done:
                actions, states = self.policy_agent.predict(obs, state=states, deterministic=False)
                original_obs = self.env.get_original_obs()
                new_obs, rewards, dones, _infos = self.env.step(actions)
                for i in range(self.sample_num_threads):
                    if not multi_thread_already_dones[i]:  # we will not store when a game is done
                        origin_obs_game[i].append(original_obs[i])
                        obs_game[i].append(obs[i])
                        acs_game[i].append(actions[i])
                        rs_game[i].append(rewards[i])
                        episode_sum_rewards[i] += rewards[i]
                        episode_lengths[i] += 1
                    if dones[i]:
                        multi_thread_already_dones[i] = True
                done = True
                for multi_thread_done in multi_thread_already_dones:
                    if not multi_thread_done:  # we will wait for all games done
                        done = False
                        break
                obs = new_obs
            origin_obs_game = [np.array(origin_obs_game[i]) for i in range(self.sample_num_threads)]
            obs_game = [np.array(obs_game[i]) for i in range(self.sample_num_threads)]
            acs_game = [np.array(acs_game[i]) for i in range(self.sample_num_threads)]
            rs_game = [np.array(rs_game[i]) for i in range(self.sample_num_threads)]
            all_orig_obs += origin_obs_game
            all_obs += obs_game
            all_acs += acs_game
            all_rs += rs_game

            sum_rewards += episode_sum_rewards
            all_lengths += episode_lengths
        # if not self.store_by_game:
        #     all_orig_obs = np.concatenate(all_orig_obs, axis=0)
        #     all_obs = np.concatenate(all_obs, axis=0)
        #     all_acs = np.concatenate(all_acs, axis=0)
        #     all_rs = np.concatenate(all_rs, axis=0)
        return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, all_lengths
