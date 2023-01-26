import math

import numpy as np
import torch
from torch.distributions import Normal
from planner.planning_agent import AbstractAgent, safe_deepcopy_env


class CEMAgent(AbstractAgent):
    """
        Cross-Entropy Method planner.
        The environment is copied and used as an oracle model to sample trajectories.
    """

    def __init__(self, env, config, cost_info_str='cost', store_by_game=False, eps=0.00001):
        super(CEMAgent, self).__init__(config)
        self.eps = eps
        self.env = env
        self.action_size = env.action_space.shape[0]
        self.cost_info_str = cost_info_str
        self.store_by_game = store_by_game

    @classmethod
    def default_config(cls):
        return dict(gamma=1.0,
                    horizon=10,
                    iterations=10,
                    candidates=100,
                    top_candidates=10,
                    std=0.1,
                    prior_lambda=1,
                    done_penalty=-1)

    # def plan_bak(self, previous_actions, prior_policy):
    #     action_distribution = Normal(
    #         loc=torch.zeros(self.config["horizon"], self.action_size),
    #         scale=torch.tensor(self.config["std"]).repeat(self.config["horizon"], self.action_size))
    #     all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
    #     sum_rewards, lengths = [], []
    #     best_orig_obs, best_obs, best_acs, best_rs, best_length, best_sum_rewards = None, None, None, None, None, None
    #     for i in range(self.config["iterations"]):
    #         sampled_actions = action_distribution.sample([self.config["candidates"]])
    #         returns = torch.zeros(self.config["candidates"])
    #         for c in range(self.config["candidates"]):
    #             obs = self.env.reset()
    #             state = None
    #             done = [False]
    #             origin_obs_game, obs_game, acs_game, rs_game = [], [], [], []
    #             for previous_action in previous_actions:
    #                 # pred_action, state = prior_policy.predict(obs, state=state, deterministic=False)
    #                 pred_action, state = prior_policy.predict(obs, state=state, deterministic=True)
    #                 obs, reward, done, _info = self.env.step(previous_action)
    #             for t in range(self.config["horizon"]+1):
    #                 if done[0] or t == self.config["horizon"]:
    #                     lengths.append(t)
    #                     # returns[c] += self.config["done_penalty"]
    #                 else:
    #                     # prior_action, state = prior_policy.predict(obs, state=state, deterministic=False)
    #                     prior_action, state = prior_policy.predict(obs, state=state, deterministic=True)
    #                     sampled_action = sampled_actions[c, t, :]
    #                     prior_lambda = self.config['prior_lambda']
    #                     action = (sampled_action + prior_lambda * prior_action) / (1 + prior_lambda)
    #                     sampled_actions[c, t, :] = action
    #                     action = action.detach().numpy()
    #                     if i == self.config["iterations"] - 1:
    #                         origin_obs_game.append(self.env.get_original_obs())
    #                         obs_game.append(obs)
    #                         acs_game.append(action)
    #                     obs, reward, done, info = self.env.step(action)
    #                     if i == self.config["iterations"] - 1:
    #                         rs_game.append(reward)
    #                     returns[c] += self.config["gamma"] ** t * (reward +
    #                                                                math.log(1 - info[0][self.cost_info_str]+self.eps))
    #             if i == self.config["iterations"] - 1:
    #                 all_orig_obs.append(np.squeeze(np.array(origin_obs_game), axis=1))
    #                 all_obs.append(np.squeeze(np.array(obs_game), axis=1))
    #                 # tmp = np.array(acs_game)
    #                 all_acs.append(np.squeeze(np.array(acs_game), axis=1))
    #                 all_rs.append(np.squeeze(np.asarray(rs_game)))
    #         # Re-fit belief to the K best action sequences
    #         _, topk = returns.topk(self.config["top_candidates"], largest=True, sorted=False)  # K ← argsort({R(j)}
    #         best_actions = sampled_actions[topk]
    #         if i == self.config["iterations"] - 1:
    #             best_orig_obs = [all_orig_obs[idx] for idx in topk]
    #             best_obs = [all_obs[idx] for idx in topk]
    #             best_acs = [all_acs[idx] for idx in topk]
    #             best_rs = [all_rs[idx] for idx in topk]
    #             best_length = [lengths[idx] for idx in topk]
    #             best_sum_rewards = [returns[idx] for idx in topk]
    #         # Update belief with new means and standard deviations
    #         mean = best_actions.mean(dim=0)
    #         std = best_actions.std(dim=0, unbiased=False)
    #         std = std.clip(min=1e-10, max=None)
    #         action_distribution = Normal(loc=mean, scale=std)
    #     # Return first action mean µ_t
    #     if self.store_by_game:
    #         return best_orig_obs, best_obs, best_acs, best_rs, best_length, best_sum_rewards
    #     else:
    #         best_orig_obs = np.squeeze(np.array(best_orig_obs), axis=1)
    #         best_obs = np.squeeze(np.array(best_obs), axis=1)
    #         best_acs = np.squeeze(np.array(best_acs), axis=1)
    #         best_rs = np.array(best_rs)
    #         best_sum_rewards = np.squeeze(np.array(best_sum_rewards), axis=1)
    #         best_length = np.array(best_length)
    #         return best_orig_obs, best_obs, best_acs, best_rs, best_length, best_sum_rewards

    def plan_multi_thread(self, previous_actions, prior_policy):
        num_threads = self.env.num_envs
        action_distribution = Normal(  # (horizon, action_size)
            loc=torch.zeros(num_threads, self.config["horizon"], self.action_size),
            scale=torch.tensor(self.config["std"]).repeat(num_threads, self.config["horizon"], self.action_size))
        reset_info = [{'write_logger': False} for i in range(num_threads)]
        for i in range(self.config["iterations"]):
            # (num_candidates, num_threads, horizons, action_dims)
            sampled_actions = action_distribution.sample([self.config["candidates"]])
            returns = torch.zeros((self.config["candidates"], num_threads))
            for c in range(self.config["candidates"]):
                obs, reset_info = self.env.reset_with_info_cost(infos=reset_info)
                state = None
                for t in range(self.config["horizon"]):
                    prior_action, state = prior_policy.predict(obs, state=state, deterministic=True)
                    sampled_action = sampled_actions[c, :, t, :]
                    prior_lambda = self.config['prior_lambda']
                    actions = (sampled_action + prior_lambda * prior_action) / (1 + prior_lambda)
                    sampled_actions[c, :, t, :] = actions
                    actions = actions.detach().numpy()
                    plan_cost = self.env.cost_function(obs.copy(),
                                                       actions.copy(),
                                                       confidence=self.config['confidence'],
                                                       force_mode=self.config['cost_mode'])
                    obs, rewards, multi_thread_dones, info = self.env.step(actions)
                    for n in range(num_threads):
                        returns[c, n] += self.config["gamma"] ** t \
                                         * (rewards[n] + math.log(1 - plan_cost[n] + self.eps))
                    done = True
                    for multi_thread_done in multi_thread_dones:
                        if not multi_thread_done:  # we will wait for all games dones
                            done = False
                    if done:
                        break

            # Re-fit belief to the K best action sequences, K ← argsort({R(j)}
            _, topk = returns.topk(self.config["top_candidates"], dim=0, largest=True, sorted=False)
            best_actions = []
            for n in range(num_threads):
                best_actions_env = sampled_actions[:, n, :, :][topk[:, n]]
                best_actions.append(best_actions_env)
            best_actions = torch.stack(best_actions, dim=1)
            # Update belief with new means and standard deviations
            b_action_mean = best_actions.mean(dim=0)
            b_action_std = best_actions.std(dim=0, unbiased=False)
            b_action_std = torch.clamp(b_action_std, min=1e-10)
            action_distribution = Normal(loc=b_action_mean, scale=b_action_std)

        return self.collect_trajectory(reset_info=reset_info,
                                       num_threads=num_threads,
                                       best_action_mean=b_action_mean)

    def plan(self, previous_actions, prior_policy):
        action_distribution = Normal(  # (horizon, action_size)
            loc=torch.zeros(self.config["horizon"], self.action_size),
            scale=torch.tensor(self.config["std"]).repeat(self.config["horizon"], self.action_size))
        reset_info = [{'write_logger': False}]
        for i in range(self.config["iterations"]):
            sampled_actions = action_distribution.sample([self.config["candidates"]])
            returns = torch.zeros(self.config["candidates"])
            for c in range(self.config["candidates"]):
                obs, reset_info = self.env.reset_with_info_cost(infos=reset_info)
                # pred_action, state = prior_policy.predict(obs, state=None, deterministic=True)
                # print(obs)
                # print(pred_action)
                state = None
                done = [False]
                # for previous_action in previous_actions:  # TODO: add previous_actions
                #     # pred_action, state = prior_policy.predict(obs, state=state, deterministic=False)
                #     pred_action, state = prior_policy.predict(obs, state=state, deterministic=True)
                #     obs, reward, done, _info = self.env.step(previous_action)
                for t in range(self.config["horizon"]):
                    # prior_action, state = prior_policy.predict(obs, state=state, deterministic=False)
                    prior_action, state = prior_policy.predict(obs, state=state, deterministic=True)
                    sampled_action = sampled_actions[c, t, :].unsqueeze(0)
                    prior_lambda = self.config['prior_lambda']
                    action = (sampled_action + prior_lambda * prior_action) / (1 + prior_lambda)
                    sampled_actions[c, t, :] = action
                    action = action.detach().numpy()
                    obs, reward, done, info = self.env.step(action)
                    returns[c] += self.config["gamma"] ** t * (reward[0] +
                                                               math.log(1 - info[0][self.cost_info_str] + self.eps))

                    if done[0]:
                        break
            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.config["top_candidates"], largest=True, sorted=False)  # K ← argsort({R(j)}
            best_actions = sampled_actions[topk]
            # Update belief with new means and standard deviations
            b_action_mean = best_actions.mean(dim=0)
            b_action_std = best_actions.std(dim=0, unbiased=False)
            b_action_std = b_action_std.clip(min=1e-10, max=None)
            action_distribution = Normal(loc=b_action_mean, scale=b_action_std)

        return self.collect_trajectory(reset_info=reset_info,
                                       num_threads=1,
                                       best_action_mean=b_action_mean.unsqueeze(dim=0))

    def collect_trajectory(self, reset_info, num_threads, best_action_mean):
        origin_obs_games = [[] for n in range(num_threads)]
        obs_games = [[] for n in range(num_threads)]
        acs_games = [[] for n in range(num_threads)]
        rs_games = [[] for n in range(num_threads)]
        sum_rewards = [0 for n in range(num_threads)]
        lengths = [1 for n in range(num_threads)]
        prev_multi_env_dones = []
        for n in range(num_threads):
            reset_info[n]['write_logger'] = True
            prev_multi_env_dones.append(False)
        obss, reset_info = self.env.reset_with_info_cost(infos=reset_info)
        for t in range(self.config["horizon"]):
            actions = best_action_mean[:, t, :].detach().numpy()
            origin_obss = self.env.get_original_obs()
            for n in range(num_threads):
                if not prev_multi_env_dones[n]:
                    origin_obs_games[n].append(origin_obss[n])
                    obs_games[n].append(obss[n])
                    acs_games[n].append(actions[n])
            obss, rewards, multi_env_dones, infos = self.env.step(actions)
            done = True
            for n in range(num_threads):
                if not prev_multi_env_dones[n]:
                    rs_games[n].append(rewards[n])
                    sum_rewards[n] += rewards[n]
                    lengths[n] += 1
                if not multi_env_dones[n]:
                    done = False
            if done:
                break
            prev_multi_env_dones = multi_env_dones
        return origin_obs_games, obs_games, acs_games, rs_games, sum_rewards, lengths

    def record(self, state, action, reward, next_state, done, info):
        pass

    def act(self, state):
        return self.plan(state, [])[0]

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False


class PytorchCEMAgent(CEMAgent):
    """
    CEM planner with Recurrent state-space models (RSSM) for transition and rewards, as in PlaNet.
    Original implementation by Kai Arulkumaran from https://github.com/Kaixhin/PlaNet/blob/master/planner.py
    Allows batch forward of many candidates (e.g. 1000)
    """

    def __init__(self, env, config, transition_model, reward_model):
        super(CEMAgent, self).__init__(config)
        self.env = env
        self.action_size = env.action_space.shape[0]
        self.transition_model = transition_model
        self.reward_model = reward_model

    def plan(self, state, belief):
        belief, state = belief.expand(self.config["candidates"], -1), state.expand(self.config["candidates"], -1)
        # Initialize factorized belief over action sequences q(a_t:t+H) ← N(0, I)
        action_distribution = Normal(torch.zeros(self.config["horizon"], self.action_size, device=belief.device),
                                     torch.ones(self.config["horizon"], self.action_size, device=belief.device))
        for i in range(self.config["iterations"]):
            # Evaluate J action sequences from the current belief (in batch)
            beliefs, states = [belief], [state]
            actions = action_distribution.sample([self.config["candidates"]])  # Sample actions
            # Sample next states
            for t in range(self.config["horizon"]):
                next_belief, next_state, _, _ = self.transition_model(states[-1], actions[:, t], beliefs[-1])
                beliefs.append(next_belief)
                states.append(next_state)
            # Calculate expected returns (batched over time x batch)
            beliefs = torch.stack(beliefs[1:], dim=0).view(self.config["horizon"] * self.config["candidates"], -1)
            states = torch.stack(states[1:], dim=0).view(self.config["horizon"] * self.config["candidates"], -1)
            returns = self.reward_model(beliefs, states).view(self.config["horizon"], self.config["candidates"]).sum(
                dim=0)
            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.config["top_candidates"], largest=True, sorted=False)  # K ← argsort({R(j)}
            best_actions = actions[topk]
            # Update belief with new means and standard deviations
            action_distribution = Normal(best_actions.mean(dim=0), best_actions.std(dim=0, unbiased=False))
        # Return first action mean µ_t
        return action_distribution.mean[0].to_list()
