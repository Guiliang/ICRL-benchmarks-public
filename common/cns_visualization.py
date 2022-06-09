import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from stable_baselines3.common import callbacks
from utils.data_utils import del_and_make
from utils.plot_utils import plot_curve


def plot_constraints(cost_function, feature_range, select_dim, obs_dim, acs_dim,
                     save_name, device='cpu', feature_data=None, feature_cost=None, feature_name=None,
                     empirical_input_means=None, num_points=1000, axis_size=24):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    selected_feature_generation = np.linspace(feature_range[0], feature_range[1], num_points)
    if empirical_input_means is None:
        input_all = np.zeros((num_points, obs_dim + acs_dim))
    else:
        assert len(empirical_input_means) == obs_dim + acs_dim
        input_all = np.expand_dims(empirical_input_means, 0).repeat(num_points, axis=0)
        # input_all = torch.tensor(input_all)
    input_all[:, select_dim] = selected_feature_generation
    with torch.no_grad():
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        preds = cost_function(obs=obs, acs=acs, force_mode='mean')  # use the mean of a distribution for visualization
    ax[0].plot(selected_feature_generation, preds, c='r', linewidth=5)
    if feature_data is not None:
        ax[0].scatter(feature_data, feature_cost)
        ax[1].hist(feature_data, bins=40, range=(feature_range[0], feature_range[1]))
        ax[1].set_axisbelow(True)
        # Turn on the minor TICKS, which are required for the minor GRID
        ax[1].minorticks_on()
        ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax[1].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax[1].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
        ax[1].set_ylabel('Frequency', fontdict={'fontsize': axis_size})
    ax[0].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
    ax[0].set_ylabel('Cost', fontdict={'fontsize': axis_size})
    ax[0].set_ylim([0, 1])
    ax[0].set_xlim(feature_range)
    ax[0].set_axisbelow(True)
    # Turn on the minor TICKS, which are required for the minor GRID
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax[0].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    fig.savefig(save_name)
    plt.close(fig=fig)


class PlotCallback(callbacks.BaseCallback):
    """
    This callback can be used/modified to fetch something from the buffer and make a
    plot using some custom plot function.
    """

    def __init__(
            self,
            train_env_id: str,
            plot_freq: int = 10000,
            log_path: str = None,
            plot_save_dir: str = None,
            verbose: int = 1,
            name_prefix: str = "model",
            plot_feature_names_dims: dict = {},
    ):
        super(PlotCallback, self).__init__(verbose)
        self.name_prefix = name_prefix
        self.log_path = log_path
        self.plot_save_dir = plot_save_dir
        self.plot_feature_names_dims = plot_feature_names_dims

    def _init_callback(self):
        # Make directory to save plots
        # del_and_make(os.path.join(self.plot_save_dir, "plots"))
        # self.plot_save_dir = os.path.join(self.plot_save_dir, "plots")
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        try:
            obs = self.model.rollout_buffer.orig_observations.copy()
        except:  # PPO uses rollout buffer which does not store orig_observations
            obs = self.model.rollout_buffer.observations.copy()
            # unormalize observations
            obs = self.training_env.unnormalize_obs(obs)
        obs = obs.reshape(-1, obs.shape[-1])  # flatten the batch size and num_envs dimensions
        rewards = self.model.rollout_buffer.rewards.copy()
        for record_info_name in self.plot_feature_names_dims.keys():
            plot_record_infos, plot_costs = zip(*sorted(zip(obs[:, self.plot_feature_names_dims[record_info_name]], rewards)))
            path = os.path.join(self.plot_save_dir, f"{self.name_prefix}_{self.num_timesteps}_steps")
            if not os.path.exists(path):
                os.mkdir(path)
            plot_curve(draw_keys=[record_info_name],
                       x_dict={record_info_name: plot_record_infos},
                       y_dict={record_info_name: plot_costs},
                       xlabel=record_info_name,
                       ylabel='cost',
                       save_name=os.path.join(path, "{0}_visual.png".format(record_info_name)),
                       apply_scatter=True
                       )
        # self.plot_fn(obs, os.path.join(self.plot_save_dir, str(self.num_timesteps) + ".png"))

# class CNSEvalCallback(EventCallback):
#     """
#     Callback for evaluating an agent.
#
#     :param eval_env: The environment used for initialization
#     :param callback_on_new_best: Callback to trigger
#         when there is a new best model according to the ``mean_reward``
#     :param n_eval_episodes: The number of episodes to test the agent
#     :param eval_freq: Evaluate the agent every eval_freq call of the callback.
#     :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
#         will be saved. It will be updated at each evaluation.
#     :param best_model_save_path: Path to a folder where the best model
#         according to performance on the eval env will be saved.
#     :param deterministic: Whether the evaluation should
#         use a stochastic or deterministic actions.
#     :param deterministic: Whether to render or not the environment during evaluation
#     :param render: Whether to render or not the environment during evaluation
#     :param verbose:
#     """
#
#     def __init__(
#             self,
#             eval_env: Union[gym.Env, VecEnv],
#             callback_on_new_best: Optional[BaseCallback] = None,
#             n_eval_episodes: int = 5,
#             eval_freq: int = 10000,
#             log_path: str = None,
#             best_model_save_path: str = None,
#             deterministic: bool = True,
#             render: bool = False,
#             verbose: int = 1,
#             record_info_names: list = [],
#             callback_for_evaluate_policy: Optional[Callable] = None
#     ):
#         super(CNSEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
#         self.n_eval_episodes = n_eval_episodes
#         self.record_info_names = record_info_names
#         self.plot_freq = eval_freq
#         self.best_mean_reward = -np.inf
#         self.last_mean_reward = -np.inf
#         self.deterministic = deterministic
#         self.render = render
#         self.callback_for_evaluate_policy = callback_for_evaluate_policy
#
#         # Convert to VecEnv for consistency
#         if not isinstance(eval_env, VecEnv):
#             eval_env = DummyVecEnv([lambda: eval_env])
#
#         if isinstance(eval_env, VecEnv):
#             assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"
#
#         self.eval_env = eval_env
#         self.best_model_save_path = best_model_save_path
#         # Logs will be written in ``evaluations.npz``
#         if log_path is not None:
#             log_path = os.path.join(log_path, "evaluations")
#         self.log_path = log_path
#         self.evaluations_results = []
#         self.evaluations_timesteps = []
#         self.evaluations_length = []
#
#     def _init_callback(self):
#         # Does not work in some corner cases, where the wrapper is not the same
#         if not isinstance(self.training_env, type(self.eval_env)):
#             warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")
#
#         # Create folders if needed
#         if self.best_model_save_path is not None:
#             os.makedirs(self.best_model_save_path, exist_ok=True)
#         if self.log_path is not None:
#             os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
#
#     def _on_step(self) -> bool:
#
#         if self.plot_freq > 0 and self.n_calls % self.plot_freq == 0:
#             # Sync training and eval env if there is VecNormalize
#             sync_envs_normalization(self.training_env, self.eval_env)
#
#             average_true_reward, std_true_reward, record_infos, costs = \
#                 evaluate_icrl_policy(self.model, self.eval_env,
#                                      record_info_names=self.record_info_names,
#                                      n_eval_episodes=self.n_eval_episodes,
#                                      deterministic=False)
#             for record_info_idx in range(len(self.record_info_names)):
#                 record_info_name = self.record_info_names[record_info_idx]
#                 plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
#                 if len(self.expert_acs.shape) == 1:
#                     empirical_input_means = np.concatenate([self.expert_obs, np.expand_dims(self.expert_acs, 1)], axis=1).mean(0)
#                 else:
#                     empirical_input_means = np.concatenate([self.expert_obs, self.expert_acs], axis=1).mean(0)
#                 plot_constraints(cost_function=constraint_net.cost_function,
#                                  feature_range=config['env']["visualize_info_ranges"][record_info_idx],
#                                  select_dim=config['env']["record_info_input_dims"][record_info_idx],
#                                  obs_dim=constraint_net.obs_dim,
#                                  acs_dim=1 if is_discrete else constraint_net.acs_dim,
#                                  device=constraint_net.device,
#                                  save_name=os.path.join(path, "{0}_visual.png".format(record_info_name)),
#                                  feature_data=plot_record_infos,
#                                  feature_cost=plot_costs,
#                                  feature_name=record_info_name,
#                                  empirical_input_means=empirical_input_means)
#
#             if self.log_path is not None:
#                 self.evaluations_timesteps.append(self.num_timesteps)
#                 self.evaluations_results.append(episode_rewards)
#                 self.evaluations_length.append(episode_lengths)
#                 np.savez(
#                     self.log_path,
#                     timesteps=self.evaluations_timesteps,
#                     results=self.evaluations_results,
#                     ep_lengths=self.evaluations_length,
#                 )
#
#             mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
#             mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
#             self.last_mean_reward = mean_reward
#
#             if self.verbose > 0:
#                 print(
#                     f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
#                 print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
#             # Add to current Logger
#             self.logger.record("eval/mean_reward", float(mean_reward))
#             self.logger.record("eval/mean_ep_length", mean_ep_length)
#             self.logger.record("eval/best_mean_reward", max(self.best_mean_reward, mean_reward))
#
#             if mean_reward > self.best_mean_reward:
#                 if self.verbose > 0:
#                     print("New best mean reward!")
#                 if self.best_model_save_path is not None:
#                     self.model.save(os.path.join(self.best_model_save_path, "best_model"))
#                 self.best_mean_reward = mean_reward
#                 # Trigger callback if needed
#                 if self.callback is not None:
#                     return self._on_event()
#
#         return True
#
#     def update_child_locals(self, locals_: Dict[str, Any]) -> None:
#         """
#         Update the references to the local variables.
#
#         :param locals_: the local variables during rollout collection
#         """
#         if self.callback:
#             self.callback.update_locals(locals_)
