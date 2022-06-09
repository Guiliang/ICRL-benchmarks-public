import sys
from itertools import accumulate
from typing import Tuple, Callable, Optional, Type, Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from utils.model_utils import handle_model_parameters


class ApproximateNet(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            expert_rs: np.ndarray,
            is_discrete: bool,
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = False,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            eps: float = 1e-5,
            dir_prior: float = 1,
            discount_factor: float = 1,
            log_std_init: float = 0.0,
            device: str = "cpu",
            log_file=None
    ):
        super(ApproximateNet, self).__init__()
        self.log_file = log_file
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim
        self._define_input_dims()

        self.expert_obs = expert_obs
        self.expert_acs = expert_acs
        self.expert_rs = expert_rs

        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.regularizer_coeff = regularizer_coeff
        self.importance_sampling = not no_importance_sampling
        self.per_step_importance_sampling = per_step_importance_sampling
        self.clip_obs = clip_obs
        self.device = device
        self.eps = eps
        self.dir_prior = dir_prior
        self.discount_factor = discount_factor

        self.train_gail_lambda = train_gail_lambda

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule

        self.current_obs_mean = initial_obs_mean
        self.current_obs_var = initial_obs_var
        self.action_low = action_low
        self.action_high = action_high

        self.target_kl_old_new = target_kl_old_new
        self.target_kl_new_old = target_kl_new_old

        self.current_progress_remaining = 1.
        self.log_std_init = log_std_init

        self._build()

    def _build(self) -> None:
        """
        build the network, what we need
        1) model the parameters of the constraint distribution following the beta distribution.
        2) model the Q function for representing the constraints.
        3) model the policy for approximating the policy represented by the Q function.
        :return:
        """

        # predict both alpha (>0) and beta (>0) parameters,
        # The mean is alpha/(alpha+beta) and variance is alpha*beta/(alpha+beta)^2*(alpha+beta+1)
        self.encoder = nn.Sequential(
            *create_mlp(self.input_dims, 2, self.hidden_sizes),
            nn.Softplus()
        )
        self.encoder.to(self.device)

        # predict the Q(s,a) values, the action is continuous
        self.q_net = nn.Sequential(
            *create_mlp(self.input_dims, 1, self.hidden_sizes)
        )
        self.q_net.to(self.device)

        # predict the policy \pi(a|s) values, the action is continuous. Actor-Critic should be employed
        self.policy_network = nn.Sequential(
            *create_mlp(self.obs_dim, self.acs_dim, self.hidden_sizes)
        )
        self.log_std = nn.Parameter(torch.ones(self.acs_dim) * self.log_std_init, requires_grad=True).to(self.device)
        self.policy_network.to(self.device)

        # build different optimizers for different models
        self.optimizers = {'ICRL': None, 'policy': None}
        param_active_key_words = {'ICRL': ['encoder', 'q_network'],
                                  'policy': ['policy_network'], }
        for key in self.optimizers.keys():
            param_frozen_list, param_active_list = \
                handle_model_parameters(model=self,
                                        active_keywords=param_active_key_words[key],
                                        model_name=key,
                                        log_file=self.log_file,
                                        set_require_grad=True)
            if self.optimizer_class is not None:
                optimizer = self.optimizer_class([{'params': param_frozen_list, 'lr': 0.0},
                                                  {'params': param_active_list,
                                                   'lr': self.lr_schedule(1)}],
                                                 lr=self.lr_schedule(1))
            else:
                optimizer = None
            self.optimizers.update({key: optimizer})

    def forward(self, x: torch.tensor) -> torch.tensor:
        alpha_beta = self.encoder(x)
        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        QValues = self.q_net(x)
        policy = self.policy_network(x)

        return alpha, beta, QValues, policy

    def cost_function(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""
        obs, acs = self.prepare_data(obs, acs)
        cost_input = torch.cat([obs, acs], dim=-1)
        with torch.no_grad():
            alpha, beta, _, _ = self.__call__(cost_input)
        # TODO: Maybe we should not use the expectation of beta distribution
        out = alpha / (alpha + beta)
        cost = 1 - out.detach().cpu().numpy()
        # return cost.squeeze(axis=-1)
        return cost

    def call_forward(self, x: np.ndarray):
        with torch.no_grad():
            out = self.__call__(torch.tensor(x, dtype=torch.float32).to(self.device))
        return out

    def dirichlet_kl_divergence(self, alpha, prior):
        """
        KL divergence between two dirichlet distribution
        There are multiple ways of modelling a dirichlet:
        1) by Laplace approximation with logistic normal: https://arxiv.org/pdf/1703.01488.pdf
        2) by directly modelling dirichlet parameters: https://arxiv.org/pdf/1901.02739.pdf
        code reference：
        1） https://github.com/sophieburkhardt/dirichlet-vae-topic-models
        2） https://github.com/is0383kk/Dirichlet-VAE
        """
        analytical_kld = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(prior, dim=1))
        analytical_kld += torch.sum(torch.lgamma(prior), dim=1)
        analytical_kld -= torch.sum(torch.lgamma(alpha), dim=1)
        minus_term = alpha - prior
        # tmp = torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
        digamma_term = torch.digamma(alpha) - \
                       torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
        test = torch.sum(torch.mul(minus_term, digamma_term), dim=1)
        analytical_kld += test
        # self.analytical_kld = self.mask * self.analytical_kld  # mask paddings
        return analytical_kld

    def predict(self, obs: torch.tensor, deterministic=False):
        mean_action = self.policy_network(obs)
        std = torch.exp(self.log_std)
        eps = torch.randn_like(std)
        if deterministic:
            action = mean_action
        else:
            action = eps * std + mean_action

        output_action = []
        for i in range(self.acs_dim):
            output_action.append(torch.clamp(action[:, i], min=self.action_low[i], max=self.action_high[i]))

        return torch.stack(output_action, dim=1)

    def train_nn(
            self,
            iterations: np.ndarray,
            # nominal_obs: np.ndarray,
            # nominal_acs: np.ndarray,
            # episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # Prepare data
        # TODO: maybe we will need the nominal data
        expert_obs, expert_acs, expert_rs = self.prepare_data(self.expert_obs, self.expert_acs, self.expert_rs)
        expert_obs, expert_acs, expert_rs = self.prepare_data(self.expert_obs, self.expert_acs, self.expert_rs)
        icrl_loss = torch.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            # TODO: maybe we do need the importance sampling
            # is_weights = torch.ones(expert_input.shape[0])

            # Do a complete pass on data, we don't have to use the get(), but let's leave it here for future usage
            for exp_batch_indices in self.get(expert_obs.shape[0], expert_obs.shape[0]):
                # Get batch data
                expert_batch_obs = expert_obs[exp_batch_indices[0]]
                expert_batch_acs = expert_acs[exp_batch_indices[0]]
                expert_batch_rs = expert_rs[exp_batch_indices[0]]
                batch_size = expert_batch_obs.shape[0]

                # Make predictions
                expert_batch_input = torch.cat([expert_batch_obs, expert_batch_acs], dim=-1)
                expert_q_values_t = self.q_net(expert_batch_input[:, 0, :]).squeeze(dim=1)
                expert_q_values_next_t = self.q_net(expert_batch_input[:, 1, :]).squeeze(dim=1)

                alpha_beta_t = self.encoder(expert_batch_input[:, 0, :])
                expert_alpha_t = alpha_beta_t[:, 0]
                expert_beta_t = alpha_beta_t[:, 1]

                # Compute KLD
                prior = (torch.ones((batch_size, 2), dtype=torch.float32) * self.dir_prior).to(self.device)
                analytical_kld = self.dirichlet_kl_divergence(alpha=torch.stack([expert_alpha_t, expert_beta_t], dim=1),
                                                              prior=prior).mean()

                # Calculate log-likelihood of exp TD error given constraint parameterisation
                expert_batch_rs_t = expert_batch_rs[:, 0]
                td = expert_q_values_t - self.discount_factor * expert_q_values_next_t - expert_batch_rs_t
                tmp1 = torch.clamp(td, max=-self.eps)  # log prob must be less than zero
                td_cost = torch.exp(torch.clamp(td, max=-self.eps))
                tmp2 = torch.distributions.Beta(expert_alpha_t, expert_beta_t).log_prob(td_cost)
                constraint_loss = - torch.distributions.Beta(expert_alpha_t, expert_beta_t).log_prob(td_cost).mean()

                # Compute reconstruction loss
                pred_act = self.predict(expert_batch_obs[:, 0, :]).detach()
                reconstruct_loss = torch.square(pred_act - expert_batch_acs[:, 0, :]).mean()

                # Calculate loss
                icrl_loss = reconstruct_loss + analytical_kld + constraint_loss

                # Update
                self.optimizers['ICRL'].zero_grad()
                icrl_loss.backward()
                self.optimizers['ICRL'].step()

                # policy network loss
                pred_act = self.policy_network(expert_batch_obs[:, 0, :])
                pred_batch_input = torch.cat([expert_batch_obs[:, 0, :], pred_act], dim=-1)
                pred_q_values = self.q_net(pred_batch_input).detach()
                policy_loss = torch.square(pred_act - torch.log(pred_q_values))

                # Update
                self.optimizers['policy'].zero_grad()
                policy_loss.backward()
                self.optimizer['policy'].step()

        bw_metrics = {"backward/icrl_loss": icrl_loss.item(),
                      # "backward/expert_loss": expert_loss.item(),
                      # "backward/unweighted_nominal_loss": torch.mean(torch.log(nominal_preds + self.eps)).item(),
                      # "backward/nominal_loss": nominal_loss.item(),
                      # "backward/regularizer_loss": regularizer_loss.item(),
                      # "backward/is_mean": torch.mean(is_weights).detach().item(),
                      # "backward/is_max": torch.max(is_weights).detach().item(),
                      # "backward/is_min": torch.min(is_weights).detach().item(),
                      # "backward/nominal_preds_max": torch.max(nominal_preds).item(),
                      # "backward/nominal_preds_min": torch.min(nominal_preds).item(),
                      # "backward/nominal_preds_mean": torch.mean(nominal_preds).item(),
                      # "backward/expert_preds_max": torch.max(expert_preds).item(),
                      # "backward/expert_preds_min": torch.min(expert_preds).item(),
                      # "backward/expert_preds_mean": torch.mean(expert_preds).item(),
                      }
        # if self.importance_sampling:
        #     stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
        #                     "backward/kl_new_old": kl_new_old.item(),
        #                     "backward/early_stop_itr": early_stop_itr}
        #     bw_metrics.update(stop_metrics)

        return bw_metrics

    def compute_is_weights(self, preds_old: torch.Tensor, preds_new: torch.Tensor,
                           episode_lengths: np.ndarray) -> torch.tensor:
        with torch.no_grad():
            n_episodes = len(episode_lengths)
            cumulative = [0] + list(accumulate(episode_lengths))

            ratio = (preds_new + self.eps) / (preds_old + self.eps)
            prod = [torch.prod(ratio[cumulative[j]:cumulative[j + 1]])
                    for j in range(n_episodes)]
            prod = torch.tensor(prod)
            normed = n_episodes * prod / (torch.sum(prod) + self.eps)

            if self.per_step_importance_sampling:
                is_weights = torch.tensor(ratio / torch.mean(ratio))
            else:
                is_weights = []
                for length, weight in zip(episode_lengths, normed):
                    is_weights += [weight] * length
                is_weights = torch.tensor(is_weights)

            # Compute KL(old, current)
            kl_old_new = torch.mean(-torch.log(prod + self.eps))
            # Compute KL(current, old)
            prod_mean = torch.mean(prod)
            kl_new_old = torch.mean((prod - prod_mean) * torch.log(prod + self.eps) / (prod_mean + self.eps))

        return is_weights.to(self.device), kl_old_new, kl_new_old

    def prepare_data(
            self,
            obs: np.ndarray,
            acs: np.ndarray,
            rs: np.ndarray = None,
            select_dim: bool = True,
    ) -> torch.tensor:

        obs = self.normalize_obs(obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)
        acs = self.reshape_actions(acs)
        acs = self.clip_actions(acs, self.action_low, self.action_high)
        if select_dim:
            obs = self.select_appropriate_dims(select_dim=self.select_obs_dim, x=obs)
            acs = self.select_appropriate_dims(select_dim=self.select_acs_dim, x=acs)
        if rs is None:
            return torch.tensor(obs, dtype=torch.float32).to(self.device), \
                   torch.tensor(acs, dtype=torch.float32).to(self.device)
        else:
            return torch.tensor(obs, dtype=torch.float32).to(self.device), \
                   torch.tensor(acs, dtype=torch.float32).to(self.device), \
                   torch.tensor(rs, dtype=torch.float32).to(self.device)

    def select_appropriate_dims(self, select_dim: list, x: Union[np.ndarray, torch.tensor]) -> Union[
        np.ndarray, torch.tensor]:
        return x[..., select_dim]

    def normalize_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
                      clip_obs: Optional[float] = None) -> np.ndarray:
        if mean is not None and var is not None:
            mean, var = mean[None], var[None]
            obs = (obs - mean) / np.sqrt(var + self.eps)
        if clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        return obs

    def reshape_actions(self, acs):
        if self.is_discrete:
            acs_ = acs.astype(int)
            if len(acs.shape) > 1:
                acs_ = np.squeeze(acs_, axis=-1)
            acs = np.zeros([acs.shape[0], self.acs_dim])
            acs[np.arange(acs_.shape[0]), acs_] = 1.

        return acs

    def clip_actions(self, acs: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
        if high is not None and low is not None:
            acs = np.clip(acs, low, high)

        return acs

    def get(self, nom_size: int, exp_size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(nom_size), np.arange(exp_size)
        else:
            size = min(nom_size, exp_size)
            indices = np.random.permutation(size)

            batch_size = self.batch_size
            # Return everything, don't create minibatches
            if batch_size is None:
                batch_size = size

            start_idx = 0
            while start_idx < size:
                batch_indices = indices[start_idx:start_idx + batch_size]
                yield batch_indices, batch_indices
                start_idx += batch_size

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        for optimizer_name in self.optimizers.keys():
            update_learning_rate(self.optimizers[optimizer_name], self.lr_schedule(current_progress_remaining))

    def save(self, save_path):
        state_dict = dict(
            cn_network=self.network.state_dict(),
            cn_optimizer=self.optimizer.state_dict(),
            obs_dim=self.obs_dim,
            acs_dim=self.acs_dim,
            is_discrete=self.is_discrete,
            obs_select_dim=self.obs_select_dim,
            acs_select_dim=self.acs_select_dim,
            clip_obs=self.clip_obs,
            obs_mean=self.current_obs_mean,
            obs_var=self.current_obs_var,
            action_low=self.action_low,
            action_high=self.action_high,
            device=self.device,
            hidden_sizes=self.hidden_sizes
        )
        torch.save(state_dict, save_path)

    def _load(self, load_path):
        state_dict = torch.load(load_path)
        if "cn_network" in state_dict:
            self.network.load_state_dict(dic["cn_network"])
        if "cn_optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(dic["cn_optimizer"])

    # Provide basic functionality to load this class
    @classmethod
    def load(
            cls,
            load_path: str,
            obs_dim: Optional[int] = None,
            acs_dim: Optional[int] = None,
            is_discrete: bool = None,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            clip_obs: Optional[float] = None,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            device: str = "auto"
    ):

        state_dict = torch.load(load_path)
        # If value isn't specified, then get from state_dict
        if obs_dim is None:
            obs_dim = state_dict["obs_dim"]
        if acs_dim is None:
            acs_dim = state_dict["acs_dim"]
        if is_discrete is None:
            is_discrete = state_dict["is_discrete"]
        if obs_select_dim is None:
            obs_select_dim = state_dict["obs_select_dim"]
        if acs_select_dim is None:
            acs_select_dim = state_dict["acs_select_dim"]
        if clip_obs is None:
            clip_obs = state_dict["clip_obs"]
        if obs_mean is None:
            obs_mean = state_dict["obs_mean"]
        if obs_var is None:
            obs_var = state_dict["obs_var"]
        if action_low is None:
            action_low = state_dict["action_low"]
        if action_high is None:
            action_high = state_dict["action_high"]
        if device is None:
            device = state_dict["device"]

        # Create network
        hidden_sizes = state_dict["hidden_sizes"]
        constraint_net = cls(
            obs_dim, acs_dim, hidden_sizes, None, None, None, None,
            is_discrete, None, obs_select_dim, acs_select_dim, None,
            None, None, clip_obs, obs_mean, obs_var, action_low, action_high,
            None, None, device
        )
        constraint_net.network.load_state_dict(state_dict["cn_network"])

        return constraint_net
