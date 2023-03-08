import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch as th

from constraint_models.constraint_net.constraint_net import ConstraintNet
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm

from utils.model_utils import dirichlet_kl_divergence_loss


class VariationalConstraintNet(ConstraintNet):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            task: str = 'VICRL',
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = True,
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
            recon_obs: bool = False,
            env_configs: dict = None,
            device: str = "cpu",
            di_prior: list = [1, 1],
            mode: str = 'sample',
            confidence: float = 0.5,
            log_file=None,
    ):
        assert 'VICRL' in task
        super().__init__(obs_dim=obs_dim,
                         acs_dim=acs_dim,
                         hidden_sizes=hidden_sizes,
                         batch_size=batch_size,
                         lr_schedule=lr_schedule,
                         expert_obs=expert_obs,
                         expert_acs=expert_acs,
                         is_discrete=is_discrete,
                         task=task,
                         regularizer_coeff=regularizer_coeff,
                         obs_select_dim=obs_select_dim,
                         acs_select_dim=acs_select_dim,
                         optimizer_class=optimizer_class,
                         optimizer_kwargs=optimizer_kwargs,
                         no_importance_sampling=no_importance_sampling,
                         per_step_importance_sampling=per_step_importance_sampling,
                         clip_obs=clip_obs,
                         initial_obs_mean=initial_obs_mean,
                         initial_obs_var=initial_obs_var,
                         action_low=action_low,
                         action_high=action_high,
                         target_kl_old_new=target_kl_old_new,
                         target_kl_new_old=target_kl_new_old,
                         train_gail_lambda=train_gail_lambda,
                         eps=eps,
                         device=device,
                         log_file=log_file,
                         recon_obs=recon_obs,
                         env_configs=env_configs,
                         )
        self.dir_prior = di_prior
        self.mode = mode
        self.confidence = confidence
        assert 'VICRL' in self.task
        # self._build()

    def _build(self) -> None:
        self.network = nn.Sequential(
            *create_mlp(self.input_dims, 2, self.hidden_sizes),
            nn.Softplus()
        )
        self.network.to(self.device)
        # Build optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.optimizer = None
        if self.train_gail_lambda:
            self.criterion = nn.BCELoss()

    def forward(self, x: th.tensor) -> th.tensor:
        alpha_beta = self.network(x)
        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        pred = torch.distributions.Beta(alpha, beta).rsample()
        return pred.unsqueeze(-1)

    def cost_function(self, obs: np.ndarray, acs: np.ndarray, force_mode: str = None, confidence: float = 0.5) -> np.ndarray:
        assert self.recon_obs or obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""
        if force_mode is None:
            mode = self.mode
        else:
            mode = force_mode
        x = self.prepare_data(obs, acs)
        with th.no_grad():
            if mode == 'sample':
                out = self.__call__(x)
                cost = 1 - out.detach().cpu().numpy()
                cost = cost.squeeze(axis=-1)
            elif mode == 'mean':
                a_b = self.network(x)
                a = a_b[:, 0]
                b = a_b[:, 1]
                out = a / (a + b)  # the mean of beta distribution
                out = out.unsqueeze(-1)
                cost = 1 - out.detach().cpu().numpy()
                cost = cost.squeeze(axis=-1)
            elif mode == 'VaR':
                a_b = self.network(x).detach().cpu().numpy()
                a = a_b[:, 0]
                b = a_b[:, 1]
                tmp1 = scipy.stats.beta.ppf(q=(1-0.1), a=a, b=b)
                tmp2 = scipy.stats.beta.ppf(q=(1-0.5), a=a, b=b)
                tmp3 = scipy.stats.beta.ppf(q=(1-0.9), a=a, b=b)
                var_values = scipy.stats.beta.ppf(q=(1 - self.confidence), a=a, b=b)
                cost = 1 - var_values
            elif mode == 'CVaR':
                a_b = self.network(x).detach().cpu().numpy()
                a = a_b[:, 0]
                b = a_b[:, 1]
                var_values = scipy.stats.beta.ppf(q=(1-self.confidence), a=a, b=b)
                cvar_values = []
                for i in range(a_b.shape[0]):
                    samples = scipy.stats.beta.rvs(a=a[i], b=b[i], size=[1000])
                    tmp = samples.mean()
                    cvar_value = samples[samples < var_values[i]].mean()
                    cvar_values.append(cvar_value)
                cost = 1 - np.asarray(cvar_values)

                tmp_out = self.__call__(x)
                tmp_cost = 1 - tmp_out.detach().cpu().numpy()
                tmp_cost = tmp_cost.squeeze(axis=-1)
            elif mode == 'hard':
                out = self.__call__(x)
                out = torch.round(out)
                cost = 1 - out.detach().cpu().numpy()
                cost = cost.squeeze(axis=-1)
            else:
                raise ValueError("Unknown cost mode {0}".format(mode))
        return cost

    def kl_regularizer_loss(self, batch_size, alpha, beta):
        # prior = (torch.ones((batch_size, 2), dtype=torch.float32) * self.dir_prior).to(self.device)
        prior = torch.tensor(np.asarray(batch_size * [self.dir_prior]), dtype=torch.float32).to(self.device)
        analytical_kld_loss = dirichlet_kl_divergence_loss(
            alpha=torch.stack([alpha, beta], dim=1),
            prior=prior).mean()
        return analytical_kld_loss

    def train_traj_nn(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            env_configs: Dict = None,
            current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var
        # Prepare data
        nominal_data_games = [self.prepare_data(nominal_obs[i], nominal_acs[i])
                              for i in range(len(nominal_obs))]
        expert_data_games = [self.prepare_data(self.expert_obs[i], self.expert_acs[i])
                             for i in range(len(self.expert_obs))]
        early_stop_itr = iterations
        # loss = th.tensor(np.inf)

        for itr in tqdm(range(iterations)):
            for gid in range(min(len(nominal_data_games), len(expert_data_games))):
                nominal_data = nominal_data_games[gid]
                expert_data = expert_data_games[gid]

                # Save current network predictions if using importance sampling
                if self.importance_sampling:
                    with th.no_grad():
                        start_preds = self.forward(nominal_data).detach()

                # Compute IS weights
                if self.importance_sampling:
                    with th.no_grad():
                        current_preds = self.forward(nominal_data).detach()
                    is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(),
                                                                                 current_preds.clone(),
                                                                                 episode_lengths)
                    # Break if kl is very large
                    if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                            (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                        early_stop_itr = itr
                        break
                else:
                    is_weights = th.ones(nominal_data.shape[0]).to(self.device)

                nominal_preds_all = []
                nominal_alpha_all = []
                nominal_beta_all = []
                expert_preds_all = []
                expert_alpha_all = []
                expert_beta_all = []
                is_batch_all = []
                for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                    # Get batch data
                    nominal_batch = nominal_data[nom_batch_indices]
                    expert_batch = expert_data[exp_batch_indices]
                    is_batch = is_weights[nom_batch_indices][..., None]
                    is_batch_all.append(is_batch)

                    # Make predictions
                    nominal_alpha_beta = self.network(nominal_batch)
                    # print(torch.max(nominal_alpha_beta).data)
                    nominal_alpha = nominal_alpha_beta[:, 0]
                    nominal_beta = nominal_alpha_beta[:, 1]
                    nominal_preds = torch.distributions.Beta(nominal_alpha, nominal_beta).rsample()
                    nominal_preds_all.append(nominal_preds)
                    nominal_alpha_all.append(nominal_alpha)
                    nominal_beta_all.append(nominal_beta)

                    expert_alpha_beta = self.network(expert_batch)
                    expert_alpha = expert_alpha_beta[:, 0]
                    expert_beta = expert_alpha_beta[:, 1]
                    expert_preds = torch.distributions.Beta(expert_alpha, expert_beta).rsample()
                    expert_preds_all.append(expert_preds)
                    expert_alpha_all.append(expert_alpha)
                    expert_beta_all.append(expert_beta)

                nominal_preds_all = th.concat(nominal_preds_all)
                nominal_alpha_all = th.concat(nominal_alpha_all)
                nominal_beta_all = th.concat(nominal_beta_all)
                expert_preds_all = th.concat(expert_preds_all)
                expert_alpha_all = th.concat(expert_alpha_all)
                expert_beta_all = th.concat(expert_beta_all)
                is_batch_all = th.concat(is_batch_all)

                # Calculate loss
                if self.train_gail_lambda:
                    nominal_loss = self.criterion(nominal_preds_all, th.zeros(*nominal_preds_all.size()))
                    expert_loss = self.criterion(expert_preds_all, th.ones(*expert_preds_all.size()))
                    regularizer_loss = th.tensor(0)
                    loss = nominal_loss + expert_loss
                else:
                    expert_preds = torch.clip(expert_preds_all, min=self.eps, max=1)
                    expert_loss = th.mean(th.log(expert_preds))
                    # expert_loss = th.mean(th.log(expert_preds + self.eps))
                    nominal_preds = torch.clip(nominal_preds_all, min=self.eps, max=1)
                    nominal_loss = th.mean(is_batch_all * th.log(nominal_preds))
                    # nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
                    nominal_batch_size = nominal_preds.shape[0]
                    expert_batch_size = expert_preds.shape[0]
                    regularizer_loss = self.kl_regularizer_loss(batch_size=nominal_batch_size,
                                                                alpha=nominal_alpha_all,
                                                                beta=nominal_beta_all,
                                                                ) + \
                                       self.kl_regularizer_loss(batch_size=expert_batch_size,
                                                                alpha=expert_alpha_all,
                                                                beta=expert_beta_all,
                                                                )
                    loss = (-expert_loss + nominal_loss) + self.regularizer_coeff * regularizer_loss
                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    # print(loss.data)
                    self.optimizer.step()

            bw_metrics = {"backward/cn_loss": loss.item(),
                          "backward/expert_loss": expert_loss.item(),
                          "backward/unweighted_nominal_loss": th.mean(th.log(nominal_preds_all + self.eps)).item(),
                          "backward/nominal_loss": nominal_loss.item(),
                          "backward/regularizer_loss": regularizer_loss.item(),
                          "backward/is_mean": th.mean(is_weights).detach().item(),
                          "backward/is_max": th.max(is_weights).detach().item(),
                          "backward/is_min": th.min(is_weights).detach().item(),
                          "backward/nominal_preds_max": th.max(nominal_preds_all).item(),
                          "backward/nominal_preds_min": th.min(nominal_preds_all).item(),
                          "backward/nominal_preds_mean": th.mean(nominal_preds_all).item(),
                          "backward/expert_preds_max": th.max(expert_preds_all).item(),
                          "backward/expert_preds_min": th.min(expert_preds_all).item(),
                          "backward/expert_preds_mean": th.mean(expert_preds_all).item(), }
            if self.importance_sampling:
                stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
                                "backward/kl_new_old": kl_new_old.item(),
                                "backward/early_stop_itr": early_stop_itr}
                bw_metrics.update(stop_metrics)

        return bw_metrics

    def train_nn(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
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
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(self.expert_obs, self.expert_acs)

        # Save current network predictions if using importance sampling
        if self.importance_sampling:
            with th.no_grad():
                start_preds = self.forward(nominal_data).detach()

        early_stop_itr = iterations
        loss = th.tensor(np.inf)

        loss_all = []
        expert_loss_all = []
        nominal_loss_all = []
        regularizer_loss_all = []
        is_weights_all = []
        nominal_preds_all = []
        expert_preds_all = []

        for itr in tqdm(range(iterations)):
            # Compute IS weights
            if self.importance_sampling:
                with th.no_grad():
                    current_preds = self.forward(nominal_data).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(),
                                                                             current_preds.clone(),
                                                                             episode_lengths)
                # Break if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                        (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = th.ones(nominal_data.shape[0]).to(self.device)

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                # Get batch data
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][..., None]

                # Make predictions
                nominal_alpha_beta = self.network(nominal_batch)
                # print(torch.max(nominal_alpha_beta).data)
                nominal_alpha = nominal_alpha_beta[:, 0]
                nominal_beta = nominal_alpha_beta[:, 1]
                nominal_preds = torch.distributions.Beta(nominal_alpha, nominal_beta).rsample()

                expert_alpha_beta = self.network(expert_batch)
                expert_alpha = expert_alpha_beta[:, 0]
                expert_beta = expert_alpha_beta[:, 1]
                expert_preds = torch.distributions.Beta(expert_alpha, expert_beta).rsample()

                # Calculate loss
                if self.train_gail_lambda:
                    nominal_loss = self.criterion(nominal_preds, th.zeros(*nominal_preds.size()))
                    expert_loss = self.criterion(expert_preds, th.ones(*expert_preds.size()))
                    regularizer_loss = th.tensor(0)
                    loss = nominal_loss + expert_loss
                else:
                    expert_preds = torch.clip(expert_preds, min=self.eps, max=1)
                    expert_loss = th.mean(th.log(expert_preds))
                    # expert_loss = th.mean(th.log(expert_preds + self.eps))
                    nominal_preds = torch.clip(nominal_preds, min=self.eps, max=1)
                    nominal_loss = th.mean(is_batch * th.log(nominal_preds))
                    # nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
                    nominal_batch_size = nominal_preds.shape[0]
                    expert_batch_size = expert_preds.shape[0]
                    regularizer_loss = self.kl_regularizer_loss(batch_size=nominal_batch_size,
                                                                alpha=nominal_alpha,
                                                                beta=nominal_beta,
                                                                ) + \
                                       self.kl_regularizer_loss(batch_size=expert_batch_size,
                                                                alpha=expert_alpha,
                                                                beta=expert_beta,
                                                                )
                    loss = (-expert_loss + nominal_loss) + self.regularizer_coeff * regularizer_loss

                loss_all.append(loss)
                expert_loss_all.append(expert_loss)
                nominal_loss_all.append(nominal_loss)
                regularizer_loss_all.append(regularizer_loss)
                is_weights_all.append(is_weights)
                expert_preds_all.append(expert_preds)
                nominal_preds_all.append(nominal_preds)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                # print(loss.data)
                self.optimizer.step()

        loss_all = torch.stack(loss_all, dim=0)
        expert_loss_all = torch.stack(expert_loss_all, dim=0)
        nominal_loss_all = torch.stack(nominal_loss_all, dim=0)
        regularizer_loss_all = torch.stack(regularizer_loss_all, dim=0)
        is_weights_all = torch.cat(is_weights_all, dim=0)
        nominal_preds_all = torch.cat(nominal_preds_all, dim=0)
        expert_preds_all = torch.cat(expert_preds_all, dim=0)

        bw_metrics = {"backward/cn_loss": th.mean(loss_all).item(),
                      "backward/expert_loss": th.mean(expert_loss_all).item(),
                      "backward/unweighted_nominal_loss": th.mean(th.log(nominal_preds_all + self.eps)).item(),
                      "backward/nominal_loss": th.mean(nominal_loss_all).item(),
                      "backward/regularizer_loss": th.mean(regularizer_loss_all).item(),
                      "backward/is_mean": th.mean(is_weights_all).detach().item(),
                      "backward/is_max": th.max(is_weights_all).detach().item(),
                      "backward/is_min": th.min(is_weights_all).detach().item(),
                      "backward/data_shape": list(nominal_data.shape),
                      "backward/nominal/preds_max": th.max(nominal_preds_all).item(),
                      "backward/nominal/preds_min": th.min(nominal_preds_all).item(),
                      "backward/nominal/preds_mean": th.mean(nominal_preds_all).item(),
                      "backward/expert/preds_max": th.max(expert_preds_all).item(),
                      "backward/expert/preds_min": th.min(expert_preds_all).item(),
                      "backward/expert/preds_mean": th.mean(expert_preds_all).item(), }
        if self.importance_sampling:
            stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
                            "backward/kl_new_old": kl_new_old.item(),
                            "backward/early_stop_itr": early_stop_itr}
            bw_metrics.update(stop_metrics)

        return bw_metrics
