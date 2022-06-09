import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
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
            device: str = "cpu",
            di_prior: list = [1, 1],
            mode: str = 'sample',
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
                         )
        self.dir_prior = di_prior
        self.mode = mode
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

    def cost_function(self, obs: np.ndarray, acs: np.ndarray, force_mode: str = None) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
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
            elif mode == 'mean':
                alpha_beta = self.network(x)
                alpha = alpha_beta[:, 0]
                beta = alpha_beta[:, 1]
                out = alpha / (alpha + beta)  # the mean of beta distribution
                out = out.unsqueeze(-1)
            elif mode == 'risk':
                pass
            else:
                raise ValueError("Unknown cost mode {0}".format(mode))
        cost = 1 - out.detach().cpu().numpy()
        return cost.squeeze(axis=-1)

    def kl_regularizer_loss(self, batch_size, alpha, beta):
        # prior = (torch.ones((batch_size, 2), dtype=torch.float32) * self.dir_prior).to(self.device)
        prior = torch.tensor(np.asarray(batch_size * [self.dir_prior]), dtype=torch.float32).to(self.device)
        analytical_kld_loss = dirichlet_kl_divergence_loss(
            alpha=torch.stack([alpha, beta], dim=1),
            prior=prior).mean()
        return analytical_kld_loss

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
                    expert_loss = th.mean(th.log(expert_preds + self.eps))
                    nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
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
