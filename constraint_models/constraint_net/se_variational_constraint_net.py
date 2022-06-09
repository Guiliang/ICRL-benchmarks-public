import sys
from itertools import accumulate
from typing import Tuple, Callable, Optional, Type, Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from constraint_models.constraint_net.variational_constraint_net import VariationalConstraintNet
from constraint_models.ss_net.aggregators import SumAggregator
from constraint_models.ss_net.conceptizers import IdentityConceptizer
from constraint_models.ss_net.neural_decision_tree import NeuralDecisionTree
from constraint_models.ss_net.parameterizers import LinearParameterizer
from stable_baselines3.common.torch_layers import create_mlp, ResBlock
from stable_baselines3.common.utils import update_learning_rate
from utils.model_utils import handle_model_parameters, dirichlet_kl_divergence_loss, stability_loss


class SelfExplainableVariationalConstraintNet(VariationalConstraintNet):
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
            regularizer_coeff: float = 0.,
            task: str = 'SSICRL',
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
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
            discount_factor: float = 1,
            log_std_init: float = 0.0,
            max_seq_len: int = 300,
            explain_model_name: str = 'senn',
            num_cut: list = [],
            temperature: float = 0.1,
            device: str = "cpu",
            di_prior: list = (1, 1),
            log_file=None
    ):
        assert 'SEVICRL' in task
        self.temperature = temperature
        self.num_cut = num_cut
        self.explain_model_name = explain_model_name
        self.log_std_init = log_std_init
        self.max_seq_len = max_seq_len
        self.discount_factor = discount_factor
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
                         di_prior=di_prior,
                         eps=eps,
                         device=device,
                         log_file=log_file)

    def _build(self) -> None:
        if self.explain_model_name == "senn":
            self.sample_conceptizer = IdentityConceptizer().to(self.device)
            self.sample_parameterizer = LinearParameterizer(num_concepts=self.input_dims,
                                                            num_classes=1,
                                                            hidden_sizes=[self.input_dims] +
                                                                         self.hidden_sizes +
                                                                         [self.input_dims]
                                                            ).to(self.device)
        elif self.explain_model_name == 'ndt':
            # the ndt cannot scale well to the large dimensions, we apply it for modelling actions
            self.ndt = NeuralDecisionTree(num_class=2,
                                          num_cut=[self.num_cut]*len(self.input_acs_dim),
                                          temperature=self.temperature,
                                          device=self.device
                                          )
            self.sample_conceptizers = []
            self.sample_parameterizers = []
            for i in range(self.ndt.num_leaf):
                # since the actions are handled by ndt, se will handle obs.
                self.sample_conceptizers.append(IdentityConceptizer().to(self.device))
                self.sample_parameterizers.append(
                    LinearParameterizer(num_concepts=len(self.input_obs_dim),
                                        num_classes=1,
                                        hidden_sizes=[len(self.input_obs_dim)] +
                                                     self.hidden_sizes +
                                                     [len(self.input_obs_dim)]
                                        ).to(self.device))
            self.sample_conceptizers = nn.ModuleList(self.sample_conceptizers)
            self.sample_parameterizers = nn.ModuleList(self.sample_parameterizers)

        self.sample_aggregator = SumAggregator(num_classes=1).to(self.device)
        self.encoder = nn.Sequential(
            *create_mlp(self.input_dims, 2, self.hidden_sizes)
        ).to(self.device)

        # build different optimizers for different models
        self.optimizers = {'ICRL': None, 'policy': None}
        param_active_key_words = {'ICRL': ['sample', 'ndt', 'encoder'],
                                  'policy': [''], }
        for key in self.optimizers.keys():
            param_frozen_list, param_active_list = \
                handle_model_parameters(model=self,
                                        active_keywords=param_active_key_words[key],
                                        model_name=key,
                                        log_file=self.log_file,
                                        set_require_grad=False)
            if self.optimizer_class is not None:
                optimizer = self.optimizer_class([{'params': param_frozen_list, 'lr': 0.0},
                                                  {'params': param_active_list,
                                                   'lr': self.lr_schedule(1)}],
                                                 lr=self.lr_schedule(1))
            else:
                optimizer = None
            self.optimizers.update({key: optimizer})

    def forward(self, x: torch.tensor) -> torch.tensor:
        preds, stab_loss = self.self_explainable_model(batch_input=x)
        return preds

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        for optimizer_name in self.optimizers.keys():
            update_learning_rate(self.optimizers[optimizer_name], self.lr_schedule(current_progress_remaining))

    # def predict(self, obs: torch.tensor, deterministic=False):
    #     mean_action = self.policy_network(obs)
    #     std = torch.exp(self.log_std)
    #     eps = torch.randn_like(std)
    #     if deterministic:
    #         action = mean_action
    #     else:
    #         action = eps * std + mean_action
    #
    #     output_action = []
    #     for i in range(self.acs_dim):
    #         output_action.append(torch.clamp(action[:, i], min=self.action_low[i], max=self.action_high[i]))
    #
    #     return torch.stack(output_action, dim=1)

    def train_traj_nn(
            self,
            iterations: int,
            nominal_obs: list,
            nominal_acs: list,
            episode_lengths: int,
            # nominal_traj_rs: list,
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

        assert iterations > 0
        for itr in tqdm(range(iterations)):
            # TODO: maybe we do need the importance sampling
            # is_weights = torch.ones(expert_input.shape[0])

            loss_all_step = []
            expert_stab_loss_all_step = []
            expert_kld_loss_all_step = []
            expert_recon_loss_all_step = []
            expert_preds_all_step = []
            nominal_stab_loss_all_step = []
            nominal_kld_loss_all_step = []
            nominal_recon_loss_all_step = []
            nominal_preds_all_step = []

            # Do a complete pass on data, we don't have to use the get(), but let's leave it here for future usage
            for batch_indices in self.get(len(self.expert_obs), len(nominal_obs)):

                # Get batch data
                batch_expert_traj_obs = [self.expert_obs[i] for i in batch_indices[0]]
                batch_expert_traj_acs = [self.expert_acs[i] for i in batch_indices[0]]
                # batch_expert_traj_rs = [self.expert_rs[i] for i in batch_indices[0]]
                batch_nominal_traj_obs = [nominal_obs[i] for i in batch_indices[1]]
                batch_nominal_traj_acs = [nominal_acs[i] for i in batch_indices[1]]
                # batch_nominal_traj_rs = [nominal_traj_rs[i] for i in batch_indices[1]]
                batch_max_seq_len = max([len(item) for item in batch_expert_traj_obs] +
                                        [len(item) for item in batch_nominal_traj_obs])
                batch_seq_len = min(batch_max_seq_len, self.max_seq_len)

                batch_expert_traj_obs, batch_expert_traj_acs = self.prepare_traj_data(
                    obs=batch_expert_traj_obs,
                    acs=batch_expert_traj_acs,
                    max_seq_len=batch_seq_len
                )

                batch_nominal_traj_obs, batch_nominal_traj_acs = self.prepare_traj_data(
                    obs=batch_nominal_traj_obs,
                    acs=batch_nominal_traj_acs,
                    max_seq_len=batch_seq_len
                )

                batch_size = batch_expert_traj_obs.shape[0]
                traj_loss = []
                for i in range(batch_seq_len):
                    batch_expert_input = torch.cat([batch_expert_traj_obs[:, i, :],
                                                    batch_expert_traj_acs[:, i, :]], dim=-1)
                    expert_stab_loss, expert_kld_loss, expert_recon_loss, expert_constraint_loss, expert_preds_t = \
                        self.compute_losses(batch_input=batch_expert_input,
                                            expert_loss=True)
                    expert_stab_loss_all_step.append(expert_stab_loss)
                    expert_kld_loss_all_step.append(expert_kld_loss)
                    expert_recon_loss_all_step.append(expert_recon_loss)
                    expert_preds_all_step.append(expert_preds_t)
                    batch_nominal_input = torch.cat([batch_nominal_traj_obs[:, i, :],
                                                     batch_nominal_traj_acs[:, i, :]], dim=-1)
                    nominal_stab_loss, nominal_kld_loss, nominal_recon_loss, nominal_constraint_loss, nominal_preds_t = \
                        self.compute_losses(batch_input=batch_nominal_input,
                                            expert_loss=False)
                    nominal_stab_loss_all_step.append(nominal_stab_loss)
                    nominal_kld_loss_all_step.append(nominal_kld_loss)
                    nominal_recon_loss_all_step.append(nominal_recon_loss)
                    nominal_preds_all_step.append(nominal_preds_t)
                    loss_t = (expert_recon_loss + nominal_recon_loss) + \
                             self.regularizer_coeff * (expert_kld_loss + nominal_kld_loss) + \
                             self.regularizer_coeff * (expert_stab_loss + nominal_stab_loss) + \
                             self.regularizer_coeff * (expert_constraint_loss + nominal_constraint_loss)
                    loss_all_step.append(loss_t)
                    traj_loss.append(loss_t)

                self.optimizers['ICRL'].zero_grad()
                ave_traj_loss = torch.mean(torch.stack(traj_loss))
                ave_traj_loss.backward()
                self.optimizers['ICRL'].step()
            ave_loss_all_step = torch.mean(torch.stack(loss_all_step))
            ave_expert_stab_loss_all_step = torch.mean(torch.stack(expert_stab_loss_all_step))
            ave_expert_kld_loss_all_step = torch.mean(torch.stack(expert_kld_loss_all_step))
            ave_expert_recon_loss_all_step = torch.mean(torch.stack(expert_recon_loss_all_step))
            expert_preds_all_step = torch.cat(expert_preds_all_step, dim=0)
            ave_nominal_stab_loss_all_step = torch.mean(torch.stack(nominal_stab_loss_all_step))
            ave_nominal_kld_loss_all_step = torch.mean(torch.stack(nominal_kld_loss_all_step))
            ave_nominal_recon_loss_all_step = torch.mean(torch.stack(nominal_recon_loss_all_step))
            nominal_preds_all_step = torch.cat(nominal_preds_all_step, dim=0)

        bw_metrics = {"backward/loss": ave_loss_all_step.item(),
                      "backward/expert/stab_loss": ave_expert_stab_loss_all_step.item(),
                      "backward/expert/kld_loss": ave_expert_kld_loss_all_step.item(),
                      "backward/expert/recon_loss": ave_expert_recon_loss_all_step.item(),
                      "backward/nominal/stab_loss": ave_nominal_stab_loss_all_step.item(),
                      "backward/nominal/kld_loss": ave_nominal_kld_loss_all_step.item(),
                      "backward/nominal/recon_loss": ave_nominal_recon_loss_all_step.item(),
                      # "backward/is_mean": torch.mean(is_weights).detach().item(),
                      # "backward/is_max": torch.max(is_weights).detach().item(),
                      # "backward/is_min": torch.min(is_weights).detach().item(),
                      "backward/nominal/preds_max": torch.max(nominal_preds_all_step).item(),
                      "backward/nominal/preds_min": torch.min(nominal_preds_all_step).item(),
                      "backward/nominal/preds_mean": torch.mean(nominal_preds_all_step).item(),
                      "backward/expert/preds_max": torch.max(expert_preds_all_step).item(),
                      "backward/expert/preds_min": torch.min(expert_preds_all_step).item(),
                      "backward/expert/preds_mean": torch.mean(expert_preds_all_step).item(),
                      }
        # if self.importance_sampling:
        #     stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
        #                     "backward/kl_new_old": kl_new_old.item(),
        #                     "backward/early_stop_itr": early_stop_itr}
        #     bw_metrics.update(stop_metrics)

        return bw_metrics

    def train_nn(
            self,
            iterations: int,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            # nominal_traj_rs: list,
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
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(self.expert_obs, self.expert_acs)

        # Save current network predictions if using importance sampling
        if self.importance_sampling:
            with torch.no_grad():
                start_preds = self.forward(nominal_data).detach()

        early_stop_itr = iterations
        loss = torch.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            # Compute IS weights
            if self.importance_sampling:
                with torch.no_grad():
                    current_preds = self.forward(nominal_data).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(), current_preds.clone(),
                                                                             episode_lengths)
                # Break if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                        (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = torch.ones(nominal_data.shape[0]).to(self.device)

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                is_batch = is_weights[nom_batch_indices][..., None]
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                expert_stab_loss, expert_kld_loss, expert_recon_loss, expert_constraint_loss, expert_preds = \
                    self.compute_losses(batch_input=expert_batch,
                                        expert_loss=True)
                nominal_stab_loss, nominal_kld_loss, nominal_recon_loss, nominal_constraint_loss, nominal_preds = \
                    self.compute_losses(batch_input=nominal_batch,
                                        expert_loss=False)
                expert_loss = expert_recon_loss.mean() + self.regularizer_coeff * (
                        expert_kld_loss + expert_stab_loss + expert_constraint_loss)
                nominal_loss = (is_batch * nominal_recon_loss).mean() + self.regularizer_coeff * (
                        nominal_kld_loss + nominal_stab_loss + nominal_constraint_loss)
                loss = torch.mean(expert_loss) + torch.mean(nominal_loss)

                self.optimizers['ICRL'].zero_grad()
                loss.backward()
                self.optimizers['ICRL'].step()

        bw_metrics = {"backward/loss": loss.item(),
                      "backward/expert/stab_loss": expert_stab_loss.item(),
                      "backward/expert/kld_loss": expert_kld_loss.item(),
                      "backward/expert/recon_loss": expert_recon_loss.mean().item(),
                      "backward/nominal/stab_loss": nominal_stab_loss.item(),
                      "backward/nominal/kld_loss": nominal_kld_loss.item(),
                      "backward/nominal/recon_loss": nominal_recon_loss.mean().item(),
                      # "backward/is_mean": torch.mean(is_weights).detach().item(),
                      # "backward/is_max": torch.max(is_weights).detach().item(),
                      # "backward/is_min": torch.min(is_weights).detach().item(),
                      "backward/nominal/preds_max": torch.max(nominal_preds).item(),
                      "backward/nominal/preds_min": torch.min(nominal_preds).item(),
                      "backward/nominal/preds_mean": torch.mean(nominal_preds).item(),
                      "backward/expert/preds_max": torch.max(expert_preds).item(),
                      "backward/expert/preds_min": torch.min(expert_preds).item(),
                      "backward/expert/preds_mean": torch.mean(expert_preds).item(),
                      }
        # if self.importance_sampling:
        #     stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
        #                     "backward/kl_new_old": kl_new_old.item(),
        #                     "backward/early_stop_itr": early_stop_itr}
        #     bw_metrics.update(stop_metrics)

        return bw_metrics

    def self_explainable_model(self, batch_input, add_loss=False):
        if self.explain_model_name == "senn":
            batch_input.requires_grad_()  # track all operations on x for jacobian calculation
            sample_concepts, _ = self.sample_conceptizer(batch_input)
            sample_relevance = self.sample_parameterizer(batch_input)
            # Both the alpha and the beta parameters should be greater than 0,
            preds = F.softplus(self.sample_aggregator(sample_concepts, sample_relevance))
            if add_loss:
                stab_loss = stability_loss(input_data=batch_input,
                                           aggregates=preds,
                                           concepts=sample_concepts,
                                           relevances=sample_relevance)
                return preds, stab_loss
            else:
                return preds, None
        elif self.explain_model_name == 'ndt':
            batch_obs_input = batch_input[:, :len(self.input_obs_dim)]
            batch_act_input = batch_input[:, len(self.input_obs_dim):]
            dt_preds, leaves_probs = self.ndt(x=batch_act_input)
            leaves_preds = []
            leaves_stab_loss = []
            for i in range(self.ndt.num_leaf):
                batch_obs_input.requires_grad_()  # track all operations on x for jacobian calculation
                sample_concepts, _ = self.sample_conceptizers[i](batch_obs_input)
                sample_relevance = self.sample_parameterizers[i](batch_obs_input)
                # Both the alpha and the beta parameters should be greater than 0,
                leaf_preds_leaf = F.sigmoid(self.sample_aggregator(sample_concepts, sample_relevance))
                leaves_preds.append(leaf_preds_leaf)
                if add_loss:
                    leaf_stab_loss = stability_loss(input_data=batch_obs_input,
                                                    aggregates=leaf_preds_leaf,
                                                    concepts=sample_concepts,
                                                    relevances=sample_relevance)
                    leaves_stab_loss.append(leaf_stab_loss)
            # leaves_probs = torch.reshape(leaves_probs, shape=[-1])
            leaves_probs = leaves_probs
            leaves_preds = torch.stack(leaves_preds, dim=1)
            preds = torch.bmm(leaves_probs.unsqueeze(1), leaves_preds).squeeze(1)
            if add_loss:
                leaves_stab_loss = torch.stack(leaves_stab_loss, dim=1)
                stab_loss = torch.bmm(leaves_probs.unsqueeze(1), leaves_stab_loss).squeeze(1)
                return preds, stab_loss
            else:
                return preds, None

    def compute_losses(self, batch_input, expert_loss):
        preds, stab_loss = self.self_explainable_model(batch_input=batch_input, add_loss=True)
        alpha_beta = self.encoder(batch_input)
        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        constraint_loss = - torch.distributions.Beta(alpha, beta).log_prob(preds).mean()
        batch_size = preds.shape[0]
        analytical_kld_loss = self.kl_regularizer_loss(batch_size, alpha=alpha, beta=beta)
        if expert_loss:
            recon_loss = -torch.log(preds + self.eps)
        else:
            recon_loss = torch.log(preds + self.eps)
        return stab_loss.mean(), analytical_kld_loss, recon_loss, constraint_loss, preds

    def padding_input(self,
                      input_data: list,
                      length: int,
                      padding_symbol: int) -> np.ndarray:
        input_data_padding = []
        for i in range(len(input_data)):
            padding_length = length - input_data[i].shape[0]
            if padding_length > 0:
                if len(input_data[i].shape) == 2:
                    padding_data = np.ones([padding_length, input_data[i].shape[1]]) * padding_symbol
                elif len(input_data[i].shape) == 1:
                    padding_data = np.ones([padding_length]) * padding_symbol
                input_sample = np.concatenate([input_data[i], padding_data], axis=0)
            else:
                if len(input_data[i].shape) == 2:
                    input_sample = input_data[i][-length:, :]
                elif len(input_data[i].shape) == 1:
                    input_sample = input_data[i][-length:]
            input_data_padding.append(input_sample)
        return np.asarray(input_data_padding)

    def prepare_traj_data(
            self,
            obs: list,
            acs: list,
            rs: list = None,
            max_seq_len: int = None,
            select_dim: bool = True,
    ) -> torch.tensor:
        bs = len(obs)
        max_seq_len = max_seq_len if max_seq_len is not None else self.max_seq_len
        obs = [self.normalize_traj_obs(obs[i], self.current_obs_mean, self.current_obs_var, self.clip_obs)
               for i in range(bs)]
        acs = [self.clip_actions(acs[i], self.action_low, self.action_high) for i in range(bs)]
        obs = self.padding_input(input_data=obs, length=max_seq_len, padding_symbol=0)
        acs = self.padding_input(input_data=acs, length=max_seq_len, padding_symbol=0)
        if rs is not None:
            rs = self.padding_input(input_data=rs, length=max_seq_len, padding_symbol=0)
        acs = self.reshape_actions(acs)
        if select_dim:
            obs = self.select_appropriate_dims(select_dim=self.input_obs_dim, x=obs)
            acs = self.select_appropriate_dims(select_dim=self.input_acs_dim, x=acs)
        if rs is None:
            return torch.tensor(obs, dtype=torch.float32).to(self.device), \
                   torch.tensor(acs, dtype=torch.float32).to(self.device)
        else:
            return torch.tensor(obs, dtype=torch.float32).to(self.device), \
                   torch.tensor(acs, dtype=torch.float32).to(self.device), \
                   torch.tensor(rs, dtype=torch.float32).to(self.device)

    def normalize_traj_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
                           clip_obs: Optional[float] = None) -> np.ndarray:
        bs = obs.shape[0]
        obs = np.reshape(obs, newshape=[-1, obs.shape[-1]])
        # tmp = np.reshape(obs, newshape=[bs, -1, obs.shape[-1]])
        if mean is not None and var is not None:
            mean, var = mean[None], var[None]
            obs = (obs - mean) / np.sqrt(var + self.eps)
        if clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        obs = np.reshape(obs, newshape=[bs, -1, obs.shape[-1]])
        return obs.squeeze(1)

    def save(self, save_path):
        state_dict = dict(
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

        for key in self.optimizers.keys():
            state_dict.update({'cn_optimizer_' + key: self.optimizers[key].state_dict()})
        tmp = self.state_dict()
        if self.explain_model_name == 'ndt':
            state_dict.update({'ndt_model': self.ndt.state_dict()})
            state_dict.update({'conceptizers_model': self.sample_conceptizers.state_dict()})
            state_dict.update({'parameterizers_model': self.sample_parameterizers.state_dict()})
            state_dict.update({'aggregator_model': self.sample_aggregator.state_dict()})
        elif self.explain_model_name == 'senn':
            state_dict.update({'conceptizer_model': self.sample_conceptizer.state_dict()})
            state_dict.update({'parameterizer_model': self.sample_parameterizer.state_dict()})
            state_dict.update({'aggregator_model': self.sample_aggregator.state_dict()})
        torch.save(state_dict, save_path)
