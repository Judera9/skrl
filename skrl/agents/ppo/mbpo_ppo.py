# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Any, Mapping, Optional, Tuple, Union, Dict

import collections
import copy
from packaging import version
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensordict import TensorDict
import warnings
import gymnasium

from skrl.agents.ppo import POMDP_PPO
from skrl.models.rwm_world_model import SystemDynamicsEnsemble
from skrl.utils.rsl_rl_utils import string_to_callable
from skrl.models import Model
from skrl.memories import Memory, RandomMemory, ReplayBuffer
from skrl.resources.preprocessors import EmpiricalNormalization
from skrl.agents import Agent

from skrl.utils.log_utils.plotter import Plotter
import matplotlib.pyplot as plt

from skrl import config, logger
from skrl.resources.schedulers import KLAdaptiveLR

# fmt: off
# [start-config-dict-torch]
MBPO_PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 5,           # number of learning epochs during each update
    "mini_batches": 4,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "entropy_scheduler": None,        # entropy scheduler class (see skrl.resources.schedulers)
    "entropy_scheduler_kwargs": {},   # entropy scheduler's kwargs (e.g. {"step_size": 1e-3})

    "use_encoder": False,                # encoder class (see skrl.models)
    "encoder_kwargs": {},           # encoder's kwargs (e.g. {"name": "encoder"})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "actor_observation_preprocessor": None, # actor observation preprocessor class (see skrl.resources.preprocessors)
    "actor_observation_preprocessor_kwargs": {}, # actor observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,        # entropy loss scale (default: 0.0)
    "grad_penalty_weight": 0.0,        # gradient penalty weight for Lipschitz continuity (default: 0.0)
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "world_model": None,
    "world_model_kwargs": {},

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)

    # "normalize_advantage_per_mini_batch": True, TODO: default operation in skrl, why not?
    }
}
# [end-config-dict-torch]
# fmt: on


class MBPO_PPO(POMDP_PPO):
    """
    Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).
    Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics (http://arxiv.org/abs/2501.10100)
    """

    world_model: SystemDynamicsEnsemble
    """World model"""

    imagination_storage: Memory
    """Imagination storage"""

    memory: Memory
    """PPO memory"""

    system_dynamics_reply_buffer: ReplayBuffer
    """System dynamics reply buffer"""

    _rwm_state_normalizer: EmpiricalNormalization
    """Normalizer for system dynamics"""

    _rwm_action_normalizer: EmpiricalNormalization
    """Normalizer for system dynamics"""

    def __init__(
        self,
        models: Mapping[str, Model],  # TODO: system_dynamics model in here
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        actor_observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,  # TODO: world_model_kwargs, *.preprocessors
        system_dynamics_reply_buffer: Optional[Memory] = None,
        imagination_storage: Optional[Memory] = None,
        multi_gpu_cfg: dict | None = None,  # Distributed training parameters
    ):
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            actor_observation_space=actor_observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

        self.world_model = self.models.get("rwm_world_model", None)
        self.checkpoint_modules["world_model"] = self.world_model
        self.world_model_kwargs = cfg.get("world_model_kwargs", {})

        self.train_world_model = self.world_model_kwargs.get("train_world_model", False)
        self.train_policy = self.world_model_kwargs.get("train_policy", False)
        self.use_imagination = self.world_model_kwargs.get("use_imagination", False)

        # set up optimizer for world model
        if self.world_model is not None:
            self.world_model_optimizer = torch.optim.Adam(
                self.world_model.parameters(),
                lr=float(self.world_model_kwargs["learning_rate"]),
                weight_decay=float(self.world_model_kwargs["weight_decay"]),
            )
            self.world_model_scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
            self.checkpoint_modules["world_model_optimizer"] = self.world_model_optimizer
        
        if not self.train_policy:
            self.policy.eval()
            for param in self.policy.parameters():
                param.requires_grad = False
            for param in self.value.parameters():
                param.requires_grad = False
            self.memory = None

        # set up normalizer for system dynamics
        self._rwm_state_normalizer = self.world_model_kwargs.get('rwm_state_normalizer', None)
        self._rwm_action_normalizer = self.world_model_kwargs.get('rwm_action_normalizer', None)

        if self._rwm_state_normalizer:
            _cfg_mean = self.world_model_kwargs.get("rwm_state_normalizer_kwargs", {}).get("mean", None)
            _cfg_std = self.world_model_kwargs.get("rwm_state_normalizer_kwargs", {}).get("std", None)
            del self.world_model_kwargs["rwm_state_normalizer_kwargs"]["mean"]
            del self.world_model_kwargs["rwm_state_normalizer_kwargs"]["std"]
            self._rwm_state_normalizer =  self._rwm_state_normalizer(**self.world_model_kwargs.get("rwm_state_normalizer_kwargs", {})).to(self.device).eval()
            state_normalizer_state_dict = {
                "_mean": torch.tensor(_cfg_mean, device=self.device).unsqueeze(0),
                "_std": torch.tensor(_cfg_std, device=self.device).unsqueeze(0),
                "_var": torch.square(torch.tensor(_cfg_std, device=self.device).unsqueeze(0)),
                "count": torch.tensor(0, dtype=torch.long),
            }
            self._rwm_state_normalizer.load_state_dict(state_normalizer_state_dict)
            self.checkpoint_modules["rwm_state_normalizer"] = self._rwm_state_normalizer
        if self._rwm_action_normalizer:
            _cfg_mean = self.world_model_kwargs.get("rwm_action_normalizer_kwargs", {}).get("mean", None)
            _cfg_std = self.world_model_kwargs.get("rwm_action_normalizer_kwargs", {}).get("std", None)
            del self.world_model_kwargs["rwm_action_normalizer_kwargs"]["mean"]
            del self.world_model_kwargs["rwm_action_normalizer_kwargs"]["std"]
            self._rwm_action_normalizer =  self._rwm_action_normalizer(**self.world_model_kwargs.get("rwm_action_normalizer_kwargs", {})).to(self.device).eval()
            action_normalizer_state_dict = {
                "_mean": torch.tensor(_cfg_mean, device=self.device).unsqueeze(0),
                "_std": torch.tensor(_cfg_std, device=self.device).unsqueeze(0),
                "_var": torch.square(torch.tensor(_cfg_std, device=self.device).unsqueeze(0)),
                "count": torch.tensor(0, dtype=torch.long),
            }
            self._rwm_action_normalizer.load_state_dict(action_normalizer_state_dict)
            self.checkpoint_modules["rwm_action_normalizer"] = self._rwm_action_normalizer

        self.num_imagination_envs = self.world_model_kwargs.get("num_imagination_envs", 0)
        self.num_imagination_steps = self.world_model_kwargs.get("num_imagination_steps", 0)
        self.command_resample_interval = self.world_model_kwargs.get("command_resample_interval", -1)
        self.uncertainty_penalty_weight = self.world_model_kwargs.get("uncertainty_penalty_weight", -0.0)

        self.sd_forecast_horizon = self.world_model_kwargs.get("forecast_horizon", 0)
        self.sd_num_mini_batches = self.world_model_kwargs.get("num_mini_batches", 0)
        self.sd_mini_batch_size = self.world_model_kwargs.get("mini_batch_size", 0)
        self.sd_loss_weights = self.world_model_kwargs.get("loss_weights", {})

        self.eval_trajectories_num = self.world_model_kwargs.get("eval_trajectories_num", 0)
        self.eval_trajectory_len = self.world_model_kwargs.get("eval_trajectory_len", 0)
        self.eval_trajectory_noise_scale = self.world_model_kwargs.get("eval_trajectory_noise_scale", 0.0)
        
        self.system_dynamics_reply_buffer = system_dynamics_reply_buffer
        self.imagination_storage = imagination_storage

        # set up system dynamics plotter
        self.plotter = Plotter()
        self.fig0, self.ax0 = plt.subplots(1, 1)
        self.plt_cfg = dict()
        self.plt_cfg["system_dynamics_state_idx_dict"] = {
            "$v$\n$[m/s]$": [0, 1, 2],
            "$\omega$\n$[rad/s]$": [3, 4, 5],
            "$g$\n$[1]$": [6, 7, 8],
            "$q$\n$[rad]$": [9, 10, 11],  # 9 - 32
            "$\dot{q}$\n$[rad/s]$": [32, 33, 34],
            # "$\\tau$\n$[Nm]$": [35, 36, 37],
        }
        self.plt_cfg["system_dynamics_num_visualizations"] = 4
        self.fig1, self.ax1 = plt.subplots(
            len(self.plt_cfg["system_dynamics_state_idx_dict"]) + 4, 
            self.plt_cfg["system_dynamics_num_visualizations"], 
            figsize=(10 * self.plt_cfg["system_dynamics_num_visualizations"], 10)
        )


    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        if self.imagination_storage is not None:
            self.imagination_storage.create_tensor(name="imagine_states", size=self.observation_space.shape, dtype=torch.float32)
            self.imagination_storage.create_tensor(name="imagine_actor_observations", size=self.actor_observation_space.shape, dtype=torch.float32)
            
            self._imagination_tensors_names = [
                "imagine_states",
                "imagine_actor_observations",
            ]

    def record_transition(
        self,
        states: torch.Tensor,
        actor_observations: torch.Tensor,
        other_observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_actor_observations: torch.Tensor,
        next_other_observations: Dict[str, torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        if self.train_world_model:
            self.record_world_model_transition(other_observations)
        if self.train_policy:
            super().record_transition(
                states=states,
                actor_observations=actor_observations,
                other_observations=other_observations,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                next_actor_observations=next_actor_observations,
                next_other_observations=next_other_observations,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            if self.train_world_model:
                self.update_world_model()
            if self.train_policy:
                self._update(timestep, timesteps)
            self.set_mode("eval")
        
        # write to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % (50 * self.write_interval):
            self.write_tracking_graph(timestep, timesteps)

        # write tracking data and checkpoints
        Agent.post_interaction(self, timestep, timesteps)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        if self._use_encoder:
            self.encoder.set_mode("eval")
            states = self._actor_observation_preprocessor(states)
            raw_encoder_output, _, _ = self.encoder.compute({"states": states}, role="policy")
            encoder_output = self._encoder_preprocessor(raw_encoder_output, train=False)
            states = torch.cat([states, encoder_output], dim=-1)
        else:
            states = self._actor_observation_preprocessor(states)

        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        log_prob = None
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            if self.train_world_model and not self.train_policy:
                actions, _, outputs = self.policy.compute({"states": states}, role="policy")
            else:
                actions, log_prob, outputs = self.policy.act({"states": states}, role="policy")
                self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_imagination_transition(
        self,
        states: torch.Tensor,
        actor_observations: torch.Tensor,
        other_observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_actor_observations: torch.Tensor,
        next_other_observations: Dict[str, torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        pass

    def record_world_model_transition(self, observation_dict: Dict[str, torch.Tensor]):
        sd_state = observation_dict.get("system_state")
        sd_action = observation_dict.get("system_action")
        sd_contact = observation_dict.get("system_contact")
        sd_termination = observation_dict.get("system_termination")
        sd_extension = observation_dict.get("system_extension")
        sd_state_norm = self._rwm_state_normalizer(sd_state)
        sd_action_norm = self._rwm_action_normalizer(sd_action)

        self.system_dynamics_reply_buffer.insert(
            [
                sd_state_norm.unsqueeze(1),
                sd_action_norm.unsqueeze(1),
                sd_extension.unsqueeze(1) if sd_extension is not None else None,
                sd_contact.unsqueeze(1) if sd_contact is not None else None,
                sd_termination.unsqueeze(1) if sd_termination is not None else None,
                ]
            )


    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        if self.use_imagination:
            values = self.imagination_storage.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=self.imagination_storage.get_tensor_by_name("rewards"),
                dones=self.imagination_storage.get_tensor_by_name("terminated") | self.imagination_storage.get_tensor_by_name("truncated"),
                values=values,
                next_values=last_values,
                discount_factor=self._discount_factor,
                lambda_coefficient=self._lambda,
            )

            self.imagination_storage.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
            self.imagination_storage.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
            self.imagination_storage.set_tensor_by_name("advantages", advantages)
        else:
            values = self.memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=self.memory.get_tensor_by_name("rewards"),
                dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
                values=values,
                next_values=last_values,
                discount_factor=self._discount_factor,
                lambda_coefficient=self._lambda,
            )

            self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
            self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
            self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        if self.use_imagination:
            sampled_batches = self.sample_combined(self.memory, self.imagination_storage)
        else:
            sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_grad_penalty = 0
        cumulative_encoder_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for batch in sampled_batches:
                if self._use_encoder:
                    sampled_states, sampled_actor_observations, sampled_actions, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages, sampled_vel_est = batch
                else:
                    sampled_states, sampled_actor_observations, sampled_actions, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages = batch

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                    sampled_actor_observations = self._actor_observation_preprocessor(sampled_actor_observations, train=not epoch)

                    # Enable gradient computation for gradient penalty
                    if self._grad_penalty_weight > 0:
                        sampled_actor_observations.requires_grad_(True)

                    if self._use_encoder:
                        self.encoder.set_mode("train")
                        raw_encoder_output, _, _ = self.encoder.compute({"states": sampled_actor_observations}, role="policy")
                        encoder_output = self._encoder_preprocessor(raw_encoder_output, train=not epoch)
                        sampled_actor_observations = torch.cat([sampled_actor_observations, encoder_output.detach()], dim=-1)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_actor_observations, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute gradient penalty for Lipschitz continuity
                    if self._grad_penalty_weight > 0:
                        gradient_penalty_loss = self._calc_grad_penalty(sampled_actor_observations, next_log_prob)
                        gradient_penalty_loss = self._grad_penalty_weight * gradient_penalty_loss
                    else:
                        gradient_penalty_loss = 0

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.entropy.mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    if self._use_encoder:
                        encoder_loss = self._update_encoder(raw_encoder_output, sampled_vel_est)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + gradient_penalty_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._grad_penalty_weight > 0:
                    cumulative_grad_penalty += gradient_penalty_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                if self._use_encoder:
                    cumulative_encoder_loss += encoder_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.learning_rate_scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.learning_rate_scheduler.step(kl.item())
                else:
                    self.learning_rate_scheduler.step()

        # update entropy scheduler
        if self._entropy_scheduler:
            self.entropy_scheduler.step()
            self._entropy_loss_scale = self.entropy_scheduler.get_value()

        # record data
        self.track_data("Loss/surrogate_loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss/value_loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss/kl_divergence", torch.tensor(kl_divergences, device=self.device).mean().item())
        self.track_data("Loss/entropy", self.policy.entropy.mean().item())
        if self._grad_penalty_weight > 0:
            self.track_data("Loss/grad_penalty", cumulative_grad_penalty / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss/entropy_loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
            if self._entropy_scheduler:
                self.track_data("Loss/entropy_loss_scale", self._entropy_loss_scale)
        self.track_data("Policy/mean_noise_std", self.policy.action_std.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Loss/learning_rate", self.learning_rate_scheduler.get_last_lr()[0])
        if self.policy._noise_generator and type(self.policy.noise_generator).__name__ == "PinkNoiseDist":
            self.track_data("Loss/noise_smoothing", self.policy.noise_generator.smoothing)
        if self._use_encoder:
            self.track_data(f"Loss/{self._encoder_kwargs['model_name']}_loss", cumulative_encoder_loss / (self._learning_epochs * self._mini_batches))

    def update_world_model(self):
        mean_sd_state_loss = 0
        mean_sd_sequence_loss = 0
        mean_sd_bound_loss = 0
        mean_sd_kl_loss = 0
        mean_sd_extension_loss = 0
        mean_sd_contact_loss = 0
        mean_sd_termination_loss = 0
        sd_generator = self.system_dynamics_reply_buffer.mini_batch_generator(
            self.world_model.history_horizon + self.sd_forecast_horizon,
            self.sd_num_mini_batches,
            self.sd_mini_batch_size,
        )
        
        for system_state_batch, system_action_batch, system_extension_batch, system_contact_batch, system_termination_batch in sd_generator:
            self.world_model.reset()
            state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss = self.world_model.compute_loss(
                system_state_batch,
                system_action_batch,
                system_extension_batch,
                system_contact_batch,
                system_termination_batch,
                bootstrap=True
            )
            loss = (
                self.sd_loss_weights["state"] * state_loss
                + self.sd_loss_weights["sequence"] * sequence_loss
                + self.sd_loss_weights["bound"] * bound_loss
                + self.sd_loss_weights["kl"] * kl_loss
                + self.sd_loss_weights["extension"] * extension_loss
                + self.sd_loss_weights["contact"] * contact_loss
                + self.sd_loss_weights["termination"] * termination_loss
            )

            # update loss
            self.world_model_optimizer.zero_grad()
            self.world_model_scaler.scale(loss).backward()
            if self._grad_norm_clip > 0:
                self.world_model_scaler.unscale_(self.world_model_optimizer)
                nn.utils.clip_grad_norm_(self.world_model.parameters(), self._grad_norm_clip)
            self.world_model_scaler.step(self.world_model_optimizer)
            self.world_model_scaler.update()

            mean_sd_state_loss += state_loss.item()
            mean_sd_sequence_loss += sequence_loss.item()
            mean_sd_bound_loss += bound_loss.item()
            mean_sd_kl_loss += kl_loss.item()
            mean_sd_extension_loss += extension_loss.item()
            mean_sd_contact_loss += contact_loss.item()
            mean_sd_termination_loss += termination_loss.item()
    
        sd_num_updates = self.sd_num_mini_batches
        mean_sd_state_loss /= sd_num_updates
        mean_sd_sequence_loss /= sd_num_updates
        mean_sd_bound_loss /= sd_num_updates
        mean_sd_kl_loss /= sd_num_updates
        mean_sd_extension_loss /= sd_num_updates
        mean_sd_contact_loss /= sd_num_updates
        mean_sd_termination_loss /= sd_num_updates


        # update tensorboard
        self.track_data("World Model/sd_state_loss", mean_sd_state_loss) if mean_sd_state_loss != 0 else None
        self.track_data("World Model/sd_sequence_loss", mean_sd_sequence_loss) if mean_sd_sequence_loss != 0 else None
        self.track_data("World Model/sd_bound_loss", mean_sd_bound_loss) if mean_sd_bound_loss != 0 else None
        self.track_data("World Model/sd_kl_loss", mean_sd_kl_loss) if mean_sd_kl_loss != 0 else None
        self.track_data("World Model/sd_extension_loss", mean_sd_extension_loss) if mean_sd_extension_loss != 0 else None
        self.track_data("World Model/sd_contact_loss", mean_sd_contact_loss) if mean_sd_contact_loss != 0 else None
        self.track_data("World Model/sd_termination_loss", mean_sd_termination_loss) if mean_sd_termination_loss != 0 else None

        return mean_sd_state_loss, mean_sd_sequence_loss, mean_sd_bound_loss, mean_sd_kl_loss, mean_sd_extension_loss, mean_sd_contact_loss, mean_sd_termination_loss
    
    def evaluate_world_model(self):  # TODO: used to eval in tensorboard log
        sd_generator = self.system_dynamics_reply_buffer.mini_batch_generator(
            self.eval_trajectory_len,
            1,
            self.eval_trajectories_num,
        )
        state_traj, action_traj, extension_traj, contact_traj, termination_traj = next(sd_generator)
        state_traj_pred, _, _, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred = self.system_dynamics_autoregressive_prediction(state_traj, action_traj, extension_traj, contact_traj, termination_traj)
        # TODO: Warning!!! should divide by epsilon to prevent inf
        traj_autoregressive_error = ((state_traj_pred[:, self.world_model.history_horizon:] - state_traj[:, self.world_model.history_horizon:]).abs().sum(dim=-1) / state_traj[:, self.world_model.history_horizon:].abs().sum(dim=-1)).mean().item()
        traj_autoregressive_error_noised_dict = {}
        for noise_scale in self.eval_trajectory_noise_scale:
            state_traj_noised = state_traj + torch.randn_like(state_traj) * noise_scale
            action_traj_noised = action_traj + torch.randn_like(action_traj) * noise_scale
            state_traj_pred_noised, _, _, _, _, _, _ = self.system_dynamics_autoregressive_prediction(state_traj_noised, action_traj_noised, extension_traj, contact_traj, termination_traj)
            traj_autoregressive_error_noised = ((state_traj_pred_noised[:, self.world_model.history_horizon:] - state_traj_noised[:, self.world_model.history_horizon:]).abs().sum(dim=-1) / state_traj_noised[:, self.world_model.history_horizon:].abs().sum(dim=-1)).mean().item()
            traj_autoregressive_error_noised_dict[noise_scale] = traj_autoregressive_error_noised

        return state_traj, action_traj, extension_traj, contact_traj, termination_traj, state_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred, traj_autoregressive_error, traj_autoregressive_error_noised_dict

    def system_dynamics_autoregressive_prediction(self, state_traj, action_traj, extension_traj, contact_traj, termination_traj):
        state_traj_pred = torch.zeros_like(state_traj, device=self.device)
        aleatoric_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        epistemic_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        action_traj_pred = action_traj.clone()
        extension_traj_pred = torch.zeros_like(extension_traj, device=self.device) if extension_traj is not None else None
        contact_traj_pred = torch.zeros_like(contact_traj, device=self.device) if contact_traj is not None else None
        termination_traj_pred = torch.zeros_like(termination_traj, device=self.device) if termination_traj is not None else None
        
        state_traj_pred[:, :self.world_model.history_horizon] = state_traj[:, :self.world_model.history_horizon]
        if extension_traj_pred is not None:
            extension_traj_pred[:, :self.world_model.history_horizon] = extension_traj[:, :self.world_model.history_horizon]
        if contact_traj_pred is not None:
            contact_traj_pred[:, :self.world_model.history_horizon] = contact_traj[:, :self.world_model.history_horizon]
        if termination_traj_pred is not None:
            termination_traj_pred[:, :self.world_model.history_horizon] = termination_traj[:, :self.world_model.history_horizon]

        self.world_model.reset()
        with torch.inference_mode():
            for i in range(self.world_model.history_horizon, self.eval_trajectory_len):
                if self.world_model.architecture_config["type"] in ["rnn", "rssm"] and i > self.world_model.history_horizon:
                    state_input = state_traj_pred[:, i - 1:i]
                    action_input = action_traj_pred[:, i - 1:i]
                else:
                    state_input = state_traj_pred[:, i - self.world_model.history_horizon:i]
                    action_input = action_traj_pred[:, i - self.world_model.history_horizon:i]
                _inputs = {"x_state_batch": state_input, "x_action_batch": action_input}
                state_pred, aleatoric_uncertainty, epistemic_uncertainty, extension_pred, contact_pred, termination_pred = self.world_model.forward(_inputs)
                state_traj_pred[:, i] = state_pred
                aleatoric_uncertainty_traj_pred[:, i] = aleatoric_uncertainty
                epistemic_uncertainty_traj_pred[:, i] = epistemic_uncertainty
                if extension_traj_pred is not None and extension_pred is not None:
                    extension_traj_pred[:, i] = extension_pred
                if contact_traj_pred is not None and contact_pred is not None:
                    contact_traj_pred[:, i] = torch.sigmoid(contact_pred).round().int()
                if termination_traj_pred is not None and termination_pred is not None:
                    termination_traj_pred[:, i] = torch.sigmoid(termination_pred).round().int()
        return state_traj_pred, aleatoric_uncertainty_traj_pred, epistemic_uncertainty_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred

    # def prepare_imagination(self):
    #     imagination_generator = self.system_replay_buffer.mini_batch_generator(self.system_dynamics.history_horizon, 1, self.imagination_storage.num_envs)
    #     imagination_state_history, imagination_action_history = next(imagination_generator)[:2]
    #     return imagination_state_history, imagination_action_history

    def sample_combined(self, real_storage:Memory, imagination_storage:Memory):  # TODO
        """Generate combined mini-batches from real and imagination storage for MBPO training.
        
        This function implements the core MBPO (Model-Based Policy Optimization) data mixing strategy
        by combining real environment experiences with imagined trajectories from the world model.
        
        Args:
            real_storage (Memory): Storage containing real environment transitions
            imagination_storage (Memory): Storage containing imagined trajectories from world model
            
        Returns:
            list: List of combined mini-batches where each element is a concatenation of real and 
                  imagined data along the batch dimension (dim=0). The combined batch size is
                  the sum of real and imagination batch sizes, allowing PPO to learn from both
                  sources simultaneously.
                  
        Note:
            - Real data provides accurate environment feedback for policy learning
            - Imagination data expands the training dataset and improves sample efficiency
            - Both storages must have compatible tensor structures and feature dimensions
            - The function handles both torch.Tensor and TensorDict data formats
        """
        real_sampled_batches = real_storage.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        imagination_sampled_batches = imagination_storage.sample_all(names=self._imagination_tensors_names, mini_batches=self._mini_batches)

        combined_batches = []
        for real_batch, imagination_batch in zip(real_sampled_batches, imagination_sampled_batches):
            combined_batch = []
            for real_batch_term, imagination_batch_term in zip(real_batch, imagination_batch):
                if isinstance(real_batch_term, torch.Tensor) and isinstance(imagination_batch_term, torch.Tensor):
                    combined_term = torch.cat([real_batch_term, imagination_batch_term], dim=0)
                elif isinstance(real_batch_term, TensorDict) and isinstance(imagination_batch_term, TensorDict):
                    combined_term = TensorDict(
                        {
                            key: torch.cat([real_batch_term[key], imagination_batch_term[key]], dim=0)
                            for key in real_batch_term.keys()
                        },
                        batch_size=[real_batch_term.batch_size[0] + imagination_batch_term.batch_size[0]],
                        device=real_batch_term.device,
                    )
                else:
                    combined_term = real_batch_term
                combined_batch.append(combined_term)
            combined_batches.append(tuple(combined_batch))
        
        return combined_batches

    def load(self, path: str, load_optimizer: bool = True) -> None:
        """Load the policy from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the policy from
        :type path: str
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        else:
            modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for name, data in modules.items():

                if name in ["world_model", "world_model_optimizer"] and self.train_world_model and not self.train_policy:
                    continue

                if not load_optimizer and "optimizer" in name:
                    continue

                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
                else:
                    logger.warning(f"Cannot load the {name} module. The agent doesn't have such an instance")

    def write_tracking_graph(self, timestep: int, timesteps: int):
        # compute iteration number
        try:
            iteration = timestep // self.cfg["rollouts"]
        except Exception:
            iteration = timestep

        state_traj, action_traj, extension_traj, contact_traj, termination_traj, \
            state_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred, \
            traj_autoregressive_error, traj_autoregressive_error_noised_dict = self.evaluate_world_model()

        state_traj = self._rwm_state_normalizer.inverse(state_traj)
        action_traj = self._rwm_action_normalizer.inverse(action_traj)
        state_traj_pred = self._rwm_state_normalizer.inverse(state_traj_pred)
        action_traj_pred = self._rwm_action_normalizer.inverse(action_traj_pred)

        # write autoregressive error
        self.writer.add_scalar("World Model/autoregressive_error", traj_autoregressive_error, iteration)

        # write trajectories
        self.plotter.plot_trajectories(
            self.ax1,
            None,
            state_traj[:self.plt_cfg["system_dynamics_num_visualizations"]],
            action_traj[:self.plt_cfg["system_dynamics_num_visualizations"]],
            extension_traj[:self.plt_cfg["system_dynamics_num_visualizations"]] if extension_traj is not None else None,
            contact_traj[:self.plt_cfg["system_dynamics_num_visualizations"]] if contact_traj is not None else None,
            termination_traj[:self.plt_cfg["system_dynamics_num_visualizations"]] if termination_traj is not None else None,
            self.plt_cfg["system_dynamics_state_idx_dict"],
            )
        self.plotter.plot_trajectories(
            self.ax1,
            self.world_model.history_horizon,
            state_traj_pred[:self.plt_cfg["system_dynamics_num_visualizations"]],
            action_traj_pred[:self.plt_cfg["system_dynamics_num_visualizations"]],
            extension_traj_pred[:self.plt_cfg["system_dynamics_num_visualizations"]] if extension_traj_pred is not None else None,
            contact_traj_pred[:self.plt_cfg["system_dynamics_num_visualizations"]] if contact_traj_pred is not None else None,
            termination_traj_pred[:self.plt_cfg["system_dynamics_num_visualizations"]] if termination_traj_pred is not None else None,
            self.plt_cfg["system_dynamics_state_idx_dict"],
            prediction=True
            )
        self.fig1.align_ylabels()
        self.writer.add_figure("World Model/trajectories", self.fig1, iteration)

        # write autoregressive error noised
        for noise_scale, value in traj_autoregressive_error_noised_dict.items():
            self.writer.add_scalar(f"World Model/autoregressive_error_noised_{noise_scale}", value, iteration)