# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from skrl.models.networks.multi_head_mlp import MLP
from skrl.models import Model
from skrl.utils.spaces import unflatten_tensorized_space


class SimpleGaussian(Model):

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        actor_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "SimpleGaussian.__init__ got unexpected arguments, which will be checked below: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(observation_space, action_space, device)

        self.share_std_backbone = kwargs.get("share_std_backbone", False)
        self.use_scale_tril = kwargs.get("use_scale_tril", False)
        self.output_action_scale = kwargs.get("output_action_scale", False)
        self.max_log_std = kwargs.get("max_log_std", 2.0)
        self.min_log_std = kwargs.get("min_log_std", -20.0)
        self.max_action = kwargs.get("max_action", 100.0)
        self.min_action = kwargs.get("min_action", -100.0)
        
        # Noise configuration
        self._noise_generator = kwargs.get("noise_generator", None)
        self._noise_generator_kwargs = kwargs.get("noise_generator_kwargs", {})
        if self._noise_generator:
            self._noise_generator_kwargs.update({
                "action_dim": self.num_actions,
                "device": self.device})
            self.noise_generator = self._noise_generator(**self._noise_generator_kwargs)

        # get the observation dimensions
        num_actor_obs = self.num_observations

        # Configure network architecture based on sharing modes
        if self.share_std_backbone:
            self.actor = MLP(num_actor_obs, self.num_actions * 2, actor_hidden_dims, activation)
                
            with torch.no_grad():
                self.actor[-1].weight[self.num_actions:, :] *= 0.01
                if hasattr(self.actor[-1], 'bias') and self.actor[-1].bias is not None:
                    # Set std head bias to 0 ( log(1.0) = 0)
                    self.actor[-1].bias[self.num_actions:].zero_()
        else:
            self.actor = MLP(num_actor_obs, self.num_actions, actor_hidden_dims, activation)

            # Initialize with identity scaling for action scale
            if self.output_action_scale:
                self.actor = MLP(num_actor_obs, self.num_actions, actor_hidden_dims, activation, last_activation="tanh")
                self.action_scale_net = MLP(num_actor_obs, self.num_actions, actor_hidden_dims, activation)
                with torch.no_grad():
                    self.action_scale_net[-1].weight[:] *= 10.0
                    if hasattr(self.action_scale_net[-1], 'bias') and self.action_scale_net[-1].bias is not None:
                        self.action_scale_net[-1].bias.fill_(1.0)
            else:
                self.actor = MLP(num_actor_obs, self.num_actions, actor_hidden_dims, activation)

            # Action noise parameters
            self.noise_std_type = noise_std_type
            
            if self.use_scale_tril:
                # Calculate dimension for scale_tril parameters: n(n+1)/2
                _std_num = self.num_actions * (self.num_actions + 1) // 2
                
                # Pre-calculate diagonal indices for packed parameter array
                self.diag_indices = []
                for i in range(self.num_actions):
                    diag_idx = i * (i + 1) // 2 + i  # diagonal element in packed format
                    self.diag_indices.append(diag_idx)
                self.diag_indices = torch.tensor(self.diag_indices, dtype=torch.long)
                
                # Create parameter tensor with proper initialization
                if self.noise_std_type == "scalar":
                    # Initialize diagonal elements with init_noise_std, off-diagonal with 0
                    scale_tril_init = torch.zeros(_std_num)
                    scale_tril_init[self.diag_indices] = init_noise_std
                    self.std = nn.Parameter(scale_tril_init)
                elif self.noise_std_type == "log":
                    # Initialize diagonal elements with log(init_noise_std), off-diagonal with log(0) -> large negative
                    scale_tril_init = torch.full((_std_num,), -20.0)  # log(0) approximation
                    scale_tril_init[self.diag_indices] = torch.log(torch.tensor(init_noise_std))
                    self.log_std = nn.Parameter(scale_tril_init)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            else:
                # Traditional mode: only diagonal elements
                _std_num = self.num_actions
                if self.noise_std_type == "scalar":
                    self.std = nn.Parameter(init_noise_std * torch.ones(_std_num))
                elif self.noise_std_type == "log":
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(_std_num)))
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        print(f"Actor MLP: {self.actor}")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        # Reset noise generator if it exists
        if self._noise_generator:
            self.noise_generator.reset(dones)

    def compute(self, inputs, role=""):
        """Compute the network output (mean action and log_std)"""
        states = inputs.get("states")
        _extra_info = {}
        
        # Handle both flattened tensors and structured observations
        if isinstance(states, dict):
            states = unflatten_tensorized_space(self.observation_space, states)
        
        # Compute mean action
        if self.share_std_backbone:
            # Use MultiHeadMLP with head-based access
            outputs = self.actor(states)
            mean = outputs[:, :self.num_actions]

            # Extract log_std from the second half
            log_std = outputs[:, self.num_actions:]
            log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
            std = torch.exp(log_std)
        else:
            # Traditional mode: separate mean and std
            mean = self.actor(states)
            
            if self.use_scale_tril:                
                # Construct lower triangular matrix for extra_info
                batch_size = mean.shape[0]
                scale_tril = torch.zeros(batch_size, self.num_actions, self.num_actions, device=self.device)
                
                # Get indices for lower triangular matrix
                rows, cols = torch.tril_indices(self.num_actions, self.num_actions, device=self.device)
                
                # Fill the matrix with parameters based on noise_std_type
                if self.noise_std_type == "scalar":
                    # Fill all elements with std parameters
                    scale_tril[:, rows, cols] = self.std.expand(batch_size, -1)
                    # Apply softplus to diagonal elements for positive definiteness
                    diag_indices = torch.arange(self.num_actions, device=self.device)
                    scale_tril[:, diag_indices, diag_indices] = F.softplus(scale_tril[:, diag_indices, diag_indices]) + 1e-6
                    # Off-diagonal elements remain as raw std parameters (can learn correlations)
                elif self.noise_std_type == "log":
                    # Fill diagonal elements with transformed log_std
                    diag_log_std = self.log_std[self.diag_indices]
                    scale_tril[:, self.diag_indices, self.diag_indices] = torch.exp(
                        torch.clamp(diag_log_std, min=self.min_log_std, max=self.max_log_std)
                    ) + 1e-6
                    # Fill off-diagonal elements with raw log_std (correlations)
                    off_diag_indices = [i for i in range(len(rows)) if rows[i] != cols[i]]
                    scale_tril[:, rows[off_diag_indices], cols[off_diag_indices]] = \
                        self.log_std[off_diag_indices].expand(batch_size, -1)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
                
                # Extract diagonal elements as std for return
                std = torch.diagonal(scale_tril, dim1=-2, dim2=-1)
                
                # Store scale_tril in extra_info
                _extra_info["scale_tril"] = scale_tril
                
            else:
                # Compute standard deviation
                if self.noise_std_type == "scalar":
                    log_std = torch.log(torch.clamp(self.std, min=torch.exp(self.min_log_std), max=torch.exp(self.max_log_std)))
                    std = log_std.exp().expand_as(mean)
                elif self.noise_std_type == "log":
                    log_std = torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)
                    std = torch.exp(log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        return mean, std, _extra_info

    def act(self, inputs, role=""):
        """Act according to the specified behavior"""
        # Compute mean and std
        mean, std, extras = self.compute(inputs, role)
        
        # Create distribution based on whether scale_tril is available
        if self.use_scale_tril:
            # Use MultivariateNormal with full covariance matrix
            self.distribution = MultivariateNormal(mean, scale_tril=extras["scale_tril"])
            actions = self.distribution.sample()
        else:
            # Use Normal distribution with diagonal std
            self.distribution = Normal(mean, std)
            
            # Sample actions
            # Only use noise generator during rollout (when taken_actions is not provided)
            if self._noise_generator and "taken_actions" not in inputs:
                if self.noise_generator.first_run:
                    self.noise_generator.init(num_envs=mean.shape[0])
                noise = self.noise_generator.sample()
                actions = mean + std * noise
            else:
                # actions = self.distribution.sample()
                actions = mean + std * torch.randn_like(std)  # fallback or during update

        # Apply action scaling if enabled
        if self.output_action_scale:
            action_scales = self.action_scale_net(inputs["states"])
            actions = actions * action_scales
        
        # Use taken_actions if provided (for PPO training), otherwise use sampled actions
        actions_for_log_prob = inputs.get("taken_actions", actions)
        
        # Calculate log probability
        if self.use_scale_tril:
            # MultivariateNormal returns shape (batch_size,), need to add dimension
            log_prob = self.distribution.log_prob(actions_for_log_prob).unsqueeze(-1)
        else:
            # Normal returns per-action log_prob, sum over actions
            log_prob = self.distribution.log_prob(actions_for_log_prob).sum(dim=-1, keepdim=True)
        
        return actions, log_prob, extras

    def act_inference(self, inputs, role=""):
        """Deterministic action for inference (returns tuple for SKRL compatibility)"""
        mean, _, _ = self.compute(inputs, role)

        # Apply action scaling if enabled
        if self.output_action_scale:
            action_scales = self.action_scale_net(inputs["states"])
            mean = mean * action_scales

        return mean, None, {}

    def get_actions_log_prob(self, actions, inputs):
        """Get log probability of actions"""
        log_prob = self.distribution.log_prob(actions)
        if isinstance(self.distribution, MultivariateNormal):
            return log_prob.unsqueeze(-1)
        else:
            return log_prob.sum(dim=-1, keepdim=True)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        if isinstance(self.distribution, MultivariateNormal):
            # For MultivariateNormal, return diagonal of covariance matrix
            return torch.sqrt(torch.diagonal(self.distribution.covariance_matrix, dim1=-2, dim2=-1))
        else:
            return self.distribution.stddev

    @property
    def entropy(self):
        if isinstance(self.distribution, MultivariateNormal):
            return self.distribution.entropy().unsqueeze(-1)
        else:
            return self.distribution.entropy().sum(dim=-1)