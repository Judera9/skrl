# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from skrl.models.torch.networks.multi_head_mlp import MLP
from skrl.models.torch import Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class SimpleGaussian(Model):
    is_recurrent = False

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
        self.output_action_scale = kwargs.get("output_action_scale", False)
        self.max_log_std = kwargs.get("max_log_std", 2.0)
        self.min_log_std = kwargs.get("min_log_std", -20.0)
        self.max_action = kwargs.get("max_action", 100.0)
        self.min_action = kwargs.get("min_action", -100.0)

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
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        print(f"Actor MLP: {self.actor}")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def compute(self, inputs, role=""):
        """Compute the network output (mean action and log_std)"""
        states = inputs.get("states")
        
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
            
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                log_std = torch.log(torch.clamp(self.std, min=torch.exp(self.min_log_std), max=torch.exp(self.max_log_std)))
                std = log_std.exp().expand_as(mean)
            elif self.noise_std_type == "log":
                log_std = torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)
                std = torch.exp(log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        return mean, std, {}

    def act(self, inputs, role=""):
        """Act according to the specified behavior"""
        # Compute mean and std
        mean, std, extras = self.compute(inputs, role)
        
        # Create distribution
        self.distribution = Normal(mean, std)
        
        # Sample actions
        actions = self.distribution.sample()

        # Apply action scaling if enabled
        if self.output_action_scale:
            action_scales = self.action_scale_net(inputs["states"])
            actions = actions * action_scales
        
        # Use taken_actions if provided (for PPO training), otherwise use sampled actions
        actions_for_log_prob = inputs.get("taken_actions", actions)
        
        # Calculate log probability
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
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)