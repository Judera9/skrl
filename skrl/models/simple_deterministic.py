# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from skrl.models.networks import MLP
from skrl.models import Model
from skrl.utils.spaces import unflatten_tensorized_space


class SimpleDeterministic(Model):
    is_recurrent = False

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        hidden_dims=[256, 256, 256],
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "SimpleDeterministic.__init__ got unexpected arguments, which will be checked below: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(observation_space, action_space, device)

        # network
        self.network = MLP(self.num_observations, self.num_actions, hidden_dims, activation)

    def reset(self, dones=None):
        pass

    def compute(self, inputs, role=""):
        """Compute the network output"""
        states = inputs.get("states")
        
        # Handle both flattened tensors and structured observations
        if isinstance(states, dict):
            states = unflatten_tensorized_space(self.observation_space, states)
        
        # Compute network output
        output = self.network(states)
        
        return output, None, {}

    def act(self, inputs, role=""):
        """Act according to the specified behavior (returns network output)"""
        return self.compute(inputs, role)