


# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.networks.mlp import MLP
from skrl.models.networks.recurrentnet import RecurrentNet
from skrl.models import Model
from skrl.utils.spaces import unflatten_tensorized_space


class Encoder(Model):
    """Encoder model for latent extraction from observations and states.
    
    Supports both MLP and Recurrent network structures for flexible latent representation learning.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        # Network configuration
        network_type="mlp",  # "mlp" or "recurrent"
        # latent_dim=64,
        # MLP configuration
        hidden_dims=[256, 128],
        activation="elu",
        # Recurrent configuration
        recurrent_type="lstm",  # "lstm" or "gru"
        num_layers=1,
        hidden_size=256,
        **kwargs,
    ):
        """Initialize the Encoder model.
        
        Args:
            observation_space: Observation space specification (Runner handles space selection)
            action_space: Action space specification ( not used for encoder)
            device: Device to run the model on
            network_type: Type of network - "mlp" or "recurrent"
            latent_dim: Dimension of the latent representation
            hidden_dims: Hidden layer dimensions for MLP
            activation: Activation function for MLP
            recurrent_type: Type of recurrent network - "lstm" or "gru"
            num_layers: Number of recurrent layers
            hidden_size: Hidden size of recurrent network
        """
        if kwargs:
            print(
                "Encoder.__init__ got unexpected arguments, which will be checked below: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(observation_space, action_space, device)

        # Store configuration
        self.network_type = network_type.lower()
        self.latent_dim = self.num_actions
        
        # Set recurrent flag based on network type
        self.is_recurrent = (self.network_type == "recurrent")
        
        # Input dimension - Runner handles space selection, so just use observation_space
        self.input_dim = observation_space.shape[-1]
        
        # Build the network
        if self.network_type == "mlp":
            self.network = MLP(
                input_dim=self.input_dim,
                output_dim=self.latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
            )
        elif self.network_type == "recurrent":
            self.network = RecurrentNet(
                input_size=self.input_dim,
                type=recurrent_type,
                num_layers=num_layers,
                hidden_size=hidden_size,
            )
            # Add output projection layer for recurrent network
            self.output_projection = nn.Linear(hidden_size, self.latent_dim)
        else:
            raise ValueError(f"Unknown network_type: {self.network_type}")

    def compute(self, inputs, role=""):
        """Compute latent representation from inputs.
        
        Args:
            inputs: Dictionary containing input data (Runner handles space selection)
            role: Role of the computation ( not used for encoder)
            
        Returns:
            Latent representation tensor
        """
        # Get states input
        x = inputs.get("states")
        if x is None:
            raise ValueError("Encoder requires 'states' in inputs")
        
        # Handle structured observations
        if isinstance(x, dict):
            x = unflatten_tensorized_space(self.observation_space, x)
        
        # Forward pass through network
        if self.network_type == "mlp":
            latent = self.network(x)
        elif self.network_type == "recurrent":
            # For recurrent network, we need to handle hidden states
            hidden_states = inputs.get("hidden_states", None)
            latent = self.network(x, hidden_states=hidden_states)
            # Project to latent dimension
            latent = self.output_projection(latent)
        
        return latent, None, {}  # Return format: (output, mean, std) - std is None for deterministic encoder

    def forward(self, inputs, role=""):
        """Forward pass wrapper.
        
        Args:
            inputs: Input dictionary
            role: Role identifier
            
        Returns:
            Latent representation
        """
        latent, _, _ = self.compute(inputs, role)
        return latent

    def reset(self, dones=None, hidden_states=None):
        """Reset the encoder state.
        
        Args:
            dones: Boolean tensor indicating which environments to reset
            hidden_states: Optional hidden states to set
        """
        if self.network_type == "recurrent":
            self.network.reset(dones=dones, hidden_states=hidden_states)

    def detach_hidden_states(self, dones=None):
        """Detach hidden states from computation graph.
        
        Args:
            dones: Boolean tensor indicating which environments to detach
        """
        if self.network_type == "recurrent":
            self.network.detach_hidden_states(dones=dones)