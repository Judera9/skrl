# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Union

from skrl.models.networks.mlp import MLP
from skrl.utils.rsl_rl_utils import resolve_nn_activation


class MultiHeadMLP(nn.Module):
    """Multi-layer perceptron with multiple output heads.
    
    This network consists of a shared backbone followed by multiple output heads.
    Each head can have different dimensions and activation functions.
    This is useful for models that need to output multiple related quantities
    (e.g., mean and std for Gaussian policies).
    
    Example:
        >>> network = MultiHeadMLP(
        ...     input_dim=78,
        ...     hidden_dims=[512, 256, 128],
        ...     activation="elu",
        ...     heads={
        ...         "mean": {"output_dim": 23, "activation": None},
        ...         "std": {"output_dim": 23, "activation": None},
        ...     }
        ... )
        >>> outputs = network(x)  # Returns {"mean": ..., "std": ...}
        >>> mean_output = network.heads["mean"](x)  # Access individual head
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str,
        heads: Dict[str, Dict[str, Union[int, str]]],
    ):
        """Initialize the MultiHeadMLP.
        
        Args:
            input_dim: Dimension of the input.
            hidden_dims: Dimensions of the hidden layers.
            activation: Activation function for hidden layers.
            heads: Dictionary of head configurations. Each head config should have:
                - "output_dim": Output dimension of the head
                - "activation": Activation function (None for linear)
        """
        super().__init__()
        
        self.head_names = list(heads.keys())
        self.head_configs = heads
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        # Create shared backbone (hidden layers only)
        backbone_layers = []
        current_dim = input_dim
        
        # Add hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            backbone_layers.append(nn.Linear(current_dim, hidden_dim))
            backbone_layers.append(resolve_nn_activation(activation))  # Apply activation after all hidden layers
            current_dim = hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        self.backbone_output_dim = current_dim
        
        # Create output heads that take the last hidden layer as input
        self.heads = nn.ModuleDict()
        for head_name, head_config in heads.items():
            output_dim = head_config["output_dim"]
            head_activation = head_config.get("activation", None)
            
            # Each head is just a linear layer (optionally with activation)
            head_layers = [nn.Linear(current_dim, output_dim)]
            if head_activation is not None:
                head_layers.append(resolve_nn_activation(head_activation))
            
            self.heads[head_name] = nn.Sequential(*head_layers)
    
    def add_head(self, head_name: str, output_dim: int, activation: str = None, init_scale: float = 0.01):
        """Add a new head to the MultiHeadMLP.
        
        Args:
            head_name: Name of the new head.
            output_dim: Output dimension of the new head.
            activation: Activation function for the head ( None for linear).
            init_scale: Scale factor for weight initialization.
        """
        if head_name in self.heads:
            raise ValueError(f"Head '{head_name}' already exists. Available heads: {list(self.heads.keys())}")
        
        # Create new head
        head_layers = [nn.Linear(self.backbone_output_dim, output_dim)]
        if activation is not None:
            head_layers.append(resolve_nn_activation(activation))
        
        new_head = nn.Sequential(*head_layers)
        
        # Initialize weights
        with torch.no_grad():
            nn.init.normal_(new_head[0].weight, std=init_scale)
            if new_head[0].bias is not None:
                nn.init.zeros_(new_head[0].bias)
        
        # Add to heads
        self.heads[head_name] = new_head
        self.head_names.append(head_name)
        self.head_configs[head_name] = {"output_dim": output_dim, "activation": activation}
    
    def remove_head(self, head_name: str):
        """Remove a head from the MultiHeadMLP.
        
        Args:
            head_name: Name of the head to remove.
        """
        if head_name not in self.heads:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {list(self.heads.keys())}")
        
        del self.heads[head_name]
        self.head_names.remove(head_name)
        del self.head_configs[head_name]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through shared backbone and all heads.
        
        Args:
            x: Input tensor.
            
        Returns:
            Dictionary mapping head names to their outputs.
        """
        # Pass through shared backbone
        backbone_features = self.backbone(x)
        
        # Pass through each head
        outputs = {}
        for head_name in self.head_names:
            outputs[head_name] = self.heads[head_name](backbone_features)
        
        return outputs
    
    def get_head_output(self, x: torch.Tensor, head_name: str) -> torch.Tensor:
        """Get output from a specific head.
        
        Args:
            x: Input tensor.
            head_name: Name of the head.
            
        Returns:
            Output from the specified head.
        """
        if head_name not in self.heads:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {list(self.heads.keys())}")
        
        backbone_features = self.backbone(x)
        return self.heads[head_name](backbone_features)
    
    def init_weights(self, scales: float | List[float]):
        """Initialize weights of backbone and heads.
        
        Args:
            scales: Scale factors for initialization. Can be a single value
                   or a list matching the number of layers.
        """
        # Initialize backbone
        self.backbone.init_weights(scales)
        
        # Initialize heads with smaller scale for stability
        head_scale = 0.01 if isinstance(scales, (int, float)) else 0.01
        for head in self.heads.values():
            head.init_weights(head_scale)
