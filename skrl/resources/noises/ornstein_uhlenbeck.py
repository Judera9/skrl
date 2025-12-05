from typing import Optional, Tuple, Union

import torch
from torch.distributions import Normal

from skrl.resources.noises import Noise


# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(
        self,
        theta: float,
        sigma: float,
        base_scale: float,
        mean: float = 0,
        std: float = 1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> None:
        """Class representing an Ornstein-Uhlenbeck noise

        :param theta: Factor to apply to current internal state
        :type theta: float
        :param sigma: Factor to apply to the normal distribution
        :type sigma: float
        :param base_scale: Factor to apply to returned noise
        :type base_scale: float
        :param mean: Mean of the normal distribution (default: ``0.0``)
        :type mean: float, optional
        :param std: Standard deviation of the normal distribution (default: ``1.0``)
        :type std: float, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional

        Example::

            >>> noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.2, base_scale=0.5)
        """
        super().__init__(device)

        self.first_run = True
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale
        
        # Store action_dim from kwargs (passed by SimpleGaussian)
        self.action_dim = kwargs.get("action_dim", 1)

        self.distribution = Normal(
            loc=torch.tensor(mean, device=self.device, dtype=torch.float32),
            scale=torch.tensor(std, device=self.device, dtype=torch.float32),
        )

    def init(self, num_envs):
        """Initialize noise state for batch environments
        
        Args:
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.first_run = False
        # Initialize state with correct action dimensions
        self.state = torch.zeros(num_envs, self.action_dim).to(self.device)

    def sample(self, sigma=1.0):
        """Sample Ornstein-Uhlenbeck noise for current batch
        
        Args:
            sigma: Scaling factor for noise (matches PinkNoiseDist interface)
            
        Returns:
            Sampled noise tensor with shape (num_envs, action_dim)
        """
        if not hasattr(self, 'num_envs'):
            raise RuntimeError("OrnsteinUhlenbeckNoise.init() must be called before sample()")
        
        # Generate noise for all environments and actions
        size = (self.num_envs, self.action_dim)
        
        if hasattr(self.state, "shape") and self.state.shape != torch.Size(size):
            self.state = torch.zeros(size).to(self.device)
            
        self.state += -self.state * self.theta + self.sigma * self.distribution.sample(size)

        return self.base_scale * self.state * sigma

    def reset(self, dones=None):
        """Reset noise state for specified environments
        
        Args:
            dones: Boolean tensor indicating which environments to reset
        """
        if not hasattr(self, 'num_envs'):
            return
            
        if dones is not None and dones.any():
            # Reset only specified environments
            dones_env_idx = dones.nonzero()[:, 0]
            self.state[dones_env_idx] = torch.zeros(len(dones_env_idx), self.action_dim).to(self.device)
        else:
            # Reset all environments
            self.state = torch.zeros(self.num_envs, self.action_dim).to(self.device)
