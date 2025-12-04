import torch
import torch.nn as nn
from skrl.resources.noises import Noise

class PinkNoiseDist(Noise):  # TODO： Actually is Red Noise
    """
    Pink Noise generator, supports batch dimension for PPO parallel environment sampling.
    """
    def __init__(
        self, 
        action_dim, 
        smoothing=0.6, 
        seq_len=1000, 
        device='cpu',
        **kwargs
    ):
        super().__init__(device)
        self.first_run = True
        self.action_dim = action_dim
        self.smoothing = smoothing

        self.add_scale = kwargs.get("add_scale", 0.0)  # 0.0 no scheduler
        self.decay_interval = kwargs.get("decay_interval", 100)
        self.max_value = kwargs.get("max_value", 0.6)

        self.step_count = 0

    def init(self, num_envs):
        self.num_envs = num_envs
        self.first_run = False
        # Precompute some parameters for Pink Noise generation
        # Here we use simplified superposition method or pre-generated buffer method, below is FFT-based reset generation logic
        # For real-time performance, we typically use state variables to maintain noise
        self.state = torch.zeros(num_envs, self.action_dim).to(self.device)

    def sample(self, sigma=1.0):
        """
        Generate next step noise.
        Note: Pink Noise is quite complex, here we provide a commonly used approximate implementation in engineering:
        Use time-correlated smooth update (similar to OU but with parameters adjusted to simulate Pink spectrum)
        """
        # A simple Pink Noise approximation: n_t = 0.5 * n_{t-1} + 0.5 * white_noise
        # Strict Pink Noise requires filters, but in RL this is commonly used as a benchmark for "Colored Noise"
        
        # Generate white noise
        white = torch.randn(self.num_envs, self.action_dim).to(self.device)

        # Mix: 0.8 is smoothing coefficient, larger means smoother (low frequency), smaller means more like white noise
        self.state = self.smoothing * self.state + (1 - self.smoothing) * white
        
        # Renormalize to maintain unit variance (important! Otherwise sigma will shrink due to smoothing)
        # Theoretically variance becomes (1-smoothing)^2 / (1 - smoothing^2)
        # Here we do simple rescaling to ensure magnitude is close to standard Gaussian
        scale_factor = torch.sqrt(torch.tensor(1 - self.smoothing**2, device=self.device)) / (1 - self.smoothing)
        
        # update schedular
        self.step_count += 1
        if self.add_scale > 0.0 and self.step_count % self.decay_interval == 0:
            self.smoothing = max(self.max_value, self.smoothing + self.add_scale)

        return self.state * scale_factor * sigma

    def reset(self, dones=None):
        if dones.any():
            # Reset only the environments that are done
            dones_env_idx = dones.nonzero()[:, 0]
            self.state[dones_env_idx] = torch.zeros(dones.sum(), self.action_dim).to(self.device)
        else:
            # Reset all environments
            self.state = torch.zeros(self.num_envs, self.action_dim).to(self.device)