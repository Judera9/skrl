import torch
import isaaclab.utils.math as math_utils

@torch.jit.script
def angle1D_to_sincos2D(angle: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

@torch.jit.script
def sincos2D_to_angle1D(sincos: torch.Tensor) -> torch.Tensor:
    return math_utils.wrap_to_pi(torch.atan2(sincos[..., 1], sincos[..., 0]))