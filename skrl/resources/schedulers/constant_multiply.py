from typing import Optional


class ConstantScheduler:
    def __init__(self, initial_value: float = 0.01, decay_factor: float = 0.99, decay_interval: int = 1000, min_value: float = 0.0):
        """Simple constant entropy scheduler with decay
        
        Args:
            initial_value: Initial entropy value
            decay_factor: Factor to multiply entropy by at each decay step
            decay_interval: Number of steps between decays
        """
        self.initial_value = initial_value
        self.current_value = initial_value
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.min_value = min_value
        self.step_count = 0
        
    def step(self) -> float:
        """Update scheduler and return current entropy value
        
        Returns:
            Current entropy value after potential decay
        """
        self.step_count += 1
        
        # Apply decay at specified intervals
        if self.step_count % self.decay_interval == 0:
            self.current_value *= self.decay_factor
            # Ensure current_value never goes below min_value
            self.current_value = max(self.current_value, self.min_value)
                
        return self.current_value

    def get_value(self) -> float:
        """Get current entropy value without stepping
        
        Returns:
            Current entropy value
        """
        return self.current_value
    
    def reset(self) -> None:
        """Reset scheduler to initial state"""
        self.current_value = self.initial_value
        self.step_count = 0