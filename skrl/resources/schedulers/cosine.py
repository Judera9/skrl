import torch
import math
from typing import Optional, Union, List

class CosineScheduler:
    def __init__(
        self, 
        initial_value: Union[float, List[float]] = 1.0, 
        min_value: Union[float, List[float]] = 0.0, 
        T_interval: Union[int, List[int]] = 1000
    ):
        """Cosine annealing scheduler with multi-stage support
        
        Args:
            initial_value: Starting value(s) - can be float or list of floats
            min_value: Minimum value(s) - can be float or list of floats  
            T_interval: Steps for each stage - can be int (single stage) or list of ints (multi-stage)
        """
        # Handle single stage mode
        if isinstance(T_interval, int):
            T_interval = [T_interval]
        
        # Convert single values to lists
        if isinstance(initial_value, (int, float)):
            initial_value = [initial_value]
        if isinstance(min_value, (int, float)):
            min_value = [min_value]
            
        # Validate inputs
        if len(initial_value) != len(min_value) or len(min_value) != len(T_interval):
            raise ValueError("initial_value, min_value, and T_interval must have same length")

        self.initial_values = initial_value
        self.min_values = min_value
        self.T_intervals = T_interval
        
        # Pre-compute cumulative steps for stage boundaries
        self.cumulative_steps = [0]
        for interval in self.T_intervals:
            self.cumulative_steps.append(self.cumulative_steps[-1] + interval)
        
        self.step_count = 0
        self.stage_count = 0  # Track current stage
        self.current_value = self.initial_values[0]  # Initialize with first value
        
    def step(self) -> float:
        """Update scheduler and return current value
        
        Returns:
            Current value after cosine annealing
        """
        self.step_count += 1
        
        # Update stage if needed
        self._update_stage(self.step_count)
        
        # Compute value based on current stage
        if self.stage_count >= len(self.T_intervals):
            # After all stages, return the last min_value
            self.current_value = self.min_values[-1]
        else:
            # Compute cosine decay for current stage
            stage_start = self.cumulative_steps[self.stage_count]
            stage_end = self.cumulative_steps[self.stage_count + 1]
            stage_duration = stage_end - stage_start
            step_in_stage = self.step_count - stage_start
            
            # Half cosine decay for this stage
            # cosine goes from 0 to pi, factor goes from 1 to 0
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_in_stage / stage_duration))
            self.current_value = (
                self.min_values[self.stage_count] + 
                (self.initial_values[self.stage_count] - self.min_values[self.stage_count]) * cosine_factor
            )
        
        return self.current_value
    
    def _update_stage(self, step_count: int) -> None:
        """Update stage count based on current step
        
        Args:
            step_count: Current step count
        """
        # Check if we need to advance to next stage
        if self.stage_count < len(self.T_intervals) and step_count >= self.cumulative_steps[self.stage_count + 1]:
            self.stage_count += 1
    
    def get_value(self) -> float:
        """Get current value without stepping
        
        Returns:
            Current value
        """
        return self.current_value
    
    def reset(self) -> None:
        """Reset scheduler to initial state"""
        self.step_count = 0
        self.stage_count = 0
        self.current_value = self.initial_values[0]