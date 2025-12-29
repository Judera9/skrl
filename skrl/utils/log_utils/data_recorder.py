import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Union
import time
import os
import threading
import torch

class DataRecorder:
    """
    A simple data recorder that manages multiple topics with deque-based data storage.
    
    Features:
    - Register topics with configurable deque length
    - Update data for registered topics
    - Track update frequency and timesteps
    - Save data to CSV or NPY formats
    - Thread-safe operations for robotics applications
    """
    
    def __init__(self, default_deque_length: int = 1000):
        """
        Initialize the DataRecorder.
        
        Args:
            default_deque_length: Default maximum length for topic deques
        """
        self.default_deque_length = default_deque_length
        self.topics: Dict[str, Dict[str, Any]] = {}
        self.update_frequencies: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.last_update_times: Dict[str, float] = {}
        self.update_counts: Dict[str, int] = {}
        self._lock = threading.Lock()  # Thread safety for robotics applications
        
    def register_topic(self, topic_name: str, deque_length: Optional[int] = None, 
                      update_frequency_hz: Optional[float] = None) -> None:
        """
        Register a new topic with a deque for data storage.
        
        Args:
            topic_name: Name of the topic to register
            deque_length: Maximum length of the deque (uses default if None)
            update_frequency_hz: Manual update frequency in Hz (optional)
        """
        with self._lock:
            if topic_name in self.topics:
                print(f"Warning: Topic '{topic_name}' already exists. Overwriting.")
                
            length = deque_length if deque_length is not None else self.default_deque_length
            
            self.topics[topic_name] = {
                'data': deque(maxlen=length),
                'timesteps': deque(maxlen=length),
                'deque_length': length
            }
            
            if update_frequency_hz is not None:
                self.update_frequencies[topic_name] = update_frequency_hz
                
            self.start_times[topic_name] = time.time()
            self.last_update_times[topic_name] = time.time()
            self.update_counts[topic_name] = 0
            
            print(f"Topic '{topic_name}' registered with deque length {length}")
    
    def register_topics(self, topic_names: list, deque_length: Optional[int] = None, 
                     update_frequency_hz: Optional[float] = None) -> None:
        """
        Register multiple topics using a list of topic names with global parameters.
        
        Args:
            topic_names: List of topic names to register
            deque_length: Maximum length of the deque for all topics (uses default if None)
            update_frequency_hz: Manual update frequency in Hz for all topics (optional)
        
        Example:
            topic_list = ['joint_positions', 'joint_velocities', 'sensor_data', 'commands']
            recorder.register_topics(topic_list, deque_length=500, update_frequency_hz=50.0)
        """
        with self._lock:
            length = deque_length if deque_length is not None else self.default_deque_length
            
            for topic_name in topic_names:
                if topic_name in self.topics:
                    print(f"Warning: Topic '{topic_name}' already exists. Overwriting.")
                    
                self.topics[topic_name] = {
                    'data': deque(maxlen=length),
                    'timesteps': deque(maxlen=length),
                    'deque_length': length
                }
                
                if update_frequency_hz is not None:
                    self.update_frequencies[topic_name] = update_frequency_hz
                    
                self.start_times[topic_name] = time.time()
                self.last_update_times[topic_name] = time.time()
                self.update_counts[topic_name] = 0
                
                print(f"Topic '{topic_name}' registered with deque length {length}")
        
        print(f"Successfully registered {len(topic_names)} topics.")
    
    def _convert_tensors_to_numpy(self, data: Any) -> Any:
        """
        Convert PyTorch tensors to numpy arrays for saving.
        
        Args:
            data: Data that may contain PyTorch tensors
            
        Returns:
            Data with tensors converted to numpy arrays
        """
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        elif isinstance(data, (list, tuple)):
            return [self._convert_tensors_to_numpy(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_tensors_to_numpy(value) for key, value in data.items()}
        else:
            return data
        
    def update_data(self, topic_name: str, data: Any, 
                   timestep: Optional[float] = None) -> None:
        """
        Update data for a registered topic.
        
        Args:
            topic_name: Name of the topic to update
            data: Data to store (can be any type)
            timestep: Timestep value (uses time-based if None)
        """
        with self._lock:
            if topic_name not in self.topics:
                raise ValueError(f"Topic '{topic_name}' not registered. Call register_topic first.")
                
            current_time = time.time()
            
            # Calculate timestep if not provided
            if timestep is None:
                if topic_name in self.update_frequencies:
                    # Use manual frequency setting
                    timestep = self.update_counts[topic_name] / self.update_frequencies[topic_name]
                else:
                    # Use actual time difference
                    timestep = current_time - self.start_times[topic_name]
                    
            # Convert tensors to numpy and store data and timestep
            converted_data = self._convert_tensors_to_numpy(data)
            self.topics[topic_name]['data'].append(converted_data)
            self.topics[topic_name]['timesteps'].append(timestep)
            
            # Update timing info
            self.last_update_times[topic_name] = current_time
            self.update_counts[topic_name] += 1
    
    def update_topics(self, topic_data: Dict[str, Any], 
                     timestep: Optional[float] = None) -> None:
        """
        Update multiple topics with data using a dictionary.
        
        Args:
            topic_data: Dictionary where keys are topic names and values are data to store
            timestep: Timestep value to use for all topics (uses individual calculation if None)
        
        Example:
            data_dict = {
                'joint_positions': np.array([0.1, 0.2, 0.3]),
                'joint_velocities': np.array([0.01, 0.02, 0.03]),
                'sensor_reading': 1.5
            }
            recorder.update_topics(data_dict)
        """
        with self._lock:
            for topic_name, data in topic_data.items():
                if topic_name not in self.topics:
                    raise ValueError(f"Topic '{topic_name}' not registered. Call register_topic first.")
                    
                current_time = time.time()
                
                # Calculate timestep for this specific topic if not provided globally
                current_timestep = timestep
                if current_timestep is None:
                    if topic_name in self.update_frequencies:
                        # Use manual frequency setting
                        current_timestep = self.update_counts[topic_name] / self.update_frequencies[topic_name]
                    else:
                        # Use actual time difference
                        current_timestep = current_time - self.start_times[topic_name]
                        
                # Convert tensors to numpy and store data and timestep
                converted_data = self._convert_tensors_to_numpy(data)
                self.topics[topic_name]['data'].append(converted_data)
                self.topics[topic_name]['timesteps'].append(current_timestep)
                
                # Update timing info
                self.last_update_times[topic_name] = current_time
                self.update_counts[topic_name] += 1
        
    def set_update_frequency(self, topic_name: str, frequency_hz: float) -> None:
        """
        Set manual update frequency for a topic.
        
        Args:
            topic_name: Name of the topic
            frequency_hz: Update frequency in Hz
        """
        with self._lock:
            if topic_name not in self.topics:
                raise ValueError(f"Topic '{topic_name}' not registered.")
                
            self.update_frequencies[topic_name] = frequency_hz
            print(f"Update frequency for '{topic_name}' set to {frequency_hz} Hz")
        
    def get_topic_data(self, topic_name: str) -> Dict[str, Any]:
        """
        Get all data for a specific topic.
        
        Args:
            topic_name: Name of the topic
            
        Returns:
            Dictionary containing data, timesteps, and metadata
        """
        with self._lock:
            if topic_name not in self.topics:
                raise ValueError(f"Topic '{topic_name}' not registered.")
                
            return {
                'data': list(self.topics[topic_name]['data']),
                'timesteps': list(self.topics[topic_name]['timesteps']),
                'update_count': self.update_counts[topic_name],
                'frequency_hz': self.update_frequencies.get(topic_name, None)
            }
        
    def save_data(self, topic_name: str, filepath: str, 
                  format: str = 'csv') -> None:
        """
        Save topic data to file.
        
        Args:
            topic_name: Name of the topic to save
            filepath: Output file path
            format: Output format ('csv' or 'npy')
        """
        # Get data with thread safety
        topic_data = self.get_topic_data(topic_name)
        data = topic_data['data']
        timesteps = topic_data['timesteps']
        
        # Convert tensors to numpy arrays before saving
        data = [self._convert_tensors_to_numpy(d) for d in data]
        
        # Create directory if it doesn't exist (fix for root-level files)
        dir_path = os.path.dirname(filepath)
        if dir_path:  # Only create if there's actually a directory path
            os.makedirs(dir_path, exist_ok=True)
        
        if format.lower() == 'csv':
            # Convert to DataFrame and save as CSV
            df_data = {'timestep': timesteps}
            
            # Handle different data types - check for empty data first
            if len(data) > 0:
                # Normalize array shapes - squeeze leading dimension 1
                normalized_data = []
                for d in data:
                    if isinstance(d, np.ndarray) and d.ndim > 1 and d.shape[0] == 1:
                        normalized_data.append(d.squeeze(0))
                    else:
                        normalized_data.append(d)
                
                # Check if all data are scalars
                if all(isinstance(d, (int, float, np.number)) and not isinstance(d, np.ndarray) or 
                      (isinstance(d, np.ndarray) and d.ndim == 0) for d in normalized_data):
                    df_data['value'] = normalized_data
                elif all(isinstance(d, (list, tuple, np.ndarray)) for d in normalized_data):
                    # Handle vector/matrix data - ensure consistent dimensions
                    try:
                        # Find the maximum dimension
                        max_dim = max(d.shape[-1] if hasattr(d, 'shape') and d.ndim > 0 else 1 
                                    for d in normalized_data)
                        
                        # Pad or reshape data to consistent dimensions
                        padded_data = []
                        for d in normalized_data:
                            if isinstance(d, np.ndarray):
                                if d.ndim == 0:
                                    # Scalar - pad to max_dim
                                    padded = np.zeros(max_dim)
                                    padded[0] = d.item()
                                    padded_data.append(padded)
                                elif d.ndim == 1:
                                    # 1D array - pad if needed
                                    if len(d) < max_dim:
                                        padded = np.zeros(max_dim)
                                        padded[:len(d)] = d
                                        padded_data.append(padded)
                                    else:
                                        padded_data.append(d)
                                else:
                                    # Multi-dimensional - flatten last dimension
                                    flattened = d.reshape(-1)
                                    if len(flattened) < max_dim:
                                        padded = np.zeros(max_dim)
                                        padded[:len(flattened)] = flattened
                                        padded_data.append(padded)
                                    else:
                                        padded_data.append(flattened[:max_dim])
                            else:
                                # Non-array scalar
                                padded = np.zeros(max_dim)
                                padded[0] = d
                                padded_data.append(padded)
                        
                        data_array = np.array(padded_data)
                        for i in range(data_array.shape[-1]):
                            df_data[f'value_{i}'] = data_array[:, i]
                    except Exception as e:
                        print(f"Warning: Could not normalize array shapes, saving as strings. Error: {e}")
                        df_data['value'] = [str(d) for d in normalized_data]
                else:
                    # Mixed types - convert to string
                    df_data['value'] = [str(d) for d in normalized_data]
                    
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
            print(f"Data for topic '{topic_name}' saved to {filepath} (CSV format)")
            
        elif format.lower() == 'npy':
            # Save as NPY format
            save_dict = {
                'data': np.array(data),
                'timesteps': np.array(timesteps),
                'update_count': topic_data['update_count'],
                'frequency_hz': topic_data['frequency_hz']
            }
            np.save(filepath, save_dict)
            print(f"Data for topic '{topic_name}' saved to {filepath} (NPY format)")
            
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'npy'.")
            
    def save_all_data(self, directory: str, format: str = 'csv') -> None:
        """
        Save all registered topics to separate files.
        
        Args:
            directory: Directory to save files
            format: Output format ('csv' or 'npy')
        """
        os.makedirs(directory, exist_ok=True)
        
        with self._lock:
            topic_names = list(self.topics.keys())
        
        for topic_name in topic_names:
            if format.lower() == 'csv':
                filepath = os.path.join(directory, f"{topic_name}.csv")
            else:
                filepath = os.path.join(directory, f"{topic_name}.npy")
                
            self.save_data(topic_name, filepath, format)
            
    def get_topic_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered topics.
        
        Returns:
            Dictionary with topic information
        """
        with self._lock:
            info = {}
            for topic_name in self.topics.keys():
                topic_data = self.topics[topic_name]
                info[topic_name] = {
                    'data_length': len(topic_data['data']),
                    'max_length': topic_data['deque_length'],
                    'update_count': self.update_counts[topic_name],
                    'frequency_hz': self.update_frequencies.get(topic_name, None),
                    'last_update': self.last_update_times[topic_name]
                }
            return info
        
    def clear_topic(self, topic_name: str) -> None:
        """
        Clear all data for a specific topic.
        
        Args:
            topic_name: Name of the topic to clear
        """
        with self._lock:
            if topic_name not in self.topics:
                raise ValueError(f"Topic '{topic_name}' not registered.")
                
            self.topics[topic_name]['data'].clear()
            self.topics[topic_name]['timesteps'].clear()
            self.update_counts[topic_name] = 0
            self.start_times[topic_name] = time.time()
            self.last_update_times[topic_name] = time.time()
            print(f"Topic '{topic_name}' data cleared.")
        
    def clear_all_topics(self) -> None:
        """
        Clear data for all topics.
        """
        with self._lock:
            for topic_name in self.topics.keys():
                self.clear_topic(topic_name)