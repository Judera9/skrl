#!/usr/bin/env python3
"""
Example usage of DataRecorder class
"""

from data_recorder import DataRecorder
import numpy as np
import time

def main():
    # Create DataRecorder with default deque length of 1000
    recorder = DataRecorder(default_deque_length=1000)
    
    # Register topics
    recorder.register_topic("joint_positions", deque_length=500, update_frequency_hz=50.0)
    recorder.register_topic("joint_velocities", deque_length=500, update_frequency_hz=50.0)
    recorder.register_topic("sensor_readings", deque_length=1000)  # No manual frequency
    
    print("=== DataRecorder Example ===")
    
    # Simulate data updates
    print("\nUpdating data...")
    for i in range(10):
        # Simulate joint positions (3D vector)
        joint_pos = np.array([np.sin(i * 0.1), np.cos(i * 0.1), i * 0.01])
        recorder.update_data("joint_positions", joint_pos)
        
        # Simulate joint velocities (3D vector)
        joint_vel = np.array([np.cos(i * 0.1) * 0.1, -np.sin(i * 0.1) * 0.1, 0.01])
        recorder.update_data("joint_velocities", joint_vel)
        
        # Simulate sensor readings (scalar)
        sensor_val = 1.0 + 0.1 * np.sin(i * 0.2)
        recorder.update_data("sensor_readings", sensor_val)
        
        time.sleep(0.02)  # Simulate 50Hz update rate
    
    # Get topic information
    print("\n=== Topic Information ===")
    info = recorder.get_topic_info()
    for topic, topic_info in info.items():
        print(f"{topic}:")
        print(f"  Data length: {topic_info['data_length']}")
        print(f"  Max length: {topic_info['max_length']}")
        print(f"  Update count: {topic_info['update_count']}")
        print(f"  Frequency Hz: {topic_info['frequency_hz']}")
    
    # Get data for a specific topic
    print("\n=== Sample Data ===")
    joint_data = recorder.get_topic_data("joint_positions")
    print(f"Joint positions shape: {np.array(joint_data['data']).shape}")
    print(f"First 3 timesteps: {joint_data['timesteps'][:3]}")
    print(f"First 3 positions: {joint_data['data'][:3]}")
    
    # Save data to CSV
    print("\n=== Saving Data ===")
    recorder.save_data("joint_positions", "./output/joint_positions.csv", format="csv")
    recorder.save_data("joint_velocities", "./output/joint_velocities.csv", format="csv")
    recorder.save_data("sensor_readings", "./output/sensor_readings.csv", format="csv")
    
    # Save data to NPY
    recorder.save_data("joint_positions", "./output/joint_positions.npy", format="npy")
    
    # Save all data at once
    recorder.save_all_data("./output/all_data", format="csv")
    
    print("\nData saved successfully!")
    
    # Clear data
    print("\n=== Clearing Data ===")
    recorder.clear_topic("sensor_readings")
    print(f"Sensor readings after clear: {len(recorder.get_topic_data('sensor_readings')['data'])}")
    
    # Clear all topics
    recorder.clear_all_topics()
    print("All topics cleared.")

if __name__ == "__main__":
    main()
