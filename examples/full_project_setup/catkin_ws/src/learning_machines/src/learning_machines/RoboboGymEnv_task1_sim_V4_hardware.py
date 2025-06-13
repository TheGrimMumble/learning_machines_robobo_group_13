import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import csv
import os
import datetime

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

class RoboboGymEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super().__init__()
        self.robobo = rob

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = open(f"/root/results/hardware_log_{timestamp}_V4.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        self.proximity_penalty = 0
        self.forward_speed_reward_ = 0

        header = ['Universal_step', 'Step_in_Episode', 'collision', 'BR', 'BL', 'FR2', 'FL2', 'FM', 'BL1', 'BM', 'FR1', 'ls', 'rs', 'proximity_penalty', 'forward_speed_reward', 'Final_reward']
        self.csv_writer.writerow(header)
        
        self.universal_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        
        # 7 IR sensors are assumed to be between 0 and 1
        # Speed is assumed to be between -1 and 1
        # obeservation space looks like this: low: [0,0,0,0,0,0,0,-1,-1], high: [1,1,1,1,1,1,1,1,1]
        self.observation_space = spaces.Box(
            low=np.array([0]*8 + [-1, -1]),
            high=np.array([1]*8 + [1, 1]),
            dtype=np.float32)
           
        
        # from when it starts punishing
        self.IR_threshold_val = 100 

        self.current_action = np.array([0,0], dtype=np.float32)
        self.timestep_duration = 100
        self.step_in_episode = 0
        self.collision_threshold = 0.8
        self.collision = False
        self.max_sensor_val = 700

        self.previous_action = np.array([0,0])
        self.current_action = np.array([0,0])

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 1200 

        self.weight_dict = {'pos_diff_weight': 20,
                            'IR_cost_weight': 1.5,
                            'forward_speed_weight': 1,
                            'turn_cost_weight': 10}
    


    # It would be nice to randomize this at the start of each episode
    # But for now it seems like this is determined by the scene that we start before running the code
    def _set_robot_initial_position(self):
        pass

    def get_info(self, obs, reward):
        
        info = {
            'Universal step': self.universal_step,
            'Step in Episode': self.step_in_episode,
            'collision': self.collision,
            'BR': obs[0] * self.max_sensor_val,
            'BL': obs[1] * self.max_sensor_val,
            'FR2': obs[2] * self.max_sensor_val,
            'FL2': obs[3] * self.max_sensor_val,
            'FM': obs[4] * self.max_sensor_val,
            'BL1': obs[5] * self.max_sensor_val,
            'BM': obs[6] * self.max_sensor_val,
            'FR1': obs[7] * self.max_sensor_val,
            'Left wheel speed': obs[8] * 100,
            'Right wheel speed': obs[9] * 100,
            'proximity_penalty': self.proximity_penalty,
            'forward_speed_reward': self.forward_speed_reward_, 
            'Final reward': reward}

        return info


    def log_normalize_sensors(self, sensor_values, epsilon=1e-6):
        """
        Log normalize IR sensor values
        
        Args:
            sensor_values: Raw sensor readings (higher = closer to obstacle)
            max_sensor_val: Maximum possible sensor value
            epsilon: Small value to avoid log(0)
        
        Returns:
            Log normalized values between 0 and 1
        """
        max_sensor_val = self.max_sensor_val
        # Clip values to valid range
        clipped = np.clip(sensor_values, epsilon, max_sensor_val)
        

        # Take log (natural log or log10 both work)
        log_values = np.log(clipped)
        
        # Normalize to 0-1 range
        log_min = np.log(epsilon)
        log_max = np.log(max_sensor_val)

        
        normalized = (log_values - log_min) / (log_max - log_min)
        
        return normalized
    
    def linear_normalize_sensors(self, sensor_values):
        min_val = 0
        max_val = self.max_sensor_val
        sensor_values = np.clip(sensor_values, min_val, max_val)
        sensor_values = (sensor_values - min_val) / (max_val - min_val)
        return sensor_values

    
    # returns IR sensor data
    def get_obs(self):
        IR_sensor_data = np.array(self.robobo.read_irs(), dtype=np.float32)
        IR_sensor_data = self.linear_normalize_sensors(IR_sensor_data)
        
        obs = np.concatenate([IR_sensor_data, self.current_action]).astype(np.float32)
        return obs

    def collision_detection(self, ir_values):
        max_sensor = np.max(ir_values)

        if max_sensor >= self.collision_threshold: 
            self.collision = True
            print('')
            print('ROBOT BUMBED INTO SOMETHING!!!')
            print('')
        else:
            self.collision = False


    def forward_speed_reward(self, obs):
        left_speed = obs[8]  
        right_speed = obs[9]
        
        if not self.collision:
            # Calculate base speed reward
            if (left_speed > 0 and right_speed > 0):
                base_reward = (left_speed + right_speed) / 2
            elif (left_speed < 0 and right_speed < 0):
                base_reward = abs((left_speed + right_speed) / 2) * 0.1
            else:
                return 0  
            
            speed_difference = abs(left_speed - right_speed)
            coordination_factor = max(0, 1 - speed_difference)  # 1 when same speed, 0 when very different
            
            return base_reward * coordination_factor
        
        return 0


    def get_reward(self, obs, left_speed, right_speed):
        IR_sensors = obs[:8]

        proximity_penalty = np.max(IR_sensors) *  self.weight_dict['IR_cost_weight']
        # IR_cost = self.compute_proximity_penalty(IR_sensors) *  self.weight_dict['IR_cost_weight']
    
        # forward_speed_reward = self.forward_speed_reward(obs) * self.weight_dict['forward_speed_weight']
        forward_speed_reward = self.forward_speed_reward(obs) * self.weight_dict['forward_speed_weight']

        # reward += pos_diff_reward
        reward = forward_speed_reward - proximity_penalty

        self.proximity_penalty = proximity_penalty
        self.forward_speed_reward_ = forward_speed_reward
        self.final_reward = reward

        return reward



    def terminate(self):
        if self.step_in_episode == self.max_steps_in_episode:
            return True
        else:
            return False
    
    def step(self, action):
        # Rescale from [-1, 1] to actual motor speeds, e.g. [-100, 100]
        action = np.array(action, dtype=np.float32)
        
        self.step_in_episode += 1
        self.universal_step += 1
        print(f'Universal step: {self.universal_step}')
        print(f'Step in Episode {self.step_in_episode}')

        
        # Save previous action BEFORE updating current action
        # self.last_action = self.current_action.copy()
        self.current_action = action

        max_speed = 100
        left_speed = int(action[0] * max_speed)
        right_speed = int(action[1] * max_speed)
        print(right_speed)
        print(left_speed)

        # Send to simulator or hardware
        self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)

        # Get new obs 
        observation = self.get_obs()

        self.collision_detection(observation[:8])  # Fix: 8 sensors not 7

        reward = self.get_reward(observation, left_speed, right_speed)
        done = self.terminate()
        info = self.get_info(observation, reward)
        
        for key, value in info.items():
            print(f"{key}: {value}")
            print('')
        
        return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        # self.robobo.stop_simulation()
        # self.robobo.play_simulation()
        
        # self.current_position = self.robobo.get_position()

        self.collision = False
        self.previous_action = np.array([0,0])
        self.current_action = np.array([0,0])

        self.proximity_penalty = 0
        self.forward_speed_reward_ = 0

        # self.previous_position = self.robobo.get_position() # set the same as the current position at the start 
        # self.current_position = self.robobo.get_position()

        self.step_in_episode = 0

        # Wait until sim is fully reset and ready (you can insert a short sleep here if needed)
        # time.sleep(0.2)

        # Get initial obs
        obs = self.get_obs()

        return obs, {}
