import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2

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

        self.universal_step = 0
        # Left and right wheel, needs rescaling
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 7 IR sensors are assumed to be between 0 and 1
        # Speed is assumed to be between -1 and 1
        # obeservation space looks like this: low: [0,0,0,0,0,0,0,-1,-1], high: [1,1,1,1,1,1,1,1,1]
        self.observation_space = spaces.Box(
            low=np.array([0]*8 + [-1, -1]),
            high=np.array([1]*8 + [1, 1]),
            dtype=np.float32)
           
        # assuming that this is the value at which it would bumb into an object, but is not based on any testing
        
        # from when it starts punishing
        self.IR_threshold_val = 100 

        self.current_action = np.array([0,0], dtype=np.float32)

        # The duration of a step in miliseconds, so each step takes a quarter of a second
        self.timestep_duration = 50

        self.step_in_episode = 0
        self.current_position = self.robobo.get_position()

        self.collision_threshold = 0.8

        self.collision = False

        self.max_sensor_val = 1000

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
            'IRS - BR': obs[0] * self.max_sensor_val,
            'IRS - BL': obs[1] * self.max_sensor_val,
            'IRS - FR2': obs[2] * self.max_sensor_val,
            'IRS - FL2': obs[3] * self.max_sensor_val,
            'IRS - FM': obs[4] * self.max_sensor_val,
            'IRS - BL1': obs[5] * self.max_sensor_val,
            'IRS - BM': obs[6] * self.max_sensor_val,
            'IRS - FR1': obs[7] * self.max_sensor_val,
            'Left wheel speed': obs[8] * 100,
            'Right wheel speed': obs[9] * 100,
            # 'reward': reward
        }

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
    
        # print("IR_sensor_data shape:", IR_sensor_data.shape)
        # print("self.current_action shape:", self.current_action.shape)

        # Normalize IR sensor values between 0 - 1 assuming max value is 1500
        # but this is an abritrary number for now
        # Vectorized min-max normalization (clip and scale)
        # min_val = 0
        # max_val = self.max_sensor_val
        # IR_sensor_data = self.log_normalize_sensors(IR_sensor_data)
        IR_sensor_data = self.linear_normalize_sensors(IR_sensor_data)
        
        obs = np.concatenate([IR_sensor_data, self.current_action]).astype(np.float32)
        return obs

    def collision_detection(self, ir_values):
        max_sensor = np.max(ir_values)

        if max_sensor >= self.collision_threshold: # this is solely an estimation
            self.collision = True
            print('')
            print('ROBOT BUMBED INTO SOMETHING!!!')
            print('')
        else:
            self.collision = False

    # def compute_proximity_penalty(self, ir_values):
    #     # Only penalize if a sensor goes above threshold
    #     max_sensor = np.max(ir_values)

    #     if max_sensor >= self.collision_threshold: # this is solely an estimation
    #         self.collision = True
    #         print('')
    #         print('ROBOT BUMBED INTO SOMETHING!!!')
    #         print('')
    #     else:
    #         self.collision = False

    #     if max_sensor <= self.IR_threshold_val:
    #         return 0.0
    #     else:
    #         # Scale penalty from 0 to 1 with linear scale
    #         normalized = (max_sensor - self.IR_threshold_val) / (self.max_sensor_val - self.IR_threshold_val)
    #         return normalized  

    # def forward_speed_reward(self, obs):

    #     # speed is normalized between -1 and 1
    #     left_speed = obs[7]
    #     right_speed = obs[8]
    #     if not self.collision:
    #         # Only reward if both wheels moving in same direction
    #         if (left_speed > 0 and right_speed > 0):
    #             forward_speed = abs((left_speed + right_speed) / 2)
    #             return forward_speed
    #         elif(left_speed < 0 and right_speed < 0):
    #             backward_speed = abs((left_speed + right_speed) / 2) * 0.25
    #             return backward_speed
    #         else:
    #             return 0  # No reward for turning in place
    #     else:
    #         return 0

    def forward_speed_reward(self, obs):
        left_speed = obs[8]  # Fix index
        right_speed = obs[9] # Fix index
        
        if not self.collision:
            # Calculate base speed reward
            if (left_speed > 0 and right_speed > 0):  # Both forward
                base_reward = (left_speed + right_speed) / 2
            elif (left_speed < 0 and right_speed < 0):  # Both backward
                base_reward = abs((left_speed + right_speed) / 2) * 0.1
            else:
                return 0  # Different directions = no reward
            
            # Scale reward based on coordination (smaller difference = higher reward)
            speed_difference = abs(left_speed - right_speed)
            coordination_factor = max(0, 1 - speed_difference)  # 1 when same speed, 0 when very different
            
            return base_reward * coordination_factor
        
        return 0

    
    # def turn_cost(self, left_speed, right_speed):
    #     turn_amount = abs(left_speed - right_speed)
    #     turn_cost = (turn_amount / 200)
    #     return turn_cost

    def get_reward(self, obs, left_speed, right_speed):
        IR_sensors = obs[:8]

        proximity_penalty = np.max(IR_sensors) *  self.weight_dict['IR_cost_weight']
        # IR_cost = self.compute_proximity_penalty(IR_sensors) *  self.weight_dict['IR_cost_weight']
    
        # forward_speed_reward = self.forward_speed_reward(obs) * self.weight_dict['forward_speed_weight']
        forward_speed_reward = self.forward_speed_reward(obs) * self.weight_dict['forward_speed_weight']

        # reward += pos_diff_reward
        reward = forward_speed_reward - proximity_penalty
        
        # left_speed = obs[7]
        # right_speed = obs[8]

        reward_dict = {'proximity_penalty': proximity_penalty,
                       'forward_speed_reward': forward_speed_reward, 
                       'Final reward': reward}
        
        for key, value in reward_dict.items():
            print(f"{key}: {value}")

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
        self.last_action = self.current_action.copy()
        self.current_action = action

        max_speed = 100
        left_speed = action[0] * max_speed 
        right_speed = action[1] * max_speed

        # Send to simulator or hardware
        self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)

        self.current_position = self.robobo.get_position()

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

    # def step(self, action):
    #     # Rescale from [-1, 1] to actual motor speeds, e.g. [-100, 100]
    #     action = np.array(action, dtype=np.float32)
        
        
    #     self.step_in_episode += 1
    #     self.universal_step += 1
    #     # print(f'Universal step: {self.universal_step}')
    #     # print(f'Step in Episode {self.step_in_episode}')

    #     self.current_action = action

    #     max_speed = 100

    #     left_speed = action[0] * max_speed 
    #     right_speed = action[1] * max_speed

    #     # Send to simulator or hardware
    #     self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)

    #     # Get new obs 
    #     observation = self.get_obs()

    #     self.collision_detection(observation[:7])

    #     reward = self.get_reward(observation, left_speed, right_speed)
    #     done = self.terminate()
    #     info = self.get_info(observation, reward)
        
    #     # for key, value in info.items():
    #     #     print(f"{key}: {value}")
    #     # print('')
    
    #     return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        # Reset simulator, not sure yet whether we should stop and start the simulator
        # for each episode, but it seemed like the savest option to start with
        self.robobo.stop_simulation()
        self.robobo.play_simulation()
        
        self.current_position = self.robobo.get_position()

        self.collision = False
        self.previous_action = np.array([0,0])
        self.current_action = np.array([0,0])

        # self.previous_position = self.robobo.get_position() # set the same as the current position at the start 
        # self.current_position = self.robobo.get_position()

        self.step_in_episode = 0

        # Wait until sim is fully reset and ready (you can insert a short sleep here if needed)
        # time.sleep(0.2)

        # Get initial obs
        obs = self.get_obs()

        return obs, {}
