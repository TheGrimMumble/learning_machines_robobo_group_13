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
        
        self.IR_threshold_val = 100 # 1000 / 4000 = 0.25. 4000 is the max value and 1000 is the cutoff value for when it is assumed
        # that the robot has bumbed into a wall
        self.current_action = np.array([0,0], dtype=np.float32)

        # The duration of a step in miliseconds, so each step takes a quarter of a second
        self.timestep_duration = 50

        self.step_in_episode = 0
        self.current_position = None

        self.max_sensor_val = 2000

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 1200 # time for one episode is 2 minutes

        self.weight_dict = {'pos_diff_weight': 20, 'IR_cost_weight': 10}

    # It would be nice to randomize this at the start of each episode
    # But for now it seems like this is determined by the scene that we start before running the code
    def _set_robot_initial_position(self):
        pass

    def get_info(self, obs, reward):
        
        info = {
            'IRS - BR': obs[0],
            'IRS - BL': obs[1],
            'IRS - FR2': obs[2],
            'IRS - FL2': obs[3],
            'IRS - FM': obs[4],
            'IRS - BL1': obs[5],
            'IRS - BM': obs[6],
            'IRS - FR1': obs[7],
            'Left wheel speed': obs[8],
            'Right wheel speed': obs[9],
            'reward': reward
        }

        return info


    # returns IR sensor data
    def get_obs(self):
        IR_sensor_data = np.array(self.robobo.read_irs(), dtype=np.float32)
    
        # print("IR_sensor_data shape:", IR_sensor_data.shape)
        # print("self.current_action shape:", self.current_action.shape)

        # Normalize IR sensor values between 0 - 1 assuming max value is 4000
        # but this is an abritrary number for now
        # Vectorized min-max normalization (clip and scale)
        min_val = 0
        max_val = self.max_sensor_val
        IR_sensor_data = np.clip(IR_sensor_data, min_val, max_val)
        IR_sensor_data = (IR_sensor_data - min_val) / (max_val - min_val)
        
        obs = np.concatenate([IR_sensor_data, self.current_action]).astype(np.float32)
        return obs

    def compute_proximity_penalty(self, ir_values):
        # Only penalize if a sensor goes above threshold
        max_sensor = np.max(ir_values)

        if max_sensor >= 800: # this is solely an estimation
            print('')
            print('ROBOT BUMBED INTO SOMETHING!!!')
            print('')

        if max_sensor <= self.IR_threshold_val:
            return 0.0
        else:
            # Scale penalty from 0 to 1 with linear scale
            normalized = (max_sensor - self.IR_threshold_val) / (self.max_sensor_val - self.IR_threshold_val)
            return normalized  

    def forward_speed_reward(self, obs):
        # Encourage forward motion over just spinning
        forward_speed = (obs[7] + obs[8]) / 2
        return forward_speed

    def get_reward(self):
        obs = self.get_obs()
        IR_sensors = obs[:7] * self.max_sensor_val
        reward = 0

        pos_diff = self.euclidean_distance(self.current_position, self.previous_position)
        # print(f'pos diff {pos_diff}')
        pos_diff_reward = pos_diff * self.weight_dict['pos_diff_weight'] 

        reward += pos_diff_reward
        
        
        # left_speed = obs[7]
        # right_speed = obs[8]

        costs = 0
        IR_cost = self.compute_proximity_penalty(IR_sensors) *  self.weight_dict['IR_cost_weight']

        costs += IR_cost
        final_reward = reward - costs

        reward_dict = {'IR cost': IR_cost, 
                       'Total costs': costs, 
                       'pos_diff_reward':pos_diff_reward, 
                       'Total reward': reward, 
                       'Final reward': final_reward}
        
        # for key, value in reward_dict.items():
        #     print(f"{key}: {value}")

        return final_reward

    def euclidean_distance(self, pos1, pos2):
        p1 = np.array([pos1.x, pos1.y])
        p2 = np.array([pos2.x, pos2.y])
        return np.linalg.norm(p2 - p1)

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
        
        self.current_action = action

        max_speed = 100

        left_speed = action[0] * max_speed 
        right_speed = action[1] * max_speed

        # Send to simulator or hardware
        self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)
        
        self.previous_position = self.current_position
        self.current_position = self.robobo.get_position()
        # print(f'previous pos {self.previous_position}')
        # print(f'current pos{self.current_position}')
        # print(f'dtype pos {type(self.current_position)}')

        # Get new obs 
        observation = self.get_obs()
        reward = self.get_reward()
        done = self.terminate()
        info = self.get_info(observation, reward)
        # print(info)
        print('')
    
        return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        # Reset simulator, not sure yet whether we should stop and start the simulator
        # for each episode, but it seemed like the savest option to start with
        self.robobo.stop_simulation()
        self.robobo.play_simulation()

        self.previous_position = self.robobo.get_position() # set the same as the current position at the start 
        self.current_position = self.robobo.get_position()

        self.step_in_episode = 0

        # Wait until sim is fully reset and ready (you can insert a short sleep here if needed)
        # time.sleep(0.2)

        # Get initial obs
        obs = self.get_obs()

        return obs, {}
