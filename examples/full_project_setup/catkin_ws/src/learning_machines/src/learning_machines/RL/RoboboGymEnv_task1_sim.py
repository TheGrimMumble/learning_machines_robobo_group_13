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
        self.IR_threshold_val = 1000
        self.current_action = np.array([0,0], dtype=np.float32)

        # The duration of a step in miliseconds, so each step takes half a second
        self.timestep_duration = 500

        self.step_in_episode = 0

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 20

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
        IR_sensor_data /= 4000.0
        
        obs = np.concatenate([IR_sensor_data, self.current_action]).astype(np.float32)
        return obs
    
    def get_reward(self):
        reward = 0
        obs = self.get_obs()
        IR_sensors = obs[:7]  
        left_speed = obs[7]
        right_speed = obs[8]
        for i in IR_sensors:
            if i >= self.IR_threshold_val:
                reward -= 1
        if left_speed == 0 or right_speed == 0:
            reward -= 20
        
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
        print(self.step_in_episode)
        self.current_action = action
        max_speed = 100
        left_speed = action[0] * max_speed
        right_speed = action[1] * max_speed

        # Send to simulator or hardware
        self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)
        
        # Get new obs 
        observation = self.get_obs()
        reward = self.get_reward()
        done = self.terminate()
        info = self.get_info(observation, reward)
        print(info)
        print('')
    
        return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        # Reset simulator, not sure yet whether we should stop and start the simulator
        # for each episode, but it seemed like the savest option to start with
        self.robobo.stop_simulation()
        self.robobo.play_simulation()

        self.step_in_episode = 0

        # Wait until sim is fully reset and ready (you can insert a short sleep here if needed)
        # time.sleep(0.2)

        # Get initial obs
        obs = self.get_obs()

        return obs, {}
