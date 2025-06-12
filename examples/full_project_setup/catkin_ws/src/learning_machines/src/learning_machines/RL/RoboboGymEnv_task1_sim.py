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
        self.IR_threshold_val = 50
        self.IR_range_max = 1_000
        self.current_action = np.array([0,0], dtype=np.float32)

        # The duration of a step in miliseconds, so each step takes half a second
        self.timestep_duration = 50

        self.step_in_episode = 0
        self.epoch_number = 0

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 512*2
        self.first_step = True
        

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
            'IRS - FL1': obs[5],
            'IRS - BM': obs[6],
            'IRS - FR1': obs[7],
            'Left wheel speed': obs[8],
            'Right wheel speed': obs[9],
            'reward': reward
        }

        return info
    
    def log_normalize(self, raw_ir):
        calibrate = np.array([0, 0, 45, 45, 0, 0, 50, 0])
        raw_ir -= calibrate
        raw_ir = np.clip(raw_ir, 1, self.IR_range_max)
        log_ir = np.log(raw_ir)
    
        # print("IR_sensor_data shape:", IR_sensor_data.shape)
        # print("self.current_action shape:", self.current_action.shape)

        # Normalize IR sensor values between 0 - 1 assuming max value is 4000
        # but this is an abritrary number for now
        log_threshold = np.log(self.IR_range_max)
        IR_sensor_data = log_ir / log_threshold
        return IR_sensor_data
    
    def linear_normalize(self, raw_ir):
        raw_ir = np.clip(raw_ir, 1, self.IR_range_max)
        raw_ir /= self.IR_range_max
        return raw_ir

    # returns IR sensor data
    def get_obs(self):
        raw_ir = np.array(self.robobo.read_irs(), dtype=np.float32)
        ir_sensor_data = self.linear_normalize(raw_ir)
        obs = np.concatenate([ir_sensor_data, self.current_action]).astype(np.float32)
        return obs
    
    def get_reward_complicated(self):
        obs = self.get_obs()
        irs_back = [obs[o] for o in (0, 1, 6)]
        back = max(irs_back)
        irs_front = [obs[o] for o in (2, 3, 4, 5, 7)]
        front = max(irs_front)
        speed = sum([obs[8], obs[9]])
        # print([obs[o] for o in (0, 1, 6)])
        # print([obs[o] for o in (2, 3, 4, 5, 7)])
        # print([obs[8], obs[9]])
        multiplier = 5
        back_up = -1 * front * speed * multiplier * 0.5
        kickoff_from_wall = back * speed * multiplier
        full_speed_ahead = (front - 0.75) * -1 * speed * multiplier
        # straight = obs[8] * obs[9] * multiplier * 2
        speed_threshold = 0
        straight = 0
        if obs[8] > speed_threshold and obs[9] > speed_threshold:
            speed_diff_inv = 1 / abs(obs[8] - obs[9])
            straight += min(25, speed_diff_inv)
        reward = sum([
            back_up,
            kickoff_from_wall,
            full_speed_ahead,
            straight
            ])
        # print(reward)
        
        return float(reward)
    
    def get_reward_medium(self):
        reward = 0
        obs = self.get_obs()

        irs = obs[:8]
        if max(irs) > 0.8:
            reward -= (max(irs) - 0.75) * 100
        
        speed_threshold = 0.6
        if obs[8] > speed_threshold and obs[9] > speed_threshold:
            speed_diff_inv = 1 / abs(obs[8] - obs[9])
            reward += min(25, speed_diff_inv)
        
        speed = sum([obs[8], obs[9]])
        if speed > 0.6:
            reward += speed * 12.5


        return float(reward)
    
    def get_reward_simple(self):
        reward = 0
        obs = self.get_obs()
        speed_threshold = 0
        if obs[8] > speed_threshold and obs[9] > speed_threshold:
            reward += (obs[8] + obs[9]) * 10
        
        irs = obs[:8]
        if max(irs) > 0.3:
            reward -= max(irs) * 5

        return reward

    def calc_dist_from_speed(self):
        pass
    
    def get_reward_med_comp(self):
        reward = 0
        obs = self.get_obs()

        irs = obs[:8]
        if max(irs) > 0.3:
            reward -= max(irs)
        
        speed_threshold = 0.2
        if obs[8] > speed_threshold and obs[9] > speed_threshold:
            reward += sum([obs[8], obs[9]]) * 2
            diff = abs(obs[8] - obs[9])
            if 0 < diff < 0.15:
                speed_inv = 1 / diff
                reward += min(20, speed_inv)

        return float(reward)
    
    def get_reward(self):
        reward = 0
        obs = self.get_obs()
        if obs[8] > 0 and  obs[9] > 0:
            speed = sum([obs[8], obs[9]])
            reward += speed * 2
        return float(reward)

    def terminate(self):
        if self.step_in_episode == self.max_steps_in_episode:
            return True
        else:
            return False

    def step(self, action):
        left_just = 10
        if self.first_step:
            info_keys = [
                'Step #',
                'BR ir',
                'BL ir',
                'FR2 ir',
                'FL2 ir',
                'FM ir',
                'FL1 ir',
                'BM ir',
                'FR1 ir',
                'L tire',
                'R tire',
                'reward'
            ]
            print("".join([key.ljust(left_just) for key in info_keys]))
            self.first_step = False
        # Rescale from [-1, 1] to actual motor speeds, e.g. [-100, 100]
        action = np.array(action, dtype=np.float32)
        self.step_in_episode += 1
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
        info_headers = [
            'IRS - BR',
            'IRS - BL',
            'IRS - FR2',
            'IRS - FL2',
            'IRS - FM',
            'IRS - FL1',
            'IRS - BM',
            'IRS - FR1',
            'Left wheel speed',
            'Right wheel speed',
            'reward'
        ]
        print(str(self.step_in_episode).ljust(left_just) + "".join(
            [str(round(info[value], 4)).ljust(left_just) for value in info_headers]
            ))
    
        return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        # Reset simulator, not sure yet whether we should stop and start the simulator
        # for each episode, but it seemed like the savest option to start with
        self.robobo.stop_simulation()
        self.robobo.play_simulation()

        self.step_in_episode = 0
        self.epoch_number += 1
        print(f" ------------ epoch: {self.epoch_number} ------------ ")


        # Wait until sim is fully reset and ready (you can insert a short sleep here if needed)
        # time.sleep(0.2)

        # Get initial obs
        obs = self.get_obs()

        return obs, {}

