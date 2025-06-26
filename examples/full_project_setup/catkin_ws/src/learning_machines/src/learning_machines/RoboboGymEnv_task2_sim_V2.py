import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import datetime
import csv

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

        # CSV logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = open(f"/root/results/sim_log_{timestamp}_V4.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        header = ['Universal_step', 
                  'Step_in_Episode', 
                  'collision', 
                  'x pos', 
                  'y pos',
                    'green_found',
                    'cx',
                    'cy',
                    'area',
                    'food',
                  'BR', 
                  'BL', 
                  'FR2', 
                  'FL2', 
                  'FM', 
                  'FL1', 
                  'BM', 
                  'FR1', 
                  'ls', 
                  'rs', 
                  'proximity_penalty', 
                  'forward_speed_reward', 
                  'Final_reward']
        self.csv_writer.writerow(header)

        self.proximity_penalty = 0
        self.forward_speed_reward_ = 0

        self.universal_step = 0

        # Left and right wheel action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 7 IR sensors are assumed to be between 0 and 1
        # Speed is assumed to be between -1 and 1
        # obeservation space looks like this: low: [0,0,0,0,0,0,0,-1,-1], high: [1,1,1,1,1,1,1,1,1]
        
        # visibility green object: float between 0 and 1
        # center x in picture is between 0 and 512 and will be normalized by dividing by 512 between 0 and 1
        # Same for center y
        # Area is area in pixels so we divide this by total 512 x 512 pixels which will also give a value between 0 and 1
        # number of collected food, max 7 food items so nr will be divided by 7
        # 8 sensor values
        self.observation_space = spaces.Box(
            low=np.array([0]*13),
            high=np.array([1]*13),
            dtype=np.float32)
        
        self.collision_threshold = 1
           
        
        # from when it starts punishing
        self.current_action = np.array([0,0], dtype=np.float32)

        # The duration of a step in miliseconds, so each step takes a quarter of a second
        self.timestep_duration = 100

        self.step_in_episode = 0
        self.current_position = self.robobo.get_position()

        # when a collision is registered
        # self.collision_threshold = 0.8

        self.collision = False

        # any value above 1000 will be clipped
        self.max_sensor_val = 1000
        self.green_found = 0

        self.previous_action = np.array([0,0])
        self.current_action = np.array([0,0])

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 600 

        self.weight_dict = {'pos_diff_weight': 20,
                            'IR_cost_weight': 1,
                            'forward_speed_weight': 0.5,
                            'turn_cost_weight': 10,
                            'area_weight': 10,
                            'food_weight': 1,
                            'detect_food_weight': 0.25,
                            'alignment_weight': 0.5}
        

        # self.nr_collected_food = self.robobo.get_nr_food_collected()

        self.image_w_h = 512
        self.total_count_food = 7
        self.nr_collected_food = 0
        self.prev_nr_collected_food = 0

    
    def detect_green_object(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        image_front = self.robobo.read_image_front()

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        green_found = False
        center_x, center_y = -1, -1
        area = 0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 300: 
                green_found = True

                x, y, w, h = cv2.boundingRect(largest)
                center_x = x + w // 2
                center_y = y + h // 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return frame, green_found, center_x, center_y, area


    # It would be nice to randomize this at the start of each episode
    # But for now it seems like this is determined by the scene that we start before running the code
    def _set_robot_initial_position(self):
        pass

    def collection_bonus(self):
        pass

    def get_info(self, obs, reward):

        obs_IR = obs[:8]
        obs_image_data = obs[8:]
        green_found = obs_image_data[0]
        cx = obs_image_data[1]
        cy = obs_image_data[2]
        area = obs_image_data[3]
        food = obs_image_data[4]

        # green_found, cx, cy, area, food = obs_image_data
        
        info = {
            'Universal step': self.universal_step,
            'Step in Episode': self.step_in_episode,
            'collision': self.collision,
            'x pos': self.current_position.x,
            'y pos': self.current_position.y,
            'green_found': green_found,
            'cx': cx,
            'cy': cy,
            'area': area,
            'food': food,
            'BR': obs[0] * self.max_sensor_val,
            'BL': obs[1] * self.max_sensor_val,
            'FR2': obs[2] * self.max_sensor_val,
            'FL2': obs[3] * self.max_sensor_val,
            'FM': obs[4] * self.max_sensor_val,
            'FL1': obs[5] * self.max_sensor_val,
            'BM': obs[6] * self.max_sensor_val,
            'FR1': obs[7] * self.max_sensor_val,
            'ls': self.current_action[0] * 100,
            'rs': self.current_action[1] * 100,
            'proximity_penalty': self.proximity_penalty,
            'forward_speed_reward': self.forward_speed_reward_, 
            'Final reward': reward}

        return info

    
    def linear_normalize_sensors(self, sensor_values):
        min_val = 0
        max_val = self.max_sensor_val
        sensor_values = np.clip(sensor_values, min_val, max_val)
        sensor_values = (sensor_values - min_val) / (max_val - min_val)
        return sensor_values

    
    def get_obs(self):
        IR_sensor_data = np.array(self.robobo.read_irs(), dtype=np.float32)
        IR_sensor_data = self.linear_normalize_sensors(IR_sensor_data)
        # print(f'IR_sensor_data, {}')
    
        front_image = self.robobo.read_image_front()
        frame, green_found, center_x, center_y, area = self.detect_green_object(front_image)
        
        self.nr_collected_food = self.robobo.get_nr_food_collected()

        if green_found:
            green_found = 1.0
        else:
            green_found = 0.0

        center_x_norm = center_x / self.image_w_h 
        center_y_norm = center_y / self.image_w_h
        area_norm = area / (self.image_w_h * self.image_w_h)
        nr_collected_food_norm = self.nr_collected_food /self.total_count_food
        
        image_data = np.array([green_found, center_x_norm, center_y_norm, area_norm, nr_collected_food_norm]).astype(np.float32)
        obs = np.concatenate([IR_sensor_data, image_data]).astype(np.float32)

        return obs

    def collision_detection(self, ir_values):
        
        max_sensor = np.max(ir_values)
        # IR_sensors_front = np.array([IR_sensors[2], IR_sensors[3], IR_sensors[4], IR_sensors[5], IR_sensors[7]])

        if max_sensor >= self.collision_threshold and self.green_found == 0:
            self.collision = True
            # print('')
            # print('ROBOT BUMBED INTO SOMETHING!!!')
            # print('')
        else:
            self.collision = False

    def forward_speed_reward(self):
        left_speed = self.current_action[0] 
        right_speed = self.current_action[1]
        
        if (left_speed > 0 and right_speed > 0):  # both forward
            base_reward = (left_speed + right_speed) / 2
        elif (left_speed < 0 and right_speed < 0):  # both backward
            base_reward = abs((left_speed + right_speed) / 2) * 0.1
        else:
            return 0  # Different directions = no reward
        
        speed_difference = abs(left_speed - right_speed)
        coordination_factor = max(0, 1 - speed_difference)  # 1 when same speed, 0 when very different
        
        return base_reward * coordination_factor


    def get_reward(self, obs):
        IR_sensors = obs[:8]
        IR_sensors_front = np.array([IR_sensors[2], IR_sensors[3], IR_sensors[4], IR_sensors[5], IR_sensors[7]])
        green_found, cx, cy, area, food = obs[8:]

        reward = 0.0
        proximity_penalty = 0.0

        if green_found == 0:
            proximity_penalty = np.max(IR_sensors_front) * self.weight_dict['IR_cost_weight']
            reward -= proximity_penalty
            self.proximity_penalty = proximity_penalty
        else:
            proximity_penalty = 0
            alignment_reward = 1.0 - abs(cx - 0.5) * 2  # center alignment
            reward += 0.1
            reward += alignment_reward * self.weight_dict['alignment_weight']

            forward_speed_reward = self.forward_speed_reward() * self.weight_dict['forward_speed_weight']
            reward += forward_speed_reward
            self.forward_speed_reward_ = forward_speed_reward

        reward += area * self.weight_dict['area_weight']
        reward += (self.nr_collected_food - self.prev_nr_collected_food) * self.weight_dict['food_weight']

        self.green_found = green_found

        return reward


    def terminate(self):
        if self.nr_collected_food == 7:
            return True
        elif self.step_in_episode == self.max_steps_in_episode:
            return True
        else:
            return False
    
    def step(self, action):
        action = np.array(action, dtype=np.float32)
        
        self.step_in_episode += 1
        self.universal_step += 1
        # print(f'Universal step: {self.universal_step}')
        # print(f'Step in Episode {self.step_in_episode}')

        self.current_action = action

        max_speed = 100
        left_speed = action[0] * max_speed 
        right_speed = action[1] * max_speed

        self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)

        self.current_position = self.robobo.get_position()

        self.prev_nr_collected_food = self.nr_collected_food
        observation = self.get_obs()

        # self.collision_detection(observation[:8])  

        reward = self.get_reward(observation)
        done = self.terminate()
        info = self.get_info(observation, reward)
        
        # for key, value in info.items():
        #     print(f"{key}: {value}")
        # print('')

        # save info
        row = [info['Universal step'], 
               info['Step in Episode'], 
               info['collision'], 
               info['x pos'], 
               info['y pos'],
                info['green_found'],
                info['cx'],
                info['cy'],
                info['area'],
                info['food'],
               info['BR'], 
               info['BL'], 
               info['FR2'], 
               info['FL2'], 
               info['FM'], 
               info['FL1'], 
               info['BM'], 
               info['FR1'], 
               info['ls'], 
               info['rs'], 
               info['proximity_penalty'], 
               info['forward_speed_reward'], 
               info['Final reward']]
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  

        self.robobo.stop_simulation()
        self.robobo.play_simulation()

        self.robobo.set_phone_tilt(100,100)
        self.robobo.set_phone_pan(180,100)

        self.nr_collected_food = 0
        self.prev_nr_collected_food = 0

        self.proximity_penalty = 0
        self.forward_speed_reward_ = 0
        
        self.current_position = self.robobo.get_position()

        # self.collision = False
        # self.previous_action = np.array([0,0])
        # self.current_action = np.array([0,0])

        self.previous_position = self.robobo.get_position() 
        self.current_position = self.robobo.get_position()

        self.step_in_episode = 0
        self.green_found = 0

        obs = self.get_obs()

        return obs, {}
