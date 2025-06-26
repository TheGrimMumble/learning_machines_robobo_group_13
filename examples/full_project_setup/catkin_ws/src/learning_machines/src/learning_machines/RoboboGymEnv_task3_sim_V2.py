import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import datetime
import csv
import os

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
                #   'collision', 
                  'x pos', 
                  'y pos',
                    'green_found',
                    'red_found'
                    'cx_red',
                    'cy_red',
                    'cx_green',
                    'cy_green',
                    'area',
                    'food',
                #   'BR', 
                #   'BL', 
                #   'FR2', 
                #   'FL2', 
                #   'FM', 
                #   'BL1', 
                #   'BM', 
                #   'FR1', 
                  'ls', 
                  'rs', 
                #   'proximity_penalty', 
                  'forward_speed_reward', 
                  'Final_reward']
        self.csv_writer.writerow(header)

        # self.proximity_penalty = 0
        # self.forward_speed_reward_ = 0

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
        self.observation_space = spaces.Box(
            low=np.array([0]*5),
            high=np.array([1]*5),
            dtype=np.float32)
            
        
        # from when it starts punishing
        self.current_action = np.array([0,0], dtype=np.float32)

        # The duration of a step in miliseconds, so each step takes a quarter of a second
        self.timestep_duration = 100

        self.step_in_episode = 0
        self.current_position = self.robobo.get_position()

        # when a collision is registered
        # self.collision_threshold = 0.8

        # self.collision = False

        # any value above 1000 will be clipped
        self.max_sensor_val = 1000

        self.previous_action = np.array([0,0])
        self.current_action = np.array([0,0])

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 600 

        self.weight_dict = {'pos_diff_weight': 20,
                            'IR_cost_weight': 1.5,
                            'forward_speed_weight': 0.5,
                            'turn_cost_weight': 10,
                            'area_weight': 10,
                            'food_weight': 1,
                            'alignment_weight': 0.5}
        

        # self.nr_collected_food = self.robobo.get_nr_food_collected()

        self.image_w_h = 512
        self.forward_speed_reward_ = 0

    
    def detect_green_object(self, frame):
        # Convert image to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # image_front = self.robobo.read_image_front()

        # # Define green color range (tweak if needed)
        # lower_green = np.array([50, 100, 100])
        # lower_green = np.array([50, 100, 100])
        # upper_green = np.array([80, 255, 255])
    
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        # Create binary mask where green is white
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Optional: clean up mask noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours (i.e., detected green blobs)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        green_found = False
        center_x, center_y = -1, -1
        area = 0

        if contours:
            # Find largest contour (assume it's the green box)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 300:  # filter out small noise
                green_found = True

                # Get bounding box and centroid
                x, y, w, h = cv2.boundingRect(largest)
                center_x = x + w // 2
                center_y = y + h // 2

                # Draw detection on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        #         # === Debug frame saving ===
        # if os.environ.get("DEBUG_GREEN_DETECTION") == "1":
        #     output_dir = "/root/results/debug_frames"
        #     os.makedirs(output_dir, exist_ok=True)
        #     frame_id = getattr(self, "debug_frame_id", 0)
        #     filepath = f"{output_dir}/frame_{frame_id:04d}.png"
        #     cv2.imwrite(filepath, frame)
        #     self.debug_frame_id = frame_id + 1


        return frame, green_found, center_x, center_y, area
    

    def detect_red_object(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        image_front = self.robobo.read_image_front()

        # Red color has two ranges in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Combine both masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_found = False
        center_x, center_y = -1, -1
        area = 0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 300:
                red_found = True
                x, y, w, h = cv2.boundingRect(largest)
                center_x = x + w // 2
                center_y = y + h // 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        return frame, red_found, center_x, center_y, area


    # It would be nice to randomize this at the start of each episode
    # But for now it seems like this is determined by the scene that we start before running the code
    def _set_robot_initial_position(self):
        pass

    def collection_bonus(self):
        pass

    def get_info(self, obs, reward):

        green_found, cx, cy, area, food = obs
        
        info = {
            'Universal step': self.universal_step,
            'Step in Episode': self.step_in_episode,
            # 'collision': self.collision,
            'x pos': self.current_position.x,
            'y pos': self.current_position.y,
            'green_found': green_found,
            'cx': cx,
            'cy': cy,
            'area': area,
            'food': food,
            # 'BR': obs[0] * self.max_sensor_val,
            # 'BL': obs[1] * self.max_sensor_val,
            # 'FR2': obs[2] * self.max_sensor_val,
            # 'FL2': obs[3] * self.max_sensor_val,
            # 'FM': obs[4] * self.max_sensor_val,
            # 'BL1': obs[5] * self.max_sensor_val,
            # 'BM': obs[6] * self.max_sensor_val,
            # 'FR1': obs[7] * self.max_sensor_val,
            'ls': self.current_action[0] * 100,
            'rs': self.current_action[1] * 100,
            # 'proximity_penalty': self.proximity_penalty,
            'forward_speed_reward': self.forward_speed_reward_, 
            'Final reward': reward}

        return info

    
    def linear_normalize_sensors(self, sensor_values):
        min_val = 0
        max_val = self.max_sensor_val
        sensor_values = np.clip(sensor_values, min_val, max_val)
        sensor_values = (sensor_values - min_val) / (max_val - min_val)
        return sensor_values

    
    # returns IR sensor data
    def get_obs(self):
        front_image = self.robobo.read_image_front()
        frame, green_found, center_x, center_y, area = self.detect_green_object(front_image)
        
        self.nr_collected_food = self.robobo.get_nr_food_collected()

        if green_found:
            green_found = 1.0
        else:
            green_found = 0.0

        center_x_norm = center_x / self.image_w_h # 512 is the amount of pixels
        center_y_norm = center_y / self.image_w_h
        area_norm = area / (self.image_w_h * self.image_w_h)
        nr_collected_food_norm = self.nr_collected_food /self.total_count_food
        
        obs = np.array([green_found, center_x_norm, center_y_norm, area_norm, nr_collected_food_norm]).astype(np.float32)
        return obs

    def collision_detection(self, ir_values):
        max_sensor = np.max(ir_values)

        if max_sensor >= self.collision_threshold:
            self.collision = True
            # print('')
            # print('ROBOT BUMBED INTO SOMETHING!!!')
            # print('')
        else:
            self.collision = False


    # def forward_speed_reward(self):
    #     left_speed = self.current_action[0] 
    #     right_speed = self.current_action[1]
        
    #     if not self.collision:
    #         # Calculate base speed reward
    #         if (left_speed > 0 and right_speed > 0):  # Both forward
    #             base_reward = (left_speed + right_speed) / 2
    #         elif (left_speed < 0 and right_speed < 0):  # Both backward
    #             base_reward = abs((left_speed + right_speed) / 2) * 0.1
    #         else:
    #             return 0  # Different directions = no reward
            
    #         speed_difference = abs(left_speed - right_speed)
    #         coordination_factor = max(0, 1 - speed_difference)  # 1 when same speed, 0 when very different
            
    #         return base_reward * coordination_factor
        
    #     return 0

    def forward_speed_reward(self):
        left_speed = self.current_action[0] 
        right_speed = self.current_action[1]
        
        
        # Calculate base speed reward
        if (left_speed > 0 and right_speed > 0):  # Both forward
            base_reward = (left_speed + right_speed) / 2
        elif (left_speed < 0 and right_speed < 0):  # Both backward
            base_reward = abs((left_speed + right_speed) / 2) * 0.1
        else:
            return 0  # Different directions = no reward
        
        speed_difference = abs(left_speed - right_speed)
        coordination_factor = max(0, 1 - speed_difference)  # 1 when same speed, 0 when very different
        
        return base_reward * coordination_factor
        
        # return 0


    def get_reward(self, obs, left_speed, right_speed):
        green_found, cx, cy, area, food = obs
        reward = 0
        alignment_reward = 1.0 - abs(cx - 0.5) * 2

        self.forward_speed_reward_ = self.forward_speed_reward()
        
        if green_found == 1:
            # reward += 0.2
            reward += self.forward_speed_reward_ * self.weight_dict['forward_speed_weight']
            reward += alignment_reward * self.weight_dict['alignment_weight']
        # else:
        #     turning = abs(left_speed - right_speed)
        #     if turning > 0.3:
        #         reward += 0.5
            # turning = abs(left_speed - right_speed)
            # if turning > 0.2:
            #     reward += 0.1

        reward += area * self.weight_dict['area_weight']
        reward += (self.nr_collected_food - self.prev_nr_collected_food) * self.weight_dict['food_weight']

        return reward



    def terminate(self):
        if self.nr_collected_food == 7:
            return True
        elif self.step_in_episode == self.max_steps_in_episode:
            return True
        else:
            return False
    
    def step(self, action):
        # Rescale from [-1, 1] to actual motor speeds, e.g. [-100, 100]
        action = np.array(action, dtype=np.float32)
        # action = np.asarray(action, dtype=np.float32)
        # action = np.random.uniform(low=-1.0, high=1.0, size=2).astype(np.float32)
        
        self.step_in_episode += 1
        self.universal_step += 1
        # print(f'Universal step: {self.universal_step}')
        # print(f'Step in Episode {self.step_in_episode}')

        self.current_action = action


        max_speed = 100
        left_speed = np.round(action[0] * max_speed, decimals=2)
        
        right_speed = np.round(action[1] * max_speed, decimals=2)

        # print(f'left speed {left_speed}')
        # print(f'right speed {right_speed}')

        # Send to simulator or hardware
        # self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)
        self.robobo.move(left_speed, right_speed, self.timestep_duration)
        # self.robobo.move_blocking(-25, 24, self.timestep_duration)

        self.current_position = self.robobo.get_position()

        # Get new obs
        self.prev_nr_collected_food = self.nr_collected_food
        observation = self.get_obs()

        # self.collision_detection(observation[:8])  # Fix: 8 sensors not 7

        reward = self.get_reward(observation, action[0], action[1])
        done = self.terminate()
        info = self.get_info(observation, reward)
        
        # for key, value in info.items():
        #     print(f"{key}: {value}")
        # print('')


        # save info
        row = [info['Universal step'], 
               info['Step in Episode'], 
            #    info['collision'], 
               info['x pos'], 
               info['y pos'],
                info['green_found'],
                info['cx'],
                info['cy'],
                info['area'],
                info['food'],
            #    info['BR'], 
            #    info['BL'], 
            #    info['FR2'], 
            #    info['FL2'], 
            #    info['FM'], 
            #    info['BL1'], 
            #    info['BM'], 
            #    info['FR1'], 
               info['ls'], 
               info['rs'], 
            #    info['proximity_penalty'], 
               info['forward_speed_reward'], 
               info['Final reward']]
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        return observation, reward, done, False, info


    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        self.robobo.stop_simulation()
        self.robobo.play_simulation()

        self.nr_collected_food = 0
        self.prev_nr_collected_food = 0

        self.robobo.set_phone_tilt(100,100)
        self.robobo.set_phone_pan(180,100)
        # self.robobo.set_phone_pan(150,100)
        #     self, tilt_position: int, tilt_speed: int, blockid: Optional[int] = None
        # ) -> int:


        # self.proximity_penalty = 0
        self.forward_speed_reward_ = 0
        
        self.current_position = self.robobo.get_position()

        # self.collision = False
        # self.previous_action = np.array([0,0])
        # self.current_action = np.array([0,0])

        self.previous_position = self.robobo.get_position() # set the same as the current position at the start 
        self.current_position = self.robobo.get_position()

        self.step_in_episode = 0

        # Get initial obs
        obs = self.get_obs()

        return obs, {}
