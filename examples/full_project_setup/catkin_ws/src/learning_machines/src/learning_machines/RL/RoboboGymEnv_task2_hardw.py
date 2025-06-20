import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import time

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
        self.robobo.set_phone_pan_blocking(177, 100)
        self.robobo.set_phone_tilt_blocking(100, 100)

        # Left and right wheel, needs rescaling
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 7 IR sensors are assumed to be between 0 and 1
        # Speed is assumed to be between -1 and 1
        # obeservation space looks like this: low: [0,0,0,0,0,0,0,-1,-1], high: [1,1,1,1,1,1,1,1,1]
        self.observation_space = spaces.Box(
            low=np.array([0]*8 + [-1]*2 + [0]*3),
            high=np.array([1]*8 + [1]*2 + [1]*3),
            dtype=np.float32)       
        
        # assuming that this is the value at which it would bumb into an object, but is not based on any testing
        self.IR_threshold_val = 50
        self.IR_range_max = 1_000
        self.previous_action = np.array([0,0], dtype=np.float32)

        # The duration of a step in miliseconds, so each step takes half a second
        self.timestep_duration = 150

        self.step_in_episode = 0
        self.global_step = 0
        self.steps_to_findall = 0

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 256*2
        self.origin_buffer_size = 10
        self.first_step = True
        self.dist_from_origin_buffer = self.reset_origin_buffer()
        self.close_call_count = 0
        self.collision_count = 0

        self.vision_size = 256
        self.total_count_food = 7
        # self.nr_collected_food = 0
        self.prev_nr_collected_food = 0

        self.green_lower = np.array([50, 100, 100])
        self.green_upper = np.array([80, 255, 255])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        self.prev_area = 0
        self.steps_since_green_found = 0
        self.steps_since_food_found = 0

        self.prnt_pos = [
            "irBL",
            "irBM",
            "irBR",
            "irFL1",
            "irFL2",
            "irFM",
            "irFR2",
            "irFR1",
            "wsL",
            "wsR",
            "green",
            "posX",
            "area",
            "Rwrd"
            ]
        self.prnt_frmt = [
            "|   ", "", "",
            "|FL1   ", "", "", "", "",
            "FR1|   ", "",
            "|   ", "", "",
            "|   "
            ]
        self.notes = ""
        
    def reset_origin_buffer(self):
        return [np.array([0, 0], dtype=np.float32)] * int(self.origin_buffer_size)
    
    # It would be nice to randomize this at the start of each episode
    # But for now it seems like this is determined by the scene that we start before running the code
    def _set_robot_initial_position(self):
        pass

    def detect_green_object(self, image):
        # 1. Resize ONCE to vision_size × vision_size
        frame = cv2.resize(image, (self.vision_size, self.vision_size))

        # 2. Convert to HSV (plain numpy array)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Mask out green in one go
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # 5. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        green_found = False
        center_x = 0.5
        area = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            raw_area = cv2.contourArea(largest)
            if raw_area > 75:
                green_found = True
                x, y, w, h = cv2.boundingRect(largest)
                # divide by current frame width, i.e. vision_size
                center_x = (x + w*0.5) / self.vision_size
            # normalize area to [0,1] relative to vision_size²
            area = raw_area / (self.vision_size * self.vision_size)

        return [green_found, center_x, area]


    def get_info(self, obs, reward):
        info = {
            "irBL": obs[1],
            "irBM": obs[6],
            "irBR": obs[0],
            "irFL1": obs[5],
            "irFL2": obs[3],
            "irFM": obs[4],
            "irFR2": obs[2],
            "irFR1": obs[7],
            "wsL": obs[8],
            "wsR": obs[9],
            "green": obs[10],
            "posX": obs[11],
            "area": obs[12],
            "Rwrd": reward
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
        calibrate = np.array([0, 0, 45, 45, 0, -30, 50, -30])
        raw_ir -= calibrate
        raw_ir = np.clip(raw_ir, 1, self.IR_range_max)
        raw_ir /= self.IR_range_max
        return raw_ir

    # returns IR sensor data
    def get_obs(self, action):
        raw_ir = np.array(self.robobo.read_irs(), dtype=np.float32)
        image = self.robobo.read_image_front()
        ir_sensor_data = self.linear_normalize(raw_ir)
        vision = np.array(self.detect_green_object(image), dtype=np.float32)
        obs = np.concatenate([ir_sensor_data, action, vision]).astype(np.float32)
        return obs
    
    def collision(self, obs):
        irs = obs[:8]
        if np.max(irs) > 0.5:
            self.notes = " Collision!"
            self.collision_count += 1
            return True
        return False
    
    def close(self, obs):
        irs = obs[:8]
        if np.max(irs) > 0.1:
            self.notes = " Too Close!"
            self.close_call_count += 1
            return True
        return False
    
    def punish_proximity(self, obs):
        irs = obs[:8]
        reward = 0
        if np.max(irs) > 0.1:
            reward += np.max(irs) # (max(irs) - 0.75) * 100
        return float(reward)
    
    def dist_from_origin_reward(self):
        coordinates = np.sum(self.dist_from_origin_buffer, axis=0)
        # print(coordinates)
        y, x = coordinates
        distance = np.sqrt(y**2 + x**2)

        angle = np.arctan2(y, x)
        ideal_angle = -np.pi / 4  # y = x
        angle_diff = np.abs(np.arctan2(
            np.sin(angle - ideal_angle),
            np.cos(angle - ideal_angle)
            ))
        direction_score = np.abs(np.cos(angle_diff))  # 1 = perfect, -1 = opposite
        
        reward = (distance * direction_score) / self.origin_buffer_size

        return float(reward)
    
    def dont_wiggle_reward(self, obs):
        action_now = np.array([obs[8], obs[9]], dtype=np.float32)

        diff = np.abs(self.previous_action - action_now)
        total_diff = np.sum(diff)

        return float((4 - total_diff) / 4)
    

    def get_reward(self, obs):
        reward = 0
        green_found = obs[10]
        food_found = False
        # food_found = self.nr_collected_food - self.prev_nr_collected_food

        if food_found > 0:
            self.steps_since_green_found = 0
            self.steps_since_food_found = 0
            reward += food_found * 10
            self.notes = " Found food!"

        elif green_found:
            self.steps_since_green_found = 0
            self.steps_since_food_found += 1
            alignment = 1.0 - abs(obs[11] - 0.5) * 2
            left_speed = obs[8]
            right_speed = obs[9]
            if left_speed > 0 and right_speed > 0:
                reward += alignment * left_speed * right_speed * 2
            reward += (obs[12] - self.prev_area) * 20

        elif self.steps_since_green_found > 2:
            self.steps_since_green_found += 1
            self.steps_since_food_found += 1
            if self.collision(obs):
                self.dist_from_origin_buffer = self.reset_origin_buffer()
            elif self.close(obs):
                index = int(np.round(np.max(obs[:8]) * 2 * self.origin_buffer_size))
                self.dist_from_origin_buffer[:index] = [
                    np.array([0,0], dtype=np.float32) for _ in range(index)
                    ]
            reward += self.dist_from_origin_reward() * 0.1
            reward -= self.punish_proximity(obs) * 0.5
            reward += self.dont_wiggle_reward(obs) * 0.1
            reward -= self.steps_since_food_found * 0.1

        else:
            self.steps_since_green_found += 1

        return float(reward)

    
    def complex_speed_reward(self, obs):
        irs_back = [obs[o] for o in (0, 1, 6)]
        back = max(irs_back)
        irs_front = [obs[o] for o in (2, 3, 4, 5, 7)]
        front = max(irs_front)
        speed = sum([obs[8], obs[9]])
        multiplier = 5

        back_up = -1 * front * speed * multiplier * 0.5
        kickoff_from_wall = back * speed * multiplier
        full_speed_ahead = (front - 0.75) * -1 * speed * multiplier

        reward = sum([
            back_up,
            kickoff_from_wall,
            full_speed_ahead
            ])
        return float(reward)
    
    def straight_reward(self, obs):
        reward = 0
        speed_threshold = 0
        speed_diff = abs(obs[8] - obs[9])
        if obs[8] > speed_threshold \
        and obs[9] > speed_threshold \
        and 0 < speed_diff < 0.15:
            speed_diff_inv = 1 / abs(obs[8] - obs[9])
            reward += min(25, speed_diff_inv)
        return float(reward)
    
    def simple_speed_reward(self, obs):
        reward = 0
        speed_threshold = 0
        if obs[8] > speed_threshold and obs[9] > speed_threshold:
            reward += (obs[8] + obs[9]) * 10
        return float(reward)
    

    def terminate(self):
        if  self.step_in_episode == self.max_steps_in_episode: # or \
            # self.total_count_food <= self.nr_collected_food:
            return True
        else:
            return False

    def step(self, action):
        # self.nr_collected_food = self.robobo.get_nr_food_collected()
        left_just = 10
        if self.first_step:
            print("Stp#".ljust(left_just) + "".join(
                [self.prnt_frmt[i] + k.ljust(left_just) for i, k in enumerate(self.prnt_pos)]
                ) + "|   Notes")
            self.first_step = False
        # Rescale from [-1, 1] to actual motor speeds, e.g. [-100, 100]
        action = np.array(action, dtype=np.float32)
        self.step_in_episode += 1
        self.global_step += 1
        self.dist_from_origin_buffer.append(np.clip(action, -0.5, 0.5))
        self.dist_from_origin_buffer.pop(0)
        max_speed = 100
        left_speed = int(round(action[0] * max_speed))
        right_speed = int(round(action[1] * max_speed))

        # Send to simulator or hardware
        self.robobo.move(left_speed, right_speed, self.timestep_duration)
        
        # Get new obs 
        observation = self.get_obs(action)
        reward = self.get_reward(observation)
        self.previous_action = action
        done = self.terminate()
        info = self.get_info(observation, reward)
        # self.prev_nr_collected_food = self.nr_collected_food

        print(str(self.global_step).ljust(left_just) + "".join(
            [self.prnt_frmt[i] + str(round(info[v], 4)).ljust(left_just) for i, v in enumerate(self.prnt_pos)]
            ) + "| " + self.notes)
        self.notes = ""
        self.steps_to_findall += 1
        self.prev_area = observation[12]
    
        return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        # Reset simulator, not sure yet whether we should stop and start the simulator
        # for each episode, but it seemed like the savest option to start with
        self.robobo.set_phone_pan_blocking(177, 100)
        self.robobo.set_phone_tilt_blocking(100, 100)

        self.step_in_episode = 0
        # self.nr_collected_food = 0
        self.prev_nr_collected_food = 0
        self.steps_to_findall = 0


        # Wait until sim is fully reset and ready (you can insert a short sleep here if needed)
        # time.sleep(0.2)

        # Get initial obs
        obs = self.get_obs(self.previous_action)

        return obs, {}

