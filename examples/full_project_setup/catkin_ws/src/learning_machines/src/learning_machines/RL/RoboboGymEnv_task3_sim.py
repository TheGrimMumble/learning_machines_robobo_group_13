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
        self.robobo.stop_simulation()
        time.sleep(0.5)
        self.robobo.play_simulation()
        time.sleep(0.5)
        self.robobo.set_phone_tilt(109, 100)

        # Left and right wheel, needs rescaling
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=np.array([0]*8 + [-1]*2 + [0] + [-1] + [0] + [0] + [-1] + [0]*2 + [0]), # irs, wheels, green, red, steps
            high=np.array([1]*8 + [1]*2 + [1]*3 + [1]*4 + [1]), # REMEMBER TO IMPLEMENT ALL STATES
            dtype=np.float32)       
        
        self.calibrate_irs = np.array([6, 6, 59, 59, 5, 5, 57, 5])
        self.irs_max = np.array([1_000, 1_000, 10_000, 10_000, 10_000, 445, 1_000, 445])
        self.previous_action = np.array([0,0], dtype=np.float32)
        self.previous_orientation_distance = {"food": None, "base": None}

        # The duration of a step in miliseconds, so each step takes half a second
        self.timestep_duration = 100

        self.step_in_episode = 0
        self.global_step = 0

        self.steps_to_red = None
        self.red_alpha = 0.05
        self.red_found = 0
        self.red_lost = 0
        self.red_captured = 0
        self.red_uncaptured = 0

        self.steps_to_green = 0
        self.green_found = 0
        self.green_lost = 0

        # The max amount of steps the robot can take per episode
        self.max_steps_in_episode = 256*2
        self.first_step = True
        self.collision_count = 0

        self.vision_size = 256
        self.prev_obs = np.array([0]*18, dtype=np.float32)
        self.terminal_state = False
        self.steps_since_red_captured = 0

        self.green_lower = np.array([50, 100, 100])
        self.green_upper = np.array([80, 255, 255])

        # Red hue wraps around the HSV range, so we define two ends of the spectrum
        self.red_lower1 = np.array([0, 70, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 70, 50])
        self.red_upper2 = np.array([180, 255, 255])

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


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
            "gXali",
            "garea",
            "red",
            "rXali",
            "rarea",
            "rcapt",
            "step",
            "Rwrd"
            ]
        self.prnt_frmt = [
            "| ", "", "",
            "|FL1 ", "", "", "", "",
            "FR1| ", "",
            "| ", "", "",
            "| ", "", "", "",
            "| ",
            "| "
            ]
        self.notes = ""
    

    def detect_helper(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        in_sight = False
        center_x_alignment = 0.0
        area = 0.0
        captured = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            raw_area = cv2.contourArea(largest)
            area = raw_area / (self.vision_size**2)
            if raw_area > (self.vision_size**2 * 0.001):
                in_sight = True
                x, y, w, h = cv2.boundingRect(largest)
                center_x = (x + w*0.5) / self.vision_size
                center_x_alignment = (center_x - 0.5) * 2
                lowest_y = (y + h) / self.vision_size
                if (-0.05 < center_x_alignment < 0.05) and (lowest_y > 0.9):
                    captured = True

        return [in_sight, center_x_alignment, area, captured]


    def detect_objects(self, image):
        frame = cv2.resize(image, (self.vision_size, self.vision_size))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        red_mask = cv2.inRange(hsv, self.red_lower1, self.red_upper1) | cv2.inRange(hsv, self.red_lower2, self.red_upper2)

        green_found, green_cx, green_area, _ = self.detect_helper(green_mask)
        red_found, red_cx, red_area, red_captured = self.detect_helper(red_mask)

        return [green_found, green_cx, green_area, red_found, red_cx, red_area, red_captured]


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
            "gXali": obs[11],
            "garea": obs[12],
            "red": obs[13],
            "rXali": obs[14],
            "rarea": obs[15],
            "rcapt": obs[16],
            "step": obs[17],
            "Rwrd": reward
            }
        return info
    
    
    def linear_normalize(self, raw_ir):
        raw_ir -= self.calibrate_irs
        clipped_ir = np.clip(raw_ir, 1, self.irs_max)
        norm_ir = clipped_ir / self.irs_max
        return norm_ir


    def get_obs(self, action):
        raw_ir = np.array(self.robobo.read_irs(), dtype=np.float32)
        image = self.robobo.read_image_front()

        ir_sensor_data = self.linear_normalize(raw_ir)
        vision = np.array(self.detect_objects(image), dtype=np.float32)
        steps = self.step_in_episode / self.max_steps_in_episode
        steps_norm = np.array([steps], dtype=np.float32)

        obs = np.concatenate([ir_sensor_data, action, vision, steps_norm]).astype(np.float32)
        return obs
    

    def punish_proximity(self, irs):
        reward = 0
        threshold = 0.7
        if np.max(irs) > threshold:
            self.notes += "Crash! "
            self.collision_count += 1
            reward += 5
        elif np.max(irs) > 0.2:
            self.notes += "Close! "
            reward += np.max(irs)*3
        return float(reward)


    def orientation_distance_reward(self, choice: str):
        assert choice in ["food", "base"]
        reward = 0

        if choice == "food":
            info = self.robobo.get_food_position()
        elif choice == "base":
            info = self.robobo.get_base_position()
        pos_triangle = [self.robobo.get_position(), self.robobo.get_LW_position(), self.robobo.get_RW_position(), self.robobo.get_BS_position()]
        
        dist_from_triangle = []
        for pos in pos_triangle:
            delta = np.array([info.x - pos.x, info.y - pos.y])
            distance = np.linalg.norm(delta)
            dist_from_triangle.append(distance)

        dist_rob, left, right, stub = dist_from_triangle
        previous_dist_rob = self.previous_orientation_distance[choice]

        direction = np.clip((left - right) / 0.145, -1, 1) # positive when to the right, negative when to the left
        left_right_mean = (left + right) / 2

        if stub > left_right_mean:
            diff = previous_dist_rob - dist_rob if previous_dist_rob else 0
            if diff > 0.005:
                reward += diff * 50
            reward -= abs(direction / 2)
        else:
            reward -= 1 - abs(direction / 2)
        self.previous_orientation_distance[choice] = dist_rob

        return float(reward)


    def get_reward(self, obs):
        reward = 0
        irs = obs[:8]
        wheels = obs[8:10]
        green_in_sight = obs[10]
        green_cx = obs[11]
        green_area = obs[12]
        red_in_sight = obs[13]
        red_cx = obs[14]
        red_area = obs[15]
        red_captured = obs[16]
        steps = obs[17]


        if self.terminal_state:
            self.steps_to_green = self.step_in_episode
            return float(100)

        ori_dist_rwrd = 0

        if red_captured:

            if not self.prev_obs[16]:
                self.steps_since_red_captured = 0
                reward += 20
                self.notes += "Rd cptrd! "
                self.steps_to_red = (
                    self.steps_to_red * (1 - self.red_alpha)) + (
                    self.step_in_episode * self.red_alpha
                    ) if self.steps_to_red else self.step_in_episode
                self.red_captured += 1
            else:
                ori_dist_rwrd = self.orientation_distance_reward("base")
                reward += 1
                self.notes += "still cptrd! "
                self.steps_since_red_captured += 1

            if green_in_sight:
                if not self.prev_obs[10]:
                    reward += 8
                    self.notes += "Gr lctd! "
                    self.green_found += 1
                else:
                    ori_dist_rwrd = self.orientation_distance_reward("base")
            else:
                if self.prev_obs[10]:
                    reward -= 10
                    self.notes += "Gr lost! "
                    self.green_lost += 1


        else:
            
            self.steps_since_red_captured += 1

            if self.prev_obs[16]:
                reward -= 10
                self.notes += "Rd Uncptrd! "
                self.red_uncaptured += 1
            else:
                ori_dist_rwrd = self.orientation_distance_reward("food")

            if red_in_sight:
                if not self.prev_obs[13]:
                    reward += 8
                    self.notes += "Rd lctd! "
                    self.red_found += 1
                else:
                    ori_dist_rwrd = self.orientation_distance_reward("food")
            else:
                if self.prev_obs[13]:
                    reward -= 10
                    self.notes += "Rd lost! "
                    self.red_lost += 1
                
                
        reward -= self.punish_proximity(irs)
        reward += ori_dist_rwrd

        # self.notes += f"#: {self.steps_since_red_captured}"
        # reward -= self.steps_since_red_captured * 0.05
        # reward -= steps
        return float(reward)
    

    def terminate(self):
        if  self.step_in_episode == self.max_steps_in_episode or \
            self.terminal_state:
            return True
        else:
            return False


    def step(self, action):
        left_just = 7
        if self.first_step:
            print("Stp#".ljust(left_just) + "".join(
                [self.prnt_frmt[i] + k.ljust(left_just) for i, k in enumerate(self.prnt_pos)]
                ) + "| Notes")
            self.first_step = False
        # Rescale from [-1, 1] to actual motor speeds, e.g. [-100, 100]
        action = np.array(action, dtype=np.float32)
        self.step_in_episode += 1
        self.global_step += 1
        max_speed = 100
        left_speed = action[0] * max_speed
        right_speed = action[1] * max_speed

        # Send to simulator or hardware
        self.robobo.move_blocking(left_speed, right_speed, self.timestep_duration)
        
        # Get new obs 
        observation = self.get_obs(action)
        self.terminal_state = self.robobo.base_detects_food()
        reward = self.get_reward(observation)
        self.previous_action = action
        done = self.terminate()
        info = self.get_info(observation, reward)

        print(str(self.global_step).ljust(left_just) + "".join(
            [self.prnt_frmt[i] + f"{info[v]:.3f}".ljust(left_just) for i, v in enumerate(self.prnt_pos)]
            ) + "| " + self.notes)
        self.notes = ""
        self.prev_obs = observation
    
        return observation, reward, done, False, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Sets random seed (Gym requirement)

        # Reset simulator, not sure yet whether we should stop and start the simulator
        # for each episode, but it seemed like the savest option to start with
        self.robobo.stop_simulation()
        time.sleep(0.5)
        self.robobo.play_simulation()
        time.sleep(0.5)
        self.robobo.set_phone_tilt(109, 100)

        self.step_in_episode = 0

        # Wait until sim is fully reset and ready (you can insert a short sleep here if needed)
        # time.sleep(0.2)

        # Get initial obs
        obs = self.get_obs(self.previous_action)

        return obs, {}

