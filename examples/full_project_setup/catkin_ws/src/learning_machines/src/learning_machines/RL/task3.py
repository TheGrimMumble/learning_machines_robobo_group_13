from .RoboboGymEnv_task3_sim import RoboboGymEnv
import time
import csv
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    Position,
    Orientation,
    SimulationRobobo,
    HardwareRobobo,
)



def make_env(rob):
    return lambda: RoboboGymEnv(rob)

# Initialize your robot interface (assuming it's simulation for now)
# from robobo_interface import SimulationRobobo     

def get_flat_params(model):
    return torch.cat([param.data.view(-1) for param in model.policy.parameters()])


def train_model(
        rob:SimulationRobobo,
        total_time_steps = 128,
        policy = 'ppo',
        version = 'test',
        multiproc = None,
        debug = True
        ):

    # Create the environment
    if multiproc:
        env_fns = [make_env(rob) for _ in range(multiproc)]
        env = SubprocVecEnv(env_fns)
    else:
        env = RoboboGymEnv(rob)

    # Check if the environment follows Gym API properly
    if debug:
        check_env(env, warn=True)

    # Define the PPO model
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.001, # default: 0.0003
        verbose=1,
        n_steps=512, #  <------------------------
        n_epochs=8) #  <------------------------
    
    initial_params = get_flat_params(model).clone()

    # Train the model
    rob.set_phone_tilt_blocking(109, 100)
    model.learn(total_timesteps=total_time_steps)

    updated_params = get_flat_params(model)

    if torch.equal(initial_params, updated_params):
        print("❌ Parameters did not change.")
    else:
        print("✅ Parameters updated.")

    # Save the model
    model.save(f"/root/results/{policy}_{total_time_steps}_{version}")

    # # Test the model
    # obs, _ = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = env.step(action)
    #     if done or truncated:
    #         obs, _ = env.reset()
    return model, env


def continue_training(
        rob:SimulationRobobo,
        path: str,
        multiproc = None
        ):
    if multiproc:
        env_fns = [make_env(rob) for _ in range(multiproc)]
        env = SubprocVecEnv(env_fns)
    else:
        env = RoboboGymEnv(rob)
    model = PPO.load(path, env=env)
    rob.set_phone_tilt_blocking(109, 100)
    return model, env


def inference(
        rob:SimulationRobobo,
        policy,
        training_steps,
        version,
        print_to_csv=True
        ):

    path = f"/root/results/{policy}_{training_steps}_{version}"
    env = RoboboGymEnv(rob)

    n_steps = 256 #  <-------------  validation trajectory length

    env.max_steps_in_episode = n_steps
    model = PPO.load(path, env=env)
    rob.set_phone_tilt_blocking(109, 100)

    obs, _ = env.reset()
    done = False
    left_speeds = []
    right_speeds = []
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # print("action:", action, "shape:", getattr(action, "shape", None))

        max_speed = 100
        left_speed = action[0] * max_speed
        right_speed = action[1] * max_speed
        left_speeds.append(left_speed)
        right_speeds.append(right_speed)

        obs, reward, done, _bool, info = env.step(action)

        rewards.append(reward)

    left_mean_speed = sum(left_speeds) / n_steps
    right_mean_speed = sum(right_speeds) / n_steps
    mean_reward = sum(rewards) / n_steps
    
    if print_to_csv:
        with open(f"/root/results/{policy}_{version}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([training_steps,
                            left_mean_speed,
                            right_mean_speed,
                            env.collision_count,
                            env.steps_to_red,
                            env.red_found,
                            env.red_lost,
                            env.red_captured,
                            env.red_uncaptured,
                            env.steps_to_green,
                            env.green_found,
                            env.green_lost,
                            mean_reward])
            

def calibrate_camera(robobo,
                     vision_size,
                     kernel,
                     blue_lower=np.array([90, 50, 50]),
                     blue_upper=np.array([130, 255, 255]),
                     pan_gain=0.25,
                     tilt_gain=0.25,
                     max_attempts=24):
    """
    Pan/tilt until two blue bars are centered in the lower-middle.
    """
    robobo.set_phone_tilt_blocking(109, 100) # max
    robobo.set_phone_pan_blocking(11, 100) # default: 179
    for _ in range(max_attempts):
        # 1) grab & preprocess
        frame = robobo.read_image_front()
        frame = cv2.resize(frame, (vision_size, vision_size))
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2) blue mask + clean
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3) find blue contours and pick the two tallest bars
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h * w > 0.01 * vision_size**2:
                bars.append((x, y, w, h))
        if len(bars) < 2:
            tilt = 332 // max_attempts
            tilt_now = robobo.read_phone_pan()
            robobo.set_phone_pan_blocking(min(343, tilt_now + tilt), 100)
            time.sleep(0.1)
            continue

        # keep the two tallest bars
        bars = sorted(bars, key=lambda b: b[3], reverse=True)[:2]
        centers = [(x + w/2, y + h/2) for x, y, w, h in bars]
        avg_cx  = sum(c[0] for c in centers) / 2.0
        avg_cy  = sum(c[1] for c in centers) / 2.0

        # compute offsets to target
        target_cx = vision_size / 2 # want the blue bars centered
        target_cy = vision_size * 0.90 # want the blue bars taking up 10% of vertical space
        dx = avg_cx - target_cx
        dy = avg_cy - target_cy

        # check success
        # print("dx", dx)
        if abs(dx) > 2:
            pan  =  max(2, int(round(pan_gain * dx)))
            # print("pan", pan)
            pan_now = robobo.read_phone_pan()
            robobo.set_phone_pan_blocking(min(343, pan_now + pan), 100)
            time.sleep(0.1)
        elif abs(dy) > 10:
            # print('dy', dy)
            tilt =  max(2, int(round(tilt_gain * dy)))
            # print("tilt", tilt)
            tilt_now = robobo.read_phone_tilt()
            robobo.set_phone_tilt_blocking(min(109, tilt_now + tilt), 100)
            time.sleep(0.1)
        else:
            return True

    return False


def print_irs():
    prnt_pos = [
        "irBL",
        "irBM",
        "irBR",
        "irFL1",
        "irFL2",
        "irFM",
        "irFR2",
        "irFR1"
        ]
    prnt_frmt = [
        "|   ", "", "",
        "|FL1   ", "", "", "", "",
        "FR1|"
        ]
    left_just = 10
    print("Stp#".ljust(left_just) + "".join(
        [prnt_frmt[i] + k.ljust(left_just) for i, k in enumerate(prnt_pos)]
        ))
    global_step = 1
    while not rob.base_detects_food():
        obs = rob.read_irs()
        info = {
            "irBL": obs[1],
            "irBM": obs[6],
            "irBR": obs[0],
            "irFL1": obs[5],
            "irFL2": obs[3],
            "irFM": obs[4],
            "irFR2": obs[2],
            "irFR1": obs[7]
            }
        print(str(global_step).ljust(left_just) + "".join(
            [prnt_frmt[i] + str(round(info[v], 4)).ljust(left_just) for i, v in enumerate(prnt_pos)]
            ))
        global_step += 1

