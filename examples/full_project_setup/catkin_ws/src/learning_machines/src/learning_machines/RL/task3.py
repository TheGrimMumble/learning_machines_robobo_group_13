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
            



