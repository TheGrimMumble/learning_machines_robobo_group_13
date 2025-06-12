from .RoboboGymEnv_task1_sim import RoboboGymEnv
import csv
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
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

# Initialize your robot interface (assuming it's simulation for now)
# from robobo_interface import SimulationRobobo     

def get_flat_params(model):
    return torch.cat([param.data.view(-1) for param in model.policy.parameters()])


def train_model(
        rob:SimulationRobobo,
        total_time_steps = 128,
        policy = 'ppo',
        version = 'test',
        ):

    # Create the environment
    env = RoboboGymEnv(rob)

    # Check if the environment follows Gym API properly
    check_env(env, warn=True)

    # Define the PPO model
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.001, # default: 0.0003
        verbose=1,
        n_steps=64, #  <------------------------
        n_epochs=32)
    
    initial_params = get_flat_params(model).clone()

    # Train the model
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


def continue_training(
        rob:SimulationRobobo,
        path: str,
        total_time_steps = 128,
        policy = 'ppo',
        version = 'test',
        ):
    env = RoboboGymEnv(rob)
    model = PPO.load(path, env=env)
    model.learn(total_timesteps=total_time_steps)
    model.save(f"/root/results/{policy}_{total_time_steps}_{version}")


def inference(
        rob:SimulationRobobo,
        path: str,
        i
        ):
    env = RoboboGymEnv(rob)

    n_steps = 64 #  <------------------------

    env.max_steps_in_episode = n_steps
    model = PPO.load(path, env=env)

    obs, _ = env.reset()
    done = False
    left_speeds = []
    right_speeds = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        max_speed = 100
        left_speed = action[0] * max_speed
        right_speed = action[1] * max_speed
        left_speeds.append(left_speed)
        right_speeds.append(right_speed)

        obs, reward, done, _bool, info = env.step(action)

    left_mean_speed = sum(left_speeds) / n_steps
    right_mean_speed = sum(right_speeds) / n_steps
    
    with open(f"{path}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([n_steps * i,
                         left_mean_speed,
                         right_mean_speed,
                         env.close_call_count,
                         env.collision_count])

