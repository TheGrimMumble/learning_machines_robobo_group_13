from .RoboboGymEnv_task1_sim import RoboboGymEnv
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

    rob = SimulationRobobo(identifier=0)

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
        n_steps=512,
        n_epochs=64)
    
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


def inference(
        rob:SimulationRobobo,
        path: str
        ):
    rob = SimulationRobobo(identifier=0)
    env = RoboboGymEnv(rob)
    env.max_steps_in_episode = 256
    model = PPO.load(path, env=env)

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, _bool, info = env.step(action)
