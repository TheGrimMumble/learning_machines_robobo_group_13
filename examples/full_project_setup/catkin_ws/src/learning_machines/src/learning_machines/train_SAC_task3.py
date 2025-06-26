import gymnasium as gym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from learning_machines.RoboboGymEnv_task3_sim_V1 import RoboboGymEnv  # adjust import path
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

def train_model_task3(rob: SimulationRobobo):
    timesteps = 1000
    # Create the log directory if it doesn't exist
    log_dir = "/root/results/robobo_task3_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    rob = SimulationRobobo()
    env = RoboboGymEnv(rob)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=60_000,         # size of replay buffer
        learning_starts=10_000,        # delay training until this many steps
        batch_size=256,
        tau=0.005,                     # target smoothing coefficient
        gamma=0.99,
        train_freq=1,                  # update frequency
        gradient_steps=1,              # number of gradient steps after each rollout
        ent_coef="auto",               # entropy coefficient
    )

    # Train the model
    model.learn(total_timesteps=timesteps)

    # Save model
    model.save(f"/root/results/SAC_task3_{timesteps}")
