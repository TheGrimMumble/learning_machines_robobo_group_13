from .RoboboGymEnv_task2_hardw import RoboboGymEnv
import time
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
    Position,
    Orientation,
    SimulationRobobo,
    HardwareRobobo,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv





def hw_inference(
        rob:HardwareRobobo,
        policy,
        training_steps,
        version
        ):

    path = f"/root/results/{policy}_{training_steps}_{version}"
    env = RoboboGymEnv(rob)

    n_steps = 512 #  <------------------------

    env.max_steps_in_episode = n_steps
    model = PPO.load(path, env=env)

    obs, _ = env.reset()
    done = False

    rob.set_phone_pan_blocking(177, 100)
    rob.set_phone_tilt_blocking(100, 100)
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _bool, info = env.step(action)

    

            


