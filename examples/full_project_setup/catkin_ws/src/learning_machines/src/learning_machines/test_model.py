from stable_baselines3 import PPO
# from learning_machines.RoboboGymEnv_task1_sim import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V2 import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V3 import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V4 import RoboboGymEnv
from learning_machines.RoboboGymEnv_task2_sim_V1 import RoboboGymEnv
from robobo_interface import SimulationRobobo
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def test_model(rob):
    # Load the trained model
    model_path = "/root/results/ppo_7200_task2_V1.zip"
    model = PPO.load(model_path)

    # Set up the robot and environment
    # rob = SimulationRobobo()
    env = RoboboGymEnv(rob)

    obs, _ = env.reset()
    done = False

    while not done:
        # Predict action using the trained model
        action, _ = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()

        # Step in the environment
        obs, reward, done, truncated, info = env.step(action)

        # Optional: add a sleep delay to slow down visualization
        # time.sleep(0.1)

    print("Test episode finished.")
