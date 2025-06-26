from stable_baselines3 import PPO
# from learning_machines.RoboboGymEnv_task1_sim import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V2 import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V3 import RoboboGymEnv
from learning_machines.RoboboGymEnv_task1_sim_V4 import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task2_sim_V2 import RoboboGymEnv
from robobo_interface import SimulationRobobo
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import gymnasium as gym
import time
from stable_baselines3 import SAC
# from learning_machines.RoboboGymEnv_task3_sim_V1 import RoboboGymEnv

def test_model(rob):
    # Load the trained model
    # model_path = "/root/results/ppo_72000_task2_V1_3_continued.zip"
    model_path = "/root/results/ppo_110000_V4_continued.zip"
    model = PPO.load(model_path)

    # Set up the robot and environment
    # rob = SimulationRobobo()
    env = RoboboGymEnv(rob)

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        print(f"Action: {action}")

        # time.sleep(.2)

        # Force action to test if sim responds
        # action = np.array([0.4, -0.3], dtype=np.float32)

        obs, reward, done, truncated, info = env.step(action)
        
        # for key, value in info.items():
        #     if key == 'rs' or 'ls':
        #         print(f"{key}: {value}")
                    
        print('')
        # print(f"Reward: {reward}, Info: {info}")
        # time.sleep(.2)

    print("Test episode finished.")

def test_model_task3(rob, model_path="/root/results/SAC_task3_1000.zip"):
    from stable_baselines3 import SAC
    from learning_machines.RoboboGymEnv_task3_sim_V1 import RoboboGymEnv

    # Use the given rob instance
    env = RoboboGymEnv(rob)

    # Load the trained model
    model = SAC.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Action: {action}")
        obs, reward, done, truncated, info = env.step(action)

        # Optionally print sensor data
        for key, value in info.items():
            if key in ['rs', 'ls']:
                print(f"{key}: {value}")

    env.close()

