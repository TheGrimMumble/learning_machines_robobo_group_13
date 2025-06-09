from learning_machines.RoboboGymEnv_task1_sim import RoboboGymEnv
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


def train_model(rob:SimulationRobobo):
    rob = SimulationRobobo()

    # Create the environment
    env = RoboboGymEnv(rob)

    # Check if the environment follows Gym API properly
    check_env(env, warn=True)

    # Define the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=2000)

    # Save the model
    model.save("/root/results/ppo_robobo")

    # # Test the model
    # obs, _ = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = env.step(action)
    #     if done or truncated:
    #         obs, _ = env.reset()