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
    total_time_steps = 100
    policy = 'ppo'
    version = 'test'

    rob = SimulationRobobo()

    # Create the environment
    env = RoboboGymEnv(rob)

    # Check if the environment follows Gym API properly
    check_env(env, warn=True)

    # Define the PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        n_steps=100,
        n_epochs=1)

    # Train the model
    model.learn(total_timesteps=total_time_steps)

    # Save the model
    model.save(f"/root/results/{policy}_{total_time_steps}_{version}")

    # # Test the model
    # obs, _ = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = env.step(action)
    #     if done or truncated:
    #         obs, _ = env.reset()