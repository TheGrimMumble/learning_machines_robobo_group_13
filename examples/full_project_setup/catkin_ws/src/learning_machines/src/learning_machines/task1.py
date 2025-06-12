# from learning_machines.RoboboGymEnv_task1_sim import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V2 import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V3 import RoboboGymEnv
from learning_machines.RoboboGymEnv_task1_sim_V4 import RoboboGymEnv
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
    total_time_steps = 100000
    policy = 'ppo'
    version = 'V4'

    rob = SimulationRobobo()

    # Create the environment
    env = RoboboGymEnv(rob)

    # Define the PPO model
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.001,
        verbose=1,
        n_steps=512,
        n_epochs=10)

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

def continue_training(rob: SimulationRobobo):
    """Continue training an existing model"""
    
    additional_steps = 100000
    model_path = '/root/results/ppo_10000_V4.zip'  # Add .zip extension!
    
    # Create the environment
    env = RoboboGymEnv(rob)
    
    # Load the existing model (SIMPLE VERSION)
    model = PPO.load(model_path, env=env)  # Only pass env, no other params!
    
    # Continue training
    model.learn(total_timesteps=additional_steps)
    
    # Save with updated name
    model.save(f"/root/results/ppo_{additional_steps + 10000}_V4_continued")
    
    return model