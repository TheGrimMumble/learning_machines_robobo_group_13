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


def train_model(rob:SimulationRobobo):
    total_time_steps = 100000
    policy = 'ppo'
    version = 'V4'

    rob = SimulationRobobo()
    env = RoboboGymEnv(rob)

    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.001,
        verbose=1,
        n_steps=512,
        n_epochs=10)

    model.learn(total_timesteps=total_time_steps)
    model.save(f"/root/results/{policy}_{total_time_steps}_{version}")


def continue_training(rob: SimulationRobobo):
    
    additional_steps = 100000
    model_path = '/root/results/ppo_10000_V4.zip' 
    env = RoboboGymEnv(rob)
    model = PPO.load(model_path, env=env)  
    model.learn(total_timesteps=additional_steps)
    model.save(f"/root/results/ppo_{additional_steps + 10000}_V4_continued")
    
    return model