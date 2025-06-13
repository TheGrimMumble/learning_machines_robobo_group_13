import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
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

class RobotEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=500, n_eval_episodes=2):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print('start test runs now')
            # Run evaluation episodes
            collision_counts = []
            rewards = []
            sensor_values = []
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_collisions = 0
                episode_sensors = []
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = self.eval_env.step(action)
                    episode_reward += reward
                    

                    if info.get('collision', False):
                        episode_collisions += 1
                    

                    max_sensor = max(obs[:8])  
                    episode_sensors.append(max_sensor)
                
                collision_counts.append(episode_collisions)
                rewards.append(episode_reward)
                sensor_values.append(np.mean(episode_sensors))
            

            avg_collisions = np.mean(collision_counts)
            avg_reward = np.mean(rewards)
            avg_sensor = np.mean(sensor_values)
            collision_rate = avg_collisions / 
            

            self.evaluations.append({
                'timestep': self.n_calls,
                'avg_reward': avg_reward,
                'avg_collisions': avg_collisions,
                'collision_rate': collision_rate,
                'avg_sensor_value': avg_sensor
            })
            
            print(f"\n=== EVALUATION at {self.n_calls} steps ===")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Avg Collisions: {avg_collisions:.1f}")
            print(f"Collision Rate: {collision_rate*100:.1f}%")
            print(f"Avg Sensor Value: {avg_sensor:.3f}")
            print("=" * 40)
            
        return True
    
    def save_evaluations(self, filepath):
        """Save evaluation data to CSV"""
        import pandas as pd
        if self.evaluations:
            df = pd.DataFrame(self.evaluations)
            df.to_csv(filepath, index=False)
            print(f"Evaluations saved to: {filepath}")

def train_model_callback(rob: SimulationRobobo):
    total_time_steps = 20000
    policy = 'ppo'
    version = 'V4_3'

    rob = SimulationRobobo()

    env = RoboboGymEnv(rob)
    
    eval_env = RoboboGymEnv(rob)
    
    eval_callback = RobotEvalCallback(
        eval_env=eval_env,
        eval_freq=1000, 
        n_eval_episodes=2  
    )

    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.001,
        verbose=1,
        n_steps=500,
        n_epochs=10)

    # Train the model WITH callback
    model.learn(total_timesteps=total_time_steps, callback=eval_callback)

    model.save(f"/root/results/{policy}_{total_time_steps}_{version}")
    
    eval_callback.save_evaluations(f"/root/results/evaluations_{version}.csv")
    
    if eval_callback.evaluations:
        print("\n=== TRAINING COMPLETE ===")
        final_eval = eval_callback.evaluations[-1]
        print(f"Final Avg Reward: {final_eval['avg_reward']:.2f}")
        print(f"Final Collision Rate: {final_eval['collision_rate']*100:.1f}%")