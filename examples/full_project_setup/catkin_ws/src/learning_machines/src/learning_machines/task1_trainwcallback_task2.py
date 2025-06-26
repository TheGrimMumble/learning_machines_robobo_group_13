import numpy as np
import time
import os
from stable_baselines3.common.callbacks import BaseCallback
from learning_machines.RoboboGymEnv_task2_sim_V2 import RoboboGymEnv
from stable_baselines3 import PPO
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
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=2, model_save_dir=None):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations = []
        self.model_save_dir = model_save_dir
        if self.model_save_dir is not None:
            os.makedirs(self.model_save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.model_save_dir is not None:
                model_path = os.path.join(self.model_save_dir, f"model_{self.n_calls}_steps")
                self.model.save(model_path)
                print(f"Model checkpoint saved to {model_path}")

            total_rewards = []
            episode_lengths = []
            food_collected_counts = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                steps = 0
                food_collected = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = self.eval_env.step(action)
                    episode_reward += reward
                    steps += 1
                    food_collected = info.get('food', 0)
                    for key, value in info.items():
                        print(f"{key}: {value}")
                    
                    print('')
                    time.sleep(0.2)

                total_rewards.append(episode_reward)
                episode_lengths.append(steps)
                food_collected_counts.append(food_collected)

            avg_reward = np.mean(total_rewards)
            avg_steps = np.mean(episode_lengths)
            avg_food_collected = np.mean(food_collected_counts)

            self.evaluations.append({
                'timestep': self.n_calls,
                'avg_reward': avg_reward,
                'avg_episode_steps': avg_steps,
                'avg_food_collected': avg_food_collected
            })

            print(f"\n=== EVALUATION at {self.n_calls} steps ===")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Avg Steps: {avg_steps:.1f}")
            print(f"Avg Food Collected: {avg_food_collected:.1f}")
            print("=" * 40)

        return True

    def save_evaluations(self, filepath):
        import pandas as pd
        if self.evaluations:
            df = pd.DataFrame(self.evaluations)
            df.to_csv(filepath, index=False)
            print(f"Evaluations saved to: {filepath}")


def train_model_callback_task2(rob: SimulationRobobo):
    total_time_steps = 30000
    policy = 'ppo'
    version = 'task2_V2'
    save_dir = f"/root/results/{policy}_{version}"

    os.makedirs(save_dir, exist_ok=True)

    rob = SimulationRobobo()
    env = RoboboGymEnv(rob)
    eval_env = RoboboGymEnv(rob)

    eval_callback = RobotEvalCallback(
        eval_env=eval_env,
        eval_freq=1800,
        n_eval_episodes=1,
        model_save_dir=save_dir
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        verbose=1,
        n_steps=300,
        n_epochs=10
    )

    model.learn(total_timesteps=total_time_steps, callback=eval_callback)

    model.save(os.path.join(save_dir, f"{policy}_{total_time_steps}_{version}"))
    eval_callback.save_evaluations(os.path.join(save_dir, f"evaluations_{version}.csv"))

    if eval_callback.evaluations:
        print("\n=== TRAINING COMPLETE ===")
        final_eval = eval_callback.evaluations[-1]
        print(f"Final Avg Reward: {final_eval['avg_reward']:.2f}")
        print(f"Final Avg Episode Steps: {final_eval['avg_episode_steps']:.1f}")
        print(f"Final Avg Food Collected: {final_eval['avg_food_collected']:.1f}")
