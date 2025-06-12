from stable_baselines3 import PPO
# from learning_machines.RoboboGymEnv_task1_sim import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V2 import RoboboGymEnv
# from learning_machines.RoboboGymEnv_task1_sim_V3 import RoboboGymEnv
from learning_machines.RoboboGymEnv_task1_sim_V4 import RoboboGymEnv
from robobo_interface import SimulationRobobo
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class RobotEvaluator:
    """Separate class for testing and evaluation - keeps env clean"""
    
    def __init__(self):
        self.reset_data()
    
    def reset_data(self):
        """Reset all tracking data"""
        self.episode_data = {
            'x_positions': [],
            'y_positions': [],
            'sensor_values': [],
            'rewards': [],
            'collisions': [],
            'timesteps': []
        }
    
    def record_step(self, position, sensor_reading, reward, collision, timestep):
        """Record data from one step"""
        self.episode_data['x_positions'].append(position.x)
        self.episode_data['y_positions'].append(position.y)
        self.episode_data['sensor_values'].append(np.max(sensor_reading))  # Max sensor
        self.episode_data['rewards'].append(reward)
        self.episode_data['collisions'].append(1 if collision else 0)
        self.episode_data['timesteps'].append(timestep)

    def test_policy(self, env, model, episodes=10, deterministic=True, policy_name="Trained"):
        """Test a policy and collect data"""
        all_episodes_data = []
        
        for episode in range(episodes):
            print(f"Testing episode {episode+1}/{episodes}")
            self.reset_data()
            
            obs, _ = env.reset()
            done = False
            timestep = 0
            
            while not done:
                if model is None:  # Random policy
                    action = env.action_space.sample()
                else:  # Trained policy
                    action, _ = model.predict(obs, deterministic=deterministic)
                
                obs, reward, done, _, _ = env.step(action)
                
                # Record step data
                position = env.current_position
                sensor_reading = obs[:8] 
                collision = env.collision
                
                self.record_step(position, sensor_reading, reward, collision, timestep)
                timestep += 1
            
            # Store episode data
            episode_summary = {
                'policy': policy_name,
                'episode': episode,
                'total_reward': sum(self.episode_data['rewards']),
                'total_collisions': sum(self.episode_data['collisions']),
                'episode_length': len(self.episode_data['x_positions']),
                'data': self.episode_data.copy()
            }
            all_episodes_data.append(episode_summary)
        
        return all_episodes_data

    def save_episode_data(self, trained_data, random_data, save_dir="/root/results"):
        """Save episode data to separate files"""
        import json
        import os
        from datetime import datetime
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trained policy data
        for i, episode_data in enumerate(trained_data):
            filename = f"trained_episode_{i}_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)
            
            # Convert numpy arrays to lists for JSON serialization
            json_data = {
                'policy': episode_data['policy'],
                'episode': episode_data['episode'],
                'total_reward': episode_data['total_reward'],
                'total_collisions': episode_data['total_collisions'],
                'episode_length': episode_data['episode_length'],
                'x_positions': episode_data['data']['x_positions'],
                'y_positions': episode_data['data']['y_positions'],
                'sensor_values': episode_data['data']['sensor_values'],
                'rewards': episode_data['data']['rewards'],
                'collisions': episode_data['data']['collisions'],
                'timesteps': episode_data['data']['timesteps']
            }
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved: {filepath}")
        
        # Save random policy data
        for i, episode_data in enumerate(random_data):
            filename = f"random_episode_{i}_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)
            
            json_data = {
                'policy': episode_data['policy'],
                'episode': episode_data['episode'],
                'total_reward': episode_data['total_reward'],
                'total_collisions': episode_data['total_collisions'],
                'episode_length': episode_data['episode_length'],
                'x_positions': episode_data['data']['x_positions'],
                'y_positions': episode_data['data']['y_positions'],
                'sensor_values': episode_data['data']['sensor_values'],
                'rewards': episode_data['data']['rewards'],
                'collisions': episode_data['data']['collisions'],
                'timesteps': episode_data['data']['timesteps']
            }
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved: {filepath}")
        
        # Also save a summary file
        summary_filename = f"evaluation_summary_{timestamp}.json"
        summary_filepath = os.path.join(save_dir, summary_filename)
        
        trained_rewards = [ep['total_reward'] for ep in trained_data]
        trained_collisions = [ep['total_collisions'] for ep in trained_data]
        random_rewards = [ep['total_reward'] for ep in random_data]
        random_collisions = [ep['total_collisions'] for ep in random_data]
        
        summary_data = {
            'timestamp': timestamp,
            'episodes_tested': len(trained_data),
            'trained_policy': {
                'avg_reward': float(np.mean(trained_rewards)),
                'std_reward': float(np.std(trained_rewards)),
                'avg_collisions': float(np.mean(trained_collisions)),
                'std_collisions': float(np.std(trained_collisions))
            },
            'random_policy': {
                'avg_reward': float(np.mean(random_rewards)),
                'std_reward': float(np.std(random_rewards)),
                'avg_collisions': float(np.mean(random_collisions)),
                'std_collisions': float(np.std(random_collisions))
            }
        }
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary: {summary_filepath}")


    def test_random_policy(self, env, episodes=10):
        """Test random policy baseline"""
        return self.test_policy(env, model=None, episodes=episodes, policy_name="Random")
    
    def plot_trajectories(self, trained_data, random_data, episode_idx=0):
        """Plot side-by-side trajectory comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Trained policy trajectory
        trained_episode = trained_data[episode_idx]['data']
        ax1.scatter(trained_episode['x_positions'], trained_episode['y_positions'], 
                   alpha=0.6, s=10, c='blue')
        ax1.plot(trained_episode['x_positions'], trained_episode['y_positions'], 
                alpha=0.3, linewidth=0.5, c='blue')
        ax1.set_title(f"Trained Policy (Episode {episode_idx})")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.axis('equal')
        
        # Random policy trajectory
        random_episode = random_data[episode_idx]['data']
        ax2.scatter(random_episode['x_positions'], random_episode['y_positions'], 
                   alpha=0.6, s=10, c='red')
        ax2.plot(random_episode['x_positions'], random_episode['y_positions'], 
                alpha=0.3, linewidth=0.5, c='red')
        ax2.set_title(f"Random Policy (Episode {episode_idx})")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, trained_data, random_data):
        """Plot key metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extract metrics
        trained_rewards = [ep['total_reward'] for ep in trained_data]
        random_rewards = [ep['total_reward'] for ep in random_data]
        
        trained_collisions = [ep['total_collisions'] for ep in trained_data]
        random_collisions = [ep['total_collisions'] for ep in random_data]
        
        # Plot rewards
        axes[0,0].boxplot([trained_rewards, random_rewards], labels=['Trained', 'Random'])
        axes[0,0].set_title('Total Reward per Episode')
        axes[0,0].set_ylabel('Reward')
        
        # Plot collisions
        axes[0,1].boxplot([trained_collisions, random_collisions], labels=['Trained', 'Random'])
        axes[0,1].set_title('Collisions per Episode')
        axes[0,1].set_ylabel('Number of Collisions')
        
        # Plot episode lengths
        trained_lengths = [ep['episode_length'] for ep in trained_data]
        random_lengths = [ep['episode_length'] for ep in random_data]
        axes[1,0].boxplot([trained_lengths, random_lengths], labels=['Trained', 'Random'])
        axes[1,0].set_title('Episode Length')
        axes[1,0].set_ylabel('Timesteps')
        
        # Plot average sensor values
        trained_sensors = [np.mean(ep['data']['sensor_values']) for ep in trained_data]
        random_sensors = [np.mean(ep['data']['sensor_values']) for ep in random_data]
        axes[1,1].boxplot([trained_sensors, random_sensors], labels=['Trained', 'Random'])
        axes[1,1].set_title('Average Sensor Value')
        axes[1,1].set_ylabel('Sensor Reading (lower=better)')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, trained_data, random_data):
        """Print summary statistics"""
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Episodes tested: {len(trained_data)}")
        
        # Trained policy stats
        trained_rewards = [ep['total_reward'] for ep in trained_data]
        trained_collisions = [ep['total_collisions'] for ep in trained_data]
        
        print(f"\nTrained Policy:")
        print(f"  Avg Reward: {np.mean(trained_rewards):.2f} ± {np.std(trained_rewards):.2f}")
        print(f"  Avg Collisions: {np.mean(trained_collisions):.1f} ± {np.std(trained_collisions):.1f}")
        
        # Random policy stats
        random_rewards = [ep['total_reward'] for ep in random_data]
        random_collisions = [ep['total_collisions'] for ep in random_data]
        
        print(f"\nRandom Policy:")
        print(f"  Avg Reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
        print(f"  Avg Collisions: {np.mean(random_collisions):.1f} ± {np.std(random_collisions):.1f}")

def evaluate_robot(rob):
    env = RoboboGymEnv(rob)
    
    model_path =  "/root/results/ppo_110000_V4_continued.zip"
    trained_model = PPO.load(model_path)
    
    evaluator = RobotEvaluator()
    
    # Test both policies
    print("Testing trained policy")
    trained_data = evaluator.test_policy(env, trained_model, episodes=3)
    
    print("Testing random policy")
    random_data = evaluator.test_random_policy(env, episodes=3)
    
    # Generate all plots and summaries
    evaluator.plot_trajectories(trained_data, random_data, episode_idx=0)
    evaluator.plot_metrics_comparison(trained_data, random_data)
    evaluator.print_summary(trained_data, random_data)
    
    return trained_data, random_data

def test_model(rob):
    # Load the trained model
    model_path = "/root/results/ppo_110000_V4_continued.zip"
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
