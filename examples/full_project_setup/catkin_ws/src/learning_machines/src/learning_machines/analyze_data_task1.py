import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_path = "/Users/lunaellinger/Documents/learning_machines/learning_machines_robobo_group_13/examples/full_project_setup/results"

# Load both datasets
trained_csv_path = f"{results_path}/sim_log_20250612_131746_V4.csv"
random_csv_path = f"{results_path}/sim_log_20250612_134812_random.csv"

trained_df = pd.read_csv(trained_csv_path)
random_df = pd.read_csv(random_csv_path)

# Count total collisions in each dataset
trained_collisions = trained_df['collision'].sum()
random_collisions = random_df['collision'].sum()
ea_collisions = 0  # Perfect EA solution

# Create bar plot comparing the three
plt.figure(figsize=(10, 6))
bars = plt.bar(['PPO', 'EA', 'Random Actions'], 
              [trained_collisions, ea_collisions, random_collisions],
              color=['blue', 'green', 'red'],
              alpha=0.7)

# Add value labels on bars
for bar in bars:
   height = bar.get_height()
   plt.text(bar.get_x() + bar.get_width()/2., height + 5,
           f'{int(height)}',
           ha='center', va='bottom')

plt.title('Total Collisions: PPO vs EA vs Random Actions')
plt.ylabel('Number of Collisions')
plt.xlabel('Policy Type')
plt.grid(axis='y', alpha=0.3)

# Calculate and show collision rates
trained_rate = (trained_collisions / len(trained_df)) * 100
random_rate = (random_collisions / len(random_df)) * 100
ea_rate = 0.0

plt.text(0.5, max(trained_collisions, random_collisions) * 0.8, 
        f'PPO: {trained_rate:.1f}%\nEA: {ea_rate:.1f}%\nRandom: {random_rate:.1f}%',
        ha='center', transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.show()

# Print summary
print(f"PPO model collisions: {trained_collisions} ({trained_rate:.1f}%)")
print(f"EA model collisions: {ea_collisions} ({ea_rate:.1f}%)")
print(f"Random actions collisions: {random_collisions} ({random_rate:.1f}%)")
print(f"PPO improvement over Random: {random_collisions - trained_collisions} fewer collisions")
print(f"EA improvement over PPO: {trained_collisions - ea_collisions} fewer collisions")


# Load evaluation data
eval_path = f"{results_path}/evaluations_V4_1.csv"
eval_df = pd.read_csv(eval_path)

# Remove rows where timestep == 0 if present
eval_df = eval_df[eval_df['timestep'] > 0]

# Plot: Average Reward over Time
plt.figure(figsize=(10, 4))
plt.plot(eval_df['timestep'], eval_df['avg_reward'], label='Average Reward', color='blue', marker='o')
plt.xlabel("Timestep")
plt.ylabel("Average Reward")
plt.title("Average Reward Over Time (PPO Evaluation)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Average Collisions over Time
plt.figure(figsize=(10, 4))
plt.plot(eval_df['timestep'], eval_df['avg_collisions'], label='Avg Collisions', color='orange', marker='o')
plt.xlabel("Timestep")
plt.ylabel("Average Collisions")
plt.title("Average Collisions Over Time (PPO Evaluation)")
plt.grid(True)
plt.tight_layout()
plt.show()



