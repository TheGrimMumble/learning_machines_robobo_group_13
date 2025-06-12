import pandas as pd
import matplotlib.pyplot as plt

results_path= "/Users/lunaellinger/Documents/learning_machines/learning_machines_robobo/examples/full_project_setup/results"


sim_df = pd.read_csv(results_path + "/sensor_log_simulation.csv")
hw_df = pd.read_csv(results_path + "/sensor_log_hardware.csv")

print(sim_df.head())
print(hw_df.head())

# Plot all sensors over time
plt.figure(figsize=(12, 6))
for column in sim_df.columns:
    if column != "Timestep":
        plt.plot(sim_df["Timestep"], sim_df[column], label=column)

# # Plot all sensors over time
# plt.figure(figsize=(12, 6))
# for column in sim_df.columns:
#     if column == "FM":
#         plt.plot(sim_df["Timestep"], sim_df[column], label=column)

plt.axhline(y=50, color='r', linestyle='-', label='threshold')

plt.ylim(0, 150)
plt.xlabel("Timestep")
plt.ylabel("Sensor Value")
plt.title("Sensor Activations Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()