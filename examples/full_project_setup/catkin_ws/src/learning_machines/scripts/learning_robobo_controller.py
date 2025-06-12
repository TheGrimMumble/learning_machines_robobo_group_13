#!/usr/bin/env python3
import sys
import csv

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
# from learning_machines.task0 import run_test_task0_actions
from learning_machines.RL.task1 import train_model, inference, continue_training


if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo(identifier=1)
        val_rob = SimulationRobobo(identifier=0)
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # run_all_actions(rob)

    # run_test_task0_actions(rob)
    
    total_time_steps = 128 #  <------------------------
    policy = 'ppo'
    version = 'all_night_v0'

    train_model(
        rob,
        total_time_steps = total_time_steps,
        policy = policy,
        version = version
        )
    with open(f"/root/results/{policy}_{total_time_steps}_{version}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Step #", "Left Speed", "Right Speed", "Close Call", "Collision"])

    inference(
        val_rob,
        f"/root/results/{policy}_{total_time_steps}_{version}",
        1
        )

    for i in range(4):
        continue_training(
            rob,
            f"/root/results/{policy}_{total_time_steps}_{version}",
            total_time_steps = total_time_steps,
            policy = policy,
            version = version
        )
        inference(
            val_rob,
            f"/root/results/{policy}_{total_time_steps}_{version}",
            i+2
            )