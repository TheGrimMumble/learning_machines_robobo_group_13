#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
# from learning_machines.task0 import run_test_task0_actions
from learning_machines.RL.task1 import train_model, inference


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
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # run_all_actions(rob)

    # run_test_task0_actions(rob)
    
    total_time_steps = 512*2
    policy = 'ppo'
    version = 'ver_001_'

    # train_model(
    #     rob,
    #     total_time_steps = total_time_steps,
    #     policy = policy,
    #     version = version
    #     )
    inference(rob, f"/root/results/{policy}_{total_time_steps}_{version}.zip")
