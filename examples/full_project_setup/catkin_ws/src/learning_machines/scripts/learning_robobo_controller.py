#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
from learning_machines.task0 import run_test_task0_actions
# from learning_machines.task1 import continue_training
from learning_machines import evaluate_robot
from learning_machines import test_model
from learning_machines import train_model
from learning_machines import continue_training
# from learning_machines.test_model import test_model
# from learning_machines.test_model import evaluate_robot


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

    # run_test_task0_actions(rob)
    # continue_training(rob)
    # train_model(rob)
    test_model(rob)
    # evaluate_robot(rob)
    # run_all_actions(rob)
