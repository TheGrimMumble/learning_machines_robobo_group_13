#!/usr/bin/env python3
import sys
import csv

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
# from learning_machines.task0 import run_test_task0_actions
from learning_machines.RL.task1 import (
    train_model,
    inference,
    continue_training,
    test_robot_sensors
    )


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
        test_rob = SimulationRobobo(identifier=2)
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # run_all_actions(rob)
    # run_all_actions(val_rob)
    # run_all_actions(test_rob)

    # test_robot_sensors(test_rob, 20)
    # test_robot_sensors(test_rob, -20)

    # run_test_task0_actions(rob)
    
    total_time_steps = 512*4 #  <------------------------
    policy = 'ppo'
    version = 'all_night_v01'

    # train_model(
    #     rob,
    #     total_time_steps = total_time_steps,
    #     policy = policy,
    #     version = version
    #     )
    # with open(f"/root/results/{policy}_{total_time_steps}_{version}.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Step #", "Left Speed", "Right Speed", "Close Call", "Collision"])

    test_vers = [
        30720,
        32768,
        34816,
        36864,
        38912,
    ]
    robs = [rob, val_rob, test_rob]
    for steps in [34816]:
        for r in robs:
            print(f"\n\nStep: {steps}\n\n")
            inference(
                r,
                f"/root/results/{policy}_{total_time_steps}_{version}_{steps}",
                total_time_steps,
                print_to_csv=False
                )

    # for i in range(51): #  <------------------------
    #     continue_training(
    #         rob,
    #         f"/root/results/{policy}_{total_time_steps}_{version}",
    #         total_time_steps = total_time_steps,
    #         policy = policy,
    #         version = version
    #     )
    #     inference(
    #         val_rob,
    #         f"/root/results/{policy}_{total_time_steps}_{version}",
    #         (total_time_steps * (i+2))
    #         )
