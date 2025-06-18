#!/usr/bin/env python3
import sys
import csv
import traceback
from datetime import datetime

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
    version = 'all_day_v03'

    # train_model(
    #     rob,
    #     total_time_steps = total_time_steps,
    #     policy = policy,
    #     version = version
    #     )
    # with open(f"/root/results/{policy}_{total_time_steps}_{version}.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Step #", "Left Speed", "Right Speed", "Close Call", "Collision", "Reward"])

    # test_vers = [
    #     30720,
    #     32768,
    #     34816,
    #     36864,
    #     38912,
    # ]
    # for steps in [34816]:
    # robs = [rob, val_rob, test_rob]
    # for r in robs:
    #     # print(f"\n\nStep: {steps}\n\n")
    #     inference(
    #         r,
    #         f"/root/results/{policy}_{24576}_{version}",
    #         total_time_steps,
    #         print_to_csv=False
    #         )





    for i in range(11, 151):  # <------------------------
        # Retry continue_training until it succeeds
        while True:
            print(f"Starting training at step {total_time_steps * (i + 1)}:")
            try:
                continue_training(
                    rob,
                    f"/root/results/{policy}_{total_time_steps * (i + 1)}_{version}",
                    i + 2,
                    total_time_steps=total_time_steps,
                    policy=policy,
                    version=version
                )
                print("Training paused")
                break  # Exit the loop if successful
            except Exception as e:
                print(f"continue_training failed at iteration {i+1}: {e}, retrying...")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open("/root/results/error_log_training.txt", "a") as f:
                    f.write(f"[{timestamp}]\n")
                    traceback.print_exc(file=f)
                    f.write(f"\n\n\n")

        # Retry inference until it succeeds
        while True:
            print(f"Starting inference at step {total_time_steps * (i + 2)}:")
            try:
                inference(
                    val_rob,
                    f"/root/results/{policy}_{total_time_steps * (i + 2)}_{version}",
                    total_time_steps * (i + 2)
                )
                print("Inference successful")
                break  # Exit the loop if successful
            except Exception as e:
                print(f"inference failed at iteration {i+1}: {e}, retrying...")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open("/root/results/error_log_inference.txt", "a") as f:
                    f.write(f"[{timestamp}]\n")
                    traceback.print_exc(file=f)
                    f.write(f"\n\n\n")




