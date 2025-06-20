#!/usr/bin/env python3
import sys
import csv
import traceback
import time
from datetime import datetime

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
# from learning_machines.task0 import run_test_task0_actions
from learning_machines.RL.task2 import (
    train_model,
    inference,
    continue_training,
    test_robot_sensors
    )
from learning_machines.RL.task2_hardw import (
    hw_inference
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
        rob = SimulationRobobo()
        rob.stop_simulation()
        time.sleep(2)
        rob.play_simulation()
        rob.set_phone_pan_blocking(177, 100)
        rob.set_phone_tilt_blocking(100, 100)
        # rob = SimulationRobobo(identifier=1)
        # val_rob = SimulationRobobo(identifier=0)
        # test_rob = SimulationRobobo(identifier=2)
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    run_all_actions(rob)
    # run_all_actions(val_rob)
    # run_all_actions(test_rob)

    # test_robot_sensors(rob, 20)
    # test_robot_sensors(rob, -20)

    # run_test_task0_actions(rob)
    
    # total_time_steps = 38912 #  <------------------------
    # policy = 'ppo'
    # version = 'task2_v03'

    # hw_inference(
    #     rob,
    #     policy,
    #     total_time_steps,
    #     version
    #     )


    # model, env = train_model(
    #     rob,
    #     total_time_steps = total_time_steps,
    #     policy = policy,
    #     version = version
    #     )
    # with open(f"/root/results/{policy}_{version}.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Step #", "Left Speed", "Right Speed", "Close Call", "Collision", "# steps to find all", "Mean reward"])

    # inference(
    #     rob,
    #     policy,
    #     total_time_steps,
    #     version
    #     )

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


    # run_all_actions(rob)

    # model, env = continue_training(
    #                     rob,
    #                     f"/root/results/{policy}_{30720}_{version}",
    #                     total_time_steps=total_time_steps,
    #                     policy=policy,
    #                     version=version
    #                     )


    # n = 1

    # for i in range(14, 51):  # <------------------------
    #     # Retry continue_training until it succeeds
    #     while True:
    #         print(f"Starting training at step {total_time_steps * (i + 1)}:")
    #         try:
    #             # continue_training(
    #             #     rob,
    #             #     f"/root/results/{policy}_{total_time_steps * (i + 1)}_{version}",
    #             #     i + 2,
    #             #     total_time_steps=total_time_steps,
    #             #     policy=policy,
    #             #     version=version
    #             # )
    #             # In-place further learning
    #             rob.set_phone_pan_blocking(177, 100)
    #             rob.set_phone_tilt_blocking(100, 100)
    #             model.learn(
    #                 total_timesteps=total_time_steps,
    #                 reset_num_timesteps=False
    #             )
    #             # Optionally save every N iterations
    #             # if i % n == 0:
    #             model.save(f"/root/results/{policy}_{total_time_steps*(i+2)}_{version}")

    #             print("Training paused")
    #             break  # Exit the loop if successful
    #         except Exception as e:
    #             print(f"continue_training failed at iteration {i+1}: {e}, retrying...")
    #             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #             with open("/root/results/error_log_training.txt", "a") as f:
    #                 f.write(f"[{timestamp}]\n")
    #                 traceback.print_exc(file=f)
    #                 f.write(f"\n\n\n")

    #     # Retry inference until it succeeds
    #     while True:
    #         print(f"Starting inference at step {total_time_steps * (i + 2)}:")
    #         try:
    #             inference(
    #                 rob,
    #                 policy,
    #                 total_time_steps * (i + 2),
    #                 version
    #             )
    #             print("Inference successful")
    #             break  # Exit the loop if successful
    #         except Exception as e:
    #             print(f"inference failed at iteration {i+1}: {e}, retrying...")
    #             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #             with open("/root/results/error_log_inference.txt", "a") as f:
    #                 f.write(f"[{timestamp}]\n")
    #                 traceback.print_exc(file=f)
    #                 f.write(f"\n\n\n")



