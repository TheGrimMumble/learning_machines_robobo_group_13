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
    total_time_steps = 512*4 #  <------------------------
    policy = 'ppo'
    version = 'task2_v03'

    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        hw_inference(
            rob,
            policy,
            total_time_steps,
            version
            )
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        rob.stop_simulation()
        time.sleep(0.5)
        rob.play_simulation()
        time.sleep(0.5)
        rob.set_phone_pan_blocking(177, 100)
        rob.set_phone_tilt_blocking(100, 100)

        with open(f"/root/results/{policy}_{version}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Step #", "Left Speed", "Right Speed", "Close Call", "Collision", "Steps to find all", "Mean reward"])

        # model, env = train_model(
        #     rob,
        #     total_time_steps = total_time_steps,
        #     policy = policy,
        #     version = version
        #     )

        resume_model_at = 40960
        model, env = continue_training(
                            rob,
                            f"/root/results/{policy}_{resume_model_at}_{version}",
                            total_time_steps=total_time_steps,
                            policy=policy,
                            version=version
                            )
        env.global_step = resume_model_at
        
        for i in range(int(resume_model_at // total_time_steps - 1), 51):
            # Retry until it succeeds
            while True:
                print(f"Starting training at step {total_time_steps * (i + 1)}:")
                try:
                    rob.set_phone_pan_blocking(177, 100)
                    rob.set_phone_tilt_blocking(100, 100)
                    model.learn(
                        total_timesteps=total_time_steps,
                        reset_num_timesteps=False
                    )

                    model.save(f"/root/results/{policy}_{total_time_steps*(i+2)}_{version}")

                    print("Training paused")
                    break 
                except Exception as e:
                    print(f"continue_training failed at iteration {i+1}: {e}, retrying...")
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open("/root/results/error_log_training.txt", "a") as f:
                        f.write(f"[{timestamp}]\n")
                        traceback.print_exc(file=f)
                        f.write(f"\n\n\n")

            while True:
                print(f"Starting inference at step {total_time_steps * (i + 2)}:")
                try:
                    inference(
                        rob,
                        policy,
                        total_time_steps * (i + 2),
                        version
                    )
                    print("Inference successful")
                    break  
                except Exception as e:
                    print(f"inference failed at iteration {i+1}: {e}, retrying...")
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open("/root/results/error_log_inference.txt", "a") as f:
                        f.write(f"[{timestamp}]\n")
                        traceback.print_exc(file=f)
                        f.write(f"\n\n\n")

    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
    
    