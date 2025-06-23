#!/usr/bin/env python3
import sys
import csv
import cv2
import numpy as np
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



def calibrate_camera(robobo,
                     vision_size,
                     kernel,
                     blue_lower=np.array([90, 50, 50]),
                     blue_upper=np.array([130, 255, 255]),
                     pan_gain=0.25,
                     tilt_gain=0.25,
                     max_attempts=24):
    """
    Pan/tilt until two blue bars are centered in the lower-middle.
    """
    robobo.set_phone_tilt_blocking(109, 100) # max
    robobo.set_phone_pan_blocking(11, 100) # default: 179
    for _ in range(max_attempts):
        # 1) grab & preprocess
        frame = robobo.read_image_front()
        frame = cv2.resize(frame, (vision_size, vision_size))
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2) blue mask + clean
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3) find blue contours and pick the two tallest bars
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h * w > 0.01 * vision_size**2:
                bars.append((x, y, w, h))
        if len(bars) < 2:
            tilt = 332 // max_attempts
            tilt_now = robobo.read_phone_pan()
            robobo.set_phone_pan_blocking(min(343, tilt_now + tilt), 100)
            time.sleep(0.1)
            continue

        # keep the two tallest bars
        bars = sorted(bars, key=lambda b: b[3], reverse=True)[:2]
        centers = [(x + w/2, y + h/2) for x, y, w, h in bars]
        avg_cx  = sum(c[0] for c in centers) / 2.0
        avg_cy  = sum(c[1] for c in centers) / 2.0

        # compute offsets to target
        target_cx = vision_size / 2 # want the blue bars centered
        target_cy = vision_size * 0.90 # want the blue bars taking up 10% of vertical space
        dx = avg_cx - target_cx
        dy = avg_cy - target_cy

        # check success
        # print("dx", dx)
        if abs(dx) > 2:
            pan  =  max(2, int(round(pan_gain * dx)))
            # print("pan", pan)
            pan_now = robobo.read_phone_pan()
            robobo.set_phone_pan_blocking(min(343, pan_now + pan), 100)
            time.sleep(0.1)
        elif abs(dy) > 10:
            # print('dy', dy)
            tilt =  max(2, int(round(tilt_gain * dy)))
            # print("tilt", tilt)
            tilt_now = robobo.read_phone_tilt()
            robobo.set_phone_tilt_blocking(min(109, tilt_now + tilt), 100)
            time.sleep(0.1)
        else:
            return True

    return False





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
    elif sys.argv[1] == "--testing":
        rob = SimulationRobobo()
        rob.stop_simulation()
        time.sleep(0.5)
        rob.play_simulation()
        time.sleep(0.5)
        # rob.set_phone_pan_blocking(177, 100)
        # rob.set_phone_tilt_blocking(100, 100)
        image = rob.read_image_front()
        cv2.imwrite('/root/results/front_camera_image_pre.png', image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # print("pan:", rob.read_phone_pan())
        if calibrate_camera(rob, 256, kernel):
            print("Success!")
        else:
            print("Failed!")
        image = rob.read_image_front()
        cv2.imwrite('/root/results/front_camera_image_post.png', image)

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
    
    