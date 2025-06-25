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
from learning_machines.RL.RoboboGymEnv_task3_sim import RoboboGymEnv
from learning_machines.RL.task3 import (
    train_model,
    inference,
    continue_training,
    calibrate_camera,
    print_irs
    )
from learning_machines.RL.task2_hardw import (
    hw_inference
    )
from robobo_interface import Orientation







if __name__ == "__main__":
    total_time_steps = 512*4 #  <------------------------
    policy = 'ppo'
    version = 'task3_v04'

    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--testing":
        rob = SimulationRobobo()
        env = RoboboGymEnv(rob)
        rob.stop_simulation()
        time.sleep(0.5)
        rob.play_simulation()
        time.sleep(0.5)
        # rob.set_phone_pan_blocking(177, 100)
        # rob.set_phone_tilt_blocking(100, 100)
        # image = rob.read_image_front()
        # cv2.imwrite('/root/results/front_camera_image_pre.png', image)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # print("pan:", rob.read_phone_pan())
        # if calibrate_camera(rob, 256, kernel):
        #     print("Success!")
        # else:
        #     print("Failed!")
        # image = rob.read_image_front()
        # cv2.imwrite('/root/results/front_camera_image_post.png', image)
        # print_irs()

        # image = rob.read_image_front()
        # green_found, green_cx, green_area, red_found, red_cx, red_area, red_captured = env.detect_objects(image)
        # while not red_captured:
        #     (food_dist, food_yaw), (base_dist, base_yaw) = env.get_relative_targets()
        #     ori = rob.get_orientation()
        #     food_ori = Orientation(yaw=food_yaw)
        #     ori.yaw = food_ori.yaw
        #     pos = rob.get_position()
        #     rob.set_position(pos, ori)
        #     time.sleep(0.5)
        #     rob.move_blocking(100, 100, 250)
        #     image = rob.read_image_front()
        #     green_found, green_cx, green_area, red_found, red_cx, red_area, red_captured = env.detect_objects(image)
        # while not rob.base_detects_food():
        #     (food_dist, food_yaw), (base_dist, base_yaw) = env.get_relative_targets()
        #     food_pos = rob.get_food_position()
        #     print(f"food_pos: x-{food_pos.x:.2f}, y-{food_pos.y:.2f}")
        #     # print(f"food_dist: {food_dist}")
        #     k_turn = 20  # turning scale
        #     k_fwd = 20   # forward speed

        #     alignment = np.cos(food_yaw)
        #     forward = k_fwd * max(0, alignment)
        #     turn = (food_yaw / np.pi) * k_turn

        #     left_speed = forward - turn
        #     right_speed = forward + turn

        #     rob.move_blocking(left_speed, right_speed, 250)
        print(f"max left right wheel dist: {abs(rob.get_LW_position() - rob.get_RW_position())}")
        while not rob.base_detects_food():
            image = rob.read_image_front()
            green_found, green_cx, green_area, red_found, red_cx, red_area, red_captured = env.detect_objects(image)
            left, right = env.compute_distance_each_wheel("food")

            turn = left - right
            k_turn = 20  # turning scale
            k_fwd = 20   # forward speed

            left_speed = k_fwd + k_turn * turn
            right_speed = k_fwd + k_turn * turn

            rob.move_blocking(left_speed, right_speed, 250)

        print("success!")
        # try:
        #     while True:
        #         rob.move_blocking(10, -10, 250)  # slow rotation

        #         pos = rob.get_position()
        #         ori = rob.get_orientation()

        #         print(f"Position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}), "
        #             f"Yaw: {np.rad2deg(ori.yaw):.2f}°, "
        #             f"Pitch: {np.rad2deg(ori.pitch):.2f}°, "
        #             f"Roll: {np.rad2deg(ori.roll):.2f}°")
                
        #         time.sleep(0.25)

        # except KeyboardInterrupt:
        #     print("Stopped by user")



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

        # with open(f"/root/results/{policy}_{version}.csv", "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerow([
        #          "Step #",
        #          "Left Speed",
        #          "Right Speed",
        #          "Collision",
        #          "#Stps to red",
        #          "#Red found",
        #          "#Red lost",
        #          "#Red captured",
        #          "#Red uncaptured",
        #          "#Stps to green",
        #          "#Green found",
        #          "#Green lost",
        #          "Mean reward"])

        # model, env = train_model(
        #     rob,
        #     total_time_steps = total_time_steps,
        #     policy = policy,
        #     version = version
        #     )

        resume_model_at = 32768
        model, env = continue_training(
                            rob,
                            f"/root/results/{policy}_{resume_model_at}_{version}"
                            )
        env.global_step = resume_model_at
        
        for i in range(int(resume_model_at // total_time_steps - 1), 151):
            # Retry until it succeeds
            while True:
                print(f"Starting training at step {total_time_steps * (i + 1)}:")
                try:
                    rob.stop_simulation()
                    time.sleep(0.1)
                    rob.play_simulation()
                    time.sleep(0.1)
                    rob.set_phone_tilt_blocking(109, 100)
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
            for _ in range(3):
                while True:
                    print(f"Starting inference at step {total_time_steps * (i + 2)}:")
                    try:
                        rob.set_phone_tilt_blocking(109, 100)
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
    
    