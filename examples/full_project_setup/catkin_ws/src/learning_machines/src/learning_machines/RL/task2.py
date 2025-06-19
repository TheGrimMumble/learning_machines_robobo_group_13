from .RoboboGymEnv_task2_sim import RoboboGymEnv
import time
import csv
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    Position,
    Orientation,
    SimulationRobobo,
    HardwareRobobo,
)

# Initialize your robot interface (assuming it's simulation for now)
# from robobo_interface import SimulationRobobo     

def get_flat_params(model):
    return torch.cat([param.data.view(-1) for param in model.policy.parameters()])


def train_model(
        rob:SimulationRobobo,
        total_time_steps = 128,
        policy = 'ppo',
        version = 'test',
        ):
    
    rob.stop_simulation()
    time.sleep(2)
    rob.play_simulation()

    # Create the environment
    env = RoboboGymEnv(rob)

    # Check if the environment follows Gym API properly
    check_env(env, warn=True)

    # Define the PPO model
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.001, # default: 0.0003
        verbose=1,
        n_steps=256, #  <------------------------
        n_epochs=8) #  <------------------------
    
    initial_params = get_flat_params(model).clone()

    # Train the model
    model.learn(total_timesteps=total_time_steps)

    updated_params = get_flat_params(model)

    if torch.equal(initial_params, updated_params):
        print("❌ Parameters did not change.")
    else:
        print("✅ Parameters updated.")

    # Save the model
    model.save(f"/root/results/{policy}_{total_time_steps}_{version}")

    # # Test the model
    # obs, _ = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = env.step(action)
    #     if done or truncated:
    #         obs, _ = env.reset()


def continue_training(
        rob:SimulationRobobo,
        path: str,
        iteration,
        total_time_steps = 128,
        policy = 'ppo',
        version = 'test',
        ):
    rob.stop_simulation()
    time.sleep(2)
    rob.play_simulation()
    env = RoboboGymEnv(rob)
    model = PPO.load(path, env=env)
    model.learn(total_timesteps=total_time_steps)
    model.save(f"/root/results/{policy}_{total_time_steps * iteration}_{version}")


def inference(
        rob:SimulationRobobo,
        policy,
        training_steps,
        version,
        print_to_csv=True
        ):
    path = f"/root/results/{policy}_{training_steps}_{version}"
    rob.stop_simulation()
    time.sleep(2)
    rob.play_simulation()
    env = RoboboGymEnv(rob)

    n_steps = 512 #  <------------------------

    env.max_steps_in_episode = n_steps
    model = PPO.load(path, env=env)

    obs, _ = env.reset()
    done = False
    left_speeds = []
    right_speeds = []
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # print("action:", action, "shape:", getattr(action, "shape", None))

        max_speed = 100
        left_speed = action[0] * max_speed
        right_speed = action[1] * max_speed
        left_speeds.append(left_speed)
        right_speeds.append(right_speed)

        obs, reward, done, _bool, info = env.step(action)
        if done:
            nmbr_steps_findall = env.steps_to_find_all

        rewards.append(reward)

    left_mean_speed = sum(left_speeds) / n_steps
    right_mean_speed = sum(right_speeds) / n_steps
    mean_reward = sum(rewards) / n_steps
    
    if print_to_csv:
        with open(f"/root/results/{policy}_{version}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([training_steps,
                            left_mean_speed,
                            right_mean_speed,
                            env.close_call_count,
                            env.collision_count,
                            nmbr_steps_findall,
                            mean_reward])
            
def format_number(n):
    # Try to format with 4 to 0 decimal places
    for i in range(4, -1, -1):
        formatted = f"{n:.{i}f}"
        if len(formatted) <= 6:
            return f"{formatted:<6}"
    # If nothing fits, round to nearest integer and return as string
    return str(round(n))


def test_robot_sensors(rob:SimulationRobobo, speed):
    rob.stop_simulation()
    time.sleep(2)
    rob.play_simulation()
    irs_pos = [
        'BR',
        'BL',
        'FR2',
        'FL2',
        'FM',
        'FL1',
        'BM',
        'FR1'
    ]
    print_pos = [
        "x",
        "y",
        "z",
        "BL",
        "BM",
        "BR",
        "FL1",
        "FL2",
        "FM",
        "FR2",
        "FR1"
    ]
    pos = rob.get_position()
    ori = rob.get_orientation()
    print(f"\n\nPosition: x = {pos.x}, y = {pos.y}, z = {pos.z}\n" + \
          f"Orientation: yaw = {ori.yaw}, pitch = {ori.pitch}, roll = {ori.roll}")
    rob.set_position(pos, Orientation(float(1), ori.pitch, ori.roll))

    while True:
        rob.move_blocking(speed, speed, 100)
        pos_new = rob.get_position()
        if pos_new == pos:
            print(f"\nCollision position: {pos_new}")
            break
        else:
            print(f"\nPosition: {pos_new}")
            pos = pos_new

    rob.set_position(pos_new, Orientation(float(1), ori.pitch, ori.roll))
    header = "".join([h.ljust(9) for h in print_pos])
    print("\n", header)

    for i in range(20):
        irs = rob.read_irs()
        pos = rob.get_position()
        vals = {**{"x": pos.x, "y": pos.y, "z": pos.z},
                **{irs_pos[i]: v for i, v in enumerate(irs)}}
        print("".join([format_number(vals[v]).ljust(9) for v in print_pos]))
        rob.move_blocking(-speed, -speed, 100)





def test_robot_sensors_deprecated(rob:SimulationRobobo):
    pos = rob.get_position()
    ori = rob.get_orientation()
    print(f"Position: x = {pos.x}, y = {pos.y}, z = {pos.z}\n" + \
          f"Orientation: yaw = {ori.yaw}, pitch = {ori.pitch}, roll = {ori.roll}")
    instruction = "Type 'p' to set position, 'o' to set orientation, 'po' to set both, and type 'exit' to stop.\n"
    act = input(instruction)
    while True:
        if act == "exit":
            break

        try:
            if act == "p":
                raw = input("Enter x,y,z: ")
                x, y, z = map(float, raw.split(","))
                rob.set_position(Position(x, y, z), ori)

            elif act == "o":
                raw = input("Enter yaw,pitch,roll: ")
                yaw, pitch, roll = map(float, raw.split(","))
                rob.set_position(pos, Orientation(yaw, pitch, roll))

            elif act == "po":
                raw = input("Enter x,y,z,yaw,pitch,roll: ")
                x, y, z, yaw, pitch, roll = map(float, raw.split(","))
                rob.set_position(Position(x, y, z), Orientation(yaw, pitch, roll))

            else:
                raise ValueError(f"Unknown action '{act}'")

        except Exception as e:
            print("Error:", e)
            act = input("Please follow the instructions:\n" + instruction)
            continue

        # Refresh and display current state
        pos = rob.get_position()
        ori = rob.get_orientation()
        print(f"Position: x={pos.x}, y={pos.y}, z={pos.z}")
        print(f"Orientation: yaw={ori.yaw}, pitch={ori.pitch}, roll={ori.roll}")

        act = input(instruction)


