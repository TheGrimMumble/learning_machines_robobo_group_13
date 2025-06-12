import cv2
import time
import csv
import numpy as np

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

def go_forward_nostop(rob:IRobobo):
    while True:
        rob.move_blocking(10, 10, 500)
        sensor_data = rob.read_irs()
        dict_sensor_values = create_sensor_dict(sensor_data)
        print(dict_sensor_values)
        print(dict_sensor_values['BM'])
        print('')
        rob.move_blocking(0, 0, 10000)
    
def go_forward_250ms_fullspeed(rob:IRobobo):
    pos = rob.get_position()
    print(f'pos before movement: {pos}')
    rob.move_blocking(100, 100, 1000)
    print(f'pos after movement of 1 sec full speed: {pos}')

def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.phone_battery())
    print("Robot battery level: ", rob.robot_battery())


def turn_right(rob:IRobobo):
    rob.move_blocking(-50, -50, 500)
    rob.move_blocking(-50, 50, 800)  # spin right

def create_sensor_dict(sensor_data):
    dict_sensor_values = {
            'BR': sensor_data[0],
            'BL': sensor_data[1],
            'FR2': sensor_data[2],
            'FL2': sensor_data[3],
            'FM': sensor_data[4],
            'BL1': sensor_data[5],
            'BM': sensor_data[6],
            'FR1': sensor_data[7]
        }
    return dict_sensor_values

def task0(rob:IRobobo):
    test_hardware(rob)
    dict_timestep = {}
    threshold = 50
    counter = 0
    counter_turn = 0

    while True:
        counter += 1
        # start moving forward asynchronously
        rob.move(10, 10, 1000) 

        sensor_data = rob.read_irs()
        dict_sensor_values = create_sensor_dict(sensor_data)

        print(f'Iteration {counter}')
        print(dict_sensor_values)
    
        # sensor_data = rob.read_irs()
        # print("Sensor data:", sensor_data)
        dict_timestep[counter] = dict_sensor_values

        if sensor_data[4] > threshold:
            counter += 1
            counter_turn += 1
            rob.move_blocking(-50, -50, 500)
            
            sensor_data = rob.read_irs()
            dict_sensor_values = create_sensor_dict(sensor_data)
            dict_timestep[counter] = dict_sensor_values

            counter += 1
            rob.move_blocking(-50, 50, 800)  # spin right
            
            sensor_data = rob.read_irs()
            dict_sensor_values = create_sensor_dict(sensor_data)
            dict_timestep[counter] = dict_sensor_values
        
        if counter_turn == 6:
            break
    
    # save data
    with open("/root/results/sensor_log_hardware_v2.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Timestep", "BR", "BL", "FR2", "FL2", "FM", "BL1", "BM", "FR1"])
        writer.writeheader()
        for timestep, sensor_vals in dict_timestep.items():
            row = {"Timestep": timestep}
            row.update(sensor_vals)
            writer.writerow(row)


def get_orientation(rob):
    print(rob.read_orientation)

def test_sensors_on_hardware(rob: IRobobo, duration_sec: float = 180):

    counter = 0
    # start_time = time.time()
    try:
        while True:
            counter += 1

            sensor_data = rob.read_irs()
            dict_sensor_values = create_sensor_dict(sensor_data)
            print(f'Iteration {counter}')
            print(dict_sensor_values)
            # print("IRS data: ", rob.read_irs())
            print('')
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    # while time.time() - start_time < duration_sec:
    #     print("IRS data: ", rob.read_irs())
    #     time.sleep(0.5)

"""
move:
val 0: left wheel speed 
val 1: right wheel speed

move_blocking:
val 0: left wheel speed 
val 1: right wheel speed
val 2: time in ms (so 1000 is one second)
"""
def go_forward(rob: IRobobo):
    rob.move(50, 50, )

def go_backward(rob: IRobobo):
    rob.move(-100, -100, 3000)

def move_back_forth(rob):
    rob.move(100, 100)
    rob.move(-100, -100)


def run_test_task0_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # test_sensors_on_hardware(rob)
    # task0(rob)
    # go_forward_1min(rob)
    go_forward_250ms_fullspeed(rob)
    # go_backward_1min(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
