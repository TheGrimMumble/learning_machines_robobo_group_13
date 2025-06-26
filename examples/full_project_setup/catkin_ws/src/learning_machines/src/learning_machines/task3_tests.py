import cv2
import time
import csv
import numpy as np
import random
import os


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

def tests_task3(rob: IRobobo):
    # Set up output directory
    image_dir = "/root/results/images"
    os.makedirs(image_dir, exist_ok=True)  # create images folder if it doesn't exist

    rob.set_phone_tilt(107, 107)
    # rob.set_phone_pan(180, 100)

    counter = 0
    while True:
        front_image = rob.read_image_front()

        # Save the image
        image_path = os.path.join(image_dir, f"frame_{counter:04d}.jpg")
        cv2.imwrite(image_path, front_image)
        print(f"Saved: {image_path}")

        rob.move_blocking(100, 100, 100)
        counter += 1
    
    

def test_camera(rob:IRobobo):
    while True:
        rw = random.randint(-100, 100)
        lw = random.randint(-100, 100)
        rob.move_blocking(lw, rw, 500)
        # base_food_distance = rob._base_food_distance()
        nr_collected_food = rob.get_nr_food_collected()
        image_front = rob.read_image_front()
        size = image_front.size
        print(f'Image: {image_front}')
        # print(f'base_food_distance: {base_food_distance}')
        print(f'nr_collected_food: {nr_collected_food}')
        print(f'Size: {size}')
        print(f'shape: {image_front.shape}')

def run_test_task3(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    tests_task3(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()