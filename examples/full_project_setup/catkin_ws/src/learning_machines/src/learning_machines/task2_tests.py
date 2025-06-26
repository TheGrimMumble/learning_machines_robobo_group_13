import cv2
import time
import csv
import numpy as np
import random


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

def test_tilt_pan(rob:IRobobo):
    while True:
        rob.set_phone_tilt(100,100)
        rob.set_phone_pan(180,100)

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

def run_test(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    test_tilt_pan(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()