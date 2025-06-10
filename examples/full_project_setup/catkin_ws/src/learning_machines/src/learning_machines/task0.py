import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from data_files import FIGURES_DIR
import asyncio
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)
import threading
import os


class SensorDataLogger:
    def __init__(self, rob: IRobobo):
        self.rob = rob
        self.back_left = 0
        self.back_right = 0
        self.front_left = 0
        self.front_right = 0
        self.front_center = 0
        self.front_right_right = 0
        self.back_center = 0
        self.front_left_left = 0

        # Data logging lists
        self.timestamps = []
        self.back_left_values = []
        self.back_right_values = []
        self.front_left_values = []
        self.front_right_values = []
        self.front_center_values = []
        self.front_right_right_values = []
        self.back_center_values = []
        self.front_left_left_values = []

        # Start time for relative timestamps
        self.start_time = time.time()

    def read_data(self):
        irs_data = (
            self.rob.read_irs()
        )  # [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]

        # Update current values
        self.back_left = irs_data[0]
        self.back_right = irs_data[1]
        self.front_left = irs_data[2]
        self.front_right = irs_data[3]
        self.front_center = irs_data[4]
        self.front_right_right = irs_data[5]
        self.back_center = irs_data[6]
        self.front_left_left = irs_data[7]

        # Log data with timestamp
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        self.back_left_values.append(self.back_left)
        self.back_right_values.append(self.back_right)
        self.front_left_values.append(self.front_left)
        self.front_right_values.append(self.front_right)
        self.front_center_values.append(self.front_center)
        self.front_right_right_values.append(self.front_right_right)
        self.back_center_values.append(self.back_center)
        self.front_left_left_values.append(self.front_left_left)

    def save_plot(self, filename="sensor_data_plot.png"):
        """Create and save a plot of all sensor values over time"""
        plt.figure(figsize=(12, 8))

        # Plot all sensor values
        plt.plot(
            self.timestamps, self.back_left_values, label="Back Left", linewidth=1.5
        )
        plt.plot(
            self.timestamps, self.back_right_values, label="Back Right", linewidth=1.5
        )
        plt.plot(
            self.timestamps, self.front_left_values, label="Front Left", linewidth=1.5
        )
        plt.plot(
            self.timestamps, self.front_right_values, label="Front Right", linewidth=1.5
        )
        plt.plot(
            self.timestamps,
            self.front_center_values,
            label="Front Center",
            linewidth=1.5,
        )
        plt.plot(
            self.timestamps,
            self.front_right_right_values,
            label="Front Right Right",
            linewidth=1.5,
        )
        plt.plot(
            self.timestamps, self.back_center_values, label="Back Center", linewidth=1.5
        )
        plt.plot(
            self.timestamps,
            self.front_left_left_values,
            label="Front Left Left",
            linewidth=1.5,
        )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Sensor Value")
        plt.title("All Sensor Values Over Time")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {filepath}")


def keep_forward_until_obstacle(rob: IRobobo, sensor: SensorDataLogger):
    print("Moving forward until obstacle detected...")
    sensor.read_data()
    while sensor.front_center < 30:
        sensor.read_data()
        print(f"Front Center: {sensor.front_center}")
        rob.move(30, 30, 300)
    rob.reset_wheels()
    print("Obstacle detected, stopping movement.")


def turn_left(rob: IRobobo):
    rob.move_blocking(-50, 50, 600)
    rob.reset_wheels()
    print("Turned left.")


def turn_right(rob: IRobobo):
    rob.move_blocking(50, -50, 600)
    rob.reset_wheels()
    print("Turned right.")


def move_back_and_turn(rob: IRobobo, sensor: SensorDataLogger):
    while sensor.front_center > 10:
        sensor.read_data()
        print(f"Front Center: {sensor.front_center}")
        rob.move_blocking(10, 0, 100)
    rob.reset_wheels()


def run_task0(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    sensor = SensorDataLogger(rob)

    # Single run instead of loop
    i = 0

    while i < 2:

        print("Walking forward until obstacle...")
        keep_forward_until_obstacle(rob, sensor)

        turn_right(rob)
        sensor.read_data()  # Log data after turn

        rob.move_blocking(30, 30, 2500)
        sensor.read_data()  # Log data after movement

        turn_left(rob)
        sensor.read_data()

        rob.move_blocking(30, 30, 3500)
        sensor.read_data()

        turn_left(rob)
        sensor.read_data()

        rob.move_blocking(30, 30, 4000)
        sensor.read_data()

        turn_left(rob)
        sensor.read_data()

        rob.move_blocking(30, 30, 4000)
        sensor.read_data()

        turn_left(rob)
        sensor.read_data()

        i += 1
    print("Run completed. Saving plot...")

    # Save the plot
    sensor.save_plot("single_run_sensor_data.png")

    print(f"Total data points collected: {len(sensor.timestamps)}")
    print(f"Run duration: {sensor.timestamps[-1]:.2f} seconds")


# Example usage:
if __name__ == "__main__":
    # Initialize your robot here (SimulationRobobo or HardwareRobobo)
    # rob = SimulationRobobo()  # or HardwareRobobo()
    # run_task0(rob)
    pass
