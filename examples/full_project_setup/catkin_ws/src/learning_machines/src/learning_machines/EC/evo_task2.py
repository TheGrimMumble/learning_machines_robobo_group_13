import cv2
import numpy as np
import random
import time
import pickle
import os
import multiprocessing
from datetime import datetime
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

import uuid
import csv


def save_top_individuals(population, gen, top_k=5, directory=FIGURES_DIR):
    """
    Save the top_k genomes and their fitness scores of generation gen.
    """
    os.makedirs(directory, exist_ok=True)
    # Assumes population is sorted by fitness descending
    for rank, individual in enumerate(population[:top_k], start=1):
        fitness = individual.fitness
        filename = os.path.join(
            directory, f"best_gen_{gen+1}_rank_{rank}_fit_{fitness:.2f}.pkl"
        )
        with open(filename, "wb") as f:
            # store both genome and fitness
            pickle.dump(
                {
                    "genome": individual.genome,
                    "fitness": fitness,
                    "food_collected": individual.food_collected,
                    "mutation_step": individual.mutation_step,
                },
                f,
            )

        print(f"Saved gen {gen+1} rank {rank} (fitness={fitness:.2f}) to {filename}")


def load_individual(filepath):
    """
    Load a genome and fitness from a pickle file and return a new Individual.
    """
    with open(os.path.join(FIGURES_DIR, filepath), "rb") as f:
        data = pickle.load(f)
    genome = data.get("genome")
    fitness = data.get("fitness", None)
    ind = Individual(len(genome))
    ind.genome = genome
    ind.fitness = fitness
    ind.food_collected = data.get("food_collected", 0)
    ind.mutation_step = data.get("mutation_step", 0.1)
    return ind


class GreenFoodDetector:
    def __init__(self, robot):
        self.robot = robot
        self.is_simulation = hasattr(robot, "_sim")

        # Green color detection parameters
        # Im not sure if this actually works
        self.green_ranges = [
            {"lower": np.array([40, 50, 50]), "upper": np.array([80, 255, 255])},
            {"lower": np.array([35, 30, 30]), "upper": np.array([85, 255, 200])},
        ]

        self.min_contour_area = 300
        self.max_contour_area = 50000
        self.debug_mode = False

        # Camera scanning parameters (pan) - Fixed positions
        self.current_pan = 177  # Center position
        self.min_pan = 11  # Left limit
        self.max_pan = 343  # Right limit

        # Camera scanning parameters (tilt) - Fixed positions
        try:
            self.current_tilt = robot.read_phone_tilt()
        except Exception:
            self.current_tilt = 67
        self.min_tilt = 26
        self.max_tilt = 109

    def get_current_image(self):
        """Get image from simulation or hardware camera"""
        try:
            if self.is_simulation:
                return self.robot.read_image_front()
            else:
                return getattr(self.robot, "_receiving_image_front", None)
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting image: {e}")
            return None

    # in the simulation they suggest the food is green
    # green food has a certain output when seen by camera
    def detect_green_food(self, image):
        """Detect green food objects in image"""
        if image is None:
            return []
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for green_range in self.green_ranges:
            mask = cv2.inRange(hsv, green_range["lower"], green_range["upper"])
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                aspect_ratio = w / h if h > 0 else 1
                shape_score = 1.0 - abs(1.0 - aspect_ratio)
                area_score = min(area / 2000.0, 1.0)
                confidence = shape_score * 0.3 + area_score * 0.7
                detections.append(
                    {
                        "center": (cx, cy),
                        "area": area,
                        "confidence": confidence,
                        "bbox": (x, y, w, h),
                        "aspect_ratio": aspect_ratio,
                    }
                )
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def get_food_sensor_data(self, image=None):
        """Get normalized food sensor data including fixed camera pan & tilt and robot orientation"""
        if image is None:
            image = self.get_current_image()
        if image is None:
            return [0.0] * 10

        detections = self.detect_green_food(image)

        # Normalize pan/tilt (fixed values)
        norm_pan = (
            (self.current_pan - self.min_pan) / (self.max_pan - self.min_pan)
        ) * 2 - 1
        norm_tilt = (
            (self.current_tilt - self.min_tilt) / (self.max_tilt - self.min_tilt)
        ) * 2 - 1

        # Get robot orientation
        # i THINK "jaw" is what we need (check datatypes.py)
        # since this is the direction robot is looking (like a compass)
        try:
            orientation = self.robot.read_orientation()
            # Normalize yaw to [-1, 1] range
            norm_yaw = (orientation.yaw % 360) / 180.0 - 1.0
        except Exception as e:
            if self.debug_mode:
                print(f"Error reading orientation: {e}")
            norm_yaw = 0.0
            orientation = None

        if not detections:
            return [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                norm_pan,
                norm_tilt,
                0.0,
                norm_yaw,
                0.0,
            ]  # Added 10th value

        best = detections[0]
        cx, cy = best["center"]
        h, w = image.shape[:2]

        # Normalized food position relative to camera
        nx = (cx / w) * 2 - 1
        ny = (cy / h) * 2 - 1
        na = min(best["area"] / 5000.0, 1.0)
        conf = best["confidence"]
        detected = 1.0

        # Calculate absolute food direction (existing code)
        camera_angle_rad = (self.current_pan - 177) * (3.14159 / 180)
        food_angle_in_camera_rad = nx * (30 * 3.14159 / 180)
        absolute_food_angle_rad = camera_angle_rad + food_angle_in_camera_rad

        if orientation is not None:
            robot_yaw_rad = orientation.yaw * (3.14159 / 180)
            food_direction_world = robot_yaw_rad + absolute_food_angle_rad
            food_direction_normalized = (
                food_direction_world % (2 * 3.14159)
            ) / 3.14159 - 1.0

            # THis can be very important i think, the angle to the food if i did it correctly here
            # We could create an extra feature combining this angle with a translation
            # of wheel speeds into angle.
            angle_to_food_rad = (
                absolute_food_angle_rad  # This is already relative to robot
            )
            # Normalize to [-1, 1] range (representing -180° to +180°)
            angle_to_food_normalized = angle_to_food_rad / 3.14159
            # Clamp to [-1, 1] range
            angle_to_food_normalized = max(-1.0, min(1.0, angle_to_food_normalized))
        else:
            food_direction_normalized = 0.0
            angle_to_food_normalized = 0.0

        return [
            nx,
            ny,
            na,
            conf,
            detected,
            norm_pan,
            norm_tilt,
            food_direction_normalized,
            norm_yaw,
            angle_to_food_normalized,
        ]


# Neural network for discrete action selection
class MultiSensorNeuralNetwork:
    def __init__(
        self, input_size=10, hidden1_size=8, hidden2_size=8, output_size=4
    ):  # 4 outputs for action selection
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        self.weights_input_hidden1 = input_size * hidden1_size
        self.biases_hidden1 = hidden1_size
        self.weights_hidden1_hidden2 = hidden1_size * hidden2_size
        self.biases_hidden2 = hidden2_size
        self.weights_hidden2_output = hidden2_size * output_size
        self.biases_output = output_size

        self.total_params = (
            self.weights_input_hidden1
            + self.biases_hidden1
            + self.weights_hidden1_hidden2
            + self.biases_hidden2
            + self.weights_hidden2_output
            + self.biases_output
        )

    def set_weights_from_genome(self, genome):
        idx = 0

        self.w_ih1 = np.array(genome[idx : idx + self.weights_input_hidden1]).reshape(
            self.input_size, self.hidden1_size
        )
        idx += self.weights_input_hidden1

        self.b_h1 = np.array(genome[idx : idx + self.biases_hidden1])
        idx += self.biases_hidden1

        self.w_h1h2 = np.array(
            genome[idx : idx + self.weights_hidden1_hidden2]
        ).reshape(self.hidden1_size, self.hidden2_size)
        idx += self.weights_hidden1_hidden2

        self.b_h2 = np.array(genome[idx : idx + self.biases_hidden2])
        idx += self.biases_hidden2

        self.w_h2o = np.array(genome[idx : idx + self.weights_hidden2_output]).reshape(
            self.hidden2_size, self.output_size
        )
        idx += self.weights_hidden2_output

        self.b_o = np.array(genome[idx : idx + self.biases_output])

    def forward(self, inputs):
        x = np.array(inputs).reshape(1, self.input_size)
        h1 = np.tanh(x.dot(self.w_ih1) + self.b_h1)
        h2 = np.tanh(h1.dot(self.w_h1h2) + self.b_h2)
        o = np.tanh(h2.dot(self.w_h2o) + self.b_o)

        return o.flatten()


class Individual:
    def __init__(self, genome_length=None):
        self.network = MultiSensorNeuralNetwork(
            10, 8, 8, 4
        )  # 4 outputs for discrete actions
        if genome_length is None:
            genome_length = self.network.total_params
        self.genome = self._xavier_initialize()
        self.fitness = 0.0
        self.survival_time = 0.0
        self.behavior_diversity = 0.0
        self.min_distance = float("inf")
        self.food_collected = 0
        self.mutation_step = 0.1

    def _xavier_initialize(self):
        params = []

        limit = np.sqrt(6.0 / (self.network.input_size + self.network.hidden1_size))
        params += list(
            np.random.uniform(-limit, limit, self.network.weights_input_hidden1)
        )

        params += list(np.random.uniform(-0.1, 0.1, self.network.biases_hidden1))

        limit = np.sqrt(6.0 / (self.network.hidden1_size + self.network.hidden2_size))
        params += list(
            np.random.uniform(-limit, limit, self.network.weights_hidden1_hidden2)
        )

        params += list(np.random.uniform(-0.1, 0.1, self.network.biases_hidden2))

        limit = np.sqrt(6.0 / (self.network.hidden2_size + self.network.output_size))
        params += list(
            np.random.uniform(-limit, limit, self.network.weights_hidden2_output)
        )

        params += list(np.random.uniform(-0.1, 0.1, self.network.biases_output))

        return params

    def get_sensor_inputs(self, food_data):
        return food_data

    def get_motor_commands(self, food_data, detector):
        inputs = self.get_sensor_inputs(food_data)
        self.network.set_weights_from_genome(self.genome)
        out = self.network.forward(inputs)

        # Convert outputs to discrete action selection (like original code)
        action_idx = np.argmax(out)  # Select action with highest output

        # Define the 4 discrete motor actions
        motor_actions = [
            (100, 100),  # Action 0: Forward
            (-100, -100),  # Action 1: Backward
            (100, -100),  # Action 2: Right turn
            (-100, 100),  # Action 3: Left turn
        ]

        lw, rw = motor_actions[action_idx]

        # Return fixed camera positions (center)
        return lw, rw, 177, 67  # Fixed pan=177, tilt=67

    def gaussian_mutate(self, mutation_rate=0.1):
        """Mutate individual with given mutation rate"""
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                noise = np.random.normal(0, self.mutation_step)
                self.genome[i] = np.clip(self.genome[i] + noise, -5.0, 5.0)

    def crossover(self, other, crossover_rate=0.7):
        """Single-point crossover with another individual"""
        if random.random() > crossover_rate:
            return self.copy(), other.copy()

        child1 = self.copy()
        child2 = other.copy()

        # Single-point crossover
        crossover_point = random.randint(1, len(self.genome) - 1)

        child1.genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        child2.genome = other.genome[:crossover_point] + self.genome[crossover_point:]

        return child1, child2

    def copy(self):
        new = Individual(len(self.genome))
        new.genome = self.genome.copy()
        new.fitness = self.fitness
        new.survival_time = self.survival_time
        new.behavior_diversity = self.behavior_diversity
        new.min_distance = self.min_distance
        new.food_collected = self.food_collected
        new.mutation_step = self.mutation_step
        return new


def fitness_evaluation(
    rob: IRobobo,
    individual: Individual,
    food_detector: GreenFoodDetector,
    initial_pos,
    initial_orient,
    max_time=30.0,
):
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        time.sleep(0.5)
        rob.play_simulation()
        time.sleep(1.0)
        rob.set_position(initial_pos, initial_orient)

    print(f"Current orientation: {initial_orient}")

    # Set camera to fixed center position
    rob.set_phone_pan_blocking(177, 50)
    rob.set_phone_tilt_blocking(67, 50)
    food_detector.current_pan = 177
    food_detector.current_tilt = 67

    start_time = time.time()
    movement_count = 0
    total_distance_traveled = 0.0
    behavior_states = []

    initial_food_count = (
        rob.get_nr_food_collected() if hasattr(rob, "get_nr_food_collected") else 0
    )

    repeat_penalty = 0
    action_prev = None
    penalty_weight = 5.0

    time_with_food = 0
    approach_bonus = 0

    total_speed_reward = 0.0

    # Counter for yaw display (similar to food detection display)
    step_counter = 0

    try:
        while time.time() - start_time < max_time:
            food_data = food_detector.get_food_sensor_data()
            lw, rw, _, _ = individual.get_motor_commands(
                food_data, food_detector
            )  # Ignore pan/tilt outputs

            # Determine which action was taken for behavior tracking
            if (lw, rw) == (100, 100):
                current_action = 0  # Forward
            elif (lw, rw) == (-100, -100):
                current_action = 1  # Backward
            elif (lw, rw) == (100, -100):
                current_action = 2  # Right
            elif (lw, rw) == (-100, 100):
                current_action = 3  # Left
            else:
                current_action = -1  # Should not happen with discrete actions

            # Behavior state tracking (action index)
            behavior_states.append(current_action)

            # Display food coordinates when detected
            if food_data[4] > 0:
                print(
                    f"Food detected at norm‐coords x={food_data[0]:.2f}, y={food_data[1]:.2f}, area={food_data[2]:.2f}"
                )
                time_with_food += 1
                if abs(lw) > 10 or abs(rw) > 10:
                    approach_bonus += 1

            # Display yaw values periodically (every 10 steps, or when food is detected)
            step_counter += 1
            if step_counter % 10 == 0 or food_data[4] > 0:
                # Convert normalized robot yaw back to degrees for display
                robot_yaw_degrees = (food_data[8] + 1.0) * 180.0
                print(
                    f"Robot yaw: {robot_yaw_degrees:.1f}° (normalized: {food_data[8]:.3f})"
                )

                # Also show food direction when food is detected
                if food_data[4] > 0:
                    food_direction_degrees = (food_data[7] + 1.0) * 180.0
                    print(
                        f"Food direction: {food_direction_degrees:.1f}° (normalized: {food_data[7]:.3f})"
                    )
                    # Show the difference between robot yaw and food direction
                    angle_diff = food_direction_degrees - robot_yaw_degrees
                    # Normalize angle difference to [-180, 180]
                    while angle_diff > 180:
                        angle_diff -= 360
                    while angle_diff < -180:
                        angle_diff += 360
                    print(f"Angle to food: {angle_diff:.1f}° (food - robot)")

                # Display current action
                action_names = ["Forward", "Backward", "Right", "Left"]
                if current_action >= 0:
                    print(f"Action: {action_names[current_action]} ({lw}, {rw})")

            # Repeat penalty for same action
            if action_prev is not None and current_action == action_prev:
                repeat_penalty += 0.0001

            action_prev = current_action

            # Speed reward calculation (same as continuous version)
            avg_speed = (lw + rw) / 2.0
            forwardness = 1.0 - abs(lw - rw) / (abs(lw) + abs(rw) + 1e-5)
            speed_reward = abs(avg_speed) * forwardness
            total_speed_reward += speed_reward * 0.15

            rob.move_blocking(lw, rw, 50)
            movement_count += 1
            total_distance_traveled += abs(lw + rw) * 0.15

    except Exception as e:
        print(f"Error during evaluation: {e}")

    rob.move_blocking(0, 0, 200)
    # Keep camera in center position
    rob.set_phone_pan_blocking(177, 50)
    rob.set_phone_tilt_blocking(67, 50)

    if isinstance(rob, SimulationRobobo):
        rob.set_position(initial_pos, initial_orient)

    survival_time = time.time() - start_time
    individual.survival_time = survival_time

    final_food_count = (
        rob.get_nr_food_collected() if hasattr(rob, "get_nr_food_collected") else 0
    )
    individual.food_collected = final_food_count - initial_food_count

    uniq = len(set(behavior_states))
    total = len(behavior_states)
    individual.behavior_diversity = uniq / max(1, total)

    base = survival_time * 2
    food_bonus = individual.food_collected * 300.0
    detect_bonus = time_with_food * 0.2
    approach = approach_bonus * 0.2
    explore = total_distance_traveled * 0.05
    diversity = individual.behavior_diversity * 0.05

    # Simplified fitness calculation (removed camera bonuses/penalties)
    repeat_norm = repeat_penalty / max(1, movement_count)
    repeat_score = repeat_norm * penalty_weight

    individual.fitness = (
        base
        + food_bonus
        + detect_bonus
        + approach
        + explore
        + diversity
        - repeat_score
        + total_speed_reward
    )

    return individual.fitness


def evaluate_individual_worker_continuous(
    port_offset, individual_queue, result_queue, initial_pos, initial_orient, max_time
):
    """Continuous worker function that processes individuals from a queue"""
    import queue

    port = 20000 + port_offset
    # print(f"Worker on port {port} starting...")

    # Create robot connection for this worker
    rob = SimulationRobobo(api_port=port)

    try:
        # Start the simulation for this worker
        rob.play_simulation()
        # print(f"Worker on port {port} simulation started")

        # Create food detector for this worker
        food_detector = GreenFoodDetector(rob)

        while True:
            try:
                # Get next individual from queue (with timeout to avoid hanging)
                individual_data, individual_index = individual_queue.get(timeout=1.0)

                # print(
                #     f"Worker on port {port} evaluating individual {individual_index+1}"
                # )

                # Recreate individual from data
                individual = Individual()
                individual.genome = individual_data["genome"]
                individual.fitness = individual_data["fitness"]
                individual.survival_time = individual_data["survival_time"]
                individual.behavior_diversity = individual_data["behavior_diversity"]
                individual.min_distance = individual_data["min_distance"]
                individual.food_collected = individual_data["food_collected"]
                individual.mutation_step = individual_data["mutation_step"]

                # Evaluate the individual
                fitness = fitness_evaluation(
                    rob,
                    individual,
                    food_detector,
                    initial_pos,
                    initial_orient,
                    max_time,
                )

                # print(
                #     f"Individual {individual_index+1} finished on port {port} with fitness {individual.fitness:.2f}"
                # )

                # Put result back
                result_data = {
                    "index": individual_index,
                    "genome": individual.genome,
                    "fitness": individual.fitness,
                    "survival_time": individual.survival_time,
                    "behavior_diversity": individual.behavior_diversity,
                    "min_distance": individual.min_distance,
                    "food_collected": individual.food_collected,
                    "mutation_step": individual.mutation_step,
                }
                result_queue.put(result_data)

            except (queue.Empty, Exception) as e:
                # No more individuals in queue or timeout - exit
                # print(f"Worker on port {port} exiting: {e}")
                break

    finally:
        # Stop the simulation for this worker
        try:
            rob.stop_simulation()
            # print(f"Worker on port {port} stopped")
        except:
            pass  # Ignore errors when stopping simulation


def tournament_selection(population, tournament_size=3):
    """Tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)


def run_individual_from_file(
    file_path, rob: IRobobo = None, max_time=6000.0, is_hardware=False
):
    """
    Runs a single loaded individual in the simulation for up to `max_time` seconds.
    Logs sensor data to a uniquely named CSV file in the same directory as `file_path`.
    Writes each row to the CSV file in real-time (not just at the end).
    """
    if rob is None:
        if is_hardware:
            rob = HardwareRobobo(camera=True)
        else:
            rob = SimulationRobobo(api_port=20000)
            rob.play_simulation()
        stop_after = True
    else:
        stop_after = False

    try:
        individual = load_individual(file_path)
        food_detector = GreenFoodDetector(rob)

        if not is_hardware:
            initial_pos = rob.get_position()
            initial_orient = rob.get_orientation()
            rob.set_position(initial_pos, initial_orient)
            rob.set_phone_pan_blocking(177, 50)
            rob.set_phone_tilt_blocking(67, 50)

        print(f"Running individual from: {file_path}")
        print("Press Ctrl+C to stop manually.\n")

        start_time = time.time()
        total_reward = 0.0
        movement_count = 0
        initial_food_count = (
            rob.get_nr_food_collected() if hasattr(rob, "get_nr_food_collected") else 0
        )

        # Prepare CSV for live logging
        base_dir = os.path.dirname(file_path)
        unique_id = uuid.uuid4().hex[:8]
        csv_filename = os.path.join(base_dir, f"food_sensor_log_{unique_id}.csv")
        csv_headers = (
            ["timestamp"]
            + [f"food_sensor_{i}" for i in range(10)]
            + ["L_speed", "R_speed", "reward", "food_detected"]
        )

        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)  # Write header once

            while time.time() - start_time < max_time:
                food_data = food_detector.get_food_sensor_data()
                print(f"Food sensor readings: {[f'{x:.2f}' for x in food_data]}")

                reward = 0
                lw, rw, _, _ = individual.get_motor_commands(food_data, food_detector)

                # Calculate reward similar to fitness evaluation
                if food_data[4] > 0:  # food detected
                    reward += 1.0

                avg_speed = (lw + rw) / 2.0
                forwardness = 1.0 - abs(lw - rw) / (abs(lw) + abs(rw) + 1e-5)
                speed_reward = abs(avg_speed) * forwardness * 0.15
                reward += speed_reward

                total_reward += reward
                movement_count += 1

                timestamp = time.time() - start_time
                writer.writerow(
                    [timestamp] + food_data + [lw, rw, reward, int(food_data[4] > 0)]
                )
                f.flush()  # Ensure data is written immediately

                current_food_count = (
                    rob.get_nr_food_collected()
                    if hasattr(rob, "get_nr_food_collected")
                    else 0
                )
                food_collected = current_food_count - initial_food_count

                print(
                    f"Time: {timestamp:.2f}s | "
                    f"Speed: L={lw} R={rw} | "
                    f"Food collected: {food_collected} | "
                    f"Reward: {total_reward:.2f}"
                )

                rob.move_blocking(lw, rw, 50)

        rob.move_blocking(0, 0, 50)
        final_food_count = (
            rob.get_nr_food_collected() if hasattr(rob, "get_nr_food_collected") else 0
        )
        total_food_collected = final_food_count - initial_food_count

        print("\n--- Evaluation Finished ---")
        print(f"Final reward: {total_reward:.2f}")
        print(f"Total food collected: {total_food_collected}")
        print(f"Survival time: {time.time() - start_time:.2f}s")
        print(f"Sensor log saved to: {csv_filename}")

    except Exception as e:
        print(f"Error while running individual: {e}")

    finally:
        if stop_after and not is_hardware:
            rob.stop_simulation()


def genetic_algorithm(rob: IRobobo, parallel=False, num_processes=10):
    """Full genetic algorithm implementation with (μ,λ) selection"""
    # GA Parameters
    MU = 30  # Number of parents
    LAMBDA = 60  # Number of offspring
    GENERATIONS = 100
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.2
    TOURNAMENT_SIZE = 3
    ELITISM_COUNT = 1  # Number of best individuals to preserve

    print(f"Running Genetic Algorithm with (μ,λ) selection")
    print(f"Parents (μ): {MU}, Offspring (λ): {LAMBDA}")
    print(f"Generations: {GENERATIONS}")
    print(f"Crossover rate: {CROSSOVER_RATE}, Mutation rate: {MUTATION_RATE}")
    print(f"Tournament size: {TOURNAMENT_SIZE}, Elitism: {ELITISM_COUNT}")
    print("Features: Fixed camera position, 4 discrete actions, food tracking")
    if parallel:
        print(f"Parallel mode: {num_processes} processes")

    # Get initial position and orientation
    if parallel:
        # For parallel mode, get initial position from a temporary connection
        print("Getting initial position from temporary CoppeliaSim connection...")
        temp_rob = SimulationRobobo(api_port=20000)
        try:
            temp_rob.play_simulation()
            start_pos = temp_rob.get_position()
            start_orient = temp_rob.get_orientation()
            temp_rob.stop_simulation()
            print(f"Initial position: {start_pos}, orientation: {start_orient}")
        except Exception as e:
            print(f"Could not get initial position: {e}")
            # Use default values if we can't get position
            start_pos = [0.0, 0.0, 0.0]
            start_orient = 0.0
    else:
        food_detector = GreenFoodDetector(rob)
        start_pos = rob.get_position()
        start_orient = rob.get_orientation()

    # Initialize population
    population = [Individual() for _ in range(MU)]

    # Track best individual
    best_individual = None
    best_fitness = float("-inf")

    # CSV logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(FIGURES_DIR, f"food_training_history_{timestamp}.csv")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Write CSV header once
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "generation",
                "index",
                "fitness",
                "survival_time",
                "food_collected",
            ],
        )
        writer.writeheader()

    # Evaluate initial population
    print("Evaluating initial population...")
    if parallel:
        import multiprocessing as mp

        # Create queues for communication
        individual_queue = mp.Queue()
        result_queue = mp.Queue()

        # Add all individuals to the queue
        for i, individual in enumerate(population):
            data = {
                "genome": individual.genome,
                "fitness": individual.fitness,
                "survival_time": individual.survival_time,
                "behavior_diversity": individual.behavior_diversity,
                "min_distance": individual.min_distance,
                "food_collected": individual.food_collected,
                "mutation_step": individual.mutation_step,
            }
            individual_queue.put((data, i))

        # Start worker processes
        processes = []
        for port_offset in range(num_processes):
            p = mp.Process(
                target=evaluate_individual_worker_continuous,
                args=(
                    port_offset,
                    individual_queue,
                    result_queue,
                    start_pos,
                    start_orient,
                    30.0,
                ),
            )
            p.start()
            processes.append(p)

        # Collect results as they come in
        results_collected = 0
        while results_collected < len(population):
            try:
                result = result_queue.get(
                    timeout=300
                )  # 5 minute timeout per individual

                # Update the individual with results
                idx = result["index"]
                population[idx].genome = result["genome"]
                population[idx].fitness = result["fitness"]
                population[idx].survival_time = result["survival_time"]
                population[idx].behavior_diversity = result["behavior_diversity"]
                population[idx].min_distance = result["min_distance"]
                population[idx].food_collected = result["food_collected"]
                population[idx].mutation_step = result["mutation_step"]

                results_collected += 1
                print(
                    f" Individual {idx+1}/{MU} completed - Fitness: {population[idx].fitness:.2f}, Food: {population[idx].food_collected} ({results_collected}/{len(population)} done)"
                )

            except Exception as e:
                print(f"Timeout or error waiting for results: {e}")
                break

        # Wait for all processes to finish and clean up
        for p in processes:
            p.terminate()
            p.join()

        print(f"Initial population evaluation complete!")
    else:
        for i, individual in enumerate(population):
            fitness = fitness_evaluation(
                rob, individual, food_detector, start_pos, start_orient
            )
            print(
                f"Individual {i+1}/{MU}: fitness={fitness:.2f}, food={individual.food_collected}"
            )

    # Find initial best
    current_best = max(population, key=lambda x: x.fitness)
    best_individual = current_best.copy()
    best_fitness = current_best.fitness
    print(
        f"Initial best fitness: {best_fitness:.2f}, food: {best_individual.food_collected}"
    )

    # Evolution loop
    for generation in range(GENERATIONS):
        print(f"\n--- Generation {generation + 1}/{GENERATIONS} ---")

        # Sort population by fitness for elitism
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Create offspring population
        offspring = []

        # Elitism: preserve best individuals
        for i in range(ELITISM_COUNT):
            offspring.append(population[i].copy())

        # Generate rest of offspring through crossover and mutation
        while len(offspring) < LAMBDA:
            # Selection
            parent1 = tournament_selection(population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)

            # Crossover
            child1, child2 = parent1.crossover(parent2, CROSSOVER_RATE)

            # Mutation
            child1.gaussian_mutate(MUTATION_RATE)
            child2.gaussian_mutate(MUTATION_RATE)

            offspring.extend([child1, child2])

        # Trim offspring to exact size
        offspring = offspring[:LAMBDA]

        # Evaluate offspring
        print(f"Evaluating {len(offspring)} offspring...")

        if parallel:
            import multiprocessing as mp

            # Create queues for communication
            individual_queue = mp.Queue()
            result_queue = mp.Queue()

            # Add all offspring to the queue (skip elite individuals)
            for i, individual in enumerate(offspring):
                if i < ELITISM_COUNT:
                    continue  # Skip evaluation for elite individuals
                data = {
                    "genome": individual.genome,
                    "fitness": individual.fitness,
                    "survival_time": individual.survival_time,
                    "behavior_diversity": individual.behavior_diversity,
                    "min_distance": individual.min_distance,
                    "food_collected": individual.food_collected,
                    "mutation_step": individual.mutation_step,
                }
                individual_queue.put((data, i))

            # Start worker processes
            processes = []
            for port_offset in range(num_processes):
                p = mp.Process(
                    target=evaluate_individual_worker_continuous,
                    args=(
                        port_offset,
                        individual_queue,
                        result_queue,
                        start_pos,
                        start_orient,
                        30.0,
                    ),
                )
                p.start()
                processes.append(p)

            # Collect results as they come in
            results_collected = 0
            offspring_to_evaluate = len(offspring) - ELITISM_COUNT
            while results_collected < offspring_to_evaluate:
                try:
                    result = result_queue.get(
                        timeout=300
                    )  # 5 minute timeout per individual

                    # Update the individual with results
                    idx = result["index"]
                    offspring[idx].genome = result["genome"]
                    offspring[idx].fitness = result["fitness"]
                    offspring[idx].survival_time = result["survival_time"]
                    offspring[idx].behavior_diversity = result["behavior_diversity"]
                    offspring[idx].min_distance = result["min_distance"]
                    offspring[idx].food_collected = result["food_collected"]
                    offspring[idx].mutation_step = result["mutation_step"]

                    results_collected += 1
                    if (
                        results_collected
                    ) % 10 == 0 or results_collected == offspring_to_evaluate:
                        print(
                            f" Offspring {results_collected}/{offspring_to_evaluate} completed - Fitness: {offspring[idx].fitness:.2f}, Food: {offspring[idx].food_collected}"
                        )

                except Exception as e:
                    print(f"Timeout or error waiting for results: {e}")
                    break

            # Wait for all processes to finish and clean up
            for p in processes:
                p.terminate()
                p.join()

            print(f"Generation {generation+1} evaluation complete!")
        else:
            for i, individual in enumerate(offspring):
                if i < ELITISM_COUNT:
                    # Skip evaluation for elite individuals (already evaluated)
                    continue
                fitness = fitness_evaluation(
                    rob, individual, food_detector, start_pos, start_orient
                )
                if (i + 1) % 10 == 0 or i == len(offspring) - 1:
                    print(
                        f"Offspring {i+1}/{len(offspring)}: fitness={fitness:.2f}, food={individual.food_collected}"
                    )

        # (μ,λ) selection: replace parents with offspring
        population = offspring[:MU]  # Select μ best offspring
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Save top individuals
        save_top_individuals(population, generation, top_k=5)

        # Update best individual
        current_best = population[0]
        if current_best.fitness > best_fitness:
            best_individual = current_best.copy()
            best_fitness = current_best.fitness
            print(
                f"NEW BEST in generation {generation + 1}: fitness={best_fitness:.2f}, food={best_individual.food_collected}"
            )

        # Statistics
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        print(f"Generation {generation + 1} stats:")
        print(
            f"  Best: {population[0].fitness:.2f} (food: {population[0].food_collected})"
        )
        print(f"  Average: {avg_fitness:.2f}")
        print(f"  Worst: {population[-1].fitness:.2f}")
        print(
            f"  Overall best: {best_fitness:.2f} (food: {best_individual.food_collected})"
        )

        # Log to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "generation",
                    "index",
                    "fitness",
                    "survival_time",
                    "food_collected",
                ],
            )
            for idx, ind in enumerate(population):
                writer.writerow(
                    {
                        "generation": generation + 1,
                        "index": idx,
                        "fitness": ind.fitness,
                        "survival_time": ind.survival_time,
                        "food_collected": ind.food_collected,
                    }
                )

    print(f"\nEvolution complete!")
    print(f"Best fitness achieved: {best_fitness:.2f}")
    print(f"Food collected by best: {best_individual.food_collected}")

    return None, best_individual


def run_neuroevolution(
    rob: IRobobo = None,
    parallel=False,
    num_processes=10,
    file_path=None,
    is_hardware=False,
):
    if parallel:
        print("Starting parallel neuroevolution with multiple CoppeliaSim instances")
        print("Each worker will manage its own simulation instance")
    elif isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    if file_path is not None:
        run_individual_from_file(file_path, rob, is_hardware=is_hardware)
    else:
        save_dir, best_individual = genetic_algorithm(rob, parallel, num_processes)

        if not parallel and isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

        return save_dir, best_individual


def run_all_actions(
    rob: IRobobo = None,
    parallel=False,
    num_processes=10,
    file_path=None,
    is_hardware=False,
):
    return run_neuroevolution(rob, parallel, num_processes, file_path, is_hardware)
