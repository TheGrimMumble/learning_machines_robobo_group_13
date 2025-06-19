# TODO:
# 1 add history
# 2 fix bug where object goes unnoticed when right in front of camera
# 3 create map of seen objects and taken objects
# 4 add as input whether or not the robot actually collected the food
# 5 avoid walls
# 6 give all detected blobs as input

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
from pathlib import Path

IMAGE_OUTPUT_DIR = Path(FIGURES_DIR) / "images"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        self.max_contour_area = 50000000
        self.debug_mode = False

        self.last_saved_time = 0  # Track last image save time

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
            # if self.is_simulation:
            return self.robot.read_image_front()
            # else:
            #     return getattr(self.robot, "_receiving_image_front", None)
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting image: {e}")
            return None

    # in the simulation they suggest the food is green
    # green food has a certain output when seen by camera
    def detect_green_food(self, image):
        """Detect green food objects in image and save all intermediate processing steps every 1 second"""
        if image is None:
            return []

        detections = []
        current_time = time.time()
        should_save = False  # current_time - self.last_saved_time >= 5.0

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if should_save:
            cv2.imwrite(str(IMAGE_OUTPUT_DIR / f"hsv_{int(current_time)}.png"), hsv)

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Process each green range
        for i, green_range in enumerate(self.green_ranges):
            mask = cv2.inRange(hsv, green_range["lower"], green_range["upper"])
            if should_save:
                cv2.imwrite(
                    str(IMAGE_OUTPUT_DIR / f"mask_range_{i}_{int(current_time)}.png"),
                    mask,
                )
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        if should_save:
            cv2.imwrite(
                str(IMAGE_OUTPUT_DIR / f"mask_cleaned_{int(current_time)}.png"),
                combined_mask,
            )

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        debug_image = image.copy()
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
                # Draw detection box
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Optionally draw all contours in blue
        cv2.drawContours(debug_image, contours, -1, (255, 0, 0), 1)

        if should_save:
            cv2.imwrite(
                str(IMAGE_OUTPUT_DIR / f"image_{int(current_time)}.png"), debug_image
            )
            self.last_saved_time = current_time

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return detections

    def get_food_sensor_data(self, ir_readings, is_hardware=False, image=None):
        def normalize(val, min_val, max_val):
            theoretical_min_log = np.log(min_val)
            theoretical_max_log = np.log(max_val)

            if val < min_val:
                val = min_val
            elif val > max_val:
                val = max_val

            # Normalize to [0, 1] using log scale
            normalized = (np.log(val) - theoretical_min_log) / (
                theoretical_max_log - theoretical_min_log
            )
            return max(0.0, min(1.0, normalized))

        if not is_hardware:
            # software
            front_left = normalize(ir_readings[2], 52, 2500)
            front_center = normalize(ir_readings[4], 5, 2500)
            front_right = normalize(ir_readings[3], 52, 2500)
            front_left_left = normalize(ir_readings[7], 5, 500)
            front_right_right = normalize(ir_readings[5], 5, 250)
            back_left = normalize(ir_readings[0], 6, 2500)
            back_right = normalize(ir_readings[1], 6, 2500)
            back_center = normalize(ir_readings[6], 57, 2500)
        else:
            # hardware
            front_left = normalize(ir_readings[2], 28, 3000)
            front_center = normalize(ir_readings[4], 9, 1500)
            front_right = normalize(ir_readings[3], 34, 3000)
            front_left_left = normalize(ir_readings[7], 7, 1500)
            front_right_right = normalize(ir_readings[5], 10, 500)
            back_left = normalize(ir_readings[0], 7, 2500)
            back_right = normalize(ir_readings[1], 10, 2500)
            back_center = normalize(ir_readings[6], 15, 2500)

        if image is None:
            image = self.get_current_image()

        detections = []
        if image is not None:
            detections = self.detect_green_food(image)

        nx, ny, na = 0.0, 0.0, 0.0
        found_food = False

        if detections:
            best = detections[0]
            cx, cy = best["center"]
            h, w = image.shape[:2]

            # Normalized food position relative to camera
            nx = (cx / w) * 2 - 1
            ny = (cy / h) * 2 - 1
            na = min(best["area"] / 5000.0, 1.0)
            found_food = True

        return [
            nx,
            ny,
            na,
            front_left,
            front_center,
            front_right,
            front_left_left,
            front_right_right,
            back_left,
            back_right,
            back_center,
        ], found_food


class MultiSensorNeuralNetwork:
    def __init__(self, input_size=11, hidden_layers=[8], output_size=2):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Calculate total number of parameters
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.shapes = [
            (layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ]

        self.weight_sizes = [in_dim * out_dim for in_dim, out_dim in self.shapes]
        self.bias_sizes = [out_dim for _, out_dim in self.shapes]

        self.total_params = sum(self.weight_sizes) + sum(self.bias_sizes)

    def set_weights_from_genome(self, genome):
        idx = 0
        self.weights = []
        self.biases = []

        for (in_dim, out_dim), w_size, b_size in zip(
            self.shapes, self.weight_sizes, self.bias_sizes
        ):
            weight = np.array(genome[idx : idx + w_size]).reshape(in_dim, out_dim)
            idx += w_size
            bias = np.array(genome[idx : idx + b_size])
            idx += b_size

            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        x = np.array(inputs).reshape(1, self.input_size)

        for i in range(len(self.weights) - 1):
            x = np.tanh(x.dot(self.weights[i]) + self.biases[i])  # Hidden layers: tanh

        # output = self.sigmoid(
        #     x.dot(self.weights[-1]) + self.biases[-1]
        # )  # Output layer: sigmoid
        output = np.tanh(
            x.dot(self.weights[-1]) + self.biases[-1]
        )  # Output layer: tanh
        return output.flatten()

    # def forward(self, inputs):
    #     x = np.array(inputs).reshape(1, self.input_size)

    #     for i in range(len(self.weights) - 1):
    #         x = np.tanh(x.dot(self.weights[i]) + self.biases[i])

    #     # Final layer
    #     output = np.tanh(x.dot(self.weights[-1]) + self.biases[-1])
    #     return output.flatten()


class Individual:
    def __init__(self, genome_length=None):
        self.network = MultiSensorNeuralNetwork()
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

        for in_dim, out_dim in self.network.shapes:
            # Xavier initialization limit for weights
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            weight_params = np.random.uniform(-limit, limit, in_dim * out_dim)
            bias_params = np.random.uniform(-0.1, 0.1, out_dim)

            params += list(weight_params)
            params += list(bias_params)

        return params

    def get_motor_commands(self, inputs):
        self.network.set_weights_from_genome(self.genome)
        out = self.network.forward(
            inputs
        )  # Output: [left_motor_activation, right_motor_activation]

        # Map output from [0, 1] to [-100, 100]
        # left_speed = int((out[0] - 0.5) * 200)
        # right_speed = int((out[1] - 0.5) * 200)
        left_speed = int(out[0] * 100)  # Scale to [-100, 100]
        right_speed = int(out[1] * 100)  # Scale to [-100, 100]
        # return 0, 0, 177, 67

        return left_speed, right_speed

    # def get_motor_commands(self, food_data, detector):
    #     inputs = self.get_sensor_inputs(food_data)
    #     self.network.set_weights_from_genome(self.genome)
    #     out = self.network.forward(inputs) * 100
    #     print(f"Network output: {out}")

    #     lw, rw = out[0], out[1]

    #     # # Convert outputs to discrete action selection
    #     # action_idx = np.argmax(out)

    #     # # Define the 4 discrete motor actions
    #     # motor_actions = [
    #     #     (100, 100),  # Forward
    #     #     (-100, -100),  # Backward
    #     #     (100, -100),  # Turn right
    #     #     (-100, 100),  # Turn left
    #     # ]

    #     # lw, rw = motor_actions[action_idx]

    #     return lw, rw, 177, 67  # Fixed camera angles

    def gaussian_mutate(self, mutation_rate=0.1):
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                noise = np.random.normal(0, self.mutation_step)
                self.genome[i] = np.clip(self.genome[i] + noise, -5.0, 5.0)

    def crossover(self, other, crossover_rate=0.7):
        if random.random() > crossover_rate:
            return self.copy(), other.copy()

        child1 = self.copy()
        child2 = other.copy()

        crossover_point = random.randint(1, len(self.genome) - 1)

        child1.genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        child2.genome = other.genome[:crossover_point] + self.genome[crossover_point:]

        return child1, child2

    def copy(self):
        new = Individual()
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
    is_parallel_worker=False,
):
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        time.sleep(0.5)
        rob.play_simulation()
        time.sleep(1.0)

    rob.set_phone_pan_blocking(177, 100)
    rob.set_phone_tilt_blocking(100, 100)
    food_detector.current_pan = 177
    food_detector.current_tilt = 100

    start_time = time.time()

    initial_food_count = (
        rob.get_nr_food_collected() if hasattr(rob, "get_nr_food_collected") else 0
    )

    forward_speed_reward = 0.0
    food_reward = 0.0
    current_food_count = initial_food_count

    collision_count = 0

    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time

        ir = rob.read_irs()
        valid = [r for r in ir if r is not None]
        collision = any(r > 300 for r in valid[0:8])

        if collision:
            collision_count += 1
        else:
            collision_count = 0

        if collision_count >= 3:
            forward_speed_reward -= 3.0

            collision_count = 0

        food_data, found_food = food_detector.get_food_sensor_data(ir)
        lw, rw = individual.get_motor_commands(food_data)
        # rob.move_blocking(lw, rw, 150)
        rob.move(lw, rw, 300)  # Non-blocking move for continuous evaluation

        # Forward speed component
        if found_food:  # rewards for moving forward only when food is detected
            speed_norm = max(0.0, (lw + rw) / 200.0)
            turn_penalty = abs(lw - rw) / 200.0
            forwardness = 1.0 - turn_penalty
            forward_speed_reward += speed_norm * forwardness

        # Check if new food has been collected
        new_food_count = (
            rob.get_nr_food_collected()
            if hasattr(rob, "get_nr_food_collected")
            else current_food_count
        )

        if new_food_count > current_food_count:
            # Give higher reward the earlier food is collected
            for _ in range(new_food_count - current_food_count):
                food_reward += 1.0 * (1.0 - elapsed / max_time)
            current_food_count = new_food_count

            if current_food_count - initial_food_count >= 7:
                food_reward += 5.0  # Bonus for collecting all food early
                break  # All food collected early

    rob.move_blocking(0, 0, 200)

    survival_time = time.time() - start_time
    individual.survival_time = survival_time

    final_food_count = (
        rob.get_nr_food_collected() if hasattr(rob, "get_nr_food_collected") else 0
    )
    individual.food_collected = final_food_count - initial_food_count

    # Final fitness: early food gets higher reward + forward motion bonus
    individual.fitness = food_reward + 0.1 * forward_speed_reward

    return individual.fitness


def evaluate_individual_worker_continuous(
    port_offset, individual_queue, result_queue, initial_pos, initial_orient
):
    """Continuous worker function that processes individuals from a queue"""
    import queue

    port = 20000 + port_offset

    # Create robot connection for this worker
    rob = SimulationRobobo(api_port=port)

    try:
        # Start the simulation for this worker
        # rob.play_simulation()

        # Create food detector for this worker
        food_detector = GreenFoodDetector(rob)

        while True:
            try:
                # Get next individual from queue (with timeout to avoid hanging)
                individual_data, individual_index = individual_queue.get(timeout=1.0)

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
                    is_parallel_worker=True,
                )

                print(
                    f"Individual {individual_index+1} finished on port {port} with fitness {individual.fitness:.2f}, food {individual.food_collected}"
                )

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
                break

    finally:
        # Stop the simulation for this worker
        try:
            rob.stop_simulation()
        except:
            pass  # Ignore errors when stopping simulation


def tournament_selection(population, tournament_size=3):
    """Tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)


def run_individual_from_file(
    file_path, rob: IRobobo = None, max_time=3 * 60, is_hardware=False
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

        # if not is_hardware:
        #     initial_pos = rob.get_position()
        #     initial_orient = rob.get_orientation()
        #     # rob.set_position(initial_pos, initial_orient)
        #     rob.set_phone_pan_blocking(177, 50)
        #     rob.set_phone_tilt_blocking(100, 50)

        rob.set_phone_pan_blocking(177, 100)  # horizontal around its axis. #11-343
        rob.set_phone_tilt_blocking(100, 100)  # veritcal, # 26-109

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
                # while True:
                ir = rob.read_irs()
                food_data, found_food = food_detector.get_food_sensor_data(
                    ir, is_hardware
                )

                lw, rw = individual.get_motor_commands(food_data)
                # time.sleep(0.25)

                # rob.move_blocking(lw, rw, 600)
                print(
                    f"Moving with L_speed={lw}, R_speed={rw}, found_food={found_food}"
                )
                rob.move(lw, rw, 300)  # Non-blocking move for continuous evaluation

                # check if food collected exceeds 7, then break
                current_food_count = (
                    rob.get_nr_food_collected()
                    if hasattr(rob, "get_nr_food_collected")
                    else initial_food_count
                )
                if current_food_count - initial_food_count >= 7:
                    print("Collected enough food, stopping evaluation.")
                    break

        rob.move_blocking(0, 0, 50)
        final_food_count = (
            rob.get_nr_food_collected() if hasattr(rob, "get_nr_food_collected") else 0
        )
        total_food_collected = final_food_count - initial_food_count

        print("\n--- Evaluation Finished ---")
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
        temp_rob = SimulationRobobo(api_port=20000)
        try:
            temp_rob.play_simulation()
            start_pos = temp_rob.get_position()
            start_orient = temp_rob.get_orientation()
            temp_rob.stop_simulation()
        except Exception as e:
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
    if parallel:
        import multiprocessing as mp

        # Create queues for communication
        individual_queue = mp.Queue()
        result_queue = mp.Queue()

        print(f"Adding {len(population)} individuals to evaluation queue...")
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
                ),
            )
            p.start()
            processes.append(p)

        print(
            f"Started {num_processes} worker processes for initial population evaluation"
        )

        # Collect results as they come in
        results_collected = 0
        while results_collected < len(population):
            try:
                result = result_queue.get(timeout=60)  # 1 minute timeout per individual
                # result = result_queue.get()

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

    else:
        for i, individual in enumerate(population):
            fitness = fitness_evaluation(
                rob,
                individual,
                food_detector,
                start_pos,
                start_orient,
                is_parallel_worker=False,
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
                # if i < ELITISM_COUNT:
                #     continue  # Skip evaluation for elite individuals
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
                        timeout=60
                    )  # 1 minute timeout per individual

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
                # if i < ELITISM_COUNT:
                #     # Skip evaluation for elite individuals (already evaluated)
                #     continue
                fitness = fitness_evaluation(
                    rob,
                    individual,
                    food_detector,
                    start_pos,
                    start_orient,
                    is_parallel_worker=False,
                )
                if (i + 1) % 10 == 0 or i == len(offspring) - 1:
                    print(
                        f"Offspring {i+1}/{len(offspring)}: fitness={fitness:.2f}, food={individual.food_collected}"
                    )

        # (μ,λ) selection: replace parents with offspring
        # after all offspring have their fitness:
        offspring.sort(key=lambda x: x.fitness, reverse=True)  # put best first
        population = offspring[:MU]  # now truly the best MU

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
