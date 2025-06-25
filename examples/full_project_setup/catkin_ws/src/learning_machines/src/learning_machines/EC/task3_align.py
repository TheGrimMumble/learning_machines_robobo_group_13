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
                    "red_objects_found": individual.red_objects_found,
                    "green_objects_found": individual.green_objects_found,
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
    ind.red_objects_found = data.get("red_objects_found", 0)
    ind.green_objects_found = data.get("green_objects_found", 0)
    ind.mutation_step = data.get("mutation_step", 0.1)
    return ind


class RedGreenObjectDetector:
    def __init__(self, robot):
        self.robot = robot
        self.is_simulation = hasattr(robot, "_sim")

        self.red_ranges = [
            {"lower": np.array([0, 80, 50]), "upper": np.array([10, 255, 255])},
            {"lower": np.array([170, 80, 50]), "upper": np.array([180, 255, 255])},
        ]

        self.green_ranges = [
            {"lower": np.array([35, 80, 50]), "upper": np.array([85, 255, 255])},
        ]

        self.min_contour_area = 20
        self.max_contour_area = 50000000
        self.debug_mode = False

        self.last_saved_time = 0

        self.current_pan = 177
        self.min_pan = 11
        self.max_pan = 343

        try:
            self.current_tilt = robot.read_phone_tilt()
        except Exception:
            self.current_tilt = 67
        self.min_tilt = 26
        self.max_tilt = 109

    def get_current_image(self):
        try:
            return self.robot.read_image_front()
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting image: {e}")
            return None

    def detect_color_objects(self, image, color_ranges, color_name):
        """Generic color object detection method"""
        if image is None:
            return []

        detections = []
        current_time = time.time()
        should_save = False  # current_time - self.last_saved_time > 3
        if should_save:
            cv2.imwrite(
                str(IMAGE_OUTPUT_DIR / f"{int(current_time)}.png"),
                image,
            )
            self.last_saved_time = current_time

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for i, color_range in enumerate(color_ranges):
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
            if should_save:
                cv2.imwrite(
                    str(
                        IMAGE_OUTPUT_DIR
                        / f"{color_name}_mask_range_{i}_{int(current_time)}.png"
                    ),
                    mask,
                )
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
                        "color": color_name,
                    }
                )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect_red_objects(self, image):
        return self.detect_color_objects(image, self.red_ranges, "red")

    def detect_green_objects(self, image):
        return self.detect_color_objects(image, self.green_ranges, "green")

    def get_sensor_data(self, ir_readings, is_hardware=False, image=None):
        def normalize(val, min_val, max_val):
            theoretical_min_log = np.log(min_val)
            theoretical_max_log = np.log(max_val)

            if val < min_val:
                val = min_val
            elif val > max_val:
                val = max_val

            normalized = (np.log(val) - theoretical_min_log) / (
                theoretical_max_log - theoretical_min_log
            )
            return max(0.0, min(1.0, normalized))

        if not is_hardware:
            front_left = normalize(ir_readings[2], 52, 2500)
            front_center = normalize(ir_readings[4], 5, 2500)
            front_right = normalize(ir_readings[3], 52, 2500)
            front_left_left = normalize(ir_readings[7], 5, 500)
            front_right_right = normalize(ir_readings[5], 5, 250)
            back_left = normalize(ir_readings[0], 6, 2500)
            back_right = normalize(ir_readings[1], 6, 2500)
            back_center = normalize(ir_readings[6], 57, 2500)
        else:
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
            if is_hardware:
                # print("Flipping")
                image = cv2.flip(image, 0)
            # save image

        red_detections = self.detect_red_objects(image)
        green_detections = self.detect_green_objects(image)

        red_x, red_y, red_a = 0.0, 0.0, 0.0
        found_red = False
        if red_detections:
            best_red = red_detections[0]
            cx, cy = best_red["center"]
            h, w = image.shape[:2]
            red_x = (cx / w) * 2 - 1
            red_y = (cy / h) * 2 - 1
            red_a = min(best_red["area"] / 20000.0, 1.0)
            found_red = True
        green_x, green_y, green_a = 0.0, 0.0, 0.0
        found_green = False
        if green_detections:
            best_green = green_detections[0]
            cx, cy = best_green["center"]
            h, w = image.shape[:2]
            green_x = (cx / w) * 2 - 1
            green_y = (cy / h) * 2 - 1
            green_a = min(best_green["area"] / 5000.0, 1.0)
            found_green = True

        # print(f"Red detected: {found_red}, Green detected: {found_green}")

        return (
            [
                red_x,
                red_y,
                red_a,  # Red object data
                green_x,
                green_y,
                green_a,  # Green object data
                front_left,
                front_center,
                front_right,
                front_left_left,
                front_right_right,
                back_left,
                back_right,
                back_center,
            ],
            found_red,
            found_green,
        )


# Just three extra inputs for the red sensors now, also increased hidden layers by 2
class MultiSensorNeuralNetwork:
    def __init__(
        self, input_size=14, hidden_layers=[10], output_size=2
    ):  # Increased input size for green
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

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
            x = np.tanh(x.dot(self.weights[i]) + self.biases[i])

        output = np.tanh(x.dot(self.weights[-1]) + self.biases[-1])
        return output.flatten()


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
        self.red_objects_found = 0
        self.green_objects_found = 0
        self.mutation_step = 0.1

    def _xavier_initialize(self):
        params = []

        for in_dim, out_dim in self.network.shapes:
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            weight_params = np.random.uniform(-limit, limit, in_dim * out_dim)
            bias_params = np.random.uniform(-0.1, 0.1, out_dim)

            params += list(weight_params)
            params += list(bias_params)

        return params

    def get_motor_commands(self, inputs):
        self.network.set_weights_from_genome(self.genome)
        out = self.network.forward(inputs)

        left_speed = int(out[0] * 100)
        right_speed = int(out[1] * 100)

        return left_speed, right_speed

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
        new.red_objects_found = self.red_objects_found
        new.green_objects_found = self.green_objects_found
        new.mutation_step = self.mutation_step
        return new


# Just focused on lining up the red object and green area
def fitness_evaluation(
    rob: IRobobo,
    individual: Individual,
    detector: RedGreenObjectDetector,
    initial_pos,
    initial_orient,
    max_time=30.0,
    is_parallel_worker=False,
):
    """
    Simplified fitness: rewards lining up red with green horizontally, and penalizes collisions.
    """
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        time.sleep(0.5)
        rob.play_simulation()
        time.sleep(1.0)

    rob.set_phone_pan_blocking(177, 100)
    rob.set_phone_tilt_blocking(109, 100)
    detector.current_pan = 177
    detector.current_tilt = 109

    start_time = time.time()

    # collision_penalty = 0.0
    # collision_count = 0
    distance_score = 0
    forward_speed_reward = 0
    prev_food_robot_distance = None
    prev_food_base_distance = None

    while time.time() - start_time < max_time:
        ir = rob.read_irs()
        # if any(r is not None and r > 300 for r in ir[:8]):
        #     collision_count += 1
        #     if collision_count >= 3:
        #         # collision_penalty += 1.0
        #         # collision_count = 0
        # else:
        #     collision_count = 0
        # detect stuck in wall by distance towards wall

        sensor_data, found_red, found_green = detector.get_sensor_data(ir)

        lw, rw = individual.get_motor_commands(sensor_data)

        # if found red
        #   if (distance > 0.16):
        #       reward for going forward
        #   else: (it has the object)
        #       got object reward (+1)
        #       if found green:
        #           reward for going forward and decreasing distance
        food_robot_distance = rob.get_robot_food_distance()
        food_base_distance = rob.get_food_base_distance()
        robot_has_food = food_robot_distance < 0.17
        red_green_score = 0
        # if found_red:
        #     distance_score += food_robot_distance * -1 + food_base_distance * -1
        #     speed_norm = max(0.0, (lw + rw) / 200.0)
        #     turn_penalty = abs(lw - rw) / 200.0
        #     forwardness = 1.0 - turn_penalty
        #     alignment_red = 1.0 - abs(sensor_data[0] - 0.5) * 2
        #     forward_speed_reward += speed_norm * forwardness * alignment_red
        #     if robot_has_food and found_green:
        #         alignment_green = 1.0 - abs(sensor_data[3] - 0.5) * 2
        #         forward_speed_reward += (
        #             speed_norm * forwardness * alignment_green
        #         ) * 2  # Extra forward reward for finding both red and green.
        # else:
        #     distance_score += -5

        if found_red:
            individual.red_objects_found += 1

        if found_green:
            individual.green_objects_found += 1
            # red_green_score += 1

        rob.move(lw, rw, 300)

        if rob.get_food_base_distance() < 0.1:
            print("Food detected by base!")
            distance_score += 30  # Reward for detecting food to base
            break

    rob.move_blocking(0, 0, 200)

    # Compute final fitness: scale alignment to 0-10, subtract penalties
    individual.fitness = distance_score + forward_speed_reward
    individual.survival_time = time.time() - start_time

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
        # Create detector for this worker
        detector = RedGreenObjectDetector(rob)

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
                individual.red_objects_found = individual_data["red_objects_found"]
                individual.green_objects_found = individual_data["green_objects_found"]
                individual.mutation_step = individual_data["mutation_step"]

                # Evaluate the individual
                fitness = fitness_evaluation(
                    rob,
                    individual,
                    detector,
                    initial_pos,
                    initial_orient,
                    is_parallel_worker=True,
                )

                print(
                    f"Individual {individual_index+1} finished on port {port} with fitness {individual.fitness:.2f}, red={individual.red_objects_found}, green={individual.green_objects_found}"
                )

                # Put result back
                result_data = {
                    "index": individual_index,
                    "genome": individual.genome,
                    "fitness": individual.fitness,
                    "survival_time": individual.survival_time,
                    "behavior_diversity": individual.behavior_diversity,
                    "min_distance": individual.min_distance,
                    "red_objects_found": individual.red_objects_found,
                    "green_objects_found": individual.green_objects_found,
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
        detector = RedGreenObjectDetector(rob)
        if is_hardware:
            # rob.set_phone_pan_blocking(177, 100)
            rob.set_phone_pan_blocking(180, 100)
            rob.set_phone_tilt_blocking(100, 100)
        else:
            rob.set_phone_pan_blocking(177, 100)
            rob.set_phone_tilt_blocking(109, 100)

        print(f"Running individual from: {file_path}")
        print("Press Ctrl+C to stop manually.\n")

        start_time = time.time()
        total_reward = 0.0
        movement_count = 0

        # Prepare CSV for live logging
        base_dir = os.path.dirname(file_path)
        unique_id = uuid.uuid4().hex[:8]
        csv_filename = os.path.join(base_dir, f"red_green_sensor_log_{unique_id}.csv")
        csv_headers = (
            ["timestamp"]
            + [
                f"sensor_{i}" for i in range(14)
            ]  # red_x, red_y, red_a, green_x, green_y, green_a, + 8 IR sensors
            + ["L_speed", "R_speed", "reward", "red_detected", "green_detected"]
        )

        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)  # Write header once

            while time.time() - start_time < max_time:
                ir = rob.read_irs()
                sensor_data, found_red, found_green = detector.get_sensor_data(
                    ir, is_hardware
                )
                # print("red: ", found_red, "green:", found_green)

                lw, rw = individual.get_motor_commands(sensor_data)

                if not is_hardware:
                    distance_to_target = rob.get_food_base_distance()
                else:
                    distance_to_target = 0

                print(
                    f"Moving with L_speed={lw}, R_speed={rw}, found_red={found_red}, found_green={found_green}, distance_to_target={distance_to_target:.2f}"
                )
                rob.move(lw, rw, 300)  # Non-blocking move for continuous evaluation

                # Calculate alignment reward
                # reward = 0.0
                # if found_red and found_green:
                #     rx, gx = sensor_data[0], sensor_data[3]  # red_x, green_x
                #     error = abs(rx - gx)
                #     alignment = max(0.0, 1.0 - error)
                #     reward = alignment

                # # Log to CSV
                # timestamp = time.time() - start_time
                # row = (
                #     [timestamp] + sensor_data + [lw, rw, reward, found_red, found_green]
                # )
                # writer.writerow(row)

                # print("robot food distance:", rob.get_robot_food_distance())

                if not is_hardware and rob.get_food_base_distance() < 0.11:
                    print("Food detected by base!")
                    break

        print("\n--- Evaluation Finished ---")
        print(f"Survival time: {time.time() - start_time:.2f}s")
        print(f"Sensor log saved to: {csv_filename}")

    except Exception as e:
        print(f"Error while running individual: {e}")

    finally:
        if stop_after and not is_hardware:
            rob.stop_simulation()


def genetic_algorithm(rob: IRobobo, parallel=False, num_processes=10):
    """Full genetic algorithm implementation with (μ,λ) selection - with parallel support"""
    MU = 10
    LAMBDA = 20
    GENERATIONS = 300
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.2
    TOURNAMENT_SIZE = 3
    ELITISM_COUNT = 3

    print(f"Running Genetic Algorithm with (μ,λ) selection")
    print(f"Parents (μ): {MU}, Offspring (λ): {LAMBDA}")
    print(f"Generations: {GENERATIONS}")
    print(f"Crossover rate: {CROSSOVER_RATE}, Mutation rate: {MUTATION_RATE}")
    print(f"Tournament size: {TOURNAMENT_SIZE}, Elitism: {ELITISM_COUNT}")
    print("Features: Red and Green object detection and alignment behavior")
    print("HSV Ranges: Red (0-10°, 170-180°), Green (35-85°) - Non-overlapping")
    if parallel:
        print(f"Parallel mode: {num_processes} processes")
    else:
        print("Running in single-threaded mode")

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
        red_green_detector = RedGreenObjectDetector(rob)
        start_pos = rob.get_position()
        start_orient = rob.get_orientation()

    population = [Individual() for _ in range(MU)]

    best_individual = None
    best_fitness = float("-inf")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(FIGURES_DIR, f"red_green_training_history_{timestamp}.csv")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "generation",
                "index",
                "fitness",
                "survival_time",
                "red_objects_found",
                "green_objects_found",
            ],
        )
        writer.writeheader()

    print(f"Evaluating initial population of {MU} individuals...")

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
                "red_objects_found": individual.red_objects_found,
                "green_objects_found": individual.green_objects_found,
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

                # Update the individual with results
                idx = result["index"]
                population[idx].genome = result["genome"]
                population[idx].fitness = result["fitness"]
                population[idx].survival_time = result["survival_time"]
                population[idx].behavior_diversity = result["behavior_diversity"]
                population[idx].min_distance = result["min_distance"]
                population[idx].red_objects_found = result["red_objects_found"]
                population[idx].green_objects_found = result["green_objects_found"]
                population[idx].mutation_step = result["mutation_step"]

                results_collected += 1
                print(
                    f" Individual {idx+1}/{MU} completed - Fitness: {population[idx].fitness:.2f}, Red: {population[idx].red_objects_found}, Green: {population[idx].green_objects_found} ({results_collected}/{len(population)} done)"
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
                red_green_detector,
                start_pos,
                start_orient,
            )
            print(
                f"Individual {i+1}/{MU}: fitness={fitness:.2f}, red={individual.red_objects_found}, green={individual.green_objects_found}"
            )

    current_best = max(population, key=lambda x: x.fitness)
    best_individual = current_best.copy()
    best_fitness = current_best.fitness
    print(
        f"Initial best fitness: {best_fitness:.2f}, red={best_individual.red_objects_found}, green={best_individual.green_objects_found}"
    )

    for generation in range(GENERATIONS):
        print(f"\n--- Generation {generation + 1}/{GENERATIONS} ---")

        population.sort(key=lambda x: x.fitness, reverse=True)
        offspring = []

        for i in range(ELITISM_COUNT):
            offspring.append(population[i].copy())

        while len(offspring) < LAMBDA:
            parent1 = tournament_selection(population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)

            child1, child2 = parent1.crossover(parent2, CROSSOVER_RATE)

            child1.gaussian_mutate(MUTATION_RATE)
            child2.gaussian_mutate(MUTATION_RATE)

            offspring.extend([child1, child2])

        offspring = offspring[:LAMBDA]

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
                    "red_objects_found": individual.red_objects_found,
                    "green_objects_found": individual.green_objects_found,
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
            offspring_to_evaluate = len(offspring)  # - ELITISM_COUNT
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
                    offspring[idx].red_objects_found = result["red_objects_found"]
                    offspring[idx].green_objects_found = result["green_objects_found"]
                    offspring[idx].mutation_step = result["mutation_step"]

                    results_collected += 1
                    if (
                        results_collected
                    ) % 10 == 0 or results_collected == offspring_to_evaluate:
                        print(
                            f" Offspring {results_collected}/{offspring_to_evaluate} completed - Fitness: {offspring[idx].fitness:.2f}, Red: {offspring[idx].red_objects_found}, Green: {offspring[idx].green_objects_found}"
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
                #     continue
                fitness = fitness_evaluation(
                    rob,
                    individual,
                    red_green_detector,
                    start_pos,
                    start_orient,
                )
                if (i + 1) % 10 == 0 or i == len(offspring) - 1:
                    print(
                        f"Offspring {i+1}/{len(offspring)}: fitness={fitness:.2f}, red={individual.red_objects_found}, green={individual.green_objects_found}"
                    )

        offspring.sort(key=lambda x: x.fitness, reverse=True)
        population = offspring[:MU]

        # Save top individuals
        save_top_individuals(population, generation, top_k=5)

        current_best = population[0]
        if current_best.fitness > best_fitness:
            best_individual = current_best.copy()
            best_fitness = current_best.fitness
            print(
                f"NEW BEST in generation {generation + 1}: fitness={best_fitness:.2f}, red={best_individual.red_objects_found}, green={best_individual.green_objects_found}"
            )

        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        print(f"Generation {generation + 1} stats:")
        print(
            f"  Best: {population[0].fitness:.2f} (red: {population[0].red_objects_found}, green: {population[0].green_objects_found})"
        )
        print(f"  Average: {avg_fitness:.2f}")
        print(f"  Worst: {population[-1].fitness:.2f}")
        print(
            f"  Overall best: {best_fitness:.2f} (red: {best_individual.red_objects_found}, green: {best_individual.green_objects_found})"
        )

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "generation",
                    "index",
                    "fitness",
                    "survival_time",
                    "red_objects_found",
                    "green_objects_found",
                ],
            )
            for idx, ind in enumerate(population):
                writer.writerow(
                    {
                        "generation": generation + 1,
                        "index": idx,
                        "fitness": ind.fitness,
                        "survival_time": ind.survival_time,
                        "red_objects_found": ind.red_objects_found,
                        "green_objects_found": ind.green_objects_found,
                    }
                )

    print(f"\nEvolution complete!")
    print(f"Best fitness achieved: {best_fitness:.2f}")
    print(f"Red objects found by best: {best_individual.red_objects_found}")
    print(f"Green objects found by best: {best_individual.green_objects_found}")

    return None, best_individual


def run_neuroevolution(
    rob: IRobobo = None,
    parallel=False,
    num_processes=10,
    file_path=None,
    is_hardware=False,
):
    """Run neuroevolution with optional parallel processing and file loading"""
    if parallel:
        print("Starting parallel neuroevolution with multiple CoppeliaSim instances")
        print("Each worker will manage its own simulation instance")
    elif isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    if file_path is not None:
        run_individual_from_file(file_path, rob, is_hardware=is_hardware)
        return None, None
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
    """Main entry point"""
    return run_neuroevolution(rob, parallel, num_processes, file_path, is_hardware)
