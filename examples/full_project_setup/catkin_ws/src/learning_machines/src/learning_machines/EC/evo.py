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
                    "sigmas": individual.sigmas,
                    "fitness": fitness,
                },
                f,
            )

        print(f"Saved gen {gen+1} rank {rank} (fitness={fitness:.2f}) to {filename}")


def load_individual(filepath):
    """
    Load a genome and fitness from a pickle file and return a new Individual.
    """
    with open(os.path.join(FIGURES_DIR, filepath), "rb") as f:
        # with open(os.path.join(os.path.join(FIGURES_DIR, "Run 1"), filepath), "rb") as f:
        data = pickle.load(f)
    genome = data.get("genome")
    fitness = data.get("fitness", None)
    ind = Individual(len(genome))
    ind.genome = genome
    ind.fitness = fitness
    ind.sigmas = data.get("sigmas", [0.10] * len(genome))
    return ind


class MultiSensorNeuralNetwork:
    # Uses 3 front input sensors, and 2 outputs (left and right wheel speed)
    # Only 1 hidden layer of size 6. Probably enough for such a simple task though
    def __init__(self, input_size=10, hidden_size=6, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = input_size * hidden_size
        self.biases_hidden = hidden_size
        self.weights_hidden_output = hidden_size * output_size
        self.biases_output = output_size

        self.total_params = (
            self.weights_input_hidden
            + self.biases_hidden
            + self.weights_hidden_output
            + self.biases_output
        )

    def set_weights_from_genome(self, genome):
        idx = 0
        self.w_ih = np.array(genome[idx : idx + self.weights_input_hidden]).reshape(
            self.input_size, self.hidden_size
        )
        idx += self.weights_input_hidden

        self.b_h = np.array(genome[idx : idx + self.biases_hidden])
        idx += self.biases_hidden

        self.w_ho = np.array(genome[idx : idx + self.weights_hidden_output]).reshape(
            self.hidden_size, self.output_size
        )
        idx += self.weights_hidden_output

        self.b_o = np.array(genome[idx : idx + self.biases_output])

    def forward(self, inputs):
        inputs = np.array(inputs).reshape(-1, 1)
        hidden = np.dot(inputs.T, self.w_ih) + self.b_h
        hidden = np.tanh(hidden)
        output = np.dot(hidden, self.w_ho) + self.b_o
        output = np.tanh(output)
        return output.flatten()


class Individual:
    def __init__(self, genome_length=None, init_sigma=0.10):
        self.network = MultiSensorNeuralNetwork(10, 6, 2)
        if genome_length is None:
            genome_length = self.network.total_params

        # genome and per-gene sigma
        self.genome = self._xavier_initialize(genome_length)
        self.sigmas = [init_sigma] * genome_length  #  NEW

        # bookkeeping
        self.fitness = 0.0
        self.survival_time = 0.0
        self.behavior_diversity = 0.0
        self.min_distance = float("inf")
        self.prev_output = [0.0, 0.0]

    def _xavier_initialize(self, genome_length):
        genome = []
        fan_in, fan_out = self.network.input_size, self.network.hidden_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        for _ in range(self.network.weights_input_hidden):
            genome.append(random.uniform(-limit, limit))
        for _ in range(self.network.biases_hidden):
            genome.append(random.uniform(-0.1, 0.1))
        fan_in, fan_out = self.network.hidden_size, self.network.output_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        for _ in range(self.network.weights_hidden_output):
            genome.append(random.uniform(-limit, limit))
        for _ in range(self.network.biases_output):
            genome.append(random.uniform(-0.1, 0.1))
        return genome

    def get_sensor_inputs(
        self,
        ir_readings,
        collision_threshold_center=1000,
        collision_threshold_others=3500,
    ):
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

        # front_left = ir_readings[2]  # minimum is 52, maximum is 2500
        # front_center = ir_readings[4] # minimum is 5, maximum is 2500
        # front_right = ir_readings[3] #minimum is 52, maximum is 2500
        # front_left_left = ir_readings[7] # minimum is 5, maximum is 500
        # front_right_right = ir_readings[5] #minimum is 5, maximum is 250
        # back_left = ir_readings[0] # minimum is 6, maximum is 2500
        # back_right = ir_readings[1] # minimum is 6, maximum is 2500
        # back_center = ir_readings[6] # minimum is 57, maximum is 2500

        front_left = normalize(ir_readings[2], 52, 2500)
        front_center = normalize(ir_readings[4], 5, 2500)
        front_right = normalize(ir_readings[3], 52, 2500)
        front_left_left = normalize(ir_readings[7], 5, 500)
        front_right_right = normalize(ir_readings[5], 5, 250)
        back_left = normalize(ir_readings[0], 6, 2500)
        back_right = normalize(ir_readings[1], 6, 2500)
        back_center = normalize(ir_readings[6], 57, 2500)

        # Append previous motor outputs, normalized from [-100,100] → [0,1]
        prev_l = (self.prev_output[0] + 100) / 200.0
        prev_r = (self.prev_output[1] + 100) / 200.0

        return [
            front_left,
            front_center,
            front_right,
            front_left_left,
            front_right_right,
            back_left,
            back_right,
            back_center,
            prev_l,
            prev_r,
        ]

    def get_motor_commands(
        self,
        ir_readings,
        collision_threshold_center=1000,
        collision_threshold_others=3500,
    ):
        inputs = self.get_sensor_inputs(
            ir_readings, collision_threshold_center, collision_threshold_others
        )
        self.network.set_weights_from_genome(self.genome)
        outputs = self.network.forward(inputs)
        l = int(outputs[0] * 100)
        r = int(outputs[1] * 100)
        l = max(-100, min(100, l))
        r = max(-100, min(100, r))

        self.prev_output = [l, r]  # store for next time step
        return l, r

    def mutate(self, sigma_min=1e-3):
        """
        Log–normal self-adaptive mutation (Bäck & Schwefel style).
        Every gene always mutates, but the step size σ co-evolves and
        quickly shrinks once a niche is found.
        """
        n = len(self.genome)
        τ, τp = 1 / np.sqrt(2 * np.sqrt(n)), 1 / np.sqrt(2 * n)
        global_noise = random.gauss(0, 1)

        for i in range(n):
            # adapt step size
            self.sigmas[i] *= np.exp(τp * global_noise + τ * random.gauss(0, 1))
            self.sigmas[i] = max(self.sigmas[i], sigma_min)

            # mutate weight with NEW σ
            self.genome[i] += random.gauss(0, self.sigmas[i])
            self.genome[i] = max(-5.0, min(5.0, self.genome[i]))

    def crossover(self, other):
        cp = random.randint(1, len(self.genome) - 1)
        c1, c2 = Individual(len(self.genome)), Individual(len(self.genome))
        c1.genome = self.genome[:cp] + other.genome[cp:]
        c2.genome = other.genome[:cp] + self.genome[cp:]
        c1.sigmas = self.sigmas[:cp] + other.sigmas[cp:]  #  NEW
        c2.sigmas = other.sigmas[:cp] + self.sigmas[cp:]  #  NEW
        return c1, c2


def read_sensor_test():
    rob = SimulationRobobo(api_port=20000)
    rob.play_simulation()

    while True:
        ir_readings = rob.read_irs()
        # round each reading to one decimal
        ir_readings = [round(r, 1) if r is not None else 0.0 for r in ir_readings]
        front_left = ir_readings[
            2
        ]  # minimum is 52, maximum is 2500, collision if > 1000
        front_center = ir_readings[
            4
        ]  # minimum is 5, maximum is 2500, collision if > 1000
        front_right = ir_readings[
            3
        ]  # minimum is 52, maximum is 2500, collision if > 1000
        front_left_left = ir_readings[
            7
        ]  # minimum is 5, maximum is 500, collision if > 300
        front_right_right = ir_readings[
            5
        ]  # minimum is 5, maximum is 250, collision > 300
        back_left = ir_readings[0]  # minimum is 6, maximum is 2500, collision if > 500
        back_right = ir_readings[1]  # minimum is 6, maximum is 2500, collision if > 500
        back_center = ir_readings[6]  # minimum is 57, maximum is 2500, collision if
        print(
            f"""Front Left: {front_left}, Front Center: {front_center}, Front Right: {front_right}, Front Left Left: {front_left_left}, Front Right Right: {front_right_right}, Back Left: {back_left}, Back Right: {back_right}, Back Center: {back_center}"""
        )
        time.sleep(1)

    rob.stop_simulation()


def fitness_evaluation(
    rob: IRobobo, individual: Individual, initial_pos, initial_orient, max_time=10.0
):
    # if isinstance(rob, SimulationRobobo):
    #     rob.set_position(initial_pos, initial_orient)

    rob.set_phone_pan_blocking(177, 50)
    rob.set_phone_tilt_blocking(67, 50)

    start_time = time.time()
    collision_threshold_center = 100
    collision_threshold_others = 100

    total_reward = 0.0
    movement_count = 0
    turning_steps = 0

    try:
        while time.time() - start_time < max_time:
            ir = rob.read_irs()
            valid = [r for r in ir if r is not None]

            # collision = False
            reward = 0

            # collision fi true if any value is > 300
            collision = any(r > 300 for r in valid[0:8])

            # # Collision detection
            # if valid and (
            #     valid[4] > collision_threshold_center  # front center
            #     or valid[2] > collision_threshold_others  # front left
            #     or valid[3] > collision_threshold_others  # front right
            #     or valid[5] > collision_threshold_others  # side right
            #     or valid[7] > collision_threshold_others  # side left
            #     or valid[0] > collision_threshold_others  # back left
            #     or valid[1] > collision_threshold_others  # back right
            #     or valid[6] > collision_threshold_center  # back center
            # ):
            #     collision = True

            # break  # collision — end episode

            # Get motor commands
            ls, rs = individual.get_motor_commands(
                ir, collision_threshold_center, collision_threshold_others
            )

            ir_sensors = individual.get_sensor_inputs(ir)[:-2]  # exclude prev outputs

            speed_norm = max(0.0, (ls + rs) / 200.0)
            # proximity_penalty = max(ir_sensors)
            proximity_penalty = sum(ir_sensors) / len(ir_sensors)

            turn_penalty = abs(ls - rs) / 200.0  # large if turning
            forwardness = 1.0 - turn_penalty  # 1.0 if straight, 0.0 if sharp turn

            clearance = 1.0 - proximity_penalty
            reward = speed_norm * clearance * forwardness
            # print("Speed values: L={}, R={}".format(ls, rs))
            # print(
            #     "Speed: {:.2f}, Proximity: {:.2f}, Clearance: {:.2f}, Reward: {:.2f}, Collision: {:.2f}".format(
            #         speed_norm, proximity_penalty, clearance, reward, int(collision)
            #     )
            # )

            # # Count turning steps (sharp turns)
            # if abs(ls - rs) > 50:
            #     turning_steps += 1

            if not collision:
                total_reward += reward

            movement_count += 1

            rob.move_blocking(ls, rs, 300)

    except Exception as e:
        print(f"Error during evaluation: {e}")

    rob.move_blocking(0, 0, 50)
    # if isinstance(rob, SimulationRobobo):
    #     rob.set_position(initial_pos, initial_orient)

    survival_time = time.time() - start_time
    # turn_ratio = turning_steps / movement_count if movement_count > 0 else 1.0

    # Final adjusted reward (penalize high turn ratios)
    # adjusted_reward = total_reward * (1.0 - turn_ratio)

    # Save metrics
    individual.fitness = total_reward  # adjusted_reward
    individual.survival_time = survival_time

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

                # Evaluate the individual
                fitness = fitness_evaluation(
                    rob, individual, initial_pos, initial_orient, max_time
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
    contenders = random.sample(population, min(tournament_size, len(population)))
    return max(contenders, key=lambda ind: ind.fitness)


def run_individual_from_file(
    file_path, rob: IRobobo = None, max_time=6000.0, is_hardware=False
):
    """
    Runs a single loaded individual in the simulation for up to `max_time` seconds.
    Logs sensor data (IRs) to a uniquely named CSV file in the same directory as `file_path`.
    Writes each row to the CSV file in real-time (not just at the end).
    """
    if rob is None:
        if is_hardware:
            rob = HardwareRobobo(camera=False)
        else:
            rob = SimulationRobobo(api_port=20000)
            rob.play_simulation()
        stop_after = True
    else:
        stop_after = False

    try:
        individual = load_individual(file_path)

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
        turning_steps = 0

        if is_hardware:
            collision_threshold_center = 75
            collision_threshold_others = 50
        else:
            collision_threshold_center = 1000
            collision_threshold_others = 3500

        # Prepare CSV for live logging
        base_dir = os.path.dirname(file_path)
        unique_id = uuid.uuid4().hex[:8]
        csv_filename = os.path.join(base_dir, f"sensor_log_{unique_id}.csv")
        csv_headers = (
            ["timestamp"]
            + [f"IR_{i}" for i in range(8)]
            + ["L_speed", "R_speed", "reward"]
        )

        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)  # Write header once

            while time.time() - start_time < max_time:
                ir = rob.read_irs()
                valid = [
                    r if r is not None else 0 for r in ir
                ]  # Replace None with 0 for logging
                print(f"IRS readings: {valid}")

                reward = 0

                if valid and (
                    valid[4] > collision_threshold_center
                    or valid[2] > collision_threshold_others
                    or valid[3] > collision_threshold_others
                    or valid[5] > collision_threshold_others
                    or valid[7] > collision_threshold_others
                    or valid[0] > collision_threshold_others
                    or valid[1] > collision_threshold_others
                    or valid[6] > collision_threshold_center
                ):
                    print("Collision detected! Stopping evaluation.")

                ls, rs = individual.get_motor_commands(
                    ir, collision_threshold_center, collision_threshold_others
                )

                if ls > 0 and rs > 0:
                    avg_speed = (ls + rs) / 2.0
                    forwardness = 1.0 - abs(ls - rs) / (ls + rs + 1e-5)
                    reward += avg_speed * forwardness

                if abs(ls - rs) > 50:
                    turning_steps += 1

                total_reward += reward
                movement_count += 1

                timestamp = time.time() - start_time
                writer.writerow([timestamp] + valid + [ls, rs, reward])
                f.flush()  # Ensure data is written immediately

                print(
                    f"Time: {timestamp:.2f}s | "
                    f"Speed: L={ls} R={rs} | "
                    f"Reward: {total_reward:.2f}"
                )

                rob.move_blocking(ls, rs, 300)

        rob.move_blocking(0, 0, 50)
        individual.fitness = total_reward
        individual.survival_time = time.time() - start_time

        print("\n--- Evaluation Finished ---")
        print(f"Final reward: {total_reward:.2f}")
        print(f"Survival time: {individual.survival_time:.2f}s")
        print(f"Sensor log saved to: {csv_filename}")

    except Exception as e:
        print(f"Error while running individual: {e}")

    finally:
        if stop_after and not is_hardware:
            rob.stop_simulation()


def evolutionary_algorithm(rob: IRobobo = None, parallel=False, num_processes=10):
    POP_SIZE = 60
    GENS = 300
    MUT_RATE = 0.15
    MUT_STR = 0.3
    ELITE = 4  # int(0.1 * POP_SIZE)  # 10% elitism

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
        start_pos = rob.get_position()
        start_orient = rob.get_orientation()

    # Initialize population
    population = [Individual() for _ in range(POP_SIZE)]
    best_hist = []
    all_time_best = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(FIGURES_DIR, f"training_history_{timestamp}.csv")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Write CSV header once
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["generation", "index", "fitness", "survival_time"]
        )
        writer.writeheader()

    for gen in range(GENS):
        print(f"\nGeneration {gen+1}/{GENS}")

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
                }
                individual_queue.put((data, i))

            # print(
            #     f"Evaluating {len(population)} individuals with {num_processes} parallel workers"
            # )
            # print(
            #     "Workers will process individuals as soon as they become available..."
            # )

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
                        10.0,
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

                    results_collected += 1
                    print(
                        f" Individual {idx+1}/{POP_SIZE} completed - Fitness: {population[idx].fitness:.2f} ({results_collected}/{len(population)} done)"
                    )

                except Exception as e:
                    print(f"Timeout or error waiting for results: {e}")
                    break

            # Wait for all processes to finish and clean up
            for p in processes:
                p.terminate()
                p.join()

            print(f"Generation {gen+1} evaluation complete!")

        else:
            # Sequential evaluation (original code)
            for i, indiv in enumerate(population):
                print(f" Individual {i+1}/{POP_SIZE}")
                fitness_evaluation(rob, indiv, start_pos, start_orient)

        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        best_hist.append(best.fitness)
        save_top_individuals(population, gen, top_k=5)

        # Calculate fitness statistics
        fitness_values = [ind.fitness for ind in population]
        best_fitness = max(fitness_values)
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        print(f" Generation {gen+1} Statistics:")
        print(f"   Best fitness: {best_fitness:.2f}")
        print(f"   Average fitness: {avg_fitness:.2f}")
        print(f"   Standard deviation: {std_fitness:.2f}")

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["generation", "index", "fitness", "survival_time"]
            )
            for idx, ind in enumerate(population):
                writer.writerow(
                    {
                        "generation": gen + 1,
                        "index": idx,
                        "fitness": ind.fitness,
                        "survival_time": ind.survival_time,
                    }
                )

        if gen < GENS - 1:
            new_pop = population[:ELITE]  # Elitism: best individual(s) survive
            while len(new_pop) < POP_SIZE:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)
                if random.random() < 0.8:
                    c1, c2 = p1.crossover(p2)
                else:
                    c1, c2 = Individual(), Individual()
                    c1.genome = p1.genome.copy()
                    c2.genome = p2.genome.copy()
                    c1.sigmas = p1.sigmas.copy()
                    c2.sigmas = p2.sigmas.copy()

                c1.mutate()
                c2.mutate()
                new_pop.extend([c1, c2])
            population = new_pop[:POP_SIZE]

    print(f"\nEvolution complete. Best fitness ever: {max(best_hist):.2f}")
    return all_time_best


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
        best_individual = evolutionary_algorithm(rob, parallel, num_processes)

        if not parallel and isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

        return None, best_individual


def run_all_actions(
    rob: IRobobo = None,
    parallel=False,
    num_processes=10,
    file_path=None,
    is_hardware=False,
):
    # read_sensor_test()
    return run_neuroevolution(rob, parallel, num_processes, file_path, is_hardware)
