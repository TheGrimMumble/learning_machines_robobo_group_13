import cv2
import numpy as np
import random
import time
import pickle
import os
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

class GreenFoodDetector:
    def __init__(self, robot):
        self.robot = robot
        self.is_simulation = hasattr(robot, '_sim')
        
        # Green color detection parameters
        #Im not sure if this actually works
        self.green_ranges = [
            {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
            {'lower': np.array([35, 30, 30]), 'upper': np.array([85, 255, 200])},
        ]
        
        self.min_contour_area = 300
        self.max_contour_area = 50000
        self.debug_mode = False
        
        # Camera scanning parameters (pan) - Fixed positions
        self.current_pan = 177  # Center position
        self.min_pan = 11      # Left limit
        self.max_pan = 343     # Right limit
        
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
                return getattr(self.robot, '_receiving_image_front', None)
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting image: {e}")
            return None

    #in the simulation they suggest the food is green 
    #green food has a certain output when seen by camera
    def detect_green_food(self, image):
        """Detect green food objects in image"""
        if image is None:
            return []
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for green_range in self.green_ranges:
            mask = cv2.inRange(hsv, green_range['lower'], green_range['upper'])
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                x,y,w,h = cv2.boundingRect(contour)
                cx, cy = x+w//2, y+h//2
                aspect_ratio = w/h if h>0 else 1
                shape_score = 1.0 - abs(1.0 - aspect_ratio)
                area_score = min(area/2000.0,1.0)
                confidence = shape_score*0.3 + area_score*0.7
                detections.append({
                    'center':(cx,cy), 'area':area, 'confidence':confidence,
                    'bbox':(x,y,w,h), 'aspect_ratio':aspect_ratio
                })
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def get_food_sensor_data(self, image=None):
        """Get normalized food sensor data including fixed camera pan & tilt and robot orientation"""
        if image is None:
            image = self.get_current_image()
        if image is None:
            return [0.0]*10  
        
        detections = self.detect_green_food(image)
        
        # Normalize pan/tilt (fixed values)
        norm_pan = ((self.current_pan - self.min_pan)/(self.max_pan-self.min_pan))*2 -1
        norm_tilt = ((self.current_tilt - self.min_tilt)/(self.max_tilt-self.min_tilt))*2 -1
        
        # Get robot orientation
        #i THINK "jaw" is what we need (check datatypes.py)
        #since this is the direction robot is looking (like a compass)
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
            return [0.0,0.0,0.0,0.0,0.0,norm_pan,norm_tilt,0.0,norm_yaw,0.0]  # Added 10th value
        
        best = detections[0]
        cx, cy = best['center']
        h, w = image.shape[:2]
        
        # Normalized food position relative to camera
        nx = (cx/w)*2 -1  
        ny = (cy/h)*2 -1  
        na = min(best['area']/5000.0, 1.0)
        conf = best['confidence']
        detected = 1.0
        
        # Calculate absolute food direction (existing code)
        camera_angle_rad = (self.current_pan - 177) * (3.14159/180)
        food_angle_in_camera_rad = nx * (30 * 3.14159/180)
        absolute_food_angle_rad = camera_angle_rad + food_angle_in_camera_rad
        
        if orientation is not None:
            robot_yaw_rad = orientation.yaw * (3.14159/180)
            food_direction_world = robot_yaw_rad + absolute_food_angle_rad
            food_direction_normalized = (food_direction_world % (2*3.14159)) / 3.14159 - 1.0
            
            #THis can be very important i think, the angle to the food if i did it correctly here
            #We could create an extra feature combining this angle with a translation
            #of wheel speeds into angle.
            angle_to_food_rad = absolute_food_angle_rad  # This is already relative to robot
            # Normalize to [-1, 1] range (representing -180° to +180°)
            angle_to_food_normalized = angle_to_food_rad / 3.14159
            # Clamp to [-1, 1] range
            angle_to_food_normalized = max(-1.0, min(1.0, angle_to_food_normalized))
        else:
            food_direction_normalized = 0.0
            angle_to_food_normalized = 0.0
        
        return [nx, ny, na, conf, detected, norm_pan, norm_tilt, 
                food_direction_normalized, norm_yaw, angle_to_food_normalized]

# Neural network for discrete action selection
class MultiSensorNeuralNetwork:
    def __init__(self, input_size=10, hidden1_size=8, hidden2_size=8, output_size=4):  # 4 outputs for action selection
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
        
        self.total_params = (self.weights_input_hidden1 + self.biases_hidden1 +
                             self.weights_hidden1_hidden2 + self.biases_hidden2 +
                             self.weights_hidden2_output + self.biases_output)

    def set_weights_from_genome(self, genome):
        idx = 0
        
        self.w_ih1 = np.array(genome[idx:idx+self.weights_input_hidden1]).reshape(self.input_size, self.hidden1_size)
        idx += self.weights_input_hidden1
        
        self.b_h1 = np.array(genome[idx:idx+self.biases_hidden1])
        idx += self.biases_hidden1
        
        self.w_h1h2 = np.array(genome[idx:idx+self.weights_hidden1_hidden2]).reshape(self.hidden1_size, self.hidden2_size)
        idx += self.weights_hidden1_hidden2
        
        self.b_h2 = np.array(genome[idx:idx+self.biases_hidden2])
        idx += self.biases_hidden2
        
        self.w_h2o = np.array(genome[idx:idx+self.weights_hidden2_output]).reshape(self.hidden2_size, self.output_size)
        idx += self.weights_hidden2_output
        
        self.b_o = np.array(genome[idx:idx+self.biases_output])

    def forward(self, inputs):
        x = np.array(inputs).reshape(1, self.input_size)
        h1 = np.tanh(x.dot(self.w_ih1) + self.b_h1)
        h2 = np.tanh(h1.dot(self.w_h1h2) + self.b_h2)
        o = np.tanh(h2.dot(self.w_h2o) + self.b_o)
        
        return o.flatten()

class Individual:
    def __init__(self, genome_length=None):
        self.network = MultiSensorNeuralNetwork(10, 8, 8, 4)  # 4 outputs for discrete actions
        if genome_length is None:
            genome_length = self.network.total_params
        self.genome = self._xavier_initialize()
        self.fitness = 0.0
        self.survival_time = 0.0
        self.behavior_diversity = 0.0
        self.min_distance = float('inf')
        self.food_collected = 0
        self.mutation_step = 0.1

    def _xavier_initialize(self):
        params = []
        
        limit = np.sqrt(6.0 / (self.network.input_size + self.network.hidden1_size))
        params += list(np.random.uniform(-limit, limit, self.network.weights_input_hidden1))
        
        params += list(np.random.uniform(-0.1, 0.1, self.network.biases_hidden1))
        
        limit = np.sqrt(6.0 / (self.network.hidden1_size + self.network.hidden2_size))
        params += list(np.random.uniform(-limit, limit, self.network.weights_hidden1_hidden2))
        
        params += list(np.random.uniform(-0.1, 0.1, self.network.biases_hidden2))
        
        limit = np.sqrt(6.0 / (self.network.hidden2_size + self.network.output_size))
        params += list(np.random.uniform(-limit, limit, self.network.weights_hidden2_output))
        
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
            (100, 100),   # Action 0: Forward
            (-100, -100), # Action 1: Backward  
            (100, -100),  # Action 2: Right turn
            (-100, 100)   # Action 3: Left turn
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
    max_time=30.0
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

    initial_food_count = rob.get_nr_food_collected() if hasattr(rob, 'get_nr_food_collected') else 0

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
            lw, rw, _, _ = individual.get_motor_commands(food_data, food_detector)  # Ignore pan/tilt outputs

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
                print(f"Food detected at norm‐coords x={food_data[0]:.2f}, y={food_data[1]:.2f}, area={food_data[2]:.2f}")
                time_with_food += 1
                if abs(lw) > 10 or abs(rw) > 10:
                    approach_bonus += 1

            # Display yaw values periodically (every 10 steps, or when food is detected)
            step_counter += 1
            if step_counter % 10 == 0 or food_data[4] > 0:
                # Convert normalized robot yaw back to degrees for display
                robot_yaw_degrees = (food_data[8] + 1.0) * 180.0
                print(f"Robot yaw: {robot_yaw_degrees:.1f}° (normalized: {food_data[8]:.3f})")
                
                # Also show food direction when food is detected
                if food_data[4] > 0:
                    food_direction_degrees = (food_data[7] + 1.0) * 180.0
                    print(f"Food direction: {food_direction_degrees:.1f}° (normalized: {food_data[7]:.3f})")
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

    final_food_count = rob.get_nr_food_collected() if hasattr(rob, 'get_nr_food_collected') else 0
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
        base + food_bonus + detect_bonus + approach + explore +
        diversity - repeat_score + total_speed_reward
    )

    return individual.fitness

def tournament_selection(population, tournament_size=3):
    """Tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)

def genetic_algorithm(rob: IRobobo):
    """Full genetic algorithm implementation with (μ,λ) selection"""
    # GA Parameters
    MU = 30          # Number of parents
    LAMBDA = 60      # Number of offspring
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

    food_detector = GreenFoodDetector(rob)
    start_pos = rob.get_position()
    start_orient = rob.get_orientation()
    
    # Initialize population
    population = [Individual() for _ in range(MU)]
    
    # Evaluate initial population
    print("Evaluating initial population...")
    for i, individual in enumerate(population):
        fitness = fitness_evaluation(rob, individual, food_detector, start_pos, start_orient)
        print(f"Individual {i+1}/{MU}: fitness={fitness:.2f}, food={individual.food_collected}")
    
    # Track best individual
    best_individual = max(population, key=lambda x: x.fitness)
    best_fitness = best_individual.fitness
    print(f"Initial best fitness: {best_fitness:.2f}, food: {best_individual.food_collected}")
    
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
        for i, individual in enumerate(offspring):
            if i < ELITISM_COUNT:
                # Skip evaluation for elite individuals (already evaluated)
                continue
            fitness = fitness_evaluation(rob, individual, food_detector, start_pos, start_orient)
            if (i + 1) % 10 == 0 or i == len(offspring) - 1:
                print(f"Offspring {i+1}/{len(offspring)}: fitness={fitness:.2f}, food={individual.food_collected}")
        
        # (μ,λ) selection: replace parents with offspring
        population = offspring[:MU]  # Select μ best offspring
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best individual
        current_best = population[0]
        if current_best.fitness > best_fitness:
            best_individual = current_best.copy()
            best_fitness = current_best.fitness
            print(f"NEW BEST in generation {generation + 1}: fitness={best_fitness:.2f}, food={best_individual.food_collected}")
        
        # Statistics
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        print(f"Generation {generation + 1} stats:")
        print(f"  Best: {population[0].fitness:.2f} (food: {population[0].food_collected})")
        print(f"  Average: {avg_fitness:.2f}")
        print(f"  Worst: {population[-1].fitness:.2f}")
        print(f"  Overall best: {best_fitness:.2f} (food: {best_individual.food_collected})")
    
    print(f"\nEvolution complete!")
    print(f"Best fitness achieved: {best_fitness:.2f}")
    print(f"Food collected by best: {best_individual.food_collected}")
    
    return None, best_individual

def run_neuroevolution(rob: IRobobo):
    if isinstance(rob, SimulationRobobo): rob.play_simulation()
    save_dir, best = genetic_algorithm(rob)
    if isinstance(rob, SimulationRobobo): rob.stop_simulation()
    return save_dir, best

def run_all_actions(rob: IRobobo):
    return run_neuroevolution(rob)
