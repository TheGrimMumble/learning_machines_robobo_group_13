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
        
        # Camera scanning parameters (pan)
        self.current_pan = 177  # Center position
        self.min_pan = 11      # Left limit
        self.max_pan = 343     # Right limit
        self.pan_speed = 100
        
        # Camera scanning parameters (tilt)
        try:
            self.current_tilt = robot.read_phone_tilt()
        except Exception:
            self.current_tilt = 67
        self.min_tilt = 26
        self.max_tilt = 109
        self.tilt_speed = 100
    

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

    def set_camera_pan(self, pan_position):
        """Set camera pan position with bounds checking"""
        pan_position = max(self.min_pan, min(self.max_pan, int(pan_position)))
        if abs(pan_position - self.current_pan) > 5:
            try:
                self.robot.set_phone_pan_blocking(pan_position, self.pan_speed)
                self.current_pan = pan_position
            except Exception as e:
                if self.debug_mode:
                    print(f"Error setting camera pan: {e}")

    def set_camera_tilt(self, tilt_position):
        """Set camera tilt position with bounds checking"""
        tilt_position = max(self.min_tilt, min(self.max_tilt, int(tilt_position)))
        if abs(tilt_position - self.current_tilt) > 5:
            try:
                self.robot.set_phone_tilt_blocking(tilt_position, self.tilt_speed)
                self.current_tilt = tilt_position
            except Exception as e:
                if self.debug_mode:
                    print(f"Error setting camera tilt: {e}")

    #in the simulation they suggest the food is green 
    #green food has a certain output when seen by camera
    def detect_green_food(self, image):
        """Detect green food objects in image"""
        if image is None:
            return []
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Combined mask
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for green_range in self.green_ranges:
            mask = cv2.inRange(hsv, green_range['lower'], green_range['upper'])
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        # Morphology
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        # Contours
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
        """Get normalized food sensor data including camera pan & tilt"""
        if image is None:
            image = self.get_current_image()
        if image is None:
            return [0.0]*8
        detections = self.detect_green_food(image)
        # Normalize pan/tilt
        norm_pan = ((self.current_pan - self.min_pan)/(self.max_pan-self.min_pan))*2 -1
        norm_tilt = ((self.current_tilt - self.min_tilt)/(self.max_tilt-self.min_tilt))*2 -1
        if not detections:
            return [0.0,0.0,0.0,0.0,0.0,norm_pan,norm_tilt,0.0]
        best = detections[0]
        cx,cy = best['center']
        h,w = image.shape[:2]
        nx = (cx/w)*2 -1
        ny = (cy/h)*2 -1
        na = min(best['area']/5000.0,1.0)
        conf = best['confidence']
        detected = 1.0
        direction = nx
        return [nx,ny,na,conf,detected,norm_pan,norm_tilt,direction]

class MultiSensorNeuralNetwork:
    def __init__(self, input_size=8, hidden_size=12, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Parameter counts
        self.weights_input_hidden = input_size * hidden_size
        self.biases_hidden = hidden_size
        self.weights_hidden_output = hidden_size * output_size
        self.biases_output = output_size
        self.total_params = (self.weights_input_hidden + self.biases_hidden +
                             self.weights_hidden_output + self.biases_output)

    def set_weights_from_genome(self, genome):
        idx=0
        self.w_ih = np.array(genome[idx:idx+self.weights_input_hidden]).reshape(self.input_size, self.hidden_size)
        idx+=self.weights_input_hidden
        self.b_h = np.array(genome[idx:idx+self.biases_hidden])
        idx+=self.biases_hidden
        self.w_ho = np.array(genome[idx:idx+self.weights_hidden_output]).reshape(self.hidden_size, self.output_size)
        idx+=self.weights_hidden_output
        self.b_o = np.array(genome[idx:idx+self.biases_output])

    def forward(self, inputs):
        x = np.array(inputs).reshape(1, self.input_size)
        h = np.tanh(x.dot(self.w_ih) + self.b_h)
        o = np.tanh(h.dot(self.w_ho) + self.b_o)
        return o.flatten()

class Individual:
    def __init__(self, genome_length=None):
        self.network = MultiSensorNeuralNetwork(8,12,4)
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
        params=[]
        # input->hidden
        limit = np.sqrt(6.0/(self.network.input_size+self.network.hidden_size))
        params += list(np.random.uniform(-limit,limit,self.network.weights_input_hidden))
        # hidden biases
        params += list(np.random.uniform(-0.1,0.1,self.network.biases_hidden))
        # hidden->output
        limit = np.sqrt(6.0/(self.network.hidden_size+self.network.output_size))
        params += list(np.random.uniform(-limit,limit,self.network.weights_hidden_output))
        # output biases
        params += list(np.random.uniform(-0.1,0.1,self.network.biases_output))
        return params

    def get_sensor_inputs(self, food_data):
        return food_data

    def get_motor_commands(self, food_data, detector):
        inputs = self.get_sensor_inputs(food_data)
        self.network.set_weights_from_genome(self.genome)
        out = self.network.forward(inputs)
        lw = max(-100, min(100, int(out[0]*100)))
        rw = max(-100, min(100, int(out[1]*100)))
        # pan
        pan_cmd = out[2]
        pan_range = detector.max_pan - detector.min_pan
        tgt_pan = detector.min_pan + ((pan_cmd+1)/2)*pan_range
        # tilt
        tilt_cmd = out[3]
        tilt_range = detector.max_tilt - detector.min_tilt
        tgt_tilt = detector.min_tilt + ((tilt_cmd+1)/2)*tilt_range
        return lw, rw, int(tgt_pan), int(tgt_tilt)

    def gaussian_mutate(self):
        noise = np.random.normal(0, self.mutation_step, len(self.genome))
        self.genome = np.clip(np.array(self.genome)+noise, -5.0, 5.0).tolist()

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
        rob.set_position(initial_pos, initial_orient)
        
    print(f"Current orientation: {initial_orient}")

    # Initialize camera to center
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
    ls_prev = rs_prev = pan_prev = tilt_prev = None
    penalty_weight = 5.0

    time_with_food = 0
    approach_bonus = 0
    camera_usage_bonus = 0
    pan_changes = 0
    tilt_changes = 0

    try:
        while time.time() - start_time < max_time:
            # Get enhanced sensor data
            food_data = food_detector.get_food_sensor_data()

            # Get commands
            ls, rs, tgt_pan, tgt_tilt = individual.get_motor_commands(food_data, food_detector)

            # Actuate camera
            food_detector.set_camera_pan(tgt_pan)
            food_detector.set_camera_tilt(tgt_tilt)

            # Track behavior
            behavior_states.append((round((ls-rs)/20), round(tgt_pan/50), round(tgt_tilt/50)))

            if food_data[4] > 0:
                print(f"Food detected at norm‐coords x={food_data[0]:.2f}, y={food_data[1]:.2f}, area={food_data[2]:.2f}")
                time_with_food += 1
                if abs(ls)>10 or abs(rs)>10:
                    approach_bonus += 1

            if pan_prev is not None and abs(tgt_pan-pan_prev)>10:
                pan_changes += 1; camera_usage_bonus +=1
            if tilt_prev is not None and abs(tgt_tilt-tilt_prev)>10:
                tilt_changes += 1; camera_usage_bonus +=1

            if (ls==ls_prev and rs==rs_prev and 
                pan_prev is not None and abs(tgt_pan-pan_prev)<5 and 
                tilt_prev is not None and abs(tgt_tilt-tilt_prev)<5):
                repeat_penalty +=1

            ls_prev, rs_prev, pan_prev, tilt_prev = ls, rs, tgt_pan, tgt_tilt

            rob.move_blocking(ls, rs, 50)
            movement_count +=1
            total_distance_traveled += abs(ls+rs)*0.3

    except Exception as e:
        print(f"Error during evaluation: {e}")

    rob.move_blocking(0,0,200)
    rob.set_phone_pan_blocking(177,50)
    rob.set_phone_tilt_blocking(67,50)

    if isinstance(rob, SimulationRobobo):
        rob.set_position(initial_pos, initial_orient)

    survival_time = time.time()-start_time
    individual.survival_time = survival_time

    final_food_count = rob.get_nr_food_collected() if hasattr(rob, 'get_nr_food_collected') else 0
    individual.food_collected = final_food_count - initial_food_count

    uniq = len(set(behavior_states))
    total = len(behavior_states)
    individual.behavior_diversity = uniq/max(1,total)

    # Compute fitness as before, adding new bonuses for tilt usage
    base = survival_time * 2
    food_bonus = individual.food_collected * 200.0
    detect_bonus = min(time_with_food*0.5,50.0)
    approach = min(approach_bonus*0.3,30.0)
    explore = min(total_distance_traveled*0.05,20.0)
    diversity = individual.behavior_diversity * 25.0
    camera_bonus = min(camera_usage_bonus*0.2,15.0)
    pan_div = min(pan_changes*0.1,10.0)
    tilt_div = min(tilt_changes*0.1,10.0)
    repeat_norm = repeat_penalty/max(1,movement_count)
    repeat_score = repeat_norm * penalty_weight

    individual.fitness = (base + food_bonus + detect_bonus + approach + explore + diversity + camera_bonus + pan_div + tilt_div - repeat_score)
    return individual.fitness

def hill_climbing_algorithm(rob: IRobobo):
    MAX_ITERATIONS = 100000
    INITIAL_STEP = 0.05
    print(f"Running Hill Climbing with Enhanced Food & Camera Control")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Initial mutation step: {INITIAL_STEP}")
    print("Features: Dynamic pan & tilt, food tracking, active scanning")

    food_detector = GreenFoodDetector(rob)
    start_pos = rob.get_position()
    start_orient = rob.get_orientation()
    
    current = Individual()
    current.mutation_step = INITIAL_STEP
    current_fitness = fitness_evaluation(rob, current, food_detector, start_pos, start_orient)
    print(f"Initial fitness: {current_fitness:.2f}")

    best_fitness = current_fitness
    improvements = 0
    window_improved = False
    best_food = current.food_collected

    for i in range(MAX_ITERATIONS):
        if i>0 and i%50==0:
            old = current.mutation_step
            if window_improved:
                current.mutation_step *= 0.999999999
            else:
                current.mutation_step *= 1.000000001
            print(f"Iteration {i}: step {old:.4f}->{current.mutation_step:.4f}, food {current.food_collected}/{best_food}")
            window_improved=False

        cand = current.copy()
        cand.gaussian_mutate()
        fit = fitness_evaluation(rob, cand, food_detector, start_pos, start_orient)
        if fit>current_fitness:
            current, current_fitness = cand, fit
            improvements+=1; window_improved=True
            if current.food_collected>best_food: best_food=current.food_collected
            print(f"Iter {i+1}: NEW BEST {fit:.2f} (food {current.food_collected})")

    print(f"Done: improvements={improvements}, final fitness={current_fitness:.2f}, food={current.food_collected}")
    return None, current

def run_neuroevolution(rob: IRobobo):
    if isinstance(rob, SimulationRobobo): rob.play_simulation()
    save_dir, best = hill_climbing_algorithm(rob)
    if isinstance(rob, SimulationRobobo): rob.stop_simulation()
    return save_dir, best

def run_all_actions(rob: IRobobo):
    return run_neuroevolution(rob)
