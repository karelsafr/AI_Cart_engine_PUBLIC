from numpy import random as np_random
import random
import numpy as np
import copy
import string

# Constants
N_INPUTS = 9   # Raycast angles
N_ACTIONS = 4  # [Gas, Brake, Left, Right]

class AIbrain_ASDF:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0
        self.x = 0
        self.y = 0
        # self.speed = 0
        
        # Lap Detection State
        self.start_pos = None
        self.max_dist_from_start = 0.0
        self.lap_bonus = 0.0
        
        # Idle Detection
        self.last_pos = None
        self.idle_timer = 0
        self.is_idle = False
        self.stopped_timer = 0  # Track how long car has been stopped
        
        # Crash Detection
        self.has_crashed = False
        
        self.init_param()

    def init_param(self):
        """Initializes model weights and biases."""
        # W shape: (4 actions, 9 inputs)
        # Larger random weights to encourage random exploration
        self.W = (np_random.rand(N_ACTIONS, N_INPUTS) - 0.5)
        
        # Biases:
        # Index 0 (Gas): Set to positive to encourage moving forward immediately
        # Index 1 (Brake): Set to negative to discourage braking initially
        # Index 2,3 (Left, Right): Near zero
        self.b = np.array([0.5, -0.5, 0.0, 0.0])
        
        self.NAME = "ASDF_"
        self.store()

    def decide(self, data):
        """Processes sensor data."""
        self.decider += 1
        x = np.asarray(data, dtype=float).ravel()
        
        # Simple Linear Model
        z = self.W.dot(x) + self.b
        
        # MAX SPEED CAP - Simple speed limit at 100
        if self.speed > 100:
            z[0] -= 1.0  # Reduce gas
            z[1] += 0.8  # Apply brake
        
        # ANTI-STOP: Force gas if car is stopped or very slow
        # Prevents cars from stopping in the center of the track
        if self.speed < 20:  # Very slow or stopped
            z[0] += 2.0  # Strong gas boost
            z[1] = -2.0  # Prevent braking
        elif self.speed < 50:  # Slow
            z[0] += 1.0  # Moderate gas boost
            z[1] = -1.0  # Discourage braking
        
        # SPEED-DEPENDENT SAFETY SYSTEM
        front_dist = x[4]
        
        # If moving fast and wall is approaching -> Brake
        if self.speed > 100 and front_dist < 4.0:
            z[0] -= 1.0  # Reduce Gas
            z[1] += 0.5  # Apply Brake
            
        # Emergency Brake for high speed
        if self.speed > 300 and front_dist < 8.0:
            z[0] = -1.0  # Cut Gas
            z[1] = 2.0   # Hard Brake
            
        return z

    def mutate(self):
        """Evolves the brain."""
        # Reduced mutation rate for more stable learning
        mutation_rate = 0.02
        
        delta_W = (np_random.rand(*self.W.shape) - 0.5) * mutation_rate
        delta_b = (np_random.rand(*self.b.shape) - 0.5) * mutation_rate
        
        self.W += delta_W
        self.b += delta_b
        
        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))
        self.store()

    def store(self):
        self.parameters = copy.deepcopy({
            "W": self.W,
            "b": self.b,
            "NAME": self.NAME,
        })

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)
        
        self.parameters = params_dict
        self.W = np.array(self.parameters["W"], dtype=float)
        self.b = np.array(self.parameters["b"], dtype=float)
        self.NAME = str(self.parameters["NAME"])
        self.NAME = "ASDF_"

    def calculate_score(self, distance, time, no):
        """Calculates fitness."""
        # Light penalty for crashes - encourages being more careful
        if self.has_crashed:
            crash_penalty = 2000.0  # Moderate penalty (not as heavy as other brains)
            self.score = distance - crash_penalty
            return
        
        # HEAVY PENALTY FOR STOPPING - prevents cars from stopping in center
        if self.stopped_timer > 0:
            stop_penalty = 1000.0 * (1 + self.stopped_timer)  # Increasing penalty for longer stops
            self.score = distance - stop_penalty
            return
        
        # Score = Distance + Lap Bonus
        # Punish heavy idling
        if self.is_idle:
            self.score = -500.0  # Increased penalty for idle (was -100.0)
        else:
            self.score = distance + self.lap_bonus
            # Small survival bonus only if moving
            if distance > 10:
                self.score += no * 0.01 

    def passcardata(self, x, y, speed, running=True):
        self.x = x
        self.y = y
        self.speed = speed
        
        # Crash detection
        if not running and not self.has_crashed:
            self.has_crashed = True
        
        # --- Stop Detection ---
        # Detect if car is stopped (speed < 15 for more than 1 second)
        if self.speed < 15:  # Stopped or very slow
            self.stopped_timer += 1
        else:
            self.stopped_timer = 0  # Reset if moving
        
        # --- Lap Detection ---
        if self.start_pos is None:
            self.start_pos = (x, y)
            self.last_pos = (x, y)
            
        dx = x - self.start_pos[0]
        dy = y - self.start_pos[1]
        dist_from_start = np.sqrt(dx*dx + dy*dy)
        
        if dist_from_start > self.max_dist_from_start:
            self.max_dist_from_start = dist_from_start
            

        # --- Idle Detection ---
        # Check movement every ~60 frames (approx 1 sec)
        if self.decider % 60 == 0 and self.last_pos:
            move_dx = x - self.last_pos[0]
            move_dy = y - self.last_pos[1]
            move_dist = np.sqrt(move_dx*move_dx + move_dy*move_dy)
            
            # If moved less than 10 pixels in 1 second, consider idle
            if move_dist < 10:
                self.idle_timer += 1
            else:
                self.idle_timer = 0 # Reset if moved
            
            # If idle for 3 seconds (3 checks)
            if self.idle_timer >= 3:
                self.is_idle = True
                
            self.last_pos = (x, y)

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)
