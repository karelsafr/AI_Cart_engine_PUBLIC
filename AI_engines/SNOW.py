from numpy import random as np_random
import random
import numpy as np
import copy
import string

# Constants
N_INPUTS = 9  # Number of raycast sensors
N_ACTIONS = 4  # [accelerate, brake, left, right]

np.random.seed(42)

# Activation functions for neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def tanh_activation(x):
    return np.tanh(x)

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh_activation,
    'linear': lambda x: x
}

class SimpleLayer:
    """Single layer of neural network for forward pass only"""
    
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = ACTIVATIONS.get(activation, relu)
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))
    
    def forward(self, x):
        """Forward pass: weighted sum + activation"""
        z = x @ self.weights + self.biases
        return self.activation(z)
    
    def set_weights(self, weights, biases):
        self.weights = weights.copy()
        self.biases = biases.copy()
    
    def get_weights(self):
        return self.weights.copy(), self.biases.copy()

class SimpleNeuralNetwork:
    """Minimal feedforward neural network for inference"""
    
    def __init__(self, layer_sizes, activations=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        if activations is None:
            activations = ['relu'] * (self.num_layers - 1) + ['sigmoid']
        
        # Create all layers
        self.layers = []
        for i in range(self.num_layers):
            layer = SimpleLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)
    
    def predict(self, x):
        """Forward pass through entire network"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
    
    def get_all_weights(self):
        return [layer.get_weights() for layer in self.layers]
    
    def set_all_weights(self, all_weights):
        for i, (weights, biases) in enumerate(all_weights):
            self.layers[i].set_weights(weights, biases)

class SNOW:
    """AI controller using evolutionary neural network"""
    
    def __init__(self):
        super().__init__()
        # Basic attributes
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0
        self.x = 0
        self.y = 0
        self.speed = 0
        self.eps = 1.0
        self.mut_cnt = 0
        
        # Gaussian mutation parameters (with decay)
        self.mutation_rate = 0.2
        self.mutation_std = 0.2
        self.mutation_decay = 0.99
        
        # Speed tracking for score calculation
        self.speed_samples = []
        self.average_speed = 0.0
        
        # Enhanced input features tracking
        self.prev_speed = 0.0
        self.prev_steering = 0.5
        
        # Input noise for robustness (data augmentation)
        self.input_noise_std = 0.02
        self.enable_input_noise = True

        self.init_param()

    # Weight initialization methods for exploration diversity
    def _init_method_xavier(self):
        """Xavier/Glorot initialization"""
        for i in range(len(self.W)):
            fan_in, fan_out = self.W[i].shape[0], self.W[i].shape[1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
    
    def _init_method_he(self):
        """He initialization (good for ReLU)"""
        for i in range(len(self.W)):
            fan_in = self.W[i].shape[0]
            scale = np.sqrt(2.0 / fan_in)
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
    
    def _init_method_centered_output(self):
        """Centered output for balanced exploration"""
        for i in range(len(self.W) - 1):
            fan_in, fan_out = self.W[i].shape[0], self.W[i].shape[1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
        
        final_idx = len(self.W) - 1
        self.W[final_idx] = np.random.randn(*self.W[final_idx].shape) * 0.01
        self.b[final_idx] = np.zeros_like(self.b[final_idx])
    
    def _init_method_small_random(self):
        """Small random weights (conservative)"""
        for i in range(len(self.W)):
            self.W[i] = np.random.randn(*self.W[i].shape) * 0.01
            self.b[i] = np.random.randn(*self.b[i].shape) * 0.01
    
    def _init_method_large_random(self):
        """Large random weights (aggressive exploration)"""
        for i in range(len(self.W)):
            self.W[i] = (np.random.rand(*self.W[i].shape) - 0.5) * 2.0
            self.b[i] = (np.random.rand(*self.b[i].shape) - 0.5) * 2.0
    
    def _init_method_sparse(self):
        """Sparse initialization (70% zeros)"""
        for i in range(len(self.W)):
            self.W[i] = np.random.randn(*self.W[i].shape) * 0.5
            mask = np.random.rand(*self.W[i].shape) < 0.7
            self.W[i][mask] = 0
            self.b[i] = np.zeros_like(self.b[i])
    
    def _init_method_positive_bias(self):
        """Positive bias to encourage action"""
        for i in range(len(self.W) - 1):
            fan_in, fan_out = self.W[i].shape[0], self.W[i].shape[1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
        
        final_idx = len(self.W) - 1
        self.W[final_idx] = np.random.randn(*self.W[final_idx].shape) * 0.1
        self.b[final_idx] = np.ones_like(self.b[final_idx]) * 0.5

    def init_param(self):
        """Initialize neural network with random weights"""
        # Network: 15 inputs -> 10 -> 4 -> 2 outputs
        self.model_net = SimpleNeuralNetwork(
            layer_sizes=[15, 10, 4, 2],
            activations=['relu', 'relu', 'sigmoid']
        )

        # Extract weights for mutation
        model_params = self.model_net.get_all_weights()
        self.W = [l[0] for l in model_params]
        self.b = [l[1] for l in model_params]
        
        # Randomly select initialization method for diversity
        init_methods = [
            ('xavier', self._init_method_xavier),
            ('he', self._init_method_he),
            ('centered', self._init_method_centered_output),
            ('small', self._init_method_small_random),
            ('large', self._init_method_large_random),
            ('sparse', self._init_method_sparse),
            ('positive', self._init_method_positive_bias),
        ]
        
        method_name, method_func = random.choice(init_methods)
        method_func()
        
        # Update network with initialized weights
        for i in range(len(self.W)):
            self.model_net.layers[i].set_weights(self.W[i], self.b[i])
        
        self.NAME = f"SNOW_"
        self.store()

    def decide(self, data):
        """
        Make driving decision based on sensor inputs.
        
        Inputs (15 total):
        - 9 raycast distances (with noise augmentation)
        - normalized speed
        - speed delta (acceleration context)
        - min distance (danger awareness)
        - safety ratio (speed vs clearance)
        - clearance bias (left vs right space)
        - previous steering (momentum)
        
        Returns: [accelerate, brake, left, right] decisions
        """
        self.decider += 1
        
        MAX_SPEED = 500.0
        EPSILON = 0.01
        
        # Add Gaussian noise to raycasts for robustness
        if self.enable_input_noise and len(data) > 0:
            raycast_data = np.array(data, dtype=float)
            noise = np.random.normal(0, self.input_noise_std, raycast_data.shape)
            raycast_data = raycast_data + noise
            raycast_data = np.clip(raycast_data, 0.0, 1.0)
            data_noisy = raycast_data.tolist()
        else:
            data_noisy = data
        
        # Feature 1: Normalized speed
        speed_normalized = self.speed / MAX_SPEED
        
        # Feature 2: Speed delta (acceleration/deceleration)
        speed_delta = (self.speed - self.prev_speed) / MAX_SPEED
        self.prev_speed = self.speed
        
        # Feature 3: Min distance (closest obstacle)
        if len(data_noisy) > 0:
            min_distance = min(data_noisy)
        else:
            min_distance = 1.0
        
        # Feature 4: Safety ratio (distance / speed)
        safety_ratio = min_distance / (speed_normalized + EPSILON)
        safety_ratio = min(10.0, max(0.0, safety_ratio))
        
        # Feature 5: Clearance bias (more space left or right?)
        if len(data_noisy) >= 9:
            left_clearance = np.mean(data_noisy[0:4])
            right_clearance = np.mean(data_noisy[5:9])
            clearance_bias = right_clearance - left_clearance
        else:
            clearance_bias = 0.0
        
        # Feature 6: Previous steering (for smooth transitions)
        prev_steering_normalized = self.prev_steering
        
        # Build input vector with all features
        x = np.asarray(
            data_noisy + 
            [speed_normalized, speed_delta, min_distance, safety_ratio, clearance_bias, prev_steering_normalized],
            dtype=float
        ).ravel()
        
        # Get network prediction
        z = self.model_net.predict(x)[0]
        
        # Extract decisions
        acc_dec = z[0]
        break_dec = 1 - acc_dec
        left_dec = z[1]
        right_dec = 1 - left_dec
        
        # Update steering history
        self.prev_steering = left_dec
        
        return np.array([acc_dec, break_dec, left_dec, right_dec])

    def mutate_weights_gaussian(self, weights, mutation_rate, mutation_std):
        """Apply Gaussian mutation to weights"""
        mask = np.random.random(weights.shape) < mutation_rate
        mutations = np.random.normal(0, mutation_std, weights.shape)
        return weights + (mask * mutations)

    def mutate(self):
        """Mutate network weights for evolution (Gaussian mutation with decay)"""
        for layer_index in range(self.model_net.num_layers):
            # Mutate weights and biases
            mutated_W = self.mutate_weights_gaussian(
                self.W[layer_index], 
                self.mutation_rate, 
                self.mutation_std
            )
            mutated_b = self.mutate_weights_gaussian(
                self.b[layer_index], 
                self.mutation_rate, 
                self.mutation_std
            )
            
            # Update network
            self.model_net.layers[layer_index].set_weights(mutated_W, mutated_b)
            self.W[layer_index] = mutated_W
            self.b[layer_index] = mutated_b

        # Update name and counters
        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))
        self.mut_cnt += 1
        self.eps = self.eps * 0.99
        
        # Decay mutation parameters for fine-tuning
        self.mutation_rate = max(0.01, self.mutation_rate * self.mutation_decay)
        self.mutation_std = max(0.001, self.mutation_std * self.mutation_decay)

        self.store()

    def store(self):
        """Save all parameters to internal storage"""
        params = {"NAME": self.NAME}
        params["num_layers"] = len(self.W)
        
        # Store weights and biases for each layer
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            params[f"W_{i}"] = w
            params[f"b_{i}"] = b
        
        # Store mutation state
        params["mutation_rate"] = self.mutation_rate
        params["mutation_std"] = self.mutation_std
        params["mut_cnt"] = self.mut_cnt
        params["eps"] = self.eps
        
        self.parameters = copy.deepcopy(params)

    def set_parameters(self, parameters):
        """Load parameters from storage"""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        # Load weights and biases
        num_layers = int(self.parameters["num_layers"])
        
        self.W = []
        self.b = []
        for i in range(num_layers):
            self.W.append(np.array(self.parameters[f"W_{i}"], dtype=float))
            self.b.append(np.array(self.parameters[f"b_{i}"], dtype=float))
        
        self.NAME = str(self.parameters["NAME"])
        
        # Load mutation state (with fallback defaults)
        self.mutation_rate = float(self.parameters.get("mutation_rate", 0.1))
        self.mutation_std = float(self.parameters.get("mutation_std", 0.1))
        self.mut_cnt = int(self.parameters.get("mut_cnt", 0))
        self.eps = float(self.parameters.get("eps", 1.0))

        self.NAME = f"SNOW_"
        # Sync with network
        for i in range(num_layers):
            self.model_net.layers[i].set_weights(self.W[i], self.b[i])

    def calculate_score(self, distance, time, no):
        """
        Calculate final score based on performance.
        Score = distance + speed_bonus + time_bonus
        """
        # Calculate average speed
        if len(self.speed_samples) > 0:
            self.average_speed = np.mean(self.speed_samples)
        else:
            self.average_speed = 0.0
        
        MAX_SPEED = 500.0
        
        # Speed bonus: rewards high average speed
        speed_bonus = (self.average_speed / MAX_SPEED) * distance * 0.5
        
        # Time bonus: rewards fast completion
        if time > 0:
            time_bonus = (distance / (time + 1.0)) * 10.0
        else:
            time_bonus = 0.0
        
        # Final score (distance is primary objective)
        self.score = distance + speed_bonus + time_bonus

    def passcardata(self, x, y, speed):
        """Receive car state data each frame"""
        self.x = x
        self.y = y
        self.speed = speed
        self.speed_samples.append(speed)

    def getscore(self):
        """Return current score"""
        return self.score

    def get_parameters(self):
        """Return copy of all parameters"""
        return copy.deepcopy(self.parameters)
