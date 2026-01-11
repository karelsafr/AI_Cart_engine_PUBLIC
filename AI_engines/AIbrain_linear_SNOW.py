from numpy import random as np_random
import random
import numpy as np
import copy
import string

# počet vstupů – ideálně = len(RAYCAST_ANGLES)
N_INPUTS = 9
N_ACTIONS = 4  # [up, down, left, right]

np.random.seed(42)

# =============================================================================
# Minimal Neural Network for Inference Only
# =============================================================================

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def tanh_activation(x):
    """Tanh activation function"""
    return np.tanh(x)

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh_activation,
    'linear': lambda x: x
}

class SimpleLayer:
    """Minimal layer for forward pass only"""
    
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = ACTIVATIONS.get(activation, relu)
        
        # Initialize weights
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))
    
    def forward(self, x):
        """Forward pass through layer"""
        z = x @ self.weights + self.biases
        return self.activation(z)
    
    def set_weights(self, weights, biases):
        """Set weights and biases"""
        self.weights = weights.copy()
        self.biases = biases.copy()
    
    def get_weights(self):
        """Get weights and biases"""
        return self.weights.copy(), self.biases.copy()

class SimpleNeuralNetwork:
    """Minimal neural network for inference only"""
    
    def __init__(self, layer_sizes, activations=None):
        """
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activations: List of activation functions for each layer
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        if activations is None:
            activations = ['relu'] * (self.num_layers - 1) + ['sigmoid']
        
        # Create layers
        self.layers = []
        for i in range(self.num_layers):
            layer = SimpleLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)
    
    def predict(self, x):
        """Forward pass through network"""
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
    
    def get_all_weights(self):
        """Get all weights as list of (weights, biases) tuples"""
        return [layer.get_weights() for layer in self.layers]
    
    def set_all_weights(self, all_weights):
        """Set all weights from list of (weights, biases) tuples"""
        for i, (weights, biases) in enumerate(all_weights):
            self.layers[i].set_weights(weights, biases)

# vždy pojmenováváme jako "AIbrain_jemnoteamu"
class AIbrain_linear_SNOW:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # pro potreby náhdných znaků
        self.decider = 0
        self.x = 0 # sem se ulozí souradnice x, max HEIGHT*1.3
        self.y = 0 # sem se ulozí souradnice y, max HEIGHT (800)
        self.speed = 0 # sem se ukládá souradnice, max MAXSPEED ( 500)
        self.eps = 1.0
        self.mut_cnt = 0
        
        # Gaussian mutation parameters
        self.mutation_rate = 0.2  # Probability that each weight will be mutated
        self.mutation_std = 0.2   # Standard deviation of Gaussian noise
        self.mutation_decay = 0.99  # Decay factor for mutation parameters
        
        # Speed tracking for penalty
        self.min_speed_threshold = 50.0  # Minimum acceptable speed (10% of MAX_SPEED=500)
        self.slow_time_accumulator = 0.0  # Track time spent below threshold
        self.speed_penalty = 0.0  # Accumulated penalty for being slow
        
        # Average speed tracking
        self.speed_samples = []  # Collect all speed samples during episode
        self.average_speed = 0.0  # Calculated at end of episode
        
        # Enhanced input features tracking
        self.prev_speed = 0.0  # For speed delta calculation
        self.prev_steering = 0.5  # For steering momentum (0=left, 1=right, 0.5=straight)
        
        # Input noise for robustness (data augmentation)
        self.input_noise_std = 0.02  # Standard deviation of Gaussian noise added to raycasts
        self.enable_input_noise = True  # Enable/disable noise augmentation

        self.init_param()

    def _init_method_xavier(self):
        """Xavier/Glorot initialization - balanced for sigmoid/tanh activations"""
        for i in range(len(self.W)):
            fan_in, fan_out = self.W[i].shape[0], self.W[i].shape[1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
    
    def _init_method_he(self):
        """He initialization - good for ReLU activations"""
        for i in range(len(self.W)):
            fan_in = self.W[i].shape[0]
            scale = np.sqrt(2.0 / fan_in)
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
    
    def _init_method_centered_output(self):
        """Initialize final layer to output ~0.5 for balanced exploration"""
        # Use Xavier for hidden layers
        for i in range(len(self.W) - 1):
            fan_in, fan_out = self.W[i].shape[0], self.W[i].shape[1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
        
        # Small weights and zero bias for final layer
        final_idx = len(self.W) - 1
        self.W[final_idx] = np.random.randn(*self.W[final_idx].shape) * 0.01
        self.b[final_idx] = np.zeros_like(self.b[final_idx])
    
    def _init_method_small_random(self):
        """Small random weights - conservative initialization"""
        for i in range(len(self.W)):
            self.W[i] = np.random.randn(*self.W[i].shape) * 0.01
            self.b[i] = np.random.randn(*self.b[i].shape) * 0.01
    
    def _init_method_large_random(self):
        """Large random weights - aggressive exploration"""
        for i in range(len(self.W)):
            self.W[i] = (np.random.rand(*self.W[i].shape) - 0.5) * 2.0
            self.b[i] = (np.random.rand(*self.b[i].shape) - 0.5) * 2.0
    
    def _init_method_sparse(self):
        """Sparse initialization - many weights start at zero"""
        for i in range(len(self.W)):
            self.W[i] = np.random.randn(*self.W[i].shape) * 0.5
            # Set 70% of weights to zero
            mask = np.random.rand(*self.W[i].shape) < 0.7
            self.W[i][mask] = 0
            self.b[i] = np.zeros_like(self.b[i])
    
    def _init_method_positive_bias(self):
        """Positive bias - encourages action (acceleration/turning)"""
        for i in range(len(self.W) - 1):
            fan_in, fan_out = self.W[i].shape[0], self.W[i].shape[1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[i] = np.random.randn(*self.W[i].shape) * scale
            self.b[i] = np.zeros_like(self.b[i])
        
        # Final layer with positive bias
        final_idx = len(self.W) - 1
        self.W[final_idx] = np.random.randn(*self.W[final_idx].shape) * 0.1
        self.b[final_idx] = np.ones_like(self.b[final_idx]) * 0.5  # Bias towards action

    def init_param(self):
        # zde si vytvoríme promnenne co potrebujeme pro nas model
        # parametry modely vzdy inicializovat v této metode
        
        self.model_net = SimpleNeuralNetwork(
            layer_sizes=[15, 10, 4, 2],
            activations=['relu', 'relu', 'sigmoid']
        )

        model_params = self.model_net.get_all_weights()
        self.W = [l[0] for l in model_params]
        self.b = [l[1] for l in model_params]
        
        # Randomly select initialization method
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
        
        # Update the model with initialized weights
        for i in range(len(self.W)):
            self.model_net.layers[i].set_weights(self.W[i], self.b[i])
        
        self.NAME = f"AIbrain_linear_SNOW_{method_name}"

        # vždy uložit!
        self.store()

    # def decide(self, data):
    #     self.decider += 1
    #     x = np.asarray(data + [self.speed / 500], dtype=float).ravel()
    #
    #     # n_w = self.W.shape[1]
    #     # if x.size < n_w:
    #     #     x = np.concatenate([x, np.zeros(n_w - x.size)])
    #     # elif x.size > n_w:
    #     #     x = x[:n_w]
    #
    #     z = self.model_net.predict(x)[0]
    #
    #     acc_dec = z[0]
    #     break_dec = 1 - acc_dec
    #     left_dec = z[1]
    #     right_dec = 1 - left_dec
    #
    #     # Override speed control if below minimum threshold
    #     # Force acceleration to prevent getting stuck
    #     # if self.speed < self.min_speed_threshold:
    #     #     acc_dec = 1.0   # Full acceleration
    #     #     break_dec = 0.0  # No braking
    #     #     # Keep left_dec and right_dec as-is (AI can still steer)
    #
    #     # lineární kombinace pro každou akci: W @ x + b
    #     # z = self.W.dot(x) + self.b
    #
    #     # vracíme přímo z; AI_car pak dělá threshold > 0.5
    #     return np.array([acc_dec, break_dec, left_dec, right_dec])

    def decide(self, data):
        """
        Enhanced decision function with additional high-impact input features.
        
        Input features (15 total):
          - 9 raycast distances (obstacle detection) + NOISE for robustness
          - 1 normalized speed (current velocity)
          - 1 speed delta (acceleration/deceleration)
          - 1 min distance (closest obstacle)
          - 1 safety ratio (speed relative to clearance)
          - 1 clearance bias (left vs right space)
          - 1 previous steering (steering momentum)
        
        Network should be: [15, 16, 2] or larger
        """
        self.decider += 1
        
        # Constants
        MAX_SPEED = 500.0
        EPSILON = 0.01  # Small value to avoid division by zero
        
        # === NOISE AUGMENTATION (for robustness) ===
        # Add Gaussian noise to raycast inputs to improve generalization
        if self.enable_input_noise and len(data) > 0:
            # Convert to numpy array for noise addition
            raycast_data = np.array(data, dtype=float)
            # Add Gaussian noise: N(0, input_noise_std)
            noise = np.random.normal(0, self.input_noise_std, raycast_data.shape)
            raycast_data = raycast_data + noise
            # Clamp to valid range [0, 1] for raycasts
            raycast_data = np.clip(raycast_data, 0.0, 1.0)
            # Convert back to list for consistency
            data_noisy = raycast_data.tolist()
        else:
            data_noisy = data
        
        # === FEATURE 1: Normalized Speed ===
        speed_normalized = self.speed / MAX_SPEED
        
        # === FEATURE 2: Speed Delta (Acceleration Context) ===
        # Tells network if accelerating (+) or decelerating (-)
        speed_delta = (self.speed - self.prev_speed) / MAX_SPEED
        self.prev_speed = self.speed
        
        # === FEATURE 3: Min Distance (Danger Awareness) ===
        # Explicit signal for closest obstacle (use noisy data)
        if len(data_noisy) > 0:
            min_distance = min(data_noisy)
        else:
            min_distance = 1.0  # Far away if no data
        
        # === FEATURE 4: Safety Ratio (Speed + Distance Context) ===
        # High value = safe (slow or far from obstacles)
        # Low value = danger (fast and close to obstacles)
        safety_ratio = min_distance / (speed_normalized + EPSILON)
        # Clamp to reasonable range [0, 10]
        safety_ratio = min(10.0, max(0.0, safety_ratio))
        
        # === FEATURE 5: Clearance Bias (Left vs Right Space) ===
        # Positive = more space on right, Negative = more space on left
        if len(data_noisy) >= 9:
            left_clearance = np.mean(data_noisy[0:4])   # Left sensors (indices 0-3)
            right_clearance = np.mean(data_noisy[5:9])  # Right sensors (indices 5-8)
            clearance_bias = right_clearance - left_clearance
        else:
            clearance_bias = 0.0
        
        # === FEATURE 6: Previous Steering (Steering Momentum) ===
        # Helps network maintain smooth steering
        prev_steering_normalized = self.prev_steering
        
        # === Build Input Vector ===
        # Order: 9 raycasts (noisy) + speed + speed_delta + min_dist + safety + clearance + prev_steer
        x = np.asarray(
            data_noisy + 
            [speed_normalized, speed_delta, min_distance, safety_ratio, clearance_bias, prev_steering_normalized],
            dtype=float
        ).ravel()
        
        # === Network Prediction ===
        z = self.model_net.predict(x)[0]
        
        acc_dec = z[0]
        break_dec = 1 - acc_dec
        left_dec = z[1]
        right_dec = 1 - left_dec
        
        # Update steering history for next iteration
        self.prev_steering = left_dec
        
        # === Return Decision ===
        return np.array([acc_dec, break_dec, left_dec, right_dec])

    def mutate_weights_gaussian(self, weights, mutation_rate, mutation_std):
        """
        Apply Gaussian mutation to weights.
        
        Args:
            weights: Weight array to mutate
            mutation_rate: Probability that each weight will be mutated
            mutation_std: Standard deviation of Gaussian noise
            
        Returns:
            Mutated weights
        """
        mask = np.random.random(weights.shape) < mutation_rate
        mutations = np.random.normal(0, mutation_std, weights.shape)
        return weights + (mask * mutations)

    def mutate(self):
        """
        Gaussian mutation: randomly selected weights are perturbed by Gaussian noise.
        Both mutation rate and std decay over time for fine-tuning.
        """
        for layer_index in range(self.model_net.num_layers):
            # Apply Gaussian mutation to weights and biases
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
            
            # Update model and store mutated weights
            self.model_net.layers[layer_index].set_weights(mutated_W, mutated_b)
            self.W[layer_index] = mutated_W
            self.b[layer_index] = mutated_b

        # Add mutation marker to name
        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))
        
        # Update counters and decay parameters
        self.mut_cnt += 1
        self.eps = self.eps * 0.99
        
        # Decay mutation parameters for fine-tuning over generations
        self.mutation_rate = max(0.01, self.mutation_rate * self.mutation_decay)
        self.mutation_std = max(0.001, self.mutation_std * self.mutation_decay)

        self.store()

    def store(self):
        # vše, co se má ukládat do .npz
        # Ukládáme váhy pro každou vrstvu zvlášť, protože mají různé tvary
        params = {"NAME": self.NAME}
        
        # Uložíme počet vrstev
        params["num_layers"] = len(self.W)
        
        # Uložíme váhy a biasy pro každou vrstvu
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            params[f"W_{i}"] = w
            params[f"b_{i}"] = b
        
        # Uložíme mutation parametry pro zachování evolučního stavu
        params["mutation_rate"] = self.mutation_rate
        params["mutation_std"] = self.mutation_std
        params["mut_cnt"] = self.mut_cnt
        params["eps"] = self.eps
        
        self.parameters = copy.deepcopy(params)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        # Načteme počet vrstev
        num_layers = int(self.parameters["num_layers"])
        
        # Načteme váhy a biasy pro každou vrstvu
        self.W = []
        self.b = []
        for i in range(num_layers):
            self.W.append(np.array(self.parameters[f"W_{i}"], dtype=float))
            self.b.append(np.array(self.parameters[f"b_{i}"], dtype=float))
        
        self.NAME = str(self.parameters["NAME"])
        
        # Načteme mutation parametry (s fallback pro starší uložené modely)
        self.mutation_rate = float(self.parameters.get("mutation_rate", 0.1))
        self.mutation_std = float(self.parameters.get("mutation_std", 0.1))
        self.mut_cnt = int(self.parameters.get("mut_cnt", 0))
        self.eps = float(self.parameters.get("eps", 1.0))
        
        # Synchronizujeme s neuronovou sítí
        for i in range(num_layers):
            self.model_net.layers[i].set_weights(self.W[i], self.b[i])


    def calculate_score(self, distance, time, no):
        # Calculate average speed over the episode
        if len(self.speed_samples) > 0:
            self.average_speed = np.mean(self.speed_samples)
        else:
            self.average_speed = 0.0
        
        # Score formula: distance is PRIMARY, speed is secondary bonus
        # Reduced speed weight from 0.5 to 0.1 so distance is prioritized
        MAX_SPEED = 500.0
        speed_bonus = (self.average_speed / MAX_SPEED) * distance * 0.5
        # speed_penalty = self.speed_penalty * distance * 0.01  # Scale penalty to be significant
        self.score = distance + speed_bonus
        # self.score = distance


        # Apply penalty for being too slow
        # Penalize heavily for spending time below minimum speed



    ##################### do těchto funkcí není potřeba zasahovat:
    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        
        # Collect speed sample for average calculation
        self.speed_samples.append(speed)
        
        # Track slow speed and accumulate penalty
        # This is called every frame, so we need dt from somewhere
        # Since we don't have dt here, we'll use a frame-based approach
        if self.speed < self.min_speed_threshold:
            self.slow_time_accumulator += 1  # Count frames
            # Apply exponential penalty: the longer you're slow, the worse it gets
            self.speed_penalty += self.slow_time_accumulator * 0.0001
        else:
            # Reset if speed is good
            self.slow_time_accumulator = 0

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)
