from numpy import random as np_random
import random
import numpy as np
import copy
import string

from NN.nn import NeuralNetwork

# počet vstupů – ideálně = len(RAYCAST_ANGLES)
N_INPUTS = 9
N_ACTIONS = 4  # [up, down, left, right]

np.random.seed(42)

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
        
        self.model_net = NeuralNetwork(
            layer_sizes=[10, 5, 2],
            activations=['relu', 'sigmoid'],
            loss='mse'
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

    def decide(self, data):
        self.decider += 1
        x = np.asarray(data + [self.speed / 500], dtype=float).ravel()

        # n_w = self.W.shape[1]
        # if x.size < n_w:
        #     x = np.concatenate([x, np.zeros(n_w - x.size)])
        # elif x.size > n_w:
        #     x = x[:n_w]

        z = self.model_net.predict(x)[0]

        acc_dec = z[0]
        break_dec = 1 - acc_dec
        left_dec = z[1]
        right_dec = 1 - left_dec

        # Override speed control if below minimum threshold
        # Force acceleration to prevent getting stuck
        if self.speed < self.min_speed_threshold:
            acc_dec = 1.0   # Full acceleration
            break_dec = 0.0  # No braking
            # Keep left_dec and right_dec as-is (AI can still steer)

        # lineární kombinace pro každou akci: W @ x + b
        # z = self.W.dot(x) + self.b

        # vracíme přímo z; AI_car pak dělá threshold > 0.5
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
        # Base score is distance traveled
        self.score = distance
        
        # Calculate average speed over the episode
        if len(self.speed_samples) > 0:
            self.average_speed = np.mean(self.speed_samples)
        else:
            self.average_speed = 0.0
        
        # Apply penalty for being too slow
        # Penalize heavily for spending time below minimum speed
        self.score -= self.speed_penalty * 100  # Scale penalty to be significant


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
