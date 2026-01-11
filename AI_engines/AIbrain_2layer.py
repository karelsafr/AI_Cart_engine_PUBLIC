from numpy import random as np_random
import numpy as np
import copy

N_SENSORS = 9
MAX_SPEED = 500.0

N_INPUTS = N_SENSORS + 1     # 10 = 9 rays + speed
N_HIDDEN = N_INPUTS          # 10
N_ACTIONS = 4

RAY_MAX_TILES = 15.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


class AIbrain_2layer:
    def __init__(self):
        super().__init__()
        self.score = 0.0

        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0

        self.mutation_sigma = 0.1

        self.init_param()

    def init_param(self):
        s1 = 1.0 / np.sqrt(N_INPUTS)
        s2 = 1.0 / np.sqrt(N_HIDDEN)

        self.W1 = np_random.randn(N_HIDDEN, N_INPUTS) * s1
        self.b1 = np_random.randn(N_HIDDEN) * s1

        self.W2 = np_random.randn(N_ACTIONS, N_HIDDEN) * s2
        self.b2 = np_random.randn(N_ACTIONS) * s2

        self.NAME = "SAFR_2layer"
        self.store()

    def decide(self, data):
        rays = np.asarray(data, dtype=float).ravel()

        if rays.size < N_SENSORS:
            rays = np.concatenate([rays, np.zeros(N_SENSORS - rays.size)])
        elif rays.size > N_SENSORS:
            rays = rays[:N_SENSORS]

        rays = np.clip(rays, 0.0, RAY_MAX_TILES) / RAY_MAX_TILES

        v = float(self.speed)
        v_norm = np.clip(v / MAX_SPEED, 0.0, 1.0) if MAX_SPEED > 0 else 0.0

        x = np.concatenate([rays, np.array([v_norm], dtype=float)])  # (10,)

        h = np.tanh(self.W1.dot(x) + self.b1)
        out = _sigmoid(self.W2.dot(h) + self.b2)
        return out

    def mutate(self):
        s = float(self.mutation_sigma)

        self.W1 = self.W1 + np_random.randn(*self.W1.shape) * s
        self.b1 = self.b1 + np_random.randn(*self.b1.shape) * s
        self.W2 = self.W2 + np_random.randn(*self.W2.shape) * s
        self.b2 = self.b2 + np_random.randn(*self.b2.shape) * s

        self.W1 = np.clip(self.W1, -5.0, 5.0)
        self.b1 = np.clip(self.b1, -5.0, 5.0)
        self.W2 = np.clip(self.W2, -5.0, 5.0)
        self.b2 = np.clip(self.b2, -5.0, 5.0)

        self.store()

    def store(self):
        self.parameters = copy.deepcopy({
            "W1": np.array(self.W1, dtype=float),
            "b1": np.array(self.b1, dtype=float),
            "W2": np.array(self.W2, dtype=float),
            "b2": np.array(self.b2, dtype=float),
            "NAME": str(self.NAME),
            #"mutation_sigma": np.array([float(self.mutation_sigma)], dtype=float),
        })

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params = {k: parameters[k] for k in parameters.files}
        else:
            params = copy.deepcopy(parameters)

        self.parameters = params

        self.W1 = np.array(params["W1"], dtype=float)
        self.b1 = np.array(params["b1"], dtype=float)
        self.W2 = np.array(params["W2"], dtype=float)
        self.b2 = np.array(params["b2"], dtype=float)
        self.NAME = str(params["NAME"])

        if "mutation_sigma" in params:
            self.mutation_sigma = float(np.asarray(params["mutation_sigma"]).ravel()[0])
        else:
            self.mutation_sigma = 0.01

    def calculate_score(self, distance, time, no):
        self.score = float(distance)

    def passcardata(self, x, y, speed):
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)

    def getscore(self):
        return self.score
