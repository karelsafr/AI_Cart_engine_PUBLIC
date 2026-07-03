import numpy as np


# --------------------------------------------------
# Utility
# --------------------------------------------------
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class AIbrain_maie:
    TEAM_NAME = "maie"

    def __init__(self):
        self.NAME = f"MAIE_"

        # -------- FITNESS --------
        self.score = 0.0
        self.best_distance_ever = 0.0
        self.progress_bonus = 300.0

        # -------- CAR DATA --------
        self.car_speed = 0.0

        # -------- STEERING MEMORY --------
        self._steer_memory = 0.0

        self.init_param()
        self.store()

    # --------------------------------------------------
    # Parameters / architecture
    # --------------------------------------------------
    def init_param(self):
        self.input_dim = None

        # network
        self.hidden1 = 16
        self.hidden2 = 12

        self.W1 = None; self.b1 = None
        self.W2 = None; self.b2 = None
        self.W3 = None; self.b3 = None

        # mutation
        self.mut_sigma_small = 0.02
        self.mut_sigma_big = 0.15
        self.mut_big_prob = 0.05
        self.reset_weight_prob = 0.002

        # steering smoothing
        self.steer_smooth = 0.85

    # --------------------------------------------------
    # Weight initialization (Xavier)
    # --------------------------------------------------
    def _init_weights(self, input_dim):
        self.input_dim = input_dim

        lim1 = np.sqrt(6 / (input_dim + self.hidden1))
        self.W1 = np.random.uniform(-lim1, lim1, (self.hidden1, input_dim))
        self.b1 = np.zeros(self.hidden1)

        lim2 = np.sqrt(6 / (self.hidden1 + self.hidden2))
        self.W2 = np.random.uniform(-lim2, lim2, (self.hidden2, self.hidden1))
        self.b2 = np.zeros(self.hidden2)

        lim3 = np.sqrt(6 / (self.hidden2 + 4))
        self.W3 = np.random.uniform(-lim3, lim3, (4, self.hidden2))
        self.b3 = np.zeros(4)

    # --------------------------------------------------
    # Decision
    # --------------------------------------------------
    def decide(self, data):
        x = np.array(data, dtype=np.float32)

        if self.W1 is None:
            self._init_weights(len(x))

        # backward compatibility (ray count)
        expected = self.W1.shape[1]
        if len(x) < expected:
            x = np.pad(x, (0, expected - len(x)))
        elif len(x) > expected:
            x = x[:expected]

        # normalize rays (close obstacle -> high value)
        mx = max(np.max(x), 1e-6)
        x = 1.0 - np.clip(x / mx, 0.0, 1.0)

        # forward pass
        h1 = np.tanh(self.W1 @ x + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        out = sigmoid(self.W3 @ h2 + self.b3)

        # -------- STEERING WITH MEMORY --------
        steer_raw = out[3] - out[2]
        self._steer_memory = (
            self.steer_smooth * self._steer_memory +
            (1.0 - self.steer_smooth) * steer_raw
        )
        self._steer_memory = np.clip(self._steer_memory, -1.0, 1.0)

        if self._steer_memory > 0:
            out[3] = 0.5 + abs(self._steer_memory)
            out[2] = 0.5 - abs(self._steer_memory)
        else:
            out[2] = 0.5 + abs(self._steer_memory)
            out[3] = 0.5 - abs(self._steer_memory)

        # -------- AUTOMATIC SLOWING IN TURNS --------
        turn_intensity = abs(self._steer_memory)

        # reduce throttle
        out[0] *= (1.0 - 0.6 * turn_intensity)

        # light braking
        out[1] = np.clip(turn_intensity * 0.4, 0.0, 1.0)

        return out.tolist()

    # --------------------------------------------------
    # Mutation
    # --------------------------------------------------
    def mutate(self):
        if self.W1 is None:
            return

        big = np.random.rand() < self.mut_big_prob
        sigma = self.mut_sigma_big if big else self.mut_sigma_small

        def mutate_arr(a):
            a = a + np.random.normal(0, sigma, a.shape)
            mask = np.random.rand(*a.shape) < self.reset_weight_prob
            if np.any(mask):
                a[mask] = np.random.uniform(-1.0, 1.0, np.sum(mask))
            return a

        self.W1 = mutate_arr(self.W1); self.b1 = mutate_arr(self.b1)
        self.W2 = mutate_arr(self.W2); self.b2 = mutate_arr(self.b2)
        self.W3 = mutate_arr(self.W3); self.b3 = mutate_arr(self.b3)

        # clamp
        np.clip(self.W1, -1.5, 1.5, out=self.W1)
        np.clip(self.W2, -1.5, 1.5, out=self.W2)
        np.clip(self.W3, -1.5, 1.5, out=self.W3)
        np.clip(self.b1, -1.0, 1.0, out=self.b1)
        np.clip(self.b2, -1.0, 1.0, out=self.b2)
        np.clip(self.b3, -1.0, 1.0, out=self.b3)

        self.store()

    # --------------------------------------------------
    # FITNESS (WITH PROGRESS BONUS)
    # --------------------------------------------------
    def calculate_score(self, distance, time, no):
        d = float(distance)
        t = float(time)

        steer_penalty = abs(self._steer_memory)

        self.score = (
            d
            - 0.02 * t
            - 2.0 * steer_penalty
        )

        # -------- PROGRESSIVE CHECKPOINT BONUS --------
        if d > self.best_distance_ever:
            improvement = d - self.best_distance_ever
            self.score += self.progress_bonus + improvement * 5.0
            self.best_distance_ever = d

    # --------------------------------------------------
    # Engine hooks
    # --------------------------------------------------
    def passcardata(self, x, y, speed):
        self.car_speed = speed

    def getscore(self):
        return self.score

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    def store(self):
        self.parameters = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "input_dim": self.input_dim,
            "best_distance_ever": self.best_distance_ever
        }

    def get_parameters(self):
        self.store()
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.W1 = parameters["W1"]; self.b1 = parameters["b1"]
        self.W2 = parameters["W2"]; self.b2 = parameters["b2"]
        self.W3 = parameters["W3"]; self.b3 = parameters["b3"]
        self.input_dim = parameters.get("input_dim", self.W1.shape[1])
        self.best_distance_ever = parameters.get("best_distance_ever", 0.0)
