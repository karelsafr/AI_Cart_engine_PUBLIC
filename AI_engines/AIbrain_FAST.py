from numpy import random as np_random
import random
import numpy as np
import copy
import string
from constants import MAX_SPEED

N_ACTIONS = 4  # [up, down, left, right]
N_INPUTS = 9

class AIbrain_FAST:
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self):
        super().__init__()
        self.score = 0.0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0

        # car state (filled by passcardata)
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0

        # heuristic params (store/load so they persist)
        self.k_gap = 0.8       # how strong "follow the biggest gap" bias is
        self.front_th = 12.0   # braking threshold in sensor units

        self.init_param()

    def init_param(self):
        # logistic-regression-like init: small weights, bias near 0
        self.W  = np_random.normal(0.0, 0.1, size=(N_ACTIONS, N_INPUTS))
        self.b  = np.zeros(N_ACTIONS, dtype=float)
        self.Ws = np_random.normal(0.0, 0.1, size=(N_ACTIONS,))

        self.NAME = "FAST_"
        self.store()

    def store(self):
        self.parameters = copy.deepcopy({
            "W": self.W,
            "b": self.b,
            "Ws": self.Ws,
            "k_gap": float(self.k_gap),
            "front_th": float(self.front_th),
            "NAME": self.NAME,
        })

    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-u))

    @staticmethod
    def _gap_dir(x: np.ndarray) -> float:
        """
        Returns steering hint in [-1, +1]:
        -1 = strongest free gap is to the LEFT, +1 = to the RIGHT.
        Follow-the-Gap style: bubble around nearest obstacle, then argmax. :contentReference[oaicite:1]{index=1}
        """
        x = np.maximum(np.asarray(x, dtype=float), 0.0)

        i_min = int(np.argmin(x))     # nearest obstacle ray
        bubble = 1                    # widen bubble by 1 neighbor on each side
        x2 = x.copy()
        x2[max(0, i_min-bubble):min(len(x2), i_min+bubble+1)] = 0.0

        i_best = int(np.argmax(x2))   # ray with largest free distance
        center = (len(x2) - 1) / 2.0  # for 9 rays => 4.0
        return float((i_best - center) / center)  # -1..+1

    def decide(self, data):
        self.decider += 1

        x = np.asarray(data, dtype=float).ravel()
        if x.size < N_INPUTS:
            x = np.concatenate([x, np.zeros(N_INPUTS - x.size)])
        elif x.size > N_INPUTS:
            x = x[:N_INPUTS]

        speed01 = np.clip(float(self.speed) / max(MAX_SPEED, 1e-6), 0.0, 1.0)

        logits = self.W.dot(x) + (self.Ws * speed01) + self.b

        # ---- GAP bias (reactive heuristic on top of learned logits) ----
        g = self._gap_dir(x)           # -1..+1
        k = float(self.k_gap)

        if g > 0:
            logits[self.RIGHT] += k * g
        else:
            logits[self.LEFT] += k * (-g)

        # optional throttle/brake hint by front sensor (index 4 is "center")
        front = float(x[4])
        if front < float(self.front_th):
            logits[self.DOWN] += 0.8
        else:
            logits[self.UP] += 0.3

        probs = self._sigmoid(logits)  # 0..1
        return probs

    def mutate(self):
        p_bitflip = 0.08
        p_flip = 0.03

        sigma = float(getattr(self, "mutation_sigma", 0.05))
        sigma = float(np.clip(sigma, 0.005, 0.12))

        if np_random.rand() < p_bitflip:
            mask_W  = (np_random.rand(*self.W.shape)  < p_flip)
            mask_b  = (np_random.rand(*self.b.shape)  < p_flip)
            mask_Ws = (np_random.rand(*self.Ws.shape) < p_flip)

            self.W[mask_W]   *= -1.0
            self.b[mask_b]   *= -1.0
            self.Ws[mask_Ws] *= -1.0

            self.NAME += "_MUT_BF_" + ''.join(random.choices(self.chars, k=3))
        else:
            sigma_W  = sigma
            sigma_b  = 0.8 * sigma
            sigma_Ws = 0.4 * sigma

            self.W  += np_random.normal(0.0, sigma_W,  size=self.W.shape)
            self.b  += np_random.normal(0.0, sigma_b,  size=self.b.shape)
            self.Ws += np_random.normal(0.0, sigma_Ws, size=self.Ws.shape)

            self.NAME += f"_MUT_G_s{sigma:.3f}_" + ''.join(random.choices(self.chars, k=3))

        # (volitelné) občas mutuj i heuristické parametry
        if np_random.rand() < 0.15:
            self.k_gap = float(np.clip(self.k_gap + np_random.normal(0.0, 0.15), 0.0, 3.0))
            self.front_th = float(max(0.0, self.front_th + np_random.normal(0.0, 1.0)))

        self.store()

    def calculate_score(self, distance, time, no):
        eps = 1e-6
        dist = float(distance)
        t_eff = max(float(time), 0.5)
        avg_speed = dist / (t_eff + eps)
        end_speed01 = np.clip(float(self.speed) / max(MAX_SPEED, 1e-6), 0.0, 1.0)

        self.score = (
            1_000_000.0 * dist +
            10_000.0    * avg_speed +
            100.0       * end_speed01 -
            t_eff
        )

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            p = {key: parameters[key] for key in parameters.files}
        else:
            p = copy.deepcopy(parameters)

        self.parameters = p
        self.W = np.array(p["W"], dtype=float)
        self.b = np.array(p["b"], dtype=float)
        self.Ws = np.array(p["Ws"], dtype=float)
        self.k_gap = float(p.get("k_gap", 0.8))
        self.front_th = float(p.get("front_th", 12.0))
        self.NAME = "FAST"
