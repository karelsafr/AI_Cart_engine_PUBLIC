from numpy import random as np_random
import random
import numpy as np
import copy
import string

# 9 paprsků z raycastu
N_SENSORS = 9

# + rychlost jako extra vstup
N_INPUTS = N_SENSORS + 1  # 9 rays + speed

# akce ve hře: [up, down, left, right]
N_ACTIONS = 4

# předpokládaná max. rychlost pro normalizaci
MAX_SPEED = 500.0


class AIbrain_shallow:
    """
    Jedna shallow neuronová síť:
    - vstup: [raycasty, normalizovaná rychlost]
    - skrytá vrstva: ReLU
    - výstup: 4 sigmoidy -> [up, down, left, right] v (0,1)

    AI_car pak dělá threshold > 0.5, takže více akcí může být aktivních současně.
    """

    def __init__(self):
        super().__init__()
        self.score = 0.0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0

        # data z auta (AI_car.passcardata)
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0

        self.init_param()

    def init_param(self):
        # velikost skryté vrstvy - klidně uprav
        self.H = 12

        # první vrstva: N_INPUTS -> H
        self.W1 = (np_random.rand(self.H, N_INPUTS) - 0.5) / np.sqrt(N_INPUTS)
        self.b1 = (np_random.rand(self.H) - 0.5)

        # druhá vrstva: H -> N_ACTIONS
        self.W2 = (np_random.rand(N_ACTIONS, self.H) - 0.5) / np.sqrt(self.H)
        self.b2 = (np_random.rand(N_ACTIONS) - 0.5)

        self.NAME = "SAFR_shallow_single"

        self.store()

    # ---------- pomocné funkce ----------

    @staticmethod
    def relu(z):
        return np.maximum(z, 0.0)

    @staticmethod
    def sigmoid(z):
        # element-wise sigmoid
        z = np.clip(z, -50, 50)  # numerická stabilita
        return 1.0 / (1.0 + np.exp(-z))

    # ---------- hlavní rozhodovací funkce ----------

    def decide(self, data):
        """
        data: 9 hodnot z raycastu
        návrat: 4 hodnoty v (0,1), [up, down, left, right]
        (AI_car na to dělá threshold > 0.5)
        """
        self.decider += 1

        # 1) připravit vstup x: raycast + rychlost
        rays = np.asarray(data, dtype=float).ravel()

        # zajistit přesně N_SENSORS prvků
        if rays.size < N_SENSORS:
            rays = np.concatenate([rays, np.zeros(N_SENSORS - rays.size)])
        elif rays.size > N_SENSORS:
            rays = rays[:N_SENSORS]

        # rychlost -> <0,1>
        v = float(self.speed)
        if MAX_SPEED > 0:
            v_norm = max(0.0, min(v / MAX_SPEED, 1.0))
        else:
            v_norm = 0.0

        x = np.concatenate([rays, np.array([v_norm], dtype=float)])  # shape (N_INPUTS,)

        # 2) dopředný průchod sítí
        # skrytá vrstva
        z1 = self.W1.dot(x) + self.b1         # (H,)
        h = self.relu(z1)                     # (H,)

        # výstupní vrstva
        z2 = self.W2.dot(h) + self.b2         # (4,)
        p = self.sigmoid(z2)                  # (4,) v (0,1)

        # p vrátíme přímo; AI_car udělá threshold > 0.5
        return p

    # ---------- mutace ----------

    def mutate(self):
        """
        Mutace: malé náhodné posuny všech parametrů.
        """
        sigma_mut = 0.005

        def noisy(arr):
            return arr + np_random.normal(loc=0.0, scale=sigma_mut, size=arr.shape)

        self.W1 = noisy(self.W1)
        self.b1 = noisy(self.b1)
        self.W2 = noisy(self.W2)
        self.b2 = noisy(self.b2)

        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))

        self.store()

    # ---------- ukládání / načítání ----------

    def store(self):
        self.parameters = copy.deepcopy({
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "H": self.H,
            "NAME": self.NAME,
        })

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        self.H = int(self.parameters["H"])
        self.NAME = str(self.parameters["NAME"])

    # ---------- skóre + data z auta ----------

    def calculate_score(self, distance, time, no):
        # můžeš později zjemnit, zatím čistě distance
        self.score = float(distance)

    def passcardata(self, x, y, speed):
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)
