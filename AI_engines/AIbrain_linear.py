from numpy import random as np_random
import random
import numpy as np
import copy
import string

# počet vstupů – ideálně = len(RAYCAST_ANGLES)
N_INPUTS = 9
N_ACTIONS = 4  # [up, down, left, right]

# vždy pojmenováváme jako "AIbrain_jemnoteamu"
class AIbrain_linear:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # pro potreby náhdných znaků
        self.decider = 0
        self.x = 0 # sem se ulozí souradnice x, max HEIGHT*1.3
        self.y = 0 # sem se ulozí souradnice y, max HEIGHT (800)
        self.speed = 0 # sem se ukládá souradnice, max MAXSPEED ( 500)

        self.init_param()

    def init_param(self):
        # zde si vytvoríme promnenne co potrebujeme pro nas model
        # parametry modely vzdy inicializovat v této metode
        self.W = (np_random.rand(N_ACTIONS, N_INPUTS) - 0.5) / N_INPUTS
        self.b = (np_random.rand(N_ACTIONS) - 0.5)

        self.NAME = "SAFR_linear"

        # vždy uložit!
        self.store()

    def decide(self, data):
        self.decider += 1

        x = np.asarray(data, dtype=float).ravel()

        n_w = self.W.shape[1]
        if x.size < n_w:
            x = np.concatenate([x, np.zeros(n_w - x.size)])
        elif x.size > n_w:
            x = x[:n_w]

        # lineární kombinace pro každou akci: W @ x + b
        z = self.W.dot(x) + self.b

        # vracíme přímo z; AI_car pak dělá threshold > 0.5
        return z

    def mutate(self):
        """
        Mutace: všechny váhy i biasy se malé náhodně posunou.
        Tím se skutečně mění lineární kombinace pro každou akci.
        """
        # náhodné perturbace ~ [-0.1, 0.1]
        delta_W = (np_random.rand(*self.W.shape) - 0.5) * 0.1
        delta_b = (np_random.rand(*self.b.shape) - 0.5) * 0.1

        self.W = self.W + delta_W
        self.b = self.b + delta_b

        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))

        self.store()

    def store(self):
        # vše, co se má ukládat do .npz
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

        # Zde nastavit co chceme ukládat:
        self.W = np.array(self.parameters["W"], dtype=float)
        self.b = np.array(self.parameters["b"], dtype=float)
        self.NAME = str(self.parameters["NAME"])


    def calculate_score(self, distance, time, no):
        self.score = distance

    ##################### do těchto funkcí není potřeba zasahovat:
    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)
