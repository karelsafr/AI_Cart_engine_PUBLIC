from numpy import random as np_random
import random
import numpy as np
import copy
import string

from constants import TILESIZE

class ONEX2:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        self.decider = 0
        self.stotal=0

        # Start tile pass counter (based on the default spawn tile position).
        # Default training spawns at (col=4,row=8) in tiles.
        self._start_tile_col = 4
        self._start_tile_row = 8
        self.start_tile_passes = 0
        self._in_start_tile = False
        self._last_printed_start_tile_passes = -1

        self.init_param()

    def init_param(self):
        self.c1 = random.random()
        self.c2 = random.random()
        self.c3 = random.random()
        self.c4 = random.random()
        self.s1=random.random()
        self.s2=random.random()
        self.s3=random.random()

        self.NAME ="ONEX_"
        self.store()

    def store(self):
        self.parameters = copy.deepcopy({
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'c4': self.c4,
            's1': self.s1,
            's2': self.s2,
            's3': self.s3,
            "NAME": "Onex2",
        })

    #in [-90, -45, -20, -5, 0, 5, 20, 45, 90]
    #out [up down left right] ie [zrychlit zpomalit left right]
    def decide(self, data):
        self.decider += 1
        self.stotal+=self.speed/(45*500)

        eps = 1e-6
        #w_left = np.array([1.0, 0.8, 0.5, 0.3])
        #w_right = np.array([0.3, 0.5, 0.8, 1.0])
        #w_left = np.array([0.9, 0.7, 0.6, 0.4])
        #w_right = np.array([0.4, 0.6, 0.7, 0.9])
        w_left = np.array([self.c1,self.c2,self.c3,self.c4])
        w_right = np.array([self.c4,self.c3,self.c2,self.c1])
        r = np.asarray(data[:9], dtype=float)

        # distances → influence (closer obstacle = stronger bias)
        left_raw = np.sum(w_left / (r[0:4] + eps))
        right_raw = np.sum(w_right / (r[5:9] + eps))

        total = left_raw + right_raw + eps
        X = left_raw / total
        Y = right_raw / total
        if ((r[4] < 1.0+((self.s1-0.5)))) and (self.speed > 70+((self.s2-0.5)*3)):  return [0, 1, float(X), float(Y)]
        if (self.speed > 100+((self.s3-0.5))):  return [0, 0.0, float(X), float(Y)]
        return [1, 0.0, float(X), float(Y)]

    def mutate(self):
        if np_random.rand(1) < 0.5:
            self.c1 = random.random()
        if np_random.rand(1)<0.5:
            self.c2 = random.random()
        if np_random.rand(1)<0.5:
            self.c3 = random.random()
        if np_random.rand(1)<0.5:
            self.c4 = random.random()

        if np_random.rand(1) < 0.5:
            self.s1 = random.random()
        if np_random.rand(1) < 0.5:
            self.s2 = random.random()
        if np_random.rand(1) < 0.5:
            self.s3 = random.random()

        self.store()

    def calculate_score(self, distance, time, no):
        # Print how many times we entered the start tile (only when it changes).
        if self.start_tile_passes != self._last_printed_start_tile_passes:
            self._last_printed_start_tile_passes = self.start_tile_passes

        self.score = distance + no + 100*self.start_tile_passes
        #print(self.score)

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

        # Count entries into the start tile.
        try:
            col = int(float(x) // TILESIZE)
            row = int(float(y) // TILESIZE)
            is_start = (col == self._start_tile_col) and (row == self._start_tile_row)
        except Exception:
            is_start = False

        if is_start and not self._in_start_tile:
            self.start_tile_passes += 1

        self._in_start_tile = bool(is_start)

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            self.parameters = copy.deepcopy(parameters)

        self.c1 = self.parameters['c1']
        self.c2 = self.parameters['c2']
        self.c3 = self.parameters['c3']
        self.c4 = self.parameters['c4']
        self.s1 = self.parameters['s1']
        self.s2 = self.parameters['s2']
        self.s3 = self.parameters['s3']
        self.NAME = "Onex2"
