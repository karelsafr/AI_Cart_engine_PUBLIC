from numpy import random as np_random
import random
import numpy as np
import copy
import string

# vzdy pojmenováváme jako "AIbrain_jemnoteamu"
class AIbrain_TeamName:
    def __init__(self):
        super().__init__()
        self.score = 0

        #self.rnd = np_random.rand
        self.chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        self.decider = 0

        self.init_param()

    def init_param(self):
        self.w1 = np_random.rand(4)
        self.w2 = np_random.rand(4)
        self.NAME ="Safr_"+''.join(random.choices(self.chars, k=5))

        # vzdy ulozit:
        self.store()

    def store(self):
        self.parameters = copy.deepcopy({
            'w1': self.w1,
            'w2': self.w2,
            "NAME": self.NAME,
        })

    def decide(self, data):
        self.decider += 1
        if np.round(self.decider) % 2 == 1:
            return np.round(self.w1)
        else:
            return np.round(self.w2)

    # mezi epochama bdue nutné mutovat, studenti si udelaj sami
    def mutate(self):
        if np_random.rand(1) < 0.5:
            self.w1 = np.array([float(i+  (np.round(np_random.rand(1))-0.5)/4) for i in self.w1])
            self.NAME += "_MUT_W1_"+''.join(random.choices(self.chars, k=3))
        else:
            self.w2 = np.array([float(i+  (np.round(np_random.rand(1))-0.5)/4) for i in self.w2])
            self.NAME += "_MUT_W2_"+''.join(random.choices(self.chars, k=3))


        self.store()


    # do techto funkcí není treba zasahovat:
    def calculate_score(self, distance, time, no):
        self.score = distance/time + no

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    #### nutné dospat!!!
    def get_parameters(self):
        """Getter pro získání všech parametrů"""
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        """Setter pro nastavení všech parametrů"""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            # Z NpzFile rovnou udělám nový slovník (tím "odstřihnu" file handler)
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            # Pokud už je to slovník (nebo jiný obyč. objekt), udělám kopii
            self.parameters = copy.deepcopy(parameters)

        self.w1 = self.parameters['w1']
        self.w2 = self.parameters['w2']
        self.NAME = self.parameters['NAME']


if __name__ == "__main__":
    aibrain = AIbrain_TeamName()
    print(aibrain.decide(np.array([1,1,2,0])))