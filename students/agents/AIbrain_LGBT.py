"""
AIbrain_TeamName - Pokročilý MLP mozek pro autonomní řízení auta

Architektura: 13 vstupů → 16 neuronů (tanh) → 4 výstupy (sigmoid)

Vstupy (13):
- 9× raycast vzdálenosti (normalizované / 10)
- 1× rychlost (normalizovaná / 500)
- 1× akcelerace (normalizovaná / 100, clamp -1 až 1)
- 2× předchozí steering akce (left, right)

Výstupy (4):
- [up, down, left, right] - práh > 0.5 aktivuje akci

Ochranné mechanismy proti degradaci:
- Bias clipping: výstupní biasy omezeny na [-2, 2]
- Up bias ochrana: minimální hodnota 0.5 (auto musí chtít jet)
- Weight clipping: váhy omezeny na [-10, 10]
- Exploration boost: 5% šance na větší mutaci pro únik z lokálních minim
"""

import numpy as np
from numpy import random as np_random
import random
import copy
import string


# Konstanty sítě
N_INPUTS = 13      # 9 raycast + speed + accel + 2 prev_steering
N_HIDDEN = 16      # počet neuronů ve skryté vrstvě
N_OUTPUTS = 4      # [up, down, left, right]

# Konstanty pro normalizaci
MAX_RAYCAST_DISTANCE = 10.0   # maximální vzdálenost raycastu v tiles
MAX_SPEED = 500.0             # maximální rychlost auta
ACCEL_NORMALIZER = 100.0      # normalizátor akcelerace

# Konstanty pro adaptivní mutaci
SIGMA_MIN = 0.01    # minimální mutation rate (zvýšeno z 0.001 pro lepší exploraci)
SIGMA_MAX = 0.5     # maximální mutation rate
TAU = 0.1           # rychlost změny mutation rate

# Konstanty pro ochranu proti degradaci
BIAS_MIN = -2.0     # minimální hodnota výstupních biasů
BIAS_MAX = 2.0      # maximální hodnota výstupních biasů
UP_BIAS_MIN = 0.5   # minimální hodnota up biasu (sigmoid(0.5) ≈ 0.62)
WEIGHT_MAX = 10.0   # maximální absolutní hodnota vah
EXPLORATION_CHANCE = 0.05  # 5% šance na exploration boost


def sigmoid(x):
    """Sigmoid aktivační funkce - výstup v rozsahu (0, 1)"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def xavier_init(n_out, n_in):
    """Xavier inicializace vah pro stabilnější start"""
    std = np.sqrt(2.0 / (n_in + n_out))
    return np_random.randn(n_out, n_in) * std


class AIbrain_LGBT:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.x = 0
        self.y = 0
        self.speed = 0
        self.prev_speed = 0
        self.prev_output = np.array([0.5, 0.5, 0.5, 0.5])
        
        self.init_param()
    
    def init_param(self):
        """Inicializace všech parametrů sítě"""
        
        # Xavier inicializace vah
        self.W1 = xavier_init(N_HIDDEN, N_INPUTS)   # 16 × 13
        self.b1 = np.zeros(N_HIDDEN)                 # 16
        self.W2 = xavier_init(N_OUTPUTS, N_HIDDEN)  # 4 × 16
        
        # Inicializace výstupních biasů - zajistí, že auta na začátku jedou
        # sigmoid(2.0) ≈ 0.88, takže "up" bude aktivní
        # sigmoid(-1.0) ≈ 0.27, takže "down" (brzda) bude neaktivní
        self.b2 = np.array([2.0, -1.0, 0.0, 0.0])    # [up, down, left, right]
        
        # Inicializace mutation rates (sigma)
        self.sigma_W1 = np.full_like(self.W1, 0.1)
        self.sigma_b1 = np.full_like(self.b1, 0.1)
        self.sigma_W2 = np.full_like(self.W2, 0.1)
        self.sigma_b2 = np.full_like(self.b2, 0.1)
        
        # Jméno mozku
        self.NAME = "LGBT_"
        
        # Reset runtime state
        self.prev_speed = 0
        self.prev_output = np.array([0.5, 0.5, 0.5, 0.5])
        
        self.store()
    
    def decide(self, data):
        """
        Rozhodovací funkce - vrací 4 hodnoty [up, down, left, right]
        
        Args:
            data: seznam 9 raycast vzdáleností
        
        Returns:
            numpy array s 4 hodnotami v rozsahu (0, 1)
        """
        
        # 1. Normalizace raycast dat (9 hodnot)
        rays = np.zeros(9)
        data_array = np.asarray(data, dtype=float).ravel()
        n_rays = min(len(data_array), 9)
        rays[:n_rays] = data_array[:n_rays]
        rays = rays / MAX_RAYCAST_DISTANCE
        
        # 2. Normalizace rychlosti
        speed_norm = self.speed / MAX_SPEED
        
        # 3. Výpočet a normalizace akcelerace
        accel = (self.speed - self.prev_speed) / ACCEL_NORMALIZER
        accel_norm = np.clip(accel, -1.0, 1.0)
        
        # 4. Předchozí steering akce
        prev_left = float(self.prev_output[2])
        prev_right = float(self.prev_output[3])
        
        # 5. Sestavení vstupního vektoru (13 hodnot)
        x = np.concatenate([
            rays,                        # 9 hodnot
            [speed_norm],                # 1 hodnota
            [accel_norm],                # 1 hodnota
            [prev_left, prev_right]      # 2 hodnoty
        ])
        
        # 6. Forward pass - skrytá vrstva s tanh aktivací
        h = np.tanh(self.W1 @ x + self.b1)
        
        # 7. Forward pass - výstupní vrstva se sigmoid aktivací
        output = sigmoid(self.W2 @ h + self.b2)
        
        # 8. Uložit pro příští frame
        self.prev_speed = self.speed
        self.prev_output = output.copy()
        
        return output
    
    def mutate(self):
        """
        Adaptivní mutace s per-weight mutation rates, sigma clipping
        a ochranami proti degradaci modelu.
        """
        
        # 0. Exploration boost - občas větší mutace pro únik z lokálních minim
        exploration_multiplier = 1.0
        if random.random() < EXPLORATION_CHANCE:
            exploration_multiplier = 3.0  # 3× větší mutace
        
        # 1. Mutovat sigma hodnoty (self-adaptation)
        self.sigma_W1 *= np.exp(TAU * np_random.randn(*self.sigma_W1.shape))
        self.sigma_b1 *= np.exp(TAU * np_random.randn(*self.sigma_b1.shape))
        self.sigma_W2 *= np.exp(TAU * np_random.randn(*self.sigma_W2.shape))
        self.sigma_b2 *= np.exp(TAU * np_random.randn(*self.sigma_b2.shape))
        
        # 2. Clipping sigma - pojistka proti degeneraci
        self.sigma_W1 = np.clip(self.sigma_W1, SIGMA_MIN, SIGMA_MAX)
        self.sigma_b1 = np.clip(self.sigma_b1, SIGMA_MIN, SIGMA_MAX)
        self.sigma_W2 = np.clip(self.sigma_W2, SIGMA_MIN, SIGMA_MAX)
        self.sigma_b2 = np.clip(self.sigma_b2, SIGMA_MIN, SIGMA_MAX)
        
        # 3. Mutovat váhy pomocí jejich sigma (s exploration multiplier)
        self.W1 += exploration_multiplier * self.sigma_W1 * np_random.randn(*self.W1.shape)
        self.b1 += exploration_multiplier * self.sigma_b1 * np_random.randn(*self.b1.shape)
        self.W2 += exploration_multiplier * self.sigma_W2 * np_random.randn(*self.W2.shape)
        self.b2 += exploration_multiplier * self.sigma_b2 * np_random.randn(*self.b2.shape)
        
        # 4. OCHRANA: Clipping vah - zabránění explodujícím vahám
        self.W1 = np.clip(self.W1, -WEIGHT_MAX, WEIGHT_MAX)
        self.W2 = np.clip(self.W2, -WEIGHT_MAX, WEIGHT_MAX)
        self.b1 = np.clip(self.b1, -WEIGHT_MAX, WEIGHT_MAX)
        
        # 5. OCHRANA: Clipping výstupních biasů - zabránění degeneraci
        self.b2 = np.clip(self.b2, BIAS_MIN, BIAS_MAX)
        
        # 6. OCHRANA: Up bias nesmí klesnout pod minimum (auto musí chtít jet!)
        self.b2[0] = max(self.b2[0], UP_BIAS_MIN)
        
        # 7. Aktualizovat jméno
        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))
        
        # 8. Reset runtime state pro novou generaci
        self.prev_speed = 0
        self.prev_output = np.array([0.5, 0.5, 0.5, 0.5])
        
        self.store()
    
    def store(self):
        """Uloží všechny parametry do slovníku pro serializaci"""
        self.parameters = copy.deepcopy({
            # Váhy sítě
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            # Mutation rates
            "sigma_W1": self.sigma_W1,
            "sigma_b1": self.sigma_b1,
            "sigma_W2": self.sigma_W2,
            "sigma_b2": self.sigma_b2,
            # Metadata
            "NAME": self.NAME,
        })
    
    def get_parameters(self):
        return copy.deepcopy(self.parameters)
    
    def set_parameters(self, parameters):
        """Načte parametry z uloženého slovníku nebo .npz souboru"""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)
        
        self.parameters = params_dict
        
        # Načíst váhy
        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        
        # Načíst mutation rates
        self.sigma_W1 = np.array(self.parameters["sigma_W1"], dtype=float)
        self.sigma_b1 = np.array(self.parameters["sigma_b1"], dtype=float)
        self.sigma_W2 = np.array(self.parameters["sigma_W2"], dtype=float)
        self.sigma_b2 = np.array(self.parameters["sigma_b2"], dtype=float)
        
        # Načíst metadata
        self.NAME = str(self.parameters["NAME"])
        
        # Reset runtime state
        self.prev_speed = 0
        self.prev_output = np.array([0.5, 0.5, 0.5, 0.5])
        self.NAME = "LGBT_"
    
    def calculate_score(self, distance, time, no):
        """
        Fitness funkce - kombinace vzdálenosti a přežití
        
        Strategie:
        - Fáze 1 (začátek): pouze distance - naučit se jezdit
        - Fáze 2 (po zvládnutí trati): distance + speed_bonus - naučit se jezdit rychle
        
        Přepni mezi fázemi odkomentováním příslušného řádku.
        """
        # === FÁZE 1: Naučit se jezdit (DOPORUČENO pro nové tratě) ===
        self.score = distance
        
        # === FÁZE 2: Naučit se jezdit rychle (až po zvládnutí trati) ===
        # avg_speed = distance / max(time, 0.1)
        # speed_bonus = avg_speed * 0.3
        # self.score = distance + speed_bonus
    
    def passcardata(self, x, y, speed):
        """Přijímá data o pozici a rychlosti auta každý frame"""
        self.x = x
        self.y = y
        self.speed = speed
    
    def getscore(self):
        """Vrátí aktuální fitness skóre"""
        return self.score
