# AGENT_INTERFACE.md

Tento dokument popisuje stabilní rozhraní mezi enginem AI Cart V2 a studentským agentem.

Je to kontrakt: pokud agent splní pravidla níže, engine ho umí automaticky najít, spustit v tréninku, uložit, načíst, použít v duelu a použít v benchmarku.

## Umístění a název

Studentský agent patří do:

```text
students/agents/
```

Soubor i třída musí mít stejné jméno:

```text
students/agents/AIbrain_MujTeam.py
class AIbrain_MujTeam
```

Název agenta se v menu, benchmark YAML a duelu zadává bez `.py`:

```text
AIbrain_MujTeam
```

Engine hledá agenty v pořadí:

1. `students/agents/`
2. `AI_engines/`

Pokud existuje stejný název v obou složkách, použije se studentská verze.

## Konstruktor

Agent musí jít vytvořit bez argumentů:

```python
brain = AIbrain_MujTeam()
```

V `__init__` má agent připravit vše, co potřebuje pro rozhodování, mutace, skóre a ukládání.

Doporučené minimum:

```python
self.NAME = "MujTeam"
self.score = 0.0
self.store()
```

## Povinné atributy

Engine očekává:

- `NAME` - lidsky čitelný název mozku pro výpisy, benchmark a debug
- `score` - aktuální fitness hodnota agenta

`score` má být číslo. Vyšší hodnota znamená lepší agent.

## Povinné metody

Registry kontroluje, že agent má tyto metody:

```python
def decide(self, data): ...
def mutate(self): ...
def store(self): ...
def get_parameters(self): ...
def set_parameters(self, parameters): ...
def calculate_score(self, distance, time, no): ...
def passcardata(self, x, y, speed): ...
def getscore(self): ...
```

Typovaný tvar stejného rozhraní je v `core/protocols.py` jako `AgentProtocol`.

## decide(data)

Volá se při řízení auta.

```python
def decide(self, data):
    return [up, down, left, right]
```

`data` je sekvence raycast vzdáleností v dlaždicích. Aktuálně engine používá 9 paprsků:

```text
[-90, -45, -20, -5, 0, 5, 20, 45, 90]
```

Výstup musí mít alespoň 4 číselné hodnoty:

```text
index 0 - plyn
index 1 - brzda
index 2 - doleva
index 3 - doprava
```

Auto akci provede, pokud je příslušná hodnota větší než `0.5`.

Příklad:

```python
return [1.0, 0.0, 0.0, 0.8]
```

Tento výstup znamená plyn a zatáčení doprava.

## passcardata(x, y, speed)

Volá se před `decide`.

```python
def passcardata(self, x, y, speed):
    self.x = x
    self.y = y
    self.speed = speed
```

Parametry:

- `x` - aktuální x pozice auta v pixelech
- `y` - aktuální y pozice auta v pixelech
- `speed` - aktuální rychlost auta

Agent tato data může použít pro rozhodování nebo skórování. Pokud je nepotřebuje, může je jen uložit nebo ignorovat.

## calculate_score(distance, time, no)

Volá se během jízdy po pohybu auta.

```python
def calculate_score(self, distance, time, no):
    self.score = distance
```

Parametry:

- `distance` - celková ujetá vzdálenost auta v dlaždicích
- `time` - čas běhu auta v sekundách
- `no` - pomocná hodnota enginu, studenti ji nemusí používat

Tato metoda má aktualizovat `self.score`. Trénink potom vybírá lepší agenty podle skóre.

## getscore()

Vrací aktuální skóre:

```python
def getscore(self):
    return self.score
```

Engine používá skóre při řazení aut v tréninku.

## mutate()

Volá se při vytváření nové generace.

```python
def mutate(self):
    ...
    self.store()
```

Metoda má náhodně změnit parametry agenta. Po změně parametrů je doporučené zavolat `self.store()`, aby se aktuální stav dal uložit do `.npz`.

Mutace nemá měnit rozhraní agenta. Agent musí po mutaci pořád umět `decide`, `get_parameters` a `set_parameters`.

## store()

Připraví parametry pro uložení:

```python
def store(self):
    self.parameters = {
        "W": self.W,
        "b": self.b,
        "NAME": self.NAME,
    }
```

Do `parameters` ukládejte jen hodnoty, které umí `numpy.savez` uložit do `.npz`: čísla, řetězce, numpy pole nebo jednoduché kombinace těchto hodnot.

## get_parameters()

Vrací parametry pro `SAVE`:

```python
def get_parameters(self):
    return self.parameters
```

Trénink při ukládání zavolá:

```python
np.savez(save_path, **brain.get_parameters())
```

Proto musí `get_parameters()` vracet slovník.

## set_parameters(parameters)

Obnoví agenta ze savu:

```python
def set_parameters(self, parameters):
    self.W = parameters["W"]
    self.b = parameters["b"]
    self.NAME = str(parameters["NAME"])
    self.store()
```

`parameters` může být:

- `numpy.lib.npyio.NpzFile` z `np.load(...)`
- slovník parametrů z jiného agenta

Metoda musí nastavit interní stav tak, aby další volání `decide` fungovalo. Pokud save neodpovídá architektuře agenta, metoda může vyhodit chybu; engine ji zachytí a vypíše hlášku.

## Save soubor

Savy patří do:

```text
students/saves/
```

Doporučený název:

```text
AIbrain_MujTeam.npz
```

Save musí odpovídat architektuře agenta. Parametry uložené z jednoho typu agenta nemusí jít načíst do jiného typu agenta.

## Minimální kostra agenta

```python
from __future__ import annotations

import copy
import numpy as np


class AIbrain_MujTeam:
    def __init__(self) -> None:
        self.NAME = "MujTeam"
        self.score = 0.0
        self.W = np.zeros((4, 9))
        self.b = np.zeros(4)
        self.store()

    def decide(self, data):
        x = np.asarray(data, dtype=float)
        y = self.W.dot(x) + self.b
        return y

    def mutate(self):
        self.W += np.random.normal(0, 0.05, self.W.shape)
        self.b += np.random.normal(0, 0.05, self.b.shape)
        self.store()

    def store(self):
        self.parameters = {
            "W": self.W,
            "b": self.b,
            "NAME": self.NAME,
        }

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        self.W = np.array(parameters["W"], dtype=float)
        self.b = np.array(parameters["b"], dtype=float)
        self.NAME = str(parameters["NAME"])
        self.store()

    def calculate_score(self, distance, time, no):
        self.score = distance

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score
```

## Co engine garantuje

Engine garantuje:

- agent je vytvořen bez argumentů
- při jízdě se volá `passcardata`, potom `decide`, potom `calculate_score`
- při ukládání se volá `get_parameters`
- při načítání se volá `set_parameters`
- při nové generaci se používá kopie parametrů a `mutate`
- agenti ze `students/agents/` mají přednost před vestavěnými agenty

## Na co se nespoléhat

Student by se neměl spoléhat na interní detaily enginu:

- přesnou implementaci `Car_manager`
- přesné pořadí aut ve sprite groupech
- absolutní cesty mimo `students/`
- přímé importy interních scén
- to, že `no` v `calculate_score` bude mít trvale stejný význam

Stabilní část je rozhraní v tomto dokumentu.

## Vzor

Nejlepší aktuální vzor je:

```text
AI_engines/AIbrain_linear.py
```

Pro hlubší kontrolu rozhraní slouží:

```text
core/protocols.py
core/AgentRegistry.py
```
