# Mapa projektu AI_Cart_engine_V2

Tento dokument popisuje aktuální strukturu projektu a hranice mezi enginem a studentskou prací.

## Přehled složek

```text
AI_Cart_engine_V2/
├── main.py                         # start aplikace, inicializace Pygame, atlasů, scén a agent registry
├── constants.py                    # konstanty okna, cest, dlaždic, rychlostí a raycastů
├── README.md                       # stručný rozcestník projektu
├── STUDENT_GUIDE.md                # hlavní návod pro studenty
├── AGENT_INTERFACE.md              # stabilní kontrakt mezi enginem a agentem
├── PROJECT_STRUCTURE.md            # tato mapa projektu
├── pyproject.toml                  # konfigurace typovacích nástrojů
├── AI_engines/                     # vestavěné vzory agentů
│   ├── AIbrain_linear.py           # výchozí agent
│   ├── AIbrain_TeamName.py         # jednoduchá šablona
│   └── AIbrain_2layer.py           # složitější ukázkový agent
├── students/                       # studentská práce a výstupy
│   ├── agents/                     # studentské .py agenty
│   ├── saves/                      # uložené .npz parametry
│   ├── maps/                       # studentské .csv mapy
│   ├── results/                    # výsledky benchmarku, logy tréninku a grafy
│   ├── benchmark_agents.yaml       # seznam agentů pro benchmark
│   └── README.md                   # stručný návod pro studentskou složku
├── core/                           # jádro enginu
├── scenes/                         # jednotlivé obrazovky aplikace
├── my_sprites/                     # auta a kolizní bloky
├── UI/                             # jednoduché UI prvky
├── DefaultSettings/                # výchozí mapy enginu
└── assets/racing-pack/             # grafická aktiva
```

## Hranice odpovědnosti

`students/` je pracovní prostor pro studenty. Sem patří vlastní agenti, natrénované savy, mapy a výsledky benchmarku.

Engine tvoří hlavně:

- `main.py`
- `constants.py`
- `core/`
- `scenes/`
- `my_sprites/`
- `UI/`
- `AI_engines/`

Studenti by engine neměli měnit. Pokud potřebují experimentovat, výsledné odevzdání má pořád být jen dvojice `.py` agenta a `.npz` savu ve `students/`.

## Běhový tok aplikace

1. `main.py` nastaví prostředí Pygame.
2. Načte grafické atlasy přes `core/TextureAtlas.py`.
3. Načte výchozí mapu přes `core/CsvTilemap.py`.
4. `core/AgentRegistry.py` automaticky objeví agenty.
5. `SceneManager` dostane mapu, atlasy, registry agentů a scény.
6. Hlavní smyčka předává eventy, update a draw aktuální scéně.

Zjednodušený tok:

```text
main.py
  ├── TextureAtlas
  ├── CsvTileMap
  ├── AgentRegistry
  └── SceneManager
        ├── MenuScene
        ├── MapEditorScene
        ├── PlayGameScene
        ├── TrainingScene
        ├── DuelScene
        └── BenchmarkScene
```

## Automatické načítání agentů

Agent registry hledá agenty v pořadí:

```text
students/agents/
AI_engines/
```

Výchozí agent je:

```text
AIbrain_linear
```

Agent je platný, pokud:

- soubor má název například `AIbrain_TeamA.py`
- třída uvnitř se jmenuje `AIbrain_TeamA`
- třída má povinné metody definované v `core/AgentRegistry.py`

Pokud je stejný název ve `students/agents/` i `AI_engines/`, použije se studentská verze.

## Cesty k mapám

Mapy řeší `SceneManager.resolve_map_path()`.

Pravidla:

1. Prázdný název znamená `DefaultRace`.
2. `DefaultRace` a `DefaultReset` se berou z `DefaultSettings/`.
3. Studentské mapy se hledají ve `students/maps/`.

Aktuální doporučené umístění:

```text
students/maps/<nazev>.csv
```

Výchozí mapy:

```text
DefaultSettings/DefaultRace.csv
DefaultSettings/DefaultReset.csv
```

## Cesty k savům

Nové savy patří do:

```text
students/saves/
```

Tréninkové tlačítko `LOAD` zkouší:

```text
students/saves/<nazev>
students/saves/<nazev>.npz
```

Pokud soubor chybí, nejde načíst nebo nepasuje k aktuálnímu agentovi, trénink vypíše chybu a nespustí se.

## Core moduly

- `core/AgentRegistry.py` - objevování a validace agentů
- `core/CsvTilemap.py` - načtení, normalizace, vykreslení a uložení CSV map
- `core/TextureAtlas.py` - načtení spritesheetů a pojmenovaných textur
- `core/TrainingLogger.py` - CSV logy generací a volitelné PNG grafy tréninku
- `core/VectorIterator.py` - cyklení dlaždic v editoru map
- `core/car_manager.py` - evoluční trénink, generace, skórování, save/load, validace vstupů a monitoring

## Scény

- `scenes/MenuScene.py` - hlavní menu, výběr mapy a agenta
- `scenes/MapEditorScene.py` - editor map, ukládá do `students/maps/`
- `scenes/PlayGameScene.py` - ruční jízda
- `scenes/TrainingScene.py` - evoluční trénink a živý monitor
- `scenes/DuelScene.py` - souboj dvou agentů
- `scenes/BenchmarkScene.py` - porovnání více agentů ze YAML/TXT seznamu
- `scenes/SceneManager.py` - přepínání scén, sdílený stav, cesty k mapám a agentům

## Sprite a fyzika

- `my_sprites/AI_car.py` - auto řízené agentem
- `my_sprites/car.py` - ručně řízené auto
- `my_sprites/block.py` - kolizní hrany mapy

AI auto:

- dostává raycast vzdálenosti
- volá `brain.decide(...)`
- prahuje výstupy přes `> 0.5`
- posílá agentovi pozici a rychlost přes `passcardata`
- aktualizuje skóre přes `calculate_score(distance, time, no)`

## UI

- `UI/Button.py` - jednoduché tlačítko s callbackem `action()`
- `UI/TextInput.py` - textové pole
- `UI/MakeGrid.py` - pomocná mřížka pro editor map

## Trénink

Trénink řídí `scenes/TrainingScene.py` a evoluční logiku drží `core/car_manager.py`.

Hlavní ovládání:

- `START` - založí nový trénink podle aktuálních polí
- `PAUSE` / `RESUME` - pozastaví nebo obnoví běžící generaci
- `NEXT GEN` - ukončí aktuální generaci stejnou logikou jako vypršení času
- `RELOAD SET` - načte nové `pocet_aut`, `cars_to_next`, `save_as`, `load_from` a `max_time`, potom přejde do další generace běžnou evoluční logikou
- `RESET` - zastaví a vymaže aktuální běh
- `SAVE` / `LOAD` - ukládání a načítání parametrů ze `students/saves/`

`RELOAD SET` nenačítá save ze souboru; jen převezme nové nastavení pro další generaci. Logika výběru nejlepších aut a mutací zůstává stejná jako u normálního konce generace.

## Benchmark

Výchozí seznam agentů:

```text
students/benchmark_agents.yaml
```

Formát:

```yaml
agents:
  - agent: AIbrain_TeamA
    save: AIbrain_TeamA.npz
```

Výsledky:

```text
students/results/
```

Rank v benchmarku je závodní pořadí. Dojetá auta jsou řazená podle času, nedojetá auta jsou až za nimi podle stavu `TIMEOUT`, `CRASH`, `OUT`. Najetá vzdálenost zůstává jen informační metrika.

## Tréninkové logy

Trénink ukládá jeden CSV řádek po každé dokončené generaci:

```text
students/results/training_logs/
```

Pokud je dostupný `matplotlib`, ukládá se i průběžný PNG graf:

```text
students/results/training_plots/
```

Interval PNG grafu řídí `TRAINING_PLOT_EVERY` v `constants.py`. Chybějící nebo nefunkční `matplotlib` trénink nezastaví.

Benchmark umí použít i ruční fallback pole ve scéně:

```text
AIbrain_TeamA:AIbrain_TeamA.npz; AIbrain_TeamB:AIbrain_TeamB.npz
```

Aktuální studentské složky:

```text
students/agents/
students/saves/
students/maps/
students/results/
```

## Dokumentace

- `README.md` - stručný rozcestník projektu
- `STUDENT_GUIDE.md` - postupy pro studenty
- `AGENT_INTERFACE.md` - přesné rozhraní studentského agenta vůči enginu
- `students/README.md` - krátký návod přímo ve studentském workspace
- `PROJECT_STRUCTURE.md` - technická mapa projektu

## Typování

- `core/protocols.py` - sdílené typové protokoly a aliasy
- `AgentProtocol` - rozhraní, které engine očekává od agentů
- `pyproject.toml` - mírná konfigurace pro `pyright` a `mypy`

Všechny Python soubory používají `from __future__ import annotations`, aby šly typy doplňovat postupně bez změny runtime chování. Typování je zavedené jako vývojová pomůcka a nemá měnit studentské API.
