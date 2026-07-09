# AI_Cart_engine_V2 Project Map

This document describes the current project structure and the boundary between the engine and student work.

## Folder overview

```text
AI_Cart_engine_V2/
├── main.py                         # application start, Pygame setup, atlases, scenes, and agent registry
├── constants.py                    # window, path, tile, speed, and raycast constants
├── README.md                       # short project index in Czech
├── README.en.md                    # short project index in English
├── STUDENT_GUIDE.md                # main student guide in Czech
├── STUDENT_GUIDE.en.md             # main student guide in English
├── AGENT_INTERFACE.md              # stable contract between engine and agent in Czech
├── AGENT_INTERFACE.en.md           # stable contract between engine and agent in English
├── PROJECT_STRUCTURE.md            # project map in Czech
├── PROJECT_STRUCTURE.en.md         # this project map in English
├── pyproject.toml                  # type checker configuration
├── AI_engines/                     # built-in example agents
│   ├── AIbrain_linear.py           # default agent
│   ├── AIbrain_TeamName.py         # simple template
│   └── AIbrain_2layer.py           # more advanced example agent
├── students/                       # student work and outputs
│   ├── agents/                     # student .py agents
│   ├── saves/                      # saved .npz parameters
│   ├── maps/                       # student .csv maps
│   ├── results/                    # benchmark results, training logs, and plots
│   ├── benchmark_agents.yaml       # list of agents for benchmark runs
│   ├── README.md                   # short guide for the student folder in Czech
│   └── README.en.md                # short guide for the student folder in English
├── core/                           # engine core
├── scenes/                         # application screens
├── my_sprites/                     # cars and collision blocks
├── UI/                             # simple UI elements
├── DefaultSettings/                # default engine maps
└── assets/racing-pack/             # graphical assets
```

## Responsibility boundary

`students/` is the workspace for students. Custom agents, trained saves, maps, and benchmark results belong there.

The engine mainly consists of:

- `main.py`
- `constants.py`
- `core/`
- `scenes/`
- `my_sprites/`
- `UI/`
- `AI_engines/`

Students should not modify the engine. If they need to experiment, the final submission should still be only the agent `.py` file and `.npz` save in `students/`.

## Application runtime flow

1. `main.py` sets up the Pygame environment.
2. It loads graphical atlases through `core/TextureAtlas.py`.
3. It loads the default map through `core/CsvTilemap.py`.
4. `core/AgentRegistry.py` automatically discovers agents.
5. `SceneManager` receives the map, atlases, agent registry, and scenes.
6. The main loop forwards events, update, and draw calls to the current scene.

Simplified flow:

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

## Automatic agent loading

The agent registry searches for agents in this order:

```text
students/agents/
AI_engines/
```

The default agent is:

```text
AIbrain_linear
```

An agent is valid if:

- the file is named for example `AIbrain_TeamA.py`
- the class inside is named `AIbrain_TeamA`
- the class has the required methods defined in `core/AgentRegistry.py`

If the same name exists in both `students/agents/` and `AI_engines/`, the student version is used.

## Map paths

Maps are resolved by `SceneManager.resolve_map_path()`.

Rules:

1. An empty name means `DefaultRace`.
2. `DefaultRace` and `DefaultReset` are loaded from `DefaultSettings/`.
3. Student maps are searched in `students/maps/`.

Current recommended location:

```text
students/maps/<name>.csv
```

Default maps:

```text
DefaultSettings/DefaultRace.csv
DefaultSettings/DefaultReset.csv
```

## Save paths

New saves belong in:

```text
students/saves/
```

The training `LOAD` button tries:

```text
students/saves/<name>
students/saves/<name>.npz
```

If the file is missing, cannot be loaded, or does not match the current agent, training prints an error and does not start.

## Core modules

- `core/AgentRegistry.py` - agent discovery and validation
- `core/CsvTilemap.py` - CSV map loading, normalization, drawing, and saving
- `core/TextureAtlas.py` - spritesheet and named texture loading
- `core/TrainingLogger.py` - generation CSV logs and optional PNG training plots
- `core/VectorIterator.py` - tile cycling in the map editor
- `core/car_manager.py` - evolutionary training, generations, scoring, save/load, input validation, and monitoring

## Scenes

- `scenes/MenuScene.py` - main menu, map selection, and agent selection
- `scenes/MapEditorScene.py` - map editor, saves into `students/maps/`
- `scenes/PlayGameScene.py` - manual driving
- `scenes/TrainingScene.py` - evolutionary training and live monitor
- `scenes/DuelScene.py` - duel between two agents
- `scenes/BenchmarkScene.py` - comparison of multiple agents from a YAML/TXT list
- `scenes/SceneManager.py` - scene switching, shared state, map paths, and agents

## Sprites and physics

- `my_sprites/AI_car.py` - car controlled by an agent
- `my_sprites/car.py` - manually controlled car
- `my_sprites/block.py` - collision edges of the map

AI car:

- receives raycast distances
- calls `brain.decide(...)`
- thresholds outputs with `> 0.5`
- passes position and speed to the agent through `passcardata`
- updates score through `calculate_score(distance, time, no)`

## UI

- `UI/Button.py` - simple button with an `action()` callback
- `UI/TextInput.py` - text input field
- `UI/MakeGrid.py` - helper grid for the map editor

## Training

Training is controlled by `scenes/TrainingScene.py`, and the evolutionary logic lives in `core/car_manager.py`.

Main controls:

- `START` - starts a new training run from the current fields
- `PAUSE` / `RESUME` - pauses or resumes the running generation
- `NEXT GEN` - ends the current generation using the same logic as a timeout
- `RELOAD SET` - reloads `pocet_aut`, `cars_to_next`, `save_as`, `load_from`, and `max_time`, then moves to the next generation using the normal evolutionary logic
- `RESET` - stops and clears the current run
- `SAVE` / `LOAD` - saves and loads parameters from `students/saves/`

`RELOAD SET` does not load a save from a file; it only applies new settings for the next generation. The selection of the best cars and mutations stays the same as during a normal generation end.

## Benchmark

Default agent list:

```text
students/benchmark_agents.yaml
```

Format:

```yaml
agents:
  - agent: AIbrain_TeamA
    save: AIbrain_TeamA.npz
```

Results:

```text
students/results/
```

Benchmark rank is racing order. Finished cars are sorted by time, and unfinished cars are placed after them by status `TIMEOUT`, `CRASH`, `OUT`. Traveled distance remains only an informational metric.

## Training logs

Training writes one CSV row after each completed generation:

```text
students/results/training_logs/
```

If `matplotlib` is available, it also saves a live PNG plot:

```text
students/results/training_plots/
```

The PNG plot interval is controlled by `TRAINING_PLOT_EVERY` in `constants.py`. Missing or broken `matplotlib` does not stop training.

The benchmark can also use the manual fallback field in the scene:

```text
AIbrain_TeamA:AIbrain_TeamA.npz; AIbrain_TeamB:AIbrain_TeamB.npz
```

Current student folders:

```text
students/agents/
students/saves/
students/maps/
students/results/
```

## Documentation

- `README.md` / `README.en.md` - short project index
- `STUDENT_GUIDE.md` / `STUDENT_GUIDE.en.md` - student procedures
- `AGENT_INTERFACE.md` / `AGENT_INTERFACE.en.md` - exact student agent interface against the engine
- `students/README.md` / `students/README.en.md` - short guide directly in the student workspace
- `PROJECT_STRUCTURE.md` / `PROJECT_STRUCTURE.en.md` - technical project map

## Typing

- `core/protocols.py` - shared type protocols and aliases
- `AgentProtocol` - interface expected from agents by the engine
- `pyproject.toml` - mild configuration for `pyright` and `mypy`

All Python files use `from __future__ import annotations`, so types can be added gradually without changing runtime behavior. Typing is introduced as a development aid and should not change the student API.
