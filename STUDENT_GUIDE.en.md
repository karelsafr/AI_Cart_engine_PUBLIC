# AI Cart V2 - Student Guide

AI Cart V2 is a simple Pygame racing engine for evolutionary training of AI agents. A student writes their own car "brain", trains it on a map, saves the parameters into an `.npz` file, and then the agent can be compared in a duel or benchmark.

The engine and student work are separated. Students normally work only inside the `students/` folder.

The main project index is in `README.en.md`. The exact agent interface contract is in `AGENT_INTERFACE.en.md`.

## Quick start

Run the project from the `AI_Cart_engine_V2/` root directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy
python main.py
```

On Windows, use the virtual environment activation command for your terminal, for example `.venv\Scripts\activate`.

## What a team submits

Each team mainly submits these two files:

```text
students/agents/AIbrain_<TeamName>.py
students/saves/AIbrain_<TeamName>.npz
```

The first file is the agent source code. The second file contains trained parameters saved from training.

Example:

```text
students/agents/AIbrain_SAFR.py
students/saves/AIbrain_SAFR.npz
```

The agent file and the class inside it must have the same name:

```python
class AIbrain_SAFR:
    ...
```

## What students may change

Students may add and edit:

- `students/agents/AIbrain_<TeamName>.py`
- `students/saves/*.npz`
- `students/maps/*.csv`
- optionally `students/benchmark_agents.yaml`, if they are preparing their own benchmark

Students should not modify the engine:

- `core/`
- `scenes/`
- `my_sprites/`
- `UI/`
- `main.py`
- `constants.py`
- `AI_engines/`

`AI_engines/` contains built-in examples and templates. It is not the place for student submissions.

## Student work structure

```text
students/
├── agents/      # Python agent files
├── saves/       # saved .npz parameters
├── maps/        # custom .csv maps
├── results/     # benchmarks, training logs, and plots
└── benchmark_agents.yaml
```

Agents are loaded automatically from two locations:

1. `students/agents/`
2. `AI_engines/`

If an agent with the same name exists in both folders, the version in `students/agents/` has priority.

The default agent after startup is `AIbrain_linear`.

## What an agent should look like

An agent is a Python class used by the car for decisions. The decision input is a list of ray distances from obstacles. The output has four values:

```text
index 0 - throttle
index 1 - brake
index 2 - left
index 3 - right
```

The car treats an action as active if the value is greater than `0.5`.

The agent must have the same interface as the example agents. The best starting reference is:

```text
AI_engines/AIbrain_linear.py
```

The exact contract between the engine and the agent is in:

```text
AGENT_INTERFACE.en.md
```

Required methods:

- `__init__(self)`
- `decide(self, data)`
- `mutate(self)`
- `store(self)`
- `get_parameters(self)`
- `set_parameters(self, parameters)`
- `calculate_score(self, distance, time, no)`
- `passcardata(self, x, y, speed)`
- `getscore(self)`

Practical minimum:

- in `__init__`, initialize parameters, `self.score`, and `self.NAME`
- in `decide`, return 4 numbers for controlling the car
- in `mutate`, randomly modify parameters
- in `store`, store parameters into `self.parameters`
- in `set_parameters`, restore state from a save
- in `calculate_score`, set fitness into `self.score`

## Main menu

After running `python main.py`, the main menu opens:

- `Hraj` - manual driving with one car
- `Mapa` - map editor
- `Trénuj` - evolutionary training of the selected agent
- `Souboj` - duel between two agents
- `Benchmark` - batch comparison of multiple agents
- `Konec` - exit the application

Controls:

- up/down arrows - select item
- Enter - confirm
- Esc - go back or exit

At the bottom of the menu there are two text fields:

- `agent_name` - agent name, for example `AIbrain_SAFR`
- map - map name, for example `DefaultRace`, `DefaultReset`, or a name from `students/maps/`

Map names are entered without a path. The `.csv` suffix is optional.

## Training an agent

Procedure:

1. Put the agent file into `students/agents/`.
2. Run `python main.py`.
3. In the menu, enter the agent name into `agent_name`, for example `AIbrain_SAFR`.
4. In the map field, enter for example `DefaultRace`.
5. Select `Trénuj`.
6. Set the training parameters.
7. Press `START`.

Fields in training:

- `pocet_aut` - number of cars in one generation
- `pocet_generaci` - number of generations
- `cars_to_next` - how many of the best cars are used for the next generation; must be between `1` and `pocet_aut`
- `save_as` - save name for the best agent
- `max_time` - maximum length of one generation in seconds
- `load_from` - save to continue from

Buttons:

- `START` - starts training from scratch with the selected agent
- `PAUSE` / `RESUME` - pauses or resumes the current run without losing the generation
- `NEXT GEN` - manually ends the current generation and selects the best cars just like when time expires
- `RELOAD SET` - reads new `pocet_aut`, `cars_to_next`, `save_as`, `load_from`, and `max_time` values from the fields, then ends the current generation with the same logic as `NEXT GEN`
- `RESET` - stops and clears the current training run
- `SAVE` - saves the current best brain according to `save_as`
- `LOAD` - loads a save from `load_from` and starts training from the loaded parameters

`RELOAD SET` does not run `LOAD` from a file. It only applies new setting values for the next generation and keeps the normal logic for selecting the best cars and mutating them.

If `cars_to_next` is outside the allowed range, the program clamps it to the nearest safe value and prints a warning to the console. `SAVE` before starting or loading training only prints a message, because there is nothing to save yet.

Recommended save names:

```text
AIbrain_<TeamName>.npz
```

For example:

```text
AIbrain_SAFR.npz
```

Saves are stored in:

```text
students/saves/
```

## Loading saves

The `LOAD` button in training searches for a save in this order:

```text
students/saves/<name>
students/saves/<name>.npz
```

This means you can enter for example `FAST` or `FAST.npz` into `load_from`.

If the save does not exist, cannot be opened, or does not match the currently selected agent, the program does not crash. It prints an error to the console and training does not start.

Important: the save must match the agent architecture. For example, parameters for `AIbrain_2layer` may not be loadable into `AIbrain_linear`.

## Training monitoring

On the right side of the training scene there is a simple live monitor:

- `Agent` - currently selected agent
- `Gen` - current generation and generation time
- `Alive` - how many cars are still driving
- `Crash` - how many cars have ended
- `Best` - best score in the current generation
- `All` - best score of the whole run
- `Avg` - average generation score
- `Med` - generation score median
- `Stag` - how many generations have passed without improving the best score
- `Save` - target save name

The monitor is informational. If `Crash` grows quickly and `Best` does not improve, the agent probably needs better decision logic, mutation, or scoring.

## Training logs and plots

Each new `START` or successful `LOAD` in training creates its own CSV log. After each completed generation, one row with metrics is appended:

- best score of the generation
- average score of the generation
- median score of the generation
- number of crashed cars
- number of active cars
- best score of the whole run
- stagnation, generation time, agent, map, and save name

CSV logs are saved into:

```text
students/results/training_logs/
```

If the `matplotlib` package is available, training also saves a live PNG plot into:

```text
students/results/training_plots/
```

The default plot interval is every 5 completed generations. The setting is in `constants.py`:

```python
TRAINING_PLOT_EVERY = 5
```

The value `10` means saving every 10 generations. The value `0` disables PNG plots. If `matplotlib` is not installed or the plot cannot be saved, the program only prints a message to the console and training continues. CSV logging does not depend on `matplotlib`.

## Maps

Default maps are in:

```text
DefaultSettings/
```

Student maps are in:

```text
students/maps/
```

Usable names in the menu:

- `DefaultRace` - default race track
- `DefaultReset` - base map for the editor
- a custom map name from `students/maps/`, for example `my_map`

Map editor:

- choose `Mapa` in the menu
- enter the map name without `.csv` into the text field
- `Save map` saves the map into `students/maps/<name>.csv`
- `Load` loads the map from `students/maps/<name>.csv`
- `RESET` loads `DefaultSettings/DefaultReset.csv`

The starting tile is:

```text
road_dirt42
```

Click in the left part of the map to change tiles. Repeated clicking on the same position cycles through the available tile types.

## Duel between two agents

Select `Souboj` in the menu.

Fields on the right:

- `map_name` - map name
- `engine1_class` - class of the first agent
- `engine1_save` - save of the first agent
- `engine2_class` - class of the second agent
- `engine2_save` - save of the second agent

Buttons:

- `LOAD MAP` - loads the map
- `START DUEL` - starts the duel

The agent is found by class/file name. The save is searched in `students/saves/`.

## Benchmarking multiple agents

Benchmark is used to compare multiple agents at once. Select `Benchmark` in the menu.

Default file with the agent list:

```text
students/benchmark_agents.yaml
```

Format:

```yaml
agents:
  - agent: AIbrain_TeamA
    save: AIbrain_TeamA.npz
  - agent: AIbrain_TeamB
    save: AIbrain_TeamB.npz
```

Fields in the benchmark scene:

- `map_name` - benchmark map
- `finish_row` - row of the finish tile
- `finish_col` - column of the finish tile
- `time_limit_s` - time limit
- `output_csv` - output CSV name
- `agents_file` - YAML or TXT file with the agent list
- `engines_fallback` - manual agent list if the file is not available

The default `agents_file` is `benchmark_agents.yaml`, which resolves to:

```text
students/benchmark_agents.yaml
```

Benchmark outputs are saved into:

```text
students/results/
```

During a running benchmark, the left side of the map shows a white list in the form `#order brain_name`. The order matches the YAML/TXT list. The same number is drawn in bright yellow with a dark outline directly over the car and is also saved into the result CSV in the `benchmark_order` column.

The final `rank` is racing order: cars that reached the finish are sorted by time. Cars that did not finish are placed after them by status `TIMEOUT`, `CRASH`, `OUT`. Traveled distance `distance_tiles` remains only an informational metric in the CSV and result output.

Manual fallback format:

```text
AIbrain_TeamA:AIbrain_TeamA.npz; AIbrain_TeamB:AIbrain_TeamB.npz
```

Supported item separators are `;` or `|`. Between agent and save, you can use `:`, `,`, or a space.

## Recommended student workflow

1. Use `AI_engines/AIbrain_linear.py` as the mental reference.
2. Create `students/agents/AIbrain_<TeamName>.py`.
3. Keep the required methods and the same file/class name.
4. Select your agent in the menu.
5. Train on `DefaultRace` and on custom maps.
6. Watch the training monitor.
7. Save the best result into `students/saves/AIbrain_<TeamName>.npz`.
8. Verify that `LOAD` can load your save.
9. Try a benchmark or duel.
10. Submit the `.py` and `.npz`.

## Common problems

`Agent was not found`

- check that the file is in `students/agents/`
- check that the file name and class name are the same
- check that the class implements the required methods

`Save was not found`

- check `students/saves/`
- in `load_from`, you can enter the name with or without `.npz`

`Save cannot be used for the current agent`

- you are probably loading parameters for a different architecture
- choose the correct agent in the menu or the correct `.npz` file

`Map does not exist`

- for default maps, use `DefaultRace` or `DefaultReset`
- for custom maps, check `students/maps/<name>.csv`

`Benchmark did not find agents`

- check `students/benchmark_agents.yaml`
- check YAML indentation
- check that the agent exists and the save is in `students/saves/`

## Code typing

The engine uses gradual, soft typing. Shared protocols are in:

```text
core/protocols.py
```

The most important one is `AgentProtocol`, which describes the methods a student agent must provide. Typing helps orientation, editor support, and safer engine changes; it does not change program behavior.

Python files in the project use `from __future__ import annotations`, so annotations can be added gradually without affecting program behavior.

Configuration for `pyright` and `mypy` is in `pyproject.toml`. Student agents are not strictly enforced by the default check so they remain flexible for teaching. If you have the tool installed, you can run for example:

```bash
pyright
mypy
```

## Audio note

`main.py` sets a dummy audio driver:

```python
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
```

This improves Pygame stability on some systems. Students do not need to change anything.
