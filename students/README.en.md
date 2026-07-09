# Students workspace

This folder is intended for student agents, saves, maps, and benchmark outputs. The engine lives outside this folder.

The detailed student guide is in `../STUDENT_GUIDE.en.md`. The exact agent contract is in `../AGENT_INTERFACE.en.md`.

```text
students/
├── agents/                 # AIbrain_<TeamName>.py
├── saves/                  # AIbrain_<TeamName>.npz
├── maps/                   # custom .csv maps
├── results/                # benchmarks, training logs, and plots
├── benchmark_agents.yaml   # list of agents for benchmark runs
└── README.md
```

## Agent

The file and the class must have the same name:

```text
students/agents/AIbrain_MyTeam.py
class AIbrain_MyTeam
```

Use this as the reference example:

```text
AI_engines/AIbrain_linear.py
```

The exact agent interface contract is in:

```text
../AGENT_INTERFACE.en.md
```

The agent must implement these methods:

- `decide`
- `mutate`
- `store`
- `get_parameters`
- `set_parameters`
- `calculate_score`
- `passcardata`
- `getscore`

## Save

Training saves `.npz` files into:

```text
students/saves/
```

Recommended name:

```text
AIbrain_<TeamName>.npz
```

When loading, you can enter the name with or without `.npz`.

During training, `cars_to_next` must be between `1` and `pocet_aut`. If you enter a value outside the allowed range, the engine clamps it to a safe value and prints a warning. `SAVE` before starting or loading a training run only prints a message, because there is nothing to save yet.

The `RELOAD SET` button during a running training session takes new values for `pocet_aut`, `cars_to_next`, `save_as`, `load_from`, and `max_time`, then creates the next generation using the same logic as `NEXT GEN`. It does not load a save from a file.

## Training logs

After each completed generation, training automatically writes one CSV row into:

```text
students/results/training_logs/
```

If `matplotlib` is available, a live PNG plot is saved into:

```text
students/results/training_plots/
```

CSV logging works on its own. If `matplotlib` is missing, training continues without plots.

## Maps

Save custom maps from the editor into:

```text
students/maps/<name>.csv
```

In the menu, you can then enter only `<name>`.

## Benchmark

The list of agents for benchmarks is in:

```text
students/benchmark_agents.yaml
```

Format:

```yaml
agents:
  - agent: AIbrain_MyTeam
    save: AIbrain_MyTeam.npz
```

Benchmark results are saved into:

```text
students/results/
```

During a running benchmark, the left side of the map shows a white list in the form `#order brain_name`. The same order number is drawn in bright yellow with a dark outline directly over the car and stored in the result CSV as `benchmark_order`.

The final `rank` favors actually reaching the finish: finished cars are sorted by time, and unfinished cars are placed after them by status `TIMEOUT`, `CRASH`, `OUT`. `distance_tiles` is only an additional informational metric.

## Submission

You typically submit:

```text
students/agents/AIbrain_<TeamName>.py
students/saves/AIbrain_<TeamName>.npz
```

Do not modify engine files outside `students/` unless that is explicitly assigned.

## Typing note

The engine has the `AgentProtocol` type protocol, which describes the required agent methods. For you, the most important thing is still to keep the same methods as in `AI_engines/AIbrain_linear.py`; typing only helps the editor and engine checks. Student agents may use annotations, but they are not strictly enforced for teaching purposes.
