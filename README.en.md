# AI Cart V2

AI Cart V2 is an educational Pygame project for evolutionary training of racing AI agents. Students write their own car "brain", train it on a map, save the parameters into an `.npz` file, and then compare agents in duels or benchmarks.

Author: Karel Šafr, PhD

## Who this project is for

- students who want to create their own agent
- teachers who want to run training sessions, benchmarks, and team comparisons
- anyone who wants to inspect a simple evolutionary racing engine in Python

## Quick start

Run the project from the `AI_Cart_engine_V2/` root directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy
python main.py
```

On Windows, use the virtual environment activation command for your terminal, for example `.venv\Scripts\activate`.

## Where to find things

- [STUDENT_GUIDE.en.md](STUDENT_GUIDE.en.md) - the main student guide: creating an agent, training, save/load, maps, duels, and benchmarks
- [AGENT_INTERFACE.en.md](AGENT_INTERFACE.en.md) - the exact contract between the engine and a student agent
- [PROJECT_STRUCTURE.en.md](PROJECT_STRUCTURE.en.md) - project map, runtime flow, and the boundary between engine code and student work
- [students/README.en.md](students/README.en.md) - a short cheat sheet directly in the student workspace

## Basic idea

A student mainly submits two files:

```text
students/agents/AIbrain_<TeamName>.py
students/saves/AIbrain_<TeamName>.npz
```

Agents are loaded automatically from `students/agents/`. Trained parameters are saved into `students/saves/`. Custom maps belong in `students/maps/`, and benchmark results, training CSV logs, and plots go into `students/results/`.

The default agent is `AIbrain_linear`.

## First orientation

If you are a student and want to start working, open [STUDENT_GUIDE.en.md](STUDENT_GUIDE.en.md).

If you only need to know which methods an agent must provide, open [AGENT_INTERFACE.en.md](AGENT_INTERFACE.en.md).

If you are modifying the engine or checking the project structure, open [PROJECT_STRUCTURE.en.md](PROJECT_STRUCTURE.en.md).
