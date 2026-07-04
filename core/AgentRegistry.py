from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from constants import PATH_STUDENT_AGENTS
from core.protocols import AgentProtocol


AgentClass: TypeAlias = type[AgentProtocol]


REQUIRED_AGENT_METHODS = (
    "decide",
    "mutate",
    "store",
    "get_parameters",
    "set_parameters",
    "calculate_score",
    "passcardata",
    "getscore",
)


@dataclass(frozen=True)
class AgentInfo:
    name: str
    module_name: str
    class_name: str
    path: Path
    source: str


class AgentRegistry:
    def __init__(self, sources: list[tuple[str, str | Path]] | None = None):
        if sources is None:
            sources = [
                ("students.agents", PATH_STUDENT_AGENTS),
                ("AI_engines", "AI_engines"),
        ]
        self.sources = [(package_name, Path(directory)) for package_name, directory in sources]
        self.agents: dict[str, AgentClass] = {}
        self.infos: dict[str, AgentInfo] = {}
        self.errors: dict[str, str] = {}

    def discover(self) -> "AgentRegistry":
        self.agents.clear()
        self.infos.clear()
        self.errors.clear()

        for package_name, directory in self.sources:
            if not directory.is_dir():
                continue

            for path in sorted(directory.glob("*.py")):
                if path.name.startswith("_"):
                    continue

                agent_name = path.stem
                if agent_name in self.agents:
                    continue

                module_name = f"{package_name}.{agent_name}"

                try:
                    module = importlib.import_module(module_name)
                except Exception as exc:
                    self.errors[f"{package_name}.{agent_name}"] = f"Import failed: {exc}"
                    continue

                try:
                    cls = getattr(module, agent_name)
                except AttributeError:
                    self.errors[f"{package_name}.{agent_name}"] = f"Missing class '{agent_name}'."
                    continue

                missing = [method for method in REQUIRED_AGENT_METHODS if not callable(getattr(cls, method, None))]
                if missing:
                    self.errors[f"{package_name}.{agent_name}"] = "Missing methods: " + ", ".join(missing)
                    continue

                self.agents[agent_name] = cls
                self.infos[agent_name] = AgentInfo(
                    name=agent_name,
                    module_name=module_name,
                    class_name=agent_name,
                    path=path,
                    source=package_name,
                )

        return self

    def names(self) -> list[str]:
        return sorted(self.agents.keys())

    def get(self, name: str | None) -> AgentClass | None:
        if not name:
            return self.default_agent()
        return self.agents.get(name.strip())

    def default_name(self) -> str | None:
        names = self.names()
        if not names:
            return None
        if "AIbrain_linear" in self.agents:
            return "AIbrain_linear"
        if "AIbrain_2layer" in self.agents:
            return "AIbrain_2layer"
        return names[0]

    def default_agent(self) -> AgentClass | None:
        name = self.default_name()
        if name is None:
            return None
        return self.agents[name]
