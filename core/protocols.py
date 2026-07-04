from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, TypeAlias

import pygame


ColorRGB: TypeAlias = tuple[int, int, int]
TileGrid: TypeAlias = list[list[str]]
AgentParameters: TypeAlias = Any


class AgentProtocol(Protocol):
    NAME: str
    score: float

    def decide(self, data: Sequence[float]) -> Sequence[float]: ...

    def mutate(self) -> None: ...

    def store(self) -> None: ...

    def get_parameters(self) -> dict[str, Any]: ...

    def set_parameters(self, parameters: AgentParameters) -> None: ...

    def calculate_score(self, distance: float, time: float, no: int) -> None: ...

    def passcardata(self, x: float, y: float, speed: float) -> None: ...

    def getscore(self) -> float: ...


class ButtonAction(Protocol):
    def action(self) -> Any: ...


class SceneProtocol(Protocol):
    active: bool

    def draw(self, screen: pygame.Surface) -> None: ...

    def update(self, dt: float, keys: Sequence[bool]) -> None: ...

    def event(self, event: pygame.event.Event) -> None: ...

    def is_active(self) -> bool: ...


class RestartableSceneProtocol(SceneProtocol, Protocol):
    def restart(self) -> None: ...


class MapSceneProtocol(RestartableSceneProtocol, Protocol):
    def get_map_name(self, mapname: str | Path) -> None: ...


class MapResolverProtocol(Protocol):
    def resolve_map_path(self, map_name: str | Path) -> Path: ...
