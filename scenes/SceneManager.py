from __future__ import annotations

from core.CsvTilemap import CsvTileMap
from pathlib import Path
from constants import EMPTY_TILE, PATH_DEFAULT_SETTINGS, PATH_STUDENT_MAPS
from core.AgentRegistry import AgentRegistry, AgentClass
from core.protocols import MapSceneProtocol, RestartableSceneProtocol, SceneProtocol

class SceneManager:
    def __init__(self, screen):
        self.screen = screen
        self.cur_scene: SceneProtocol
        self.menu_scene: SceneProtocol
        self.mapeditor_scene: SceneProtocol
        self.playgame_scene: MapSceneProtocol
        self.training_scene: MapSceneProtocol
        self.duel_scene: RestartableSceneProtocol
        self.benchmark_scene: RestartableSceneProtocol
        self.cur_tmap: CsvTileMap
        self.atlas_tmap = None
        self.tile_path_default_setting_csv = ""
        self.TILESIZE = 0
        self.agent_registry: AgentRegistry
        self.brain: AgentClass | None = None
        self.brain_name: str | None = None

    def add_menu(self, menu_scene: SceneProtocol) -> None:
        self.cur_scene = menu_scene
        self.menu_scene = menu_scene

    def set_menu(self) -> None:
        self.cur_scene = self.menu_scene

    def add_mapeditor(self, mapeditor_scene: SceneProtocol) -> None:
        self.mapeditor_scene = mapeditor_scene

    def set_mapa(self) -> None:
        self.cur_scene = self.mapeditor_scene

    def add_playgame(self, playgame_scene: MapSceneProtocol) -> None:
        self.playgame_scene = playgame_scene

    def set_playgame(self, mapname: str | Path) -> None:
        self.cur_scene = self.playgame_scene
        self.playgame_scene.get_map_name(mapname)
        self.playgame_scene.restart()

    def add_training(self, training_scene: MapSceneProtocol) -> None:
        self.training_scene = training_scene

    def set_training(self,mapname: str | Path) -> None:
        self.cur_scene = self.training_scene
        self.training_scene.get_map_name(mapname)
        self.training_scene.restart()

    def set_cur_tmap(self, tmap: CsvTileMap) -> None:
        self.cur_tmap = tmap

    def set_atlas_tmap(self,atlas) -> None:
        self.atlas_tmap = atlas

    def set_default_tmap_name(self, tile_path_default_setting_csv: str) -> None:
        self.tile_path_default_setting_csv = tile_path_default_setting_csv

    def set_TILESIZE(self,TILESIZE: int) -> None:
        self.TILESIZE = TILESIZE

    def set_vehicle_atlas(self, vehicles_atlas) -> None:
        self.vehicles_atlas = vehicles_atlas

    def load_tmap(self, map_file: str | Path) -> None:
        self.cur_tmap = CsvTileMap(
            self.atlas_tmap,
            map_file,
            tile_w=self.TILESIZE,
            tile_h=self.TILESIZE,
            base_tile=EMPTY_TILE,  # základní výplň mapy
            empty_symbol="."  # POZOR! prázdné buňky v CSV znamenají „jen base“
        )
        self.cur_tmap.prerender()

    def draw(self) -> None:
        self.cur_scene.draw(self.screen)

    def update(self, dt: float, keys) -> None:
        if not self.cur_scene.is_active():
            self.active = False
        self.cur_scene.update(dt, keys)

    def event(self, event) -> None:
        self.cur_scene.event(event)

    def is_active(self) -> bool:
        return self.cur_scene.is_active()

    def add_agent_registry(self, agent_registry: AgentRegistry) -> None:
        self.agent_registry = agent_registry
        self.brain_name = agent_registry.default_name()
        self.brain = agent_registry.default_agent()

        if self.brain_name is None:
            print("ERROR: nebyl nalezen zadny platny agent")
        else:
            print(f"Vychozi agent pro trening: {self.brain_name}")

    def add_brain(self,brain: AgentClass) -> None:
        self.brain = brain

    def get_brain(self) -> AgentClass | None:
        return self.brain

    def set_brain(self, brain_name: str | None) -> bool:
        if not hasattr(self, "agent_registry"):
            print("ERROR: AgentRegistry neni inicializovany")
            return False

        selected_name = (brain_name or "").strip() or self.agent_registry.default_name()
        brain = self.agent_registry.get(selected_name)
        if brain is None:
            print(f"ERROR: agent '{brain_name}' nebyl nalezen. Dostupni agenti: {', '.join(self.get_agent_names())}")
            return False

        self.brain = brain
        self.brain_name = selected_name
        print(f"Agent pro trening nastaven: {self.brain_name}")
        return True

    def get_brain_name(self) -> str | None:
        return getattr(self, "brain_name", None)

    def get_agent_names(self) -> list[str]:
        if not hasattr(self, "agent_registry"):
            return []
        return self.agent_registry.names()

    def get_agent_class(self, agent_name: str | None) -> AgentClass | None:
        if not hasattr(self, "agent_registry"):
            return None
        return self.agent_registry.get(agent_name)

    def resolve_map_path(self, map_name: str | Path) -> Path:
        name = (map_name or "").strip()
        if not name:
            name = "DefaultRace"

        if name.endswith(".csv"):
            explicit_path = Path(name)
            if explicit_path.is_file():
                return explicit_path
            name = explicit_path.stem

        if name in ("DefaultRace", "DefaultReset"):
            return Path(PATH_DEFAULT_SETTINGS) / f"{name}.csv"

        student_path = Path(PATH_STUDENT_MAPS) / f"{name}.csv"
        if student_path.is_file():
            return student_path

        return student_path

    def add_duel(self, duel_scene: RestartableSceneProtocol) -> None:
        self.duel_scene = duel_scene

    def set_duel(self) -> None:
        self.cur_scene = self.duel_scene
        self.duel_scene.restart()

    def add_benchmark(self, benchmark_scene: RestartableSceneProtocol) -> None:
        self.benchmark_scene = benchmark_scene

    def set_benchmark(self) -> None:
        self.cur_scene = self.benchmark_scene
        self.benchmark_scene.restart()
