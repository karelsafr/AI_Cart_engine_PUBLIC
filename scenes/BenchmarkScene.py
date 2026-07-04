from __future__ import annotations

import os
import csv
from pathlib import Path
from datetime import datetime

import pygame
import numpy as np

from constants import (
    BLACK, WHITE,
    WIDTH, TILESIZE,
    MAP_MENUSIZE, MAP_BUTTON_IDENT, MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT, MAP_BUTTON_FONTSIZE,
    tilesides, PATH_SAVES, PATH_BENCHMARK_AGENTS, PATH_BENCHMARK_RESULTS
)
from my_sprites.AI_car import AI_car
from my_sprites.block import Blocks
from UI.TextInput import TextInput
from UI.Button import Button
from core.protocols import AgentProtocol


class _LoadMapButton:
    def __init__(self, scene: "BenchmarkScene"):
        self.scene = scene

    def action(self) -> None:
        self.scene.load_map()


class _StartBenchmarkButton:
    def __init__(self, scene: "BenchmarkScene"):
        self.scene = scene

    def action(self) -> None:
        self.scene.start_benchmark()


class BenchmarkScene:
    """
    Benchmark více mozků na jedné mapě.

    Doporučený workflow:
      seznam agentů načíst z students/benchmark_agents.yaml

    Podporovaný YAML formát:
      agents:
        - agent: AIbrain_A
          save: AIbrain_A.npz
        - agent: AIbrain_B
          save: AIbrain_B.npz

    Engines input formát (jedno pole):
      "AIbrain_A:AIbrain_A.npz; AIbrain_B:AIbrain_B.npz; AIbrain_linear:userbrain.npz"

    Separátory položek: ; nebo |
    Separátor třída-soubor: : nebo , nebo mezera

    Cíl je definován dlaždicí (finish_row, finish_col).
    Auto je FINISHED, jakmile jeho střed (car.pos) vstoupí do cílové dlaždice.
    """

    def __init__(self, scene_manager):
        self.scene_manager = scene_manager

        self.font = pygame.font.SysFont(None, MAP_BUTTON_FONTSIZE)
        self.font_small = pygame.font.SysFont(None, int(MAP_BUTTON_FONTSIZE * 0.7))
        self.font_car_number = pygame.font.SysFont(None, max(23, int(MAP_BUTTON_FONTSIZE * 0.68)), bold=True)
        self.font_benchmark_label = pygame.font.SysFont(None, max(16, int(MAP_BUTTON_FONTSIZE * 0.45)))

        self.active = True

        x0 = WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT

        # map name
        self.input_map = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 0.0, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "map_name"
        )
        self.input_map.set_default("DefaultRace")

        # finish tile row/col
        self.input_finish_row = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 0.5, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "finish_row"
        )
        self.input_finish_row.set_default("8")

        self.input_finish_col = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 1.0, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "finish_col"
        )
        self.input_finish_col.set_default("5")

        # time limit
        self.input_time_limit = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 1.5, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "time_limit_s"
        )
        self.input_time_limit.set_default("30")

        # output CSV name
        self.input_output = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 2.0, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "output_csv"
        )
        self.input_output.set_default("benchmark_results.csv")

        # agents config file
        self.input_agents_file = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 2.5, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "agents_file"
        )
        self.input_agents_file.set_default("benchmark_agents.yaml")

        # fallback engines list
        self.input_engines = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 3.0, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "engines_fallback"
        )

        # buttons
        self.buttons = [
            Button(
                (x0, 0 + TILESIZE * 4.0),
                (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                "LOAD MAP",
                self.font,
                _LoadMapButton(self)
            ),
            Button(
                (x0, 0 + TILESIZE * 5.0),
                (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                "START BENCHMARK",
                self.font,
                _StartBenchmarkButton(self)
            ),
        ]

        # map and blocks
        self.Blocks: Blocks | None = None
        self.map_w_px = 0
        self.map_h_px = 0

        # start pos
        self.start_pos: tuple[float, float] | None = None

        # finish rect
        self.finish_rect: pygame.Rect | None = None
        self.finish_row = 0
        self.finish_col = 9

        # cars
        self.cars_group = pygame.sprite.Group()
        self.cars: list[AI_car] = []

        # run state
        self.benchmark_active = False
        self.benchmark_time = 0.0
        self.time_limit_s = 30.0

        self.results: list[dict] = []
        self.results_lines: list[str] = []
        self.last_saved_path: Path | None = None

        # načti mapu hned na start
        self.load_map()

    def restart(self) -> None:
        self.active = True
        self._reset_run_state(keep_map=True)

    def _reset_run_state(self, *, keep_map: bool) -> None:
        self.cars_group.empty()
        self.cars.clear()

        self.benchmark_active = False
        self.benchmark_time = 0.0

        self.results.clear()
        self.results_lines.clear()
        self.last_saved_path = None

        if not keep_map:
            self.Blocks = None
            self.start_pos = None
            self.finish_rect = None

    def load_map(self) -> None:
        name = self.input_map.get_text("DefaultRace")
        map_file = self.scene_manager.resolve_map_path(name)
        if not os.path.exists(map_file):
            print(f"ERROR: mapa {map_file} neexistuje")
            return

        self.scene_manager.load_tmap(map_file)
        AI_car.set_atlas(self.scene_manager.vehicles_atlas)

        self.Blocks = Blocks(TILESIZE, 50, self.scene_manager.cur_tmap.grid, tilesides)
        self.Blocks.constructBG()

        self.map_w_px, self.map_h_px = self.Blocks.image.get_size()

        self.start_pos = self._find_start_pos()

        # finish tile
        self.finish_row = int(self.input_finish_row.get_int(0) or 0)
        self.finish_col = int(self.input_finish_col.get_int(9) or 9)
        self.finish_rect = self._finish_tile_rect(self.finish_row, self.finish_col)

        self._reset_run_state(keep_map=True)

        print(f"Benchmark mapa načtena: {map_file} | start_pos={self.start_pos} | finish=({self.finish_row},{self.finish_col})")

    def _find_start_pos(self) -> tuple[float, float]:
        grid = self.scene_manager.cur_tmap.grid
        for r, row in enumerate(grid):
            for c, tile_name in enumerate(row):
                if tile_name == "road_dirt42":
                    x = c * TILESIZE + TILESIZE // 2
                    y = r * TILESIZE + TILESIZE // 2
                    return (x, y)

        # fallback jako v tréninku
        print("VAROVÁNÍ: nenašel jsem start tile road_dirt42, používám fallback souřadnice")
        return (TILESIZE * 4 + TILESIZE // 2, TILESIZE * 8 + TILESIZE // 2)

    def _finish_tile_rect(self, row: int, col: int) -> pygame.Rect:
        grid = self.scene_manager.cur_tmap.grid
        max_r = len(grid) - 1
        max_c = len(grid[0]) - 1

        row = max(0, min(max_r, row))
        col = max(0, min(max_c, col))

        x = col * TILESIZE
        y = row * TILESIZE
        return pygame.Rect(x, y, TILESIZE, TILESIZE)

    @staticmethod
    def _split_entries(text: str) -> list[str]:
        s = (text or "").strip()
        if not s:
            return []
        # povolíme oddělovače ; a |
        parts = []
        for chunk in s.replace("|", ";").split(";"):
            t = chunk.strip()
            if t:
                parts.append(t)
        return parts

    def _parse_engines(self, text: str) -> list[tuple[str, str]]:
        """
        Vrací list (engine_class, save_file).
        Formáty jedné položky:
          "AIbrain_X:AIbrain_X.npz"
          "AIbrain_X,AIbrain_X.npz"
          "AIbrain_X AIbrain_X.npz"
          "AIbrain_X"  -> save default 'userbrain.npz'
        """
        entries = self._split_entries(text)
        out = []
        for e in entries:
            # normalizace vícenásobných mezer
            e2 = " ".join(e.split())

            engine = None
            save = None

            if ":" in e2:
                a, b = e2.split(":", 1)
                engine, save = a.strip(), b.strip()
            elif "," in e2:
                a, b = e2.split(",", 1)
                engine, save = a.strip(), b.strip()
            else:
                parts = e2.split(" ")
                if len(parts) >= 2:
                    engine, save = parts[0].strip(), parts[1].strip()
                elif len(parts) == 1:
                    engine, save = parts[0].strip(), "userbrain.npz"

            if engine:
                if not save:
                    save = "userbrain.npz"
                out.append((engine, save))

        return out

    @staticmethod
    def _strip_comment(line: str) -> str:
        return line.split("#", 1)[0].strip()

    @staticmethod
    def _clean_yaml_value(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
        return value

    def _resolve_agents_file_path(self, raw_path: str) -> Path:
        raw = (raw_path or "").strip()
        if not raw:
            raw = PATH_BENCHMARK_AGENTS

        path = Path(raw)
        if not path.is_absolute() and path.parent == Path("."):
            path = Path(PATH_BENCHMARK_AGENTS).parent / path
        return path

    def _parse_agents_yaml(self, text: str) -> list[tuple[str, str]]:
        """
        Minimalni YAML parser pro jednoduche benchmark konfiguraky.
        Schvalne nepouziva PyYAML, aby studenti nemuseli nic doinstalovavat.
        """
        rows: list[dict[str, str]] = []
        current: dict[str, str] | None = None

        for raw_line in text.splitlines():
            line = self._strip_comment(raw_line)
            if not line or line == "agents:":
                continue

            if line.startswith("-"):
                if current:
                    rows.append(current)
                current = {}

                rest = line[1:].strip()
                if rest:
                    if ":" in rest:
                        key, value = rest.split(":", 1)
                        key = key.strip()
                        value = self._clean_yaml_value(value)
                        if key in ("agent", "engine", "class", "name"):
                            current["agent"] = value
                        elif key in ("save", "save_file", "file", "npz"):
                            current["save"] = value
                        elif value:
                            current["agent"] = key
                            current["save"] = value
                    else:
                        current["agent"] = rest
                continue

            if current is not None and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = self._clean_yaml_value(value)
                if key in ("agent", "engine", "class", "name"):
                    current["agent"] = value
                elif key in ("save", "save_file", "file", "npz"):
                    current["save"] = value

        if current:
            rows.append(current)

        out: list[tuple[str, str]] = []
        for row in rows:
            agent = (row.get("agent") or "").strip()
            save = (row.get("save") or "userbrain.npz").strip()
            if agent:
                out.append((agent, save))

        return out

    def _load_engines_from_file(self, raw_path: str) -> list[tuple[str, str]]:
        path = self._resolve_agents_file_path(raw_path)
        if not path.is_file():
            print(f"VAROVÁNÍ: benchmark soubor {path.as_posix()} neexistuje, použiji ruční fallback pole")
            return []

        text = path.read_text(encoding="utf-8-sig")
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            engines = self._parse_agents_yaml(text)
        else:
            engines = self._parse_engines(text.replace("\n", ";"))

        if engines:
            print(f"Benchmark načetl {len(engines)} agentů ze souboru {path.as_posix()}")
        else:
            print(f"VAROVÁNÍ: benchmark soubor {path.as_posix()} neobsahuje žádné agenty")

        return engines

    def _get_benchmark_engines(self) -> list[tuple[str, str]]:
        engines = self._load_engines_from_file(self.input_agents_file.get_text("benchmark_agents.yaml") or "")
        if engines:
            return engines
        return self._parse_engines(self.input_engines.get_text("") or "")

    def _load_brain(self, engine_class_name: str, save_file_name: str) -> AgentProtocol | None:
        try:
            BrainClass = self.scene_manager.get_agent_class(engine_class_name)
            if BrainClass is None:
                raise ImportError(f"agent '{engine_class_name}' neni v registry")
        except Exception as e:
            print(f"ERROR: nelze nacist agenta {engine_class_name}: {e}")
            return None

        brain = BrainClass()

        params_path = Path(PATH_SAVES) / save_file_name

        if params_path.is_file():
            try:
                params = np.load(params_path)
                brain.set_parameters(params)
                print(f"Načítám parametry z {params_path}")
            except Exception as e:
                print(f"VAROVÁNÍ: načtení parametrů z {params_path} selhalo: {e}")
        else:
            print(f"VAROVÁNÍ: soubor {params_path} neexistuje, používám náhodný mozek")

        brain._engine_class = engine_class_name
        brain._source_file = save_file_name

        return brain

    def start_benchmark(self) -> None:
        if self.Blocks is None or self.start_pos is None:
            print("ERROR: nejdřív načti mapu")
            return

        # načti time limit a finish rect (pro případ, že uživatel změnil inputy)
        self.time_limit_s = float(self.input_time_limit.get_float(30.0) or 30.0)
        self.finish_row = int(self.input_finish_row.get_int(self.finish_row) or self.finish_row)
        self.finish_col = int(self.input_finish_col.get_int(self.finish_col) or self.finish_col)
        self.finish_rect = self._finish_tile_rect(self.finish_row, self.finish_col)

        engines = self._get_benchmark_engines()
        if not engines:
            print("ERROR: seznam agentů pro benchmark je prázdný")
            return

        self._reset_run_state(keep_map=True)

        x0, y0 = self.start_pos

        # rozprostření aut v ose Y, aby nestála přesně na sobě (bez velkých zásahů)
        # pokud je aut hodně, budou se překrývat, ale ve hře neřešíte car-car kolize, takže to nevadí.
        spacing = 1 # pár pixelů až malé % z tile
        offset0 = -0.5 * (len(engines) - 1) * spacing

        for i, (engine_class, save_file) in enumerate(engines):
            brain = self._load_brain(engine_class, save_file)
            if brain is None:
                continue

            y = y0 + offset0 + i * spacing
            car = AI_car(x0, y, 10, 20, brain)

            car.running = True
            car.has_finished = False
            car.finish_time = None
            car.status = "RUNNING"
            car.benchmark_order = i + 1

            self.cars_group.add(car)
            self.cars.append(car)

        if not self.cars:
            print("ERROR: nepodařilo se vytvořit žádné auto")
            return

        self.benchmark_active = True
        self.benchmark_time = 0.0

        print(f"Benchmark startuje: aut={len(self.cars)} | limit={self.time_limit_s:.2f}s | finish=({self.finish_row},{self.finish_col})")

    def _is_out_of_bounds(self, car: AI_car) -> bool:
        return (
            car.pos.x < 0
            or car.pos.y < 0
            or car.pos.x >= self.map_w_px
            or car.pos.y >= self.map_h_px
        )

    def _check_finish(self, car: AI_car) -> bool:
        if self.finish_rect is None:
            return False
        return self.finish_rect.collidepoint(int(car.pos.x), int(car.pos.y))

    def _finalize(self, *, reason: str) -> None:
        # zastav a označ zbylá auta
        for car in self.cars:
            if getattr(car, "has_finished", False):
                continue
            car.running = False
            car.has_finished = True
            car.finish_time = None
            car.status = reason

        self.benchmark_active = False

        self._build_results()
        self._save_results()

    def _build_results(self) -> None:
        self.results = []
        for car in self.cars:
            brain = car.brain
            brain_name = getattr(brain, "NAME", "<noname>")
            engine_class = getattr(brain, "_engine_class", "<unknown>")
            source_file = getattr(brain, "_source_file", "<unknown>")
            benchmark_order = int(getattr(car, "benchmark_order", 0))

            finish_time = car.finish_time
            status = getattr(car, "status", "UNKNOWN")
            dist_tiles = float(getattr(car, "logs_distance", 0.0))

            self.results.append({
                "benchmark_order": benchmark_order,
                "brain_name": str(brain_name),
                "engine_class": str(engine_class),
                "save_file": str(source_file),
                "status": str(status),
                "finish_time_s": float(finish_time) if finish_time is not None else None,
                "distance_tiles": dist_tiles
            })

        # Závodní pořadí:
        # 1) FINISHED podle času
        # 2) TIMEOUT
        # 3) CRASH
        # 4) OUT
        # distance_tiles zůstává jen jako diagnostická metrika, ne jako rank.
        self.results.sort(key=self._result_sort_key)

        # lines pro vykreslení
        self.results_lines = []
        for idx, r in enumerate(self.results, start=1):
            t = r["finish_time_s"]
            t_str = f"{t:.2f}s" if t is not None else "-"
            self.results_lines.append(
                f"{idx}. #{r['benchmark_order']} {r['brain_name']} | {r['status']} | time={t_str} | dist={r['distance_tiles']:.2f} | {r['engine_class']} | {r['save_file']}"
            )

    @staticmethod
    def _result_sort_key(result: dict) -> tuple[int, float, int]:
        order = result.get("benchmark_order", 999999)
        try:
            order = int(order)
        except (TypeError, ValueError):
            order = 999999

        status = str(result.get("status", "UNKNOWN"))
        finish_time = result.get("finish_time_s")
        if status == "FINISHED" and finish_time is not None:
            return (0, float(finish_time), order)

        status_priority = {
            "TIMEOUT": 1,
            "CRASH": 2,
            "OUT": 3,
        }.get(status, 4)

        return (status_priority, 0.0, order)

    def _resolve_output_path(self) -> Path:
        raw = (self.input_output.get_text("benchmark_results.csv") or "").strip()

        if not raw:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw = f"benchmark_{self.input_map.get_text('map')}_{stamp}.csv"

        p = Path(raw)
        if p.suffix.lower() != ".csv":
            p = p.with_suffix(".csv")

        # pokud uživatel nezadal složku, ulož do students/results/
        if not p.is_absolute() and p.parent == Path("."):
            p = Path(PATH_BENCHMARK_RESULTS) / p

        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _save_results(self) -> None:
        if not self.results:
            return

        out_path = self._resolve_output_path()

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "rank",
                "benchmark_order",
                "map_name",
                "finish_row",
                "finish_col",
                "time_limit_s",
                "brain_name",
                "engine_class",
                "save_file",
                "status",
                "finish_time_s",
                "distance_tiles"
            ])

            map_name = self.input_map.get_text("DefaultRace")
            for rank, r in enumerate(self.results, start=1):
                w.writerow([
                    rank,
                    r["benchmark_order"],
                    map_name,
                    self.finish_row,
                    self.finish_col,
                    self.time_limit_s,
                    r["brain_name"],
                    r["engine_class"],
                    r["save_file"],
                    r["status"],
                    r["finish_time_s"] if r["finish_time_s"] is not None else "",
                    f"{r['distance_tiles']:.6f}",
                ])

        self.last_saved_path = out_path
        print(f"Benchmark výsledky uloženy do: {out_path.as_posix()}")

    def update(self, dt: float, keys) -> None:
        for inp in (
            self.input_map,
            self.input_finish_row, self.input_finish_col,
            self.input_time_limit,
            self.input_output,
            self.input_agents_file,
            self.input_engines
        ):
            inp.update(dt)

        if not self.benchmark_active:
            return

        self.benchmark_time += dt

        # timeout
        if self.benchmark_time >= self.time_limit_s:
            self._finalize(reason="TIMEOUT")
            return

        # update cars
        for car in self.cars:
            if not car.running:
                continue

            car.update(dt, keys, self.Blocks)

            # kolize se zdí
            hit = pygame.sprite.spritecollideany(car, self.Blocks.sprites)
            if hit is not None:
                car.running = False
                car.has_finished = True
                car.finish_time = None
                car.status = "CRASH"
                continue

            # mimo mapu
            if self._is_out_of_bounds(car):
                car.running = False
                car.has_finished = True
                car.finish_time = None
                car.status = "OUT"
                continue

            # dojel
            if self._check_finish(car):
                car.running = False
                car.has_finished = True
                car.finish_time = self.benchmark_time
                car.status = "FINISHED"
                continue

        # pokud jsou všichni hotovi, finalize
        if all(getattr(c, "has_finished", False) for c in self.cars):
            self._build_results()
            self._save_results()
            self.benchmark_active = False

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BLACK)

        # mapa
        if self.scene_manager.cur_tmap is not None:
            self.scene_manager.cur_tmap.draw(screen)

        # bloky (volitelné, ale držím styl vašich scén)
        if self.Blocks is not None:
            self.Blocks.draw(screen)

        # zvýraznění cílové dlaždice
        if self.finish_rect is not None:
            pygame.draw.rect(screen, (0, 200, 0), self.finish_rect, 2)

        # auta
        self.cars_group.draw(screen)
        self._draw_car_numbers(screen)

        # UI
        for inp in (
            self.input_map,
            self.input_finish_row, self.input_finish_col,
            self.input_time_limit,
            self.input_output,
            self.input_agents_file,
            self.input_engines
        ):
            inp.draw(screen)

        for b in self.buttons:
            b.draw(screen)

        # stav
        if self.benchmark_active:
            txt = f"t = {self.benchmark_time:.2f} s / limit = {self.time_limit_s:.2f} s"
        else:
            saved = self.last_saved_path.as_posix() if self.last_saved_path else "-"
            txt = f"Benchmark idle | last CSV: {saved}"

        surf = self.font_small.render(txt, True, WHITE)
        screen.blit(surf, (20, 20))

        if self.benchmark_active:
            self._draw_benchmark_labels(screen, 50)
        else:
            # výsledky (prvních pár řádků)
            y = 50
            for line in self.results_lines[:12]:
                s = self.font_small.render(line, True, WHITE)
                screen.blit(s, (20, y))
                y += self.font_small.get_linesize()

    def _draw_car_numbers(self, screen: pygame.Surface) -> None:
        if not self.cars:
            return

        for car in self.cars:
            order = getattr(car, "benchmark_order", None)
            if order is None:
                continue

            pos = getattr(car, "pos", None)
            if pos is not None:
                x = int(pos.x)
                y = int(pos.y)
            else:
                x, y = car.rect.center

            if self.map_w_px and self.map_h_px:
                if x < 0 or y < 0 or x >= self.map_w_px or y >= self.map_h_px:
                    continue

            text = str(order)
            outline = self.font_car_number.render(text, True, BLACK)
            outline_rect = outline.get_rect(center=(x, y))
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)):
                screen.blit(outline, outline_rect.move(dx, dy))

            label = self.font_car_number.render(text, True, (255, 245, 0))
            rect = label.get_rect(center=(x, y))
            screen.blit(label, rect)

    def _draw_benchmark_labels(self, screen: pygame.Surface, y_start: int) -> None:
        if not self.cars:
            return

        x = 20
        y = y_start
        max_width = max(120, WIDTH - MAP_MENUSIZE - x - 20)
        line_h = self.font_benchmark_label.get_linesize()
        bottom = screen.get_height() - line_h

        cars = sorted(self.cars, key=self._benchmark_order_key)
        for car in cars:
            order = getattr(car, "benchmark_order", None)
            if order is None:
                continue

            text = f"#{order} {self._car_brain_name(car)}"
            text = self._fit_text(text, self.font_benchmark_label, max_width)
            label = self.font_benchmark_label.render(text, True, WHITE)
            screen.blit(label, (x, y))

            y += line_h
            if y > bottom:
                break

    @staticmethod
    def _car_brain_name(car: AI_car) -> str:
        brain = getattr(car, "brain", None)
        if brain is None:
            return "<noname>"
        return str(getattr(brain, "NAME", None) or getattr(brain, "_engine_class", "<noname>"))

    @staticmethod
    def _benchmark_order_key(car: AI_car) -> int:
        order = getattr(car, "benchmark_order", None)
        if order is None:
            return 999999
        try:
            return int(order)
        except (TypeError, ValueError):
            return 999999

    @staticmethod
    def _fit_text(text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text

        suffix = "..."
        available = max(0, max_width - font.size(suffix)[0])
        trimmed = ""
        for ch in text:
            if font.size(trimmed + ch)[0] > available:
                break
            trimmed += ch
        return trimmed.rstrip() + suffix

    def event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.scene_manager.set_menu()

        for b in self.buttons:
            b.handle_event(event)

        for inp in (
            self.input_map,
            self.input_finish_row, self.input_finish_col,
            self.input_time_limit,
            self.input_output,
            self.input_agents_file,
            self.input_engines
        ):
            inp.handle_event(event)

    def is_active(self) -> bool:
        return self.active

    def reset(self) -> None:
        pass
