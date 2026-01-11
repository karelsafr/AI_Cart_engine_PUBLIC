import os
import csv
import importlib
from pathlib import Path
from datetime import datetime

import pygame
import numpy as np

from constants import (
    BLACK, WHITE,
    WIDTH, TILESIZE,
    MAP_MENUSIZE, MAP_BUTTON_IDENT, MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT, MAP_BUTTON_FONTSIZE,
    tilesides, PATH_SAVES
)
from my_sprites.AI_car import AI_car
from my_sprites.block import Blocks
from UI.TextInput import TextInput
from UI.Button import Button


class _LoadMapButton:
    def __init__(self, scene):
        self.scene = scene

    def action(self):
        self.scene.load_map()


class _StartBenchmarkButton:
    def __init__(self, scene):
        self.scene = scene

    def action(self):
        self.scene.start_benchmark()


class BenchmarkScene:
    """
    Benchmark více mozků na jedné mapě.

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

        # engines list
        self.input_engines = TextInput(
            pygame.Rect(x0, 5 + TILESIZE * 2.5, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_small,
            "engines"
        )
        self.input_engines.set_default("ONEX2:ONEX2.npz; AIbrain_Zero:zero.npz;AIbrain_QERQ:QERQ.npz;ONEX:ONEX.npz;AIbrain_FAST:FAST.npz;AIbrain_ASDF:asdf.npz;SNOW:SNOW.npz;AIbrain_vers:vers.npz;AIbrain_LGBT:LGBT.npz;AIbrain_maie:maie.npz")

        # buttons
        self.buttons = [
            Button(
                (x0, 0 + TILESIZE * 3.2),
                (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                "LOAD MAP",
                self.font,
                _LoadMapButton(self)
            ),
            Button(
                (x0, 0 + TILESIZE * 4.2),
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

    def restart(self):
        self.active = True
        self._reset_run_state(keep_map=True)

    def _reset_run_state(self, *, keep_map: bool):
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

    def load_map(self):
        name = self.input_map.get_text("DefaultRace")

        if name in ("DefaultRace", "DefaultReset"):
            prefix = "DefaultSettings/"
        else:
            prefix = "UserData/"

        map_file = prefix + name + ".csv"
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

    def _load_brain(self, engine_class_name: str, save_file_name: str):
        try:
            module = importlib.import_module(f"AI_engines.{engine_class_name}")
            BrainClass = getattr(module, engine_class_name)
        except Exception as e:
            print(f"ERROR: nelze importovat AI_engines.{engine_class_name}: {e}")
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

    def start_benchmark(self):
        if self.Blocks is None or self.start_pos is None:
            print("ERROR: nejdřív načti mapu")
            return

        # načti time limit a finish rect (pro případ, že uživatel změnil inputy)
        self.time_limit_s = float(self.input_time_limit.get_float(30.0) or 30.0)
        self.finish_row = int(self.input_finish_row.get_int(self.finish_row) or self.finish_row)
        self.finish_col = int(self.input_finish_col.get_int(self.finish_col) or self.finish_col)
        self.finish_rect = self._finish_tile_rect(self.finish_row, self.finish_col)

        engines = self._parse_engines(self.input_engines.get_text("") or "")
        if not engines:
            print("ERROR: seznam enginů je prázdný")
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

    def _finalize(self, *, reason: str):
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

    def _build_results(self):
        self.results = []
        for car in self.cars:
            brain = car.brain
            brain_name = getattr(brain, "NAME", "<noname>")
            engine_class = getattr(brain, "_engine_class", "<unknown>")
            source_file = getattr(brain, "_source_file", "<unknown>")

            finish_time = car.finish_time
            status = getattr(car, "status", "UNKNOWN")
            dist_tiles = float(getattr(car, "logs_distance", 0.0))

            self.results.append({
                "brain_name": str(brain_name),
                "engine_class": str(engine_class),
                "save_file": str(source_file),
                "status": str(status),
                "finish_time_s": float(finish_time) if finish_time is not None else None,
                "distance_tiles": dist_tiles
            })

        # seřazení:
        # 1) FINISHED podle času
        # 2) ostatní až za nimi, pro přehled je řadím podle distance_tiles sestupně
        def key(r):
            if r["status"] == "FINISHED" and r["finish_time_s"] is not None:
                return (0, r["finish_time_s"])
            return (1, -r["distance_tiles"])

        self.results.sort(key=key)

        # lines pro vykreslení
        self.results_lines = []
        for idx, r in enumerate(self.results, start=1):
            t = r["finish_time_s"]
            t_str = f"{t:.2f}s" if t is not None else "-"
            self.results_lines.append(
                f"{idx}. {r['brain_name']} | {r['status']} | time={t_str} | dist={r['distance_tiles']:.2f} | {r['engine_class']} | {r['save_file']}"
            )

    def _resolve_output_path(self) -> Path:
        raw = (self.input_output.get_text("benchmark_results.csv") or "").strip()

        if not raw:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw = f"benchmark_{self.input_map.get_text('map')}_{stamp}.csv"

        p = Path(raw)
        if p.suffix.lower() != ".csv":
            p = p.with_suffix(".csv")

        # pokud uživatel nezadal složku, ulož do UserData/
        if not p.is_absolute() and p.parent == Path("."):
            p = Path("UserData") / p

        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _save_results(self):
        if not self.results:
            return

        out_path = self._resolve_output_path()

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "rank",
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

    def update(self, dt, keys):
        for inp in (
            self.input_map,
            self.input_finish_row, self.input_finish_col,
            self.input_time_limit,
            self.input_output,
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

    def draw(self, screen):
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

        # UI
        for inp in (
            self.input_map,
            self.input_finish_row, self.input_finish_col,
            self.input_time_limit,
            self.input_output,
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

        # výsledky (prvních pár řádků)
        y = 50
        for line in self.results_lines[:12]:
            s = self.font_small.render(line, True, WHITE)
            screen.blit(s, (20, y))
            y += self.font_small.get_linesize()

    def event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.scene_manager.set_menu()

        for b in self.buttons:
            b.handle_event(event)

        for inp in (
            self.input_map,
            self.input_finish_row, self.input_finish_col,
            self.input_time_limit,
            self.input_output,
            self.input_engines
        ):
            inp.handle_event(event)

    def is_active(self):
        return self.active

    def reset(self):
        pass
