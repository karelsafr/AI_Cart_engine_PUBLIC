from __future__ import annotations

import pygame, os
import numpy as np
from pathlib import Path

from constants import (
    BLACK, WHITE,
    WIDTH, HEIGHT,
    TILESIZE,
    MAP_MENUSIZE, MAP_BUTTON_IDENT, MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT, MAP_BUTTON_FONTSIZE,
    tilesides, PATH_SAVES
)
from my_sprites.AI_car import AI_car
from my_sprites.block import Blocks
from UI.TextInput import TextInput
from UI.Button import Button
from core.protocols import AgentProtocol


class _LoadMapButton:
    def __init__(self, duel_scene: "DuelScene"):
        self.duel_scene = duel_scene

    def action(self) -> None:
        self.duel_scene.load_map()


class _StartRaceButton:
    def __init__(self, duel_scene: "DuelScene"):
        self.duel_scene = duel_scene

    def action(self) -> None:
        self.duel_scene.start_race()


class DuelScene:
    def __init__(self, scene_manager):
        self.scene_manager = scene_manager
        self.font = pygame.font.SysFont(None, MAP_BUTTON_FONTSIZE)
        self.font_small = pygame.font.SysFont(None, int(MAP_BUTTON_FONTSIZE * 0.7))

        self.active = True

        # -------- Text inputy vpravo --------
        # mapa
        self.input_map = TextInput(
            pygame.Rect(
                WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT,
                5 + TILESIZE * 0,
                MAP_BUTTON_WIDTH,
                int(MAP_BUTTON_HEIGHT * 0.7)
            ),
            self.font_small,
            "map_name"
        )
        self.input_map.set_default("DefaultRace")

        # engine 1 - třída agenta, třída se jmenuje stejně jako soubor
        self.input_engine1 = TextInput(
            pygame.Rect(
                WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT,
                5 + TILESIZE * 0.5,
                MAP_BUTTON_WIDTH,
                int(MAP_BUTTON_HEIGHT * 0.7)
            ),
            self.font_small,
            "engine1_class"
        )
        self.input_engine1.set_default("AIbrain_TeamName")

        # engine 1 - uložený mozek
        self.input_engine1_save = TextInput(
            pygame.Rect(
                WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT,
                5 + TILESIZE * 1.0,
                MAP_BUTTON_WIDTH,
                int(MAP_BUTTON_HEIGHT * 0.7)
            ),
            self.font_small,
            "engine1_save"
        )
        self.input_engine1_save.set_default("userbrain.npz")

        # engine 2 - třída
        self.input_engine2 = TextInput(
            pygame.Rect(
                WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT,
                5 + TILESIZE * 1.5,
                MAP_BUTTON_WIDTH,
                int(MAP_BUTTON_HEIGHT * 0.7)
            ),
            self.font_small,
            "engine2_class"
        )
        self.input_engine2.set_default("AIbrain_TeamName")

        # engine 2 - uložený mozek
        self.input_engine2_save = TextInput(
            pygame.Rect(
                WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT,
                5 + TILESIZE * 2.0,
                MAP_BUTTON_WIDTH,
                int(MAP_BUTTON_HEIGHT * 0.7)
            ),
            self.font_small,
            "engine2_save"
        )
        self.input_engine2_save.set_default("userbrain.npz")

        # -------- Tlačítka --------
        self.buttons = [
            Button(
                (WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE * 3),
                (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                "LOAD MAP",
                self.font,
                _LoadMapButton(self)
            ),
            Button(
                (WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE * 4),
                (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                "START DUEL",
                self.font,
                _StartRaceButton(self)
            )
        ]

        # -------- Stav závodu --------
        self.Blocks: Blocks | None = None
        self.cars_group = pygame.sprite.Group()
        self.cars: list[AI_car] = []

        self.race_active = False
        self.race_time = 0.0
        self.finished_count = 0
        self.results_lines: list[str] = []

        # čáry start a cíl
        self.kill_x: float | None = None
        self.finish_x: float | None = None
        self.start_pos: tuple[float, float] | None = None

        self.gate_y_top: float | None = None
        self.gate_y_bottom: float | None = None

        # Načti defaultní mapu
        self.load_map()

    def restart(self) -> None:
        """Resetuje duel, ale nechává načtenou mapu."""
        self.active = True
        self.cars_group.empty()
        self.cars.clear()
        self.race_active = False
        self.race_time = 0.0
        self.finished_count = 0
        self.results_lines.clear()

        if self.Blocks is None:
            self.load_map()

    def load_map(self) -> None:
        """Načte mapu podle textinputu a připraví bloky a kontrolní čáry."""
        name = self.input_map.get_text("DefaultRace")
        map_file = self.scene_manager.resolve_map_path(name)

        if not os.path.exists(map_file):
            print(f"ERROR: mapa {map_file} neexistuje")
            return

        self.scene_manager.load_tmap(map_file)
        AI_car.set_atlas(self.scene_manager.vehicles_atlas)

        self.Blocks = Blocks(TILESIZE, 50, self.scene_manager.cur_tmap.grid, tilesides)
        self.Blocks.constructBG()

        self._init_start_and_gates()

        print(f"Mapa pro duel načtena: {map_file}")

    def _init_start_and_gates(self) -> None:
        """Najde start tile road_dirt42 a určí souřadnice startu a bran."""
        grid = self.scene_manager.cur_tmap.grid
        start_x = None
        start_y = None
        start_row = None
        start_col = None

        for i, row in enumerate(grid):
            for j, tile_name in enumerate(row):
                if tile_name == "road_dirt42":
                    start_x = j * TILESIZE + TILESIZE // 2
                    start_y = i * TILESIZE + TILESIZE // 2
                    start_row = i
                    start_col = j
                    break
            if start_x is not None:
                break

        if start_x is None:
            # fallback - stejné jako ve tréninku
            start_x = TILESIZE * 4 + TILESIZE // 2
            start_y = TILESIZE * 8 + TILESIZE // 2
            print("VAROVÁNÍ: nenašel jsem start tile road_dirt42, používám fallback souřadnice")
            # vezmeme řádek podle start_y
            start_row = int(start_y // TILESIZE)

        self.start_pos = (start_x, start_y)

        # vertikální rozsah bran - jen v řádku start tile
        self.gate_y_top = start_row * TILESIZE
        self.gate_y_bottom = self.gate_y_top + TILESIZE

        # AUTO MÁ ÚHEL 180°, TJ. JDE DOLEVA
        # -> dopředu = menší x (vlevo)
        # -> dozadu = větší x (vpravo)

        offset = int(TILESIZE * 0.3)

        # Cíl (zelená) před autem - vlevo od startu
        self.finish_x = start_x + offset+30
        # Kill brána za autem - vpravo od startu
        self.kill_x = start_x + offset

    def _load_brain(self, engine_class_name: str, save_file_name: str) -> AgentProtocol | None:
        """Dynamicky načte třídu brainu a případně nahraje parametry z .npz."""
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

        # tady si k mozku uložíme název souboru, odkud se vzal
        brain._source_file = save_file_name

        return brain

    def start_race(self) -> None:
        """Vytvoří dvě auta, načte mozky a spustí duel."""
        if self.Blocks is None or self.start_pos is None:
            print("ERROR: nejdřív načti mapu (LOAD MAP)")
            return

        engine1_name = self.input_engine1.get_text("AIbrain_TeamName")
        engine2_name = self.input_engine2.get_text("AIbrain_TeamName")
        save1_name = self.input_engine1_save.get_text("userbrain.npz")
        save2_name = self.input_engine2_save.get_text("userbrain.npz")

        brain1 = self._load_brain(engine1_name, save1_name)
        brain2 = self._load_brain(engine2_name, save2_name)

        if brain1 is None or brain2 is None:
            print("ERROR: alespoň jeden mozek se nepodařilo vytvořit")
            return

        self.cars_group.empty()
        self.cars.clear()

        x, y = self.start_pos

        # dvě auta vedle sebe na startu
        car1 = AI_car(x, y - TILESIZE * 0.1, 10, 20, brain1)
        car2 = AI_car(x, y + TILESIZE * 0.1, 10, 20, brain2)

        for car in (car1, car2):
            car.running = True
            car.has_finished = False
            car.finish_time = None
            car.order = None
            car.no = 0  # sem pak zapíšeme pořadí

            self.cars_group.add(car)
            self.cars.append(car)

        self.race_time = 0.0
        self.finished_count = 0
        self.results_lines.clear()
        self.race_active = True

        print("Duel startuje")

    def _register_finish(self, car: AI_car) -> None:
        """Nastaví pořadí, čas a uloží textový výsledek."""
        if getattr(car, "has_finished", False):
            return

        car.has_finished = True
        self.finished_count += 1
        car.order = self.finished_count

        # 1. místo -> 2, 2. místo -> 1 (podle vaší logiky s car.no)
        car.no = 3 - self.finished_count

        # jméno souboru, odkud pochází mozek
        source_file = getattr(car.brain, "_source_file", "<random>")

        # čas – pokud má car.finish_time, použijeme ho, jinak aktuální čas závodu
        event_time = car.finish_time if car.finish_time is not None else self.race_time

        # stav: dojel / nedojel
        status = "DOJEL" if car.finish_time is not None else "NEDOJEL"

        line = (
            f"Brain: {car.brain.NAME} | "
            f"soubor: {source_file} | "
            f"čas: {event_time:.2f} s | "
            f"stav: {status} | "
            f"pořadí: {car.order}"
        )

        self.results_lines.append(line)

        if self.finished_count == len(self.cars):
            self.race_active = False
            self._print_results_to_console()

    def _print_results_to_console(self) -> None:
        print("===== VÝSLEDKY DUELU =====")
        for line in self.results_lines:
            print(line)
        print("==========================")


    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BLACK)

        # mapa
        if self.scene_manager.cur_tmap is not None:
            self.scene_manager.cur_tmap.draw(screen)

        # bloky
        if self.Blocks is not None:
            self.Blocks.draw(screen)

        # start a cíl jako dvě krátké svislé čáry jen v jednom tile řádku
        if (
            self.kill_x is not None
            and self.finish_x is not None
            and self.gate_y_top is not None
            and self.gate_y_bottom is not None
        ):
            pygame.draw.line(
                screen,
                (255, 0, 0),
                (self.kill_x, self.gate_y_top),
                (self.kill_x, self.gate_y_bottom),
                2,
            )
            pygame.draw.line(
                screen,
                (0, 255, 0),
                (self.finish_x, self.gate_y_top),
                (self.finish_x, self.gate_y_bottom),
                2,
            )

        # auta
        self.cars_group.draw(screen)

        # UI
        for inp in (
            self.input_map,
            self.input_engine1, self.input_engine1_save,
            self.input_engine2, self.input_engine2_save
        ):
            inp.draw(screen)

        for button in self.buttons:
            button.draw(screen)

        # čas
        if self.race_active:
            txt = f"t = {self.race_time:.2f} s"
        else:
            txt = "Zmáčkni START DUEL pro nový souboj"

        surf = self.font.render(txt, True, WHITE)
        screen.blit(surf, (20, 20))

        # výsledky
        y = 60
        for line in self.results_lines:
            s = self.font_small.render(line, True, WHITE)
            screen.blit(s, (20, y))
            y += self.font_small.get_linesize()

    def update(self, dt: float, keys) -> None:
        # text input
        for inp in (
            self.input_map,
            self.input_engine1, self.input_engine1_save,
            self.input_engine2, self.input_engine2_save
        ):
            inp.update(dt)

        if not self.race_active:
            return

        self.race_time += dt

        for car in self.cars:
            if not car.running:
                continue

            prev_x = car.pos.x
            prev_y = car.pos.y

            car.update(dt, keys, self.Blocks)

            # náraz do zdi
            hit = pygame.sprite.spritecollideany(car, self.Blocks.sprites)
            if hit is not None:
                car.running = False
                car.finish_time = None
                self._register_finish(car)
                continue

            # brány platí jen v daném vertikálním rozsahu (9. řádek = řádek startu)
            in_gate_row = (
                    self.gate_y_top is not None
                    and self.gate_y_bottom is not None
                    and self.gate_y_top <= car.pos.y <= self.gate_y_bottom
            )

            if in_gate_row:
                # KILL BRÁNA – za autem (vpravo od startu)
                # pokud auto jede dozadu (x roste) a překročí čáru zleva doprava
                if (
                        self.kill_x is not None
                        and prev_x < self.kill_x <= car.pos.x
                ):
                    car.running = False
                    car.finish_time = None
                    self._register_finish(car)
                    continue

                # CÍLOVÁ BRÁNA – před autem (vlevo od startu)
                # pokud auto jede dopředu (x klesá) a překročí čáru zprava doleva
                if (
                        self.finish_x is not None
                        and prev_x > self.finish_x >= car.pos.x
                ):
                    car.running = False
                    car.finish_time = self.race_time
                    self._register_finish(car)
                    continue

        # pokud se závod ještě tváří aktivní, ale všechna auta stojí, ukonči
        if self.race_active and all((not c.running) for c in self.cars):
            self.race_active = False

    def event(self, event: pygame.event.Event) -> None:
        # ESC - zpět do menu
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.scene_manager.set_menu()

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            self.check_mouse_click(x, y)

        for button in self.buttons:
            button.handle_event(event)

        for inp in (
            self.input_map,
            self.input_engine1, self.input_engine1_save,
            self.input_engine2, self.input_engine2_save
        ):
            inp.handle_event(event)

    def check_mouse_click(self, x: int, y: int) -> None:
        # zatím nic speciálního
        pass

    def is_active(self) -> bool:
        return self.active

    def reset(self) -> None:
        pass
