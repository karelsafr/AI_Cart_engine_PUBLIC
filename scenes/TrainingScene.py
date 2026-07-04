from __future__ import annotations

import pygame, sys, os
from pathlib import Path

from constants import BLACK, WHITE, TILESIZE, SPEED, MAX_SPEED, tilesides, MAP_BUTTON_FONTSIZE
from constants import BLACK, GREY, WIDTH, HEIGHT, MAP_MENUSIZE, MAP_BUTTON_IDENT, MAP_BUTTON_HEIGHT, MAP_BUTTON_WIDTH
from my_sprites.car import car
from my_sprites.block import Blocks
from UI.TextInput import TextInput
from UI.Button import Button
from core.car_manager import Car_manager
from my_sprites.AI_car import AI_car


class load_tmap_raw:
    def __init__(self, scenemanager, filename: str | Path):
        self.scenemanager = scenemanager
        self.filename = filename
    def action(self) -> None:
        if os.path.exists(self.filename):
            self.scenemanager.load_tmap(self.filename)
        else:
            print(f"ERROR: {self.filename} non existent")


class startbutton:
    def __init__(self, training_scene: "Training"):
        self.training_scene = training_scene

    def action(self) -> None:
        for input in self.training_scene.text_inputs:
            if input.placeholder == "pocet_aut":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "pocet_generaci":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "max_time":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "cars_to_next":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "save_as":
                self.training_scene.input_data[input.placeholder] = input.get_text("userbrain.npz")
            elif input.placeholder == "load_from":
                self.training_scene.input_data[input.placeholder] = input.get_text("userbrain.npz")


        self.training_scene.start()

class savebutton:
    def __init__(self, training_scene: "Training"):
        self.training_scene = training_scene
    def action(self) -> None:
        for input in self.training_scene.text_inputs:
            if input.placeholder == "save_as":
                self.training_scene.input_data[input.placeholder] = input.get_text("userbrain.npz")
        self.training_scene.cars_manager.save(self.training_scene.input_data["save_as"])

class loadbutton:
    def __init__(self, training_scene: "Training"):
        self.training_scene = training_scene
    def action(self) -> None:
        for input in self.training_scene.text_inputs:
            if input.placeholder == "pocet_aut":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "pocet_generaci":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "max_time":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "cars_to_next":
                self.training_scene.input_data[input.placeholder] = input.get_int(10)
            elif input.placeholder == "save_as":
                self.training_scene.input_data[input.placeholder] = input.get_text("userbrain.npz")
            elif input.placeholder == "load_from":
                self.training_scene.input_data[input.placeholder] = input.get_text("userbrain.npz")
        self.training_scene.cars_manager.setup(**self.training_scene.input_data)
        self.training_scene.cars_manager.add_defaultbrain(self.training_scene.scene_manager.get_brain())
        self.training_scene.cars_manager.set_run_metadata(
            self.training_scene.scene_manager.get_brain_name(),
            getattr(self.training_scene, "map_name", "DefaultRace"),
        )
        self.training_scene.cars_manager.load(self.training_scene.input_data["load_from"])

class pausebutton:
    def __init__(self, training_scene: "Training"):
        self.training_scene = training_scene

    def action(self) -> None:
        self.training_scene.pause_resume()

class nextgenerationbutton:
    def __init__(self, training_scene: "Training"):
        self.training_scene = training_scene

    def action(self) -> None:
        self.training_scene.next_generation()

class resetrunbutton:
    def __init__(self, training_scene: "Training"):
        self.training_scene = training_scene

    def action(self) -> None:
        self.training_scene.reset_run()

class reloadsettingsbutton:
    def __init__(self, training_scene: "Training"):
        self.training_scene = training_scene

    def action(self) -> None:
        self.training_scene.reload_settings()

# testovací mapa:
class Training:
    def __init__(self, scene_manager):
        self.scene_manager = scene_manager
        self.font = pygame.font.SysFont(None, 50)
        self.text  = "JSME V PlayGame"
        self.active = True
        # cars, inicializace atlasu!
        AI_car.set_atlas(self.scene_manager.vehicles_atlas)
        self.Blocks = Blocks(TILESIZE, 50, self.scene_manager.cur_tmap.grid, tilesides)
        self.Blocks.constructBG()

        self.font = pygame.font.SysFont(None, MAP_BUTTON_FONTSIZE)
        self.font_textinput = pygame.font.SysFont(None, int(MAP_BUTTON_FONTSIZE * 0.75))
        self.font_monitor = pygame.font.SysFont(None, max(16, int(MAP_BUTTON_FONTSIZE * 0.42)))
        self.font_button_small = pygame.font.SysFont(None, max(18, int(MAP_BUTTON_FONTSIZE * 0.52)))

        # inputy
        self.input_data = {"pocet_aut": 10, "pocet_generaci": 3, "max_time": 5, "cars_to_next":3, "save_as":"usersave.npz","load_from": "userbrain.npz"}
        self.cars_manager = Car_manager(**self.input_data)

        self.text_inputs = [
            TextInput(
            pygame.Rect(WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 5+TILESIZE*0, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
            self.font_textinput, "pocet_aut"),
            TextInput(
                pygame.Rect(WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 5+TILESIZE*0.5, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT * 0.7)),
                self.font_textinput, "pocet_generaci"),
            TextInput(
                pygame.Rect(WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 5 + TILESIZE * 1, MAP_BUTTON_WIDTH,
                            int(MAP_BUTTON_HEIGHT * 0.7)),
                self.font_textinput, "cars_to_next"),
            TextInput(
                pygame.Rect(WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 5 + TILESIZE * 1.5, MAP_BUTTON_WIDTH,
                            int(MAP_BUTTON_HEIGHT * 0.7)),
                self.font_textinput, "save_as"),
            TextInput(
                pygame.Rect(WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 5 + TILESIZE * 2, MAP_BUTTON_WIDTH,
                            int(MAP_BUTTON_HEIGHT * 0.7)),
                self.font_textinput, "max_time"),
            TextInput(
                pygame.Rect(WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 5 + TILESIZE * 8, MAP_BUTTON_WIDTH,
                            int(MAP_BUTTON_HEIGHT * 0.7)),
                self.font_textinput, "load_from"),
        ]


        bx = WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT
        by = int(5 + TILESIZE * 4.5)
        gap = 6
        button_h = int(MAP_BUTTON_HEIGHT * 0.72)
        half_w = int((MAP_BUTTON_WIDTH - gap) / 2)
        row_step = button_h + gap

        self.buttons = [
            Button((bx, by), (half_w, button_h), "START", self.font_button_small, startbutton(self)),
            Button((bx + half_w + gap, by), (half_w, button_h), "PAUSE", self.font_button_small, pausebutton(self), "pause_toggle"),
            Button((bx, by + row_step), (half_w, button_h), "NEXT GEN", self.font_button_small, nextgenerationbutton(self)),
            Button((bx + half_w + gap, by + row_step), (half_w, button_h), "RESET", self.font_button_small, resetrunbutton(self)),
            Button((bx, by + row_step * 2), (half_w, button_h), "SAVE", self.font_button_small, savebutton(self)),
            Button((bx + half_w + gap, by + row_step * 2), (half_w, button_h), "LOAD", self.font_button_small, loadbutton(self)),
            Button((bx, by + row_step * 3), (MAP_BUTTON_WIDTH, button_h), "RELOAD SET", self.font_button_small, reloadsettingsbutton(self)),
        ]

    def restart(self) -> None:
        self.cars_manager = Car_manager(**self.input_data)
        self.Blocks = Blocks(TILESIZE, 50, self.scene_manager.cur_tmap.grid, tilesides)
        self.Blocks.constructBG()

    def start(self) -> None:
        self.cars_manager.setup(**self.input_data)
        self.cars_manager.add_defaultbrain(self.scene_manager.get_brain())
        self.cars_manager.set_run_metadata(
            self.scene_manager.get_brain_name(),
            getattr(self, "map_name", "DefaultRace"),
        )
        self.cars_manager.start()

    def stop(self) -> None:
        self.cars_manager.stop()

    def pause_resume(self) -> None:
        self.cars_manager.toggle_pause()

    def next_generation(self) -> None:
        self.cars_manager.next_generation()

    def reset_run(self) -> None:
        self.cars_manager.reset_run()

    def reload_settings(self) -> None:
        for input in self.text_inputs:
            if input.placeholder == "pocet_aut":
                self.input_data[input.placeholder] = input.get_int(self.input_data["pocet_aut"])
            elif input.placeholder == "max_time":
                self.input_data[input.placeholder] = input.get_int(self.input_data["max_time"])
            elif input.placeholder == "cars_to_next":
                self.input_data[input.placeholder] = input.get_int(self.input_data["cars_to_next"])
            elif input.placeholder == "save_as":
                self.input_data[input.placeholder] = input.get_text(self.input_data["save_as"])
            elif input.placeholder == "load_from":
                self.input_data[input.placeholder] = input.get_text(self.input_data["load_from"])

        self.cars_manager.reload_settings_and_next_generation(
            pocet_aut=self.input_data["pocet_aut"],
            max_time=self.input_data["max_time"],
            cars_to_next=self.input_data["cars_to_next"],
            save_as=self.input_data["save_as"],
            load_from=self.input_data["load_from"],
        )

    def get_map_name(self,mapname: str | Path) -> None:
        self.map_name = mapname
        self.loader = load_tmap_raw(self.scene_manager, self.map_name)
        self.loader.action()

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BLACK)

        self.scene_manager.cur_tmap.draw(screen)

        y = 150
        surf = self.font.render(self.map_name, True, WHITE)
        rect = surf.get_rect(center=(screen.get_width() // 2, y))
        screen.blit(surf, rect)


        #self.cars.draw(screen)
        self.Blocks.draw(screen)# vykreslení bloků!

        # vykrlesení prvků:
        for text in self.text_inputs:
            text.draw(screen)

        self.update_control_labels()
        for button in self.buttons:
            button.draw(screen)

        if self.cars_manager.running:
            self.cars_manager.draw(screen)

        self.draw_training_monitor(screen)

    def update(self, dt: float, keys) -> None:
        #self.cars.update(dt, keys, self.Blocks)

        for text in self.text_inputs:
            text.update(dt)

        self.cars_manager.update(dt, keys, self.Blocks)

    def _fit_monitor_text(self, text: str, max_width: int) -> str:
        if self.font_monitor.size(text)[0] <= max_width:
            return text

        suffix = "..."
        while text and self.font_monitor.size(text + suffix)[0] > max_width:
            text = text[:-1]
        return text + suffix if text else suffix

    def draw_training_monitor(self, screen: pygame.Surface) -> None:
        stats = self.cars_manager.get_training_stats()

        x = WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT
        y = int(5 + TILESIZE * 2.45)
        width = MAP_BUTTON_WIDTH
        line_h = self.font_monitor.get_linesize()

        total_generations = max(1, int(stats["generations_total"]))
        current_generation = int(stats["generation"]) + 1 if stats["training_active"] else int(stats["generation"])
        current_generation = max(0, min(current_generation, total_generations))
        state = "PAUSED" if stats["paused"] else "RUN" if stats["running"] else "IDLE"

        agent_name = self.scene_manager.get_brain_name() or "<agent>"
        lines = [
            ("MONITOR", (120, 180, 255)),
            (f"State: {state}", WHITE),
            (f"Agent: {agent_name}", WHITE),
            (f"Gen: {current_generation}/{total_generations} T:{stats['time']:.1f}/{stats['max_time']:.0f}", WHITE),
            (f"Alive: {stats['cars_active']}/{stats['cars_total']} Crash:{stats['cars_crashed']}", WHITE),
            (f"Best: {stats['best_score']:.2f} All:{stats['best_score_all']:.2f}", WHITE),
            (f"Avg:{stats['avg_score']:.2f} Med:{stats['median_score']:.2f}", WHITE),
            (f"Stag:{stats['stagnation_epochs']} Save:{stats['save_as']}", WHITE),
        ]
        panel_h = line_h * len(lines) + 8

        pygame.draw.rect(screen, (18, 18, 18), (x - 2, y - 4, width + 4, panel_h))
        pygame.draw.rect(screen, (80, 80, 80), (x - 2, y - 4, width + 4, panel_h), 1)

        for i, (text, color) in enumerate(lines):
            shown = self._fit_monitor_text(text, width - 6)
            surf = self.font_monitor.render(shown, True, color)
            screen.blit(surf, (x + 3, y + i * line_h))

    def update_control_labels(self) -> None:
        for button in self.buttons:
            if button.ButtonID == "pause_toggle":
                button.set_text("RESUME" if self.cars_manager.paused else "PAUSE")

    def event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            self.check_mouse_click(x, y)
        if event.type == pygame.KEYDOWN:
            if event.key  == pygame.K_ESCAPE:
                self.scene_manager.set_menu()

        # predávám zatím zbytecne kazdej event do všech tlačítek:
        for button in self.buttons:
            button.handle_event(event)

        # text input:
        for text in self.text_inputs:
            text.handle_event(event)

        # nejde o updaty pro akce ve hre, jen jednorázové zálezitosti zde!!

    def check_mouse_click(self, x: int, y: int) -> None:
        pass

    def is_active(self) -> bool:
        return self.active

    def reset(self) -> None:
        pass
