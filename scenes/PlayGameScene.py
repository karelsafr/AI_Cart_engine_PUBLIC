from __future__ import annotations

import pygame, sys, os
from pathlib import Path

from constants import BLACK, WHITE, TILESIZE, SPEED, MAX_SPEED, tilesides
from my_sprites.car import car
from my_sprites.block import Blocks

class load_tmap_raw:
    def __init__(self, scenemanager, filename: str | Path):
        self.scenemanager = scenemanager
        self.filename = filename
    def action(self) -> None:
        if os.path.exists(self.filename):
            self.scenemanager.load_tmap(self.filename)
        else:
            print(f"ERROR: {self.filename} non existent")

# testovací mapa:
class PlayGame:
    def __init__(self, scene_manager):
        self.scene_manager = scene_manager
        self.font = pygame.font.SysFont(None, 50)
        self.text  = "JSME V PlayGame"
        self.active = True
        # cars, inicializace atlasu!
        car.set_atlas(self.scene_manager.vehicles_atlas)
        self.cars = pygame.sprite.Group()
        self.Blocks = Blocks(TILESIZE, 50, self.scene_manager.cur_tmap.grid, tilesides)
        self.Blocks.constructBG()

    def restart(self) -> None:
        self.cars = pygame.sprite.Group()
        self.cars.add(car(TILESIZE * 4+int(TILESIZE/2), TILESIZE * 8+int(TILESIZE/2), 10, 20))
        self.Blocks = Blocks(TILESIZE, 50, self.scene_manager.cur_tmap.grid, tilesides)
        self.Blocks.constructBG()

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


        self.cars.draw(screen)
        self.Blocks.draw(screen)# vykreslení bloků!

    def update(self, dt: float, keys) -> None:
        self.cars.update(dt, keys, self.Blocks)

    def event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            self.check_mouse_click(x, y)

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                for car in self.cars:
                    car.kill()
                self.scene_manager.set_menu()

        # nejde o updaty pro akce ve hre, jen jednorázové zálezitosti zde!!

    def check_mouse_click(self, x: int, y: int) -> None:
        pass

    def is_active(self) -> bool:
        return self.active

    def reset(self) -> None:
        pass
