from __future__ import annotations

import pygame, os
from pathlib import Path
from constants import BLACK, WHITE,GREY, WIDTH,HEIGHT, MAP_MENUSIZE, MAP_BUTTON_IDENT, MAP_BUTTON_HEIGHT, MAP_BUTTON_WIDTH
from constants import tile_path_default_setting_csv, MAP_BUTTON_FONTSIZE, PATH_STUDENT_MAPS
from UI.MakeGrid import MakeGrid
from constants import TILESIZE
from core.VectorIterator import VectorIterator
from UI.Button import Button
from UI.TextInput import TextInput

class savemap:
    def __init__(self, scene_manager,filename: str | Path):
        self.scene_manager = scene_manager
        self.mapname = filename
    def action(self) -> None:
        print("zmacklo se SAVE tlačítko")
        Path(self.mapname).parent.mkdir(parents=True, exist_ok=True)
        self.scene_manager.cur_tmap.save_csv(self.mapname)
    def update_text(self,filename: str) -> None:
        self.mapname = str(Path(PATH_STUDENT_MAPS) / f"{filename}.csv")


class load_tmap:
    def __init__(self, scenemanager, filename: str | Path):
        self.scenemanager = scenemanager
        self.filename = filename
    def action(self) -> None:
        if os.path.exists(self.filename):
            self.scenemanager.load_tmap(self.filename)
        else:
            print(f"ERROR: {self.filename} non existent")
    def update_text(self,filename: str) -> None:
        self.filename = str(Path(PATH_STUDENT_MAPS) / f"{filename}.csv")

# jen testovací mapa:
class MapEditor:
    def __init__(self, scene_manager, tilecicle: list[str]):
        self.scene_manager = scene_manager
        self.font = pygame.font.SysFont(None, MAP_BUTTON_FONTSIZE)
        self.font_map_name = pygame.font.SysFont(None, int(MAP_BUTTON_FONTSIZE*0.75))
        self.text  = "JSME V MAPE"
        self.active = True
        self.iter_tiles = VectorIterator(tilecicle)
        self.last_x = -1
        self.last_y = -1


        self.set_first_tile() # pokud zacínáme od znova

        self.buttons = [
            Button((WIDTH-MAP_MENUSIZE+ MAP_BUTTON_IDENT, 0+TILESIZE), (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                   "Save map",self.font,
                   savemap(self.scene_manager, str(Path(PATH_STUDENT_MAPS) / "DefaultTiles_USER.csv")), "text_update"),
            Button((WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE*2), (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                   "Load", self.font,
                   load_tmap(self.scene_manager, str(Path(PATH_STUDENT_MAPS) / "DefaultTiles_USER.csv")),# sem pak konkrenti jmeno
                   "text_update"),
            Button((WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE*3), (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                   "RESET", self.font,
                   load_tmap(self.scene_manager, tile_path_default_setting_csv)
                   )
        ]


        self.map_name = TextInput(
            pygame.Rect(WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 5, MAP_BUTTON_WIDTH, int(MAP_BUTTON_HEIGHT*0.7)),
            self.font_map_name, "your_map_name")

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BLACK)

        self.scene_manager.cur_tmap.draw(screen) # vykrelsí mapu
        MakeGrid(screen, TILESIZE, TILESIZE, 10, 10,GREY, HEIGHT, WIDTH-MAP_MENUSIZE)

        # Vykreslení tlačítek
        for button in self.buttons:
            button.draw(screen)

        self.map_name.draw(screen)

    def set_first_tile(self) -> None:
        self.scene_manager.cur_tmap.set_tile(8, 4, "road_dirt42")

    def update(self, dt: float, keys) -> None:
        self.map_name.update(dt)

        for button in self.buttons:
            if button.ButtonID == "text_update":
                button.on_release.update_text(self.map_name.text)

    def event(self, event: pygame.event.Event) -> None:
        # myš
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            self.check_mouse_click(x, y)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.scene_manager.set_menu()

        # predávám zatím zbytecne kazdej event do všech tlačítek:
        for button in self.buttons:
            button.handle_event(event)

        # text input:
        self.map_name.handle_event(event)

    def change_tile(self, x: int, y: int) -> None:
        if (self.last_x, self.last_y) == (x, y):
            self.scene_manager.cur_tmap.set_tile(x, y, next(self.iter_tiles))
            self.scene_manager.cur_tmap.prerender()
        else:
            self.iter_tiles.reset()
            self.scene_manager.cur_tmap.set_tile(x, y, next(self.iter_tiles))
            self.scene_manager.cur_tmap.prerender()
            self.last_x, self.last_y = x, y


    def check_mouse_click(self, x: int, y: int) -> None:
        if x<(WIDTH-MAP_MENUSIZE):
            self.change_tile(y//TILESIZE, x//TILESIZE)

    def is_active(self) -> bool:
        return self.active

    def reset(self) -> None:
        pass
