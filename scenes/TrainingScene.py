import pygame, sys, os

from constants import BLACK, WHITE, TILESIZE, SPEED, MAX_SPEED, tilesides, MAP_BUTTON_FONTSIZE
from constants import BLACK, GREY, WIDTH, HEIGHT, MAP_MENUSIZE, MAP_BUTTON_IDENT, MAP_BUTTON_HEIGHT, MAP_BUTTON_WIDTH
from my_sprites.car import car
from my_sprites.block import Blocks
from UI.TextInput import TextInput
from UI.Button import Button
from core.car_manager import Car_manager
from my_sprites.AI_car import AI_car


class load_tmap_raw:
    def __init__(self, scenemanager, filename):
        self.scenemanager = scenemanager
        self.filename = filename
    def action(self):
        if os.path.exists(self.filename):
            self.scenemanager.load_tmap(self.filename)
        else:
            print(f"ERROR: {self.filename} non existent")


class startbutton:
    def __init__(self, training_scene):
        self.training_scene = training_scene

    def action(self):
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
    def __init__(self, training_scene):
        self.training_scene = training_scene
    def action(self):
        self.training_scene.cars_manager.save(self.training_scene.input_data["save_as"])

class loadbutton:
    def __init__(self, training_scene):
        self.training_scene = training_scene
    def action(self):
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
        self.training_scene.cars_manager.load(self.training_scene.input_data["load_from"])

class stopbutton:
    def __init__(self, training_scene):
        self.training_scene = training_scene

    def action(self):
        self.training_scene.stop()

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


        self.buttons = [
            Button((WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE * 4), (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                   "START", self.font,
                   startbutton(self)
                   ),
            Button((WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE * 5), (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                   "STOP", self.font,
                   stopbutton(self)
                   ),
            Button((WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE * 6), (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                   "SAVE AS", self.font,
                   savebutton(self)
                   ),
            Button((WIDTH - MAP_MENUSIZE + MAP_BUTTON_IDENT, 0 + TILESIZE * 7), (MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT),
                   "LOAD & PLAY", self.font,
                   loadbutton(self)
                   )
        ]

    def restart(self):
        self.cars_manager = Car_manager(**self.input_data)
        self.Blocks = Blocks(TILESIZE, 50, self.scene_manager.cur_tmap.grid, tilesides)
        self.Blocks.constructBG()

    def start(self):
        self.cars_manager.setup(**self.input_data)
        self.cars_manager.add_defaultbrain(self.scene_manager.get_brain())
        self.cars_manager.start()

    def stop(self):
        self.cars_manager.stop()

    def get_map_name(self,mapname):
        self.map_name = mapname
        self.loader = load_tmap_raw(self.scene_manager, self.map_name)
        self.loader.action()

    def draw(self, screen):
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

        for button in self.buttons:
            button.draw(screen)

        if self.cars_manager.running:
            self.cars_manager.draw(screen)

    def update(self, dt, keys):
        #self.cars.update(dt, keys, self.Blocks)

        for text in self.text_inputs:
            text.update(dt)

        self.cars_manager.update(dt, keys, self.Blocks)

    def event(self, event):
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

    def check_mouse_click(self, x, y):
        pass

    def is_active(self):
        return self.active

    def reset(self):
        pass