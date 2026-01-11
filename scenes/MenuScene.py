import pygame, sys
from constants import BLACK, WHITE, BLUE, WIDTH, MENU_SCENE_START_Y_POS, MENU_FONT_SIZE, MENU_MENU_IDENT, HEIGHT
from UI.TextInput import TextInput


class MenuScene:
    def __init__(self, sceene_manager):
        self.name = "MENU"
        self.sceene_manager = sceene_manager
        self.font = pygame.font.SysFont(None, MENU_FONT_SIZE)
        self.font_input = pygame.font.SysFont(None, int(MENU_FONT_SIZE*0.5))

        # položky menu
        self.items = ["Hraj", "Mapa", "Trénuj", "Souboj", "Benchmark", "Konec"]
        self.selected = 0   # vybrana polozka

        self.active = True

        self.textinput = TextInput(pygame.Rect(int(WIDTH/2)-MENU_MENU_IDENT*2, HEIGHT-MENU_MENU_IDENT, MENU_MENU_IDENT*4, int(MENU_FONT_SIZE*0.7)),
                                   self.font_input, "DefaultRace")
        self.textinput.set_default("DefaultRace")

    def draw(self, screen):
        screen.fill(BLACK)

        y = MENU_SCENE_START_Y_POS
        for i, text in enumerate(self.items):
            color = BLUE if i == self.selected else WHITE
            surf = self.font.render(text, True, color)
            rect = surf.get_rect(center=(screen.get_width()//2, y))
            screen.blit(surf, rect)
            y += MENU_MENU_IDENT

        self.textinput.draw(screen)

    def update(self, dt, keys):
        self.textinput.update(dt)

    def event(self, event):
        # myš
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            self.check_mouse_click(x, y)

        # klávesy
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN,):
                self.selected = (self.selected + 1) % len(self.items)
            elif event.key in (pygame.K_UP,):
                self.selected = (self.selected - 1) % len(self.items)
            elif event.key in (pygame.K_RETURN,):
                self.activate(self.selected)
            elif event.key in (pygame.K_ESCAPE,):
                self.active = False

        self.textinput.handle_event(event)

    def check_mouse_click(self, x, y):
        y_pos = MENU_SCENE_START_Y_POS
        for i, text in enumerate(self.items):
            surf = self.font.render(text, True, WHITE)
            rect = surf.get_rect(center=(WIDTH/2, y_pos))  # 400 = polovina WIDTH (800)
            if rect.collidepoint(x, y):
                self.activate(i)
            y_pos += MENU_MENU_IDENT

    def activate(self, index):
        chosen = self.items[index]
        if chosen == "Konec":
            self.active = False

        if chosen == "Mapa":
            self.sceene_manager.set_mapa()

        if chosen == "Hraj":
            if self.textinput.text in ("DefaultRace", "DefaultReset"):
                prefix = "DefaultSettings/"
            elif self.textinput.text == "": #prazdana bunka pri mazání!
                prefix = "DefaultSettings/"
                self.textinput.text = "DefaultRace"
            else:
                prefix = "UserData/"

            self.sceene_manager.set_playgame(prefix + self.textinput.text+".csv")

        if chosen == "Trénuj":
            if self.textinput.text in ("DefaultRace", "DefaultReset"):
                prefix = "DefaultSettings/"
            elif self.textinput.text == "": #prazdana bunka pri mazání!
                prefix = "DefaultSettings/"
                self.textinput.text = "DefaultRace"
            else:
                prefix = "UserData/"

            self.sceene_manager.set_training(prefix + self.textinput.text+".csv")

        if chosen == "Souboj":
            # zde nic nepotřebuješ předávat, mapa se řeší v DuelScene přes vlastní TextInput
            self.sceene_manager.set_duel()

        if chosen == "Benchmark":
            self.sceene_manager.set_benchmark()

    def is_active(self):
        return self.active

    def reset(self):
        pass
