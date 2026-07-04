from __future__ import annotations

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

        default_agent = self.sceene_manager.get_brain_name() or "AIbrain_linear"
        self.agent_input = TextInput(
            pygame.Rect(
                int(WIDTH / 2) - MENU_MENU_IDENT * 2,
                int(HEIGHT - MENU_MENU_IDENT * 1.75),
                MENU_MENU_IDENT * 4,
                int(MENU_FONT_SIZE * 0.7)
            ),
            self.font_input,
            "agent_name"
        )
        self.agent_input.set_default(default_agent)

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BLACK)

        y = MENU_SCENE_START_Y_POS
        for i, text in enumerate(self.items):
            color = BLUE if i == self.selected else WHITE
            surf = self.font.render(text, True, color)
            rect = surf.get_rect(center=(screen.get_width()//2, y))
            screen.blit(surf, rect)
            y += MENU_MENU_IDENT

        self.agent_input.draw(screen)
        self.textinput.draw(screen)

    def update(self, dt: float, keys) -> None:
        self.agent_input.update(dt)
        self.textinput.update(dt)

    def event(self, event: pygame.event.Event) -> None:
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

        self.agent_input.handle_event(event)
        self.textinput.handle_event(event)

    def check_mouse_click(self, x: int, y: int) -> None:
        y_pos = MENU_SCENE_START_Y_POS
        for i, text in enumerate(self.items):
            surf = self.font.render(text, True, WHITE)
            rect = surf.get_rect(center=(WIDTH/2, y_pos))  # 400 = polovina WIDTH (800)
            if rect.collidepoint(x, y):
                self.activate(i)
            y_pos += MENU_MENU_IDENT

    def activate(self, index: int) -> None:
        chosen = self.items[index]
        if chosen == "Konec":
            self.active = False

        if chosen == "Mapa":
            self.sceene_manager.set_mapa()

        if chosen == "Hraj":
            map_name = self.textinput.get_text("DefaultRace")
            self.sceene_manager.set_playgame(str(self.sceene_manager.resolve_map_path(map_name)))

        if chosen == "Trénuj":
            agent_name = self.agent_input.get_text(self.sceene_manager.get_brain_name())
            if not self.sceene_manager.set_brain(agent_name):
                current_name = self.sceene_manager.get_brain_name()
                if current_name:
                    self.agent_input.set_default(current_name)
                return

            map_name = self.textinput.get_text("DefaultRace")
            self.sceene_manager.set_training(str(self.sceene_manager.resolve_map_path(map_name)))

        if chosen == "Souboj":
            # zde nic nepotřebuješ předávat, mapa se řeší v DuelScene přes vlastní TextInput
            self.sceene_manager.set_duel()

        if chosen == "Benchmark":
            self.sceene_manager.set_benchmark()

    def is_active(self) -> bool:
        return self.active

    def reset(self) -> None:
        pass
