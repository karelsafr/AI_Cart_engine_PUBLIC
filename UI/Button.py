from __future__ import annotations

import pygame
from core.protocols import ButtonAction

class Button:
    def __init__(
        self,
        posxy: tuple[int, int] | tuple[float, float],
        wh: tuple[int, int] | tuple[float, float],
        text: str,
        font: pygame.font.Font,
        on_release: ButtonAction | None,
        ButtonID: str = "",
    ):
        # Rect: (x, y, w, h)
        self.rect = pygame.Rect(posxy, wh)
        self.text = text
        self.font = font
        self.on_release = on_release  # callback volaný při MOUSEBUTTONUP uvnitř
        self.hover = False
        self.pressed_inside = False
        self.ButtonID = ButtonID

        # předpočítat text surface
        self._txt_surf = self.font.render(self.text, True, (20, 20, 20))
        self._txt_rect = self._txt_surf.get_rect(center=self.rect.center)

    def set_text(self, text: str) -> None:
        if self.text == text:
            return
        self.text = text
        self._txt_surf = self.font.render(self.text, True, (20, 20, 20))
        self._txt_rect = self._txt_surf.get_rect(center=self.rect.center)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
            # udrž text vycentrovaný i když by se rect hýbal
            self._txt_rect = self._txt_surf.get_rect(center=self.rect.center)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed_inside = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed_inside and self.rect.collidepoint(event.pos):
                # click = DOWN uvnitř + UP uvnitř
                if self.on_release:
                    self.on_release.action()
            self.pressed_inside = False

    def draw(self, screen: pygame.Surface) -> None:
        color = (220, 220, 220) if self.hover else (200, 200, 200)
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (60, 60, 60), self.rect, 2, border_radius=8)
        screen.blit(self._txt_surf, self._txt_rect)
