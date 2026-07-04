from __future__ import annotations

import pygame

class TextInput:
    def __init__(self, rect: pygame.Rect, font: pygame.font.Font, placeholder: str = ""):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.text = ""
        self.placeholder = placeholder
        self.focused = False
        self.caret_pos = 0  # index v textu
        self.caret_visible = True
        self.caret_timer = 0.0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
            # jednoduché: skoč kurzorem na konec
            if self.focused:
                self.caret_pos = len(self.text)

        if not self.focused:
            return

        if event.type == pygame.TEXTINPUT:
            # vlož znak(y) na caret
            self.text = self.text[:self.caret_pos] + event.text + self.text[self.caret_pos:]
            self.caret_pos += len(event.text)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE and self.caret_pos > 0:
                self.text = self.text[:self.caret_pos-1] + self.text[self.caret_pos:]
                self.caret_pos -= 1
            elif event.key == pygame.K_DELETE and self.caret_pos < len(self.text):
                self.text = self.text[:self.caret_pos] + self.text[self.caret_pos+1:]
            elif event.key == pygame.K_LEFT and self.caret_pos > 0:
                self.caret_pos -= 1
            elif event.key == pygame.K_RIGHT and self.caret_pos < len(self.text):
                self.caret_pos += 1
            elif event.key == pygame.K_RETURN:
                pass  # potvrzení (např. zavolat callback s textem)

    def update(self, dt: float) -> None:
        self.caret_timer += dt
        if self.caret_timer >= 0.5:
            self.caret_timer = 0.0
            self.caret_visible = not self.caret_visible

    def set_default(self, text: str) -> None:
        self.text = text
        self.caret_pos = len(self.text)

    def draw(self, surf: pygame.Surface) -> None:
        # rámeček
        pygame.draw.rect(surf, (245,245,245), self.rect, border_radius=6)
        pygame.draw.rect(surf, (80,80,80), self.rect, 2, border_radius=6)

        shown = self.text if self.text else self.placeholder
        color = (20,20,20) if self.text else (130,130,130)
        txt_surf = self.font.render(shown, True, color)
        surf.blit(txt_surf, (self.rect.x+8, self.rect.y+6))

        # caret - jen když focused a máme text surface
        if self.focused and self.caret_visible:
            caret_x = self.rect.x + 8 + self.font.size(self.text[:self.caret_pos])[0]
            pygame.draw.line(surf, (30,30,30), (caret_x, self.rect.y+6), (caret_x, self.rect.y+self.rect.height-6), 1)

    def get_int(self, default=None) -> int | None:
        s = self.text.strip()
        if not s:
            return default
        try:
            return int(s)
        except ValueError:
            return default

    def get_float(self, default=None) -> float | None:
        s = self.text.strip()
        if not s:
            return default

        s = s.replace(",", ".")

        try:
            return float(s)
        except ValueError:
            return default

    def get_text(self, default="userbrain.npz") -> str | None:
        if self.text == "" or self.text is None:
            return default
        return self.text
