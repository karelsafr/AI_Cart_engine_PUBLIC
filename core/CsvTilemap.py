# core/CsvTileMap.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import csv
import pygame
from constants import EMPTY_TILE
from core.TextureAtlas import TextureAtlas   # uprav dle svého názvu souboru

class CsvTileMap:
    def __init__(
        self,
        atlas: TextureAtlas,
        csv_path: str | Path,
        *,
        tile_w: Optional[int] = None,
        tile_h: Optional[int] = None,
        base_tile: Optional[str] = EMPTY_TILE,       # Jméno dlaždice pro „výplň“
        empty_symbol: str = ".",               #  Jak značíme „prázdno“ v CSV
        scale_mode: str = "smooth",
    ):
        self.atlas = atlas
        self.csv_path = Path(csv_path)
        self.grid: List[List[str]] = []
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.surface: Optional[pygame.Surface] = None
        self.width_px = 0
        self.height_px = 0
        self.scale_mode = scale_mode

        self.base_tile = base_tile
        self.empty_symbol = empty_symbol
        self._base_tile_cached: Optional[pygame.Surface] = None

    def load_csv(self) -> None:
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            grid = []
            for row in reader:
                # NEODSTRANUJI prázdné položky – chceme mít přesnou mřížku.
                if not row:
                    continue
                items = [cell.strip() for cell in row]
                grid.append(items)

        if not grid:
            raise ValueError(f"CSV '{self.csv_path}' je prázdné.")

        w0 = len(grid[0])
        for i, r in enumerate(grid):
            if len(r) != w0:
                raise ValueError(f"Řádek {i} má jinou délku ({len(r)}) než první řádek ({w0}).")
        self.grid = grid

    def _ensure_tile_size(self) -> None:
        if self.tile_w is None or self.tile_h is None:
            # vezmeme první nenulový název v mřížce
            first_name = None
            for row in self.grid:
                for name in row:
                    if name and name != self.empty_symbol:
                        first_name = name
                        break
                if first_name:
                    break
            if first_name is None:
                raise ValueError("Nelze odvodit tile size – mřížka má jen prázdné položky.")
            s0 = self.atlas.get(first_name)
            self.tile_w, self.tile_h = s0.get_width(), s0.get_height()

    def _cache_base_tile(self) -> None:
        if self.base_tile:
            t = self.atlas.get(self.base_tile)
            # (většinou už jsou všechny dlaždice škálované v atlasu)
            if self.tile_w is not None and self.tile_h is not None:
                if (t.get_width(), t.get_height()) != (self.tile_w, self.tile_h):
                    t = pygame.transform.smoothscale(t, (self.tile_w, self.tile_h))
            self._base_tile_cached = t
        else:
            self._base_tile_cached = None

    def prerender(self, *, fill_color: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        if not self.grid:
            self.load_csv()

        self._ensure_tile_size()
        rows = len(self.grid)
        cols = len(self.grid[0])
        self.width_px = cols * self.tile_w
        self.height_px = rows * self.tile_h

        bg = pygame.Surface((self.width_px, self.height_px), flags=pygame.SRCALPHA)
        bg.fill(fill_color)

        # vyplň základní dlaždicí (base) – celý rastr
        self._cache_base_tile()
        if self._base_tile_cached is not None:
            base = self._base_tile_cached
            for r in range(rows):
                py = r * self.tile_h
                for c in range(cols):
                    px = c * self.tile_w
                    bg.blit(base, (px, py))

        # overlay z CSV (přeskoč prázdné symboly)
        for r, row in enumerate(self.grid):
            py = r * self.tile_h
            for c, name in enumerate(row):
                if not name or name == self.empty_symbol:
                    continue
                px = c * self.tile_w
                try:
                    tile = self.atlas.get(name)
                except KeyError:
                    tile = self._missing_tile(self.tile_w, self.tile_h)
                bg.blit(tile, (px, py))

        try:
            bg = bg.convert_alpha()
        except pygame.error:
            bg = bg.convert()
        self.surface = bg

    def draw(self, screen: pygame.Surface, dest: tuple[int, int] = (0, 0)) -> None:
        if self.surface is None:
            self.prerender()
        screen.blit(self.surface, dest)

    def set_tile(self, row: int, col: int, name: str, update_surface: bool = True) -> None:
        """Změní jednu buňku (overlay). Nejdřív vrátí base do buňky, pak položí novou dlaždici."""
        assert 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])
        self.grid[row][col] = name

        if update_surface and self.surface is not None:
            if self.tile_w is None or self.tile_h is None:
                return
            px = col * self.tile_w
            py = row * self.tile_h
            rect = pygame.Rect(px, py, self.tile_w, self.tile_h)

            # obnov base v buňce (nebo vyčisti do transparentní)
            if self._base_tile_cached is not None:
                self.surface.blit(self._base_tile_cached, (px, py))
            else:
                # čistě transparentní vyčištění
                self.surface.fill((0,0,0,0), rect)

            # polož overlay, pokud není prázdný symbol
            if name and name != self.empty_symbol:
                tile = self.atlas.get(name)
                if (tile.get_width(), tile.get_height()) != (self.tile_w, self.tile_h):
                    tile = pygame.transform.smoothscale(tile, (self.tile_w, self.tile_h))
                self.surface.blit(tile, (px, py))

    def save_csv(self, path: str | Path) -> None:
        out = Path(path)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in self.grid:
                writer.writerow(row)

    @staticmethod
    def _missing_tile(w: int, h: int) -> pygame.Surface:
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((255, 0, 255))
        pygame.draw.line(s, (0, 0, 0), (0, 0), (w, h), 2)
        pygame.draw.line(s, (0, 0, 0), (0, h), (w, 0), 2)
        try:
            s = s.convert_alpha()
        except pygame.error:
            pass
        return s

    # Změna base dlaždice za běhu
    def set_base_tile(self, name: Optional[str], rebuild: bool = True) -> None:
        self.base_tile = name
        self._cache_base_tile()
        if rebuild:
            self.prerender()
