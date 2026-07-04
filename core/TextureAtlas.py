# core/atlas.py
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional
import pygame

class TextureAtlas:
    def __init__(
        self,
        image_path: str | Path,
        xml_path: str | Path,
        *,
        add_alias_without_ext: bool = True,
        scale_to: tuple[int, int] | None = None,
        scale_mode: str = "smooth",
    ):
        self.image_path = Path(image_path)
        self.xml_path = Path(xml_path)
        self.add_alias_without_ext = add_alias_without_ext

        self.sheet: Optional[pygame.Surface] = None
        self.surfaces: Dict[str, pygame.Surface] = {}   # name -> Surface (convert_alpha když to jde)
        self.rects: Dict[str, pygame.Rect] = {}         # name -> Rect v atlasu (pro debug)
        self.scale_to = scale_to
        self.scale_mode = scale_mode

    def load(self) -> None:
        # načti sheet
        sheet = pygame.image.load(self.image_path.as_posix())
        # máme průhlednost? u atlasů obvykle ano
        try:
            sheet = sheet.convert_alpha()
        except pygame.error:
            sheet = sheet.convert()
        self.sheet = sheet

        # parse XML
        root = ET.parse(self.xml_path.as_posix()).getroot()
        assert root.tag in ("TextureAtlas", "atlas"), f"Neočekávaný kořen tagu: {root.tag}"

        count = 0
        for st in root.findall("SubTexture"):
            name = st.attrib["name"]
            x = int(st.attrib["x"])
            y = int(st.attrib["y"])
            w = int(st.attrib["width"])
            h = int(st.attrib["height"])

            rect = pygame.Rect(x, y, w, h)
            # vytvoříme nezávislou Surface s alfa kanálem

            tile = pygame.Surface((w, h), flags=pygame.SRCALPHA)
            tile.blit(sheet, (0, 0), rect)

            # --- zde škálujeme rovnou při načtení ---
            if self.scale_to is not None and (w, h) != self.scale_to:
                if self.scale_mode == "smooth":
                    tile = pygame.transform.smoothscale(tile, self.scale_to)
                else:
                    tile = pygame.transform.scale(tile, self.scale_to)

            if pygame.display.get_surface() is not None:
                try:
                    tile = tile.convert_alpha()
                except pygame.error:
                    tile = tile.convert()

            self.surfaces[name] = tile
            self.rects[name] = rect
            count += 1

            if self.add_alias_without_ext and name.endswith(".png"):
                alias = name[:-4]
                self.surfaces[alias] = tile
                self.rects[alias] = rect

        if count == 0:
            raise ValueError("V XML nebyly nalezeny žádné <SubTexture> položky.")

    def convert_all(self) -> None:
        """Zkonvertuje všechny Surface na formát displeje (po set_mode)."""
        if pygame.display.get_surface() is None:
            return
        new_surfaces: Dict[str, pygame.Surface] = {}
        for k, s in self.surfaces.items():
            try:
                new_surfaces[k] = s.convert_alpha()
            except pygame.error:
                new_surfaces[k] = s.convert()
        self.surfaces = new_surfaces

    def get(self, name: str) -> pygame.Surface:
        try:
            return self.surfaces[name]
        except KeyError:
            raise KeyError(f"Dlaždice '{name}' v atlasu nenalezena.")

    def names(self) -> Iterable[str]:
        return self.surfaces.keys()
