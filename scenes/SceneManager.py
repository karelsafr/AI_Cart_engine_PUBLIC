from core.CsvTilemap import CsvTileMap
from constants import EMPTY_TILE

class SceneManager:
    def __init__(self, screen):
        self.screen = screen

    def add_menu(self, menu_scene):
        self.cur_scene = menu_scene
        self.menu_scene = menu_scene

    def set_menu(self):
        self.cur_scene = self.menu_scene

    def add_mapeditor(self, mapeditor_scene):
        self.mapeditor_scene = mapeditor_scene

    def set_mapa(self):
        self.cur_scene = self.mapeditor_scene

    def add_playgame(self, playgame_scene):
        self.playgame_scene = playgame_scene

    def set_playgame(self, mapname):
        self.cur_scene = self.playgame_scene
        self.cur_scene.get_map_name(mapname)
        self.cur_scene.restart()

    def add_training(self, training_scene):
        self.training_scene = training_scene

    def set_training(self,mapname):
        self.cur_scene = self.training_scene
        self.cur_scene.get_map_name(mapname)
        self.cur_scene.restart()

    def set_cur_tmap(self, tmap):
        self.cur_tmap = tmap

    def set_atlas_tmap(self,atlas):
        self.atlas_tmap = atlas

    def set_default_tmap_name(self, tile_path_default_setting_csv):
        self.tile_path_default_setting_csv = tile_path_default_setting_csv

    def set_TILESIZE(self,TILESIZE):
        self.TILESIZE = TILESIZE

    def set_vehicle_atlas(self, vehicles_atlas):
        self.vehicles_atlas = vehicles_atlas

    def load_tmap(self, map_file):
        self.cur_tmap = CsvTileMap(
            self.atlas_tmap,
            map_file,
            tile_w=self.TILESIZE,
            tile_h=self.TILESIZE,
            base_tile=EMPTY_TILE,  # základní výplň mapy
            empty_symbol="."  # POZOR! prázdné buňky v CSV znamenají „jen base“
        )
        self.cur_tmap.prerender()

    def draw(self):
        self.cur_scene.draw(self.screen)

    def update(self, dt, keys):
        if not self.cur_scene.is_active():
            self.active = False
        self.cur_scene.update(dt, keys)

    def event(self, event):
        self.cur_scene.event(event)

    def is_active(self):
        return self.cur_scene.is_active()

    def add_brain(self,brain):
        self.brain = brain

    def get_brain(self):
        return self.brain

    def add_duel(self, duel_scene):
        self.duel_scene = duel_scene

    def set_duel(self):
        self.cur_scene = self.duel_scene
        self.cur_scene.restart()

    def add_benchmark(self, benchmark_scene):
        self.benchmark_scene = benchmark_scene

    def set_benchmark(self):
        self.cur_scene = self.benchmark_scene
        self.cur_scene.restart()