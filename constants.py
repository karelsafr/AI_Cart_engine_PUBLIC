# Barvy:
BLACK = (0, 0, 0)
GREY = (192, 192, 192)
PURPLE = (150, 10, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# Setting okna - vse od výšky !!! (určuje počet dlazdic aj!)
HEIGHT = int(800)


# Dopocitavam vse uz:
WIDTH = int(HEIGHT*1.3) # SIRKA OBRAZOVKY - prostor navíc pro menu o 30%

# VELIKOST TILE:
TILESIZE = int(HEIGHT//10) # 10 dlazdic na výšku # DLAZDICE

# MENU SCENE
MENU_FONT_SIZE  = int(HEIGHT*0.1)   # VELIKOST FONTU V MENU SCENE
MENU_SCENE_START_Y_POS = int(HEIGHT*0.2) # ODSUNUTI od shora v MENU SCENE
MENU_MENU_IDENT = int(MENU_FONT_SIZE) # velikost prostoru mezi talčítky

## Editor MAPA setting / MAP SCENE
# Sirka a výška tlačítek v mapě, jejich odsazení aj:
MAP_MENUSIZE = int(HEIGHT*0.3)  # MAP EDITOR MENU sírka
MAP_BUTTON_FONTSIZE = int(MAP_MENUSIZE//6) # MAP SCENE VELIKOST TLACITKA
MAP_BUTTON_WIDTH = int(MAP_MENUSIZE*0.8)  # MAP_SCENE width tlacitka
MAP_BUTTON_HEIGHT = int(MAP_MENUSIZE*0.2) # MAP SCENE výska tlacítka
MAP_BUTTON_IDENT = int(MAP_MENUSIZE*0.1) # MAP SCENE ODSAZENI

FPS = int(45)

# Paths / assets
tile_path_png = "assets/racing-pack/Spritesheets/spritesheet_tiles.png"
tile_path_xml = "assets/racing-pack/Spritesheets/spritesheet_tiles.xml"
car_path_png = "assets/racing-pack/Spritesheets/spritesheet_vehicles.png"
car_path_xml = "assets/racing-pack/Spritesheets/spritesheet_vehicles.xml"
tile_path_default_setting_csv = "DefaultSettings/DefaultReset.csv"
tile_path_default_race_csv = "DefaultRace/DefaultRace.csv"
PATH_SAVES = "UserData/SAVES/"
# Tiles cycle:
tilecicle = ["road_dirt01", "road_dirt90","road_dirt38", "road_dirt02",  "road_dirt04", "road_dirt40", "land_grass04"]
EMPTY_TILE = "land_grass04"

tilesides= {"road_dirt01": ["T", "B"],
            "road_dirt90": ["R", "L"],
            "road_dirt38": ["L", "B"],
            "road_dirt02": ["L", "T"],
            "road_dirt04": ["T", "R"],
            "road_dirt40": ["R", "B"],
            "road_dirt42": ["T", "B"],# start tile!!!
            "land_grass04": ["T", "B", "R", "L"]}

# CAR, speed na čtverec
SPEED = 100
MAX_SPEED = 500
TURN_SPEED = 100
BREAK_SPEED =  100
FRICTION_SPEED = 100
RAYCAST_ANGLES = [-90, -45, -20, -5, 0, 5, 20, 45, 90]