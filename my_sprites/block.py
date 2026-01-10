import pygame
from constants import WHITE, TILESIZE, tilesides

class BlockV(pygame.sprite.Sprite):
    def __init__(self, x, y, width=2, height=TILESIZE, alpha=51):
        super().__init__()
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        r, g, b = WHITE
        self.image.fill((r, g, b, alpha))
        self.rect = self.image.get_rect(topleft=(x, y))


class BlockH(pygame.sprite.Sprite):
    def __init__(self, x, y, width=TILESIZE, height=2, alpha=51):
        super().__init__()
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        r, g, b = WHITE
        self.image.fill((r, g, b, alpha))
        self.rect = self.image.get_rect(topleft=(x, y))


class Blocks(pygame.sprite.Sprite):
    def __init__(self, tile_size, alpha, tilegrid, tilesides):
        super().__init__()
        self.tile_size = tile_size
        self.alpha = alpha
        self.tilesides = tilesides
        self.tilegrid = tilegrid
        self.sprites = pygame.sprite.Group()

        self.image = pygame.Surface(
            (len(tilegrid[0]) * tile_size, len(tilegrid) * tile_size),
            pygame.SRCALPHA
        )
        self.rect = self.image.get_rect(topleft=(0, 0))

        self.mask: pygame.mask.Mask | None = None

    def constructBG(self):
        # vyčistit
        self.image.fill((0, 0, 0, 0))
        self.sprites.empty()

        for i in range(len(self.tilegrid)):              # i = řádek (y)
            for j in range(len(self.tilegrid[i])):       # j = sloupec (x)
                onetile = self.tilegrid[i][j]
                sides = self.tilesides[onetile]

                x = j * self.tile_size
                y = i * self.tile_size

                for side in sides:
                    block = None
                    if side == "T":
                        block = BlockH(x, y, alpha=self.alpha)                    # horní hrana
                    elif side == "B":
                        block = BlockH(x, y + self.tile_size-1, alpha=self.alpha)   # spodní hrana
                    elif side == "L":
                        block = BlockV(x, y, alpha=self.alpha)                    # levá hrana
                    elif side == "R":
                        block = BlockV(x + self.tile_size-1, y, alpha=self.alpha)   # pravá hrana

                    if block is not None:
                        self.sprites.add(block)
                        # zároveň kreslíme blok do image, ze které pak uděláme masku
                        self.image.blit(block.image, block.rect)

        # vytvoření masky pro raycasting
        self.mask = pygame.mask.from_surface(self.image, 1)

    def draw(self, screen):
        self.sprites.draw(screen)
        # případně by stačilo:
        # screen.blit(self.image, self.rect)
