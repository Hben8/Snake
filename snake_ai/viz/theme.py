"""Colores y fuentes compartidos por todas las vistas."""

import pygame

BG = (14, 17, 26)
PANEL = (22, 26, 39)
PANEL_BORDER = (44, 51, 74)
TEXT = (226, 231, 245)
TEXT_DIM = (126, 135, 163)

BOARD_A = (26, 32, 48)
BOARD_B = (30, 37, 55)
SNAKE_HEAD = (120, 240, 140)
SNAKE_BODY = (52, 190, 96)
SNAKE_TAIL = (28, 110, 66)
APPLE = (240, 80, 90)
LEAF = (90, 200, 110)

NEURON_OFF = (38, 44, 64)
NEURON_ON = (255, 214, 90)
EXCITE = (80, 170, 255)    # conexión que empuja a activar la neurona siguiente
INHIBIT = (255, 95, 95)    # conexión que empuja a apagarla
DECISION = (110, 240, 150)  # la red decide
RANDOM = (255, 160, 70)     # exploración al azar

_font_cache: dict[tuple[int, bool], pygame.font.Font] = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont("segoeui,arial,helvetica,dejavusans", size, bold=bold)
    return _font_cache[key]


def lerp_color(a, b, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_text(surface, text, pos, size=16, color=TEXT, bold=False, anchor="topleft"):
    img = font(size, bold).render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect
