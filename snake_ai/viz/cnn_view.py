"""Dibuja una red CONVOLUCIONAL "pensando": en lugar de neuronas sueltas, MAPAS de activación.

Cada filtro de una convolución produce un mapa del tamaño del tablero: brilla donde ese filtro ha
detectado "su" patrón (un borde, una cabeza, un hueco cerrado...). Se ven en este orden:

  Entrada (el tablero como imagen, girado: la serpiente siempre mira arriba)
    -> 16 mapas de la 1ª convolución -> 32 mapas de la 2ª -> capa densa (128 neuronas) -> 3 salidas (Q)

Cuanto más brillante un píxel, más activo. Cada mapa se normaliza por separado para que se vea su patrón.
"""

import numpy as np
import pygame

from . import theme

_OFF = np.array(theme.NEURON_OFF, dtype=np.float32)
_ON = np.array(theme.NEURON_ON, dtype=np.float32)
_INPUT_COLORS = ((130, 140, 170), (52, 190, 96), (160, 255, 170), (240, 80, 90))  # pared, cuerpo, cabeza, comida


def _blit_map(surface: pygame.Surface, level: np.ndarray, pos: tuple[int, int], size: int, on=_ON) -> None:
    """Pinta un mapa (alto × ancho, valores 0..1) como un cuadradito de `size` píxeles."""
    rgb = _OFF + (np.asarray(on, dtype=np.float32) - _OFF) * level[..., None]
    img = pygame.surfarray.make_surface(rgb.astype(np.uint8).transpose(1, 0, 2))
    surface.blit(pygame.transform.scale(img, (size, size)), pos)
    pygame.draw.rect(surface, theme.PANEL_BORDER, (*pos, size, size), 1)


def _grid(surface, maps: np.ndarray, origin: tuple[int, int], cols: int, size: int, gap: int,
          on_colors=None) -> None:
    for k, m in enumerate(maps):
        peak = float(m.max())
        level = m / peak if peak > 1e-9 else np.zeros_like(m)
        row, col = divmod(k, cols)
        pos = (origin[0] + col * (size + gap), origin[1] + row * (size + gap))
        _blit_map(surface, level, pos, size, on_colors[k] if on_colors else _ON)


def draw_cnn(surface: pygame.Surface, rect: pygame.Rect, network, obs: np.ndarray, output_labels,
             chosen: int | None = None, explored: bool = False) -> None:
    pygame.draw.rect(surface, theme.PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, theme.PANEL_BORDER, rect, 1, border_radius=10)
    theme.draw_text(surface, "Cerebro: red convolucional", (rect.x + 16, rect.y + 10), 18, bold=True)
    theme.draw_text(surface, "más brillante = más activo", (rect.right - 16, rect.y + 14), 13,
                    theme.TEXT_DIM, anchor="topright")

    image, conv1, conv2, dense, q = network.activations(obs)
    x0, y0 = rect.x, rect.y

    # --- fila de arriba: entrada y mapas de las dos convoluciones
    top = y0 + 70
    theme.draw_text(surface, "Entrada: el tablero", (x0 + 24, top - 22), 13, theme.TEXT_DIM)
    _grid(surface, image, (x0 + 24, top), 2, 48, 20, on_colors=_INPUT_COLORS)
    for k, name in enumerate(("pared", "cuerpo", "cabeza", "comida")):
        col, row = k % 2, k // 2
        theme.draw_text(surface, name, (x0 + 24 + col * 68 + 24, top + row * 68 + 50), 12, theme.TEXT_DIM,
                        anchor="midtop")
    theme.draw_text(surface, "1ª convolución (16 filtros)", (x0 + 180, top - 22), 13, theme.TEXT_DIM)
    _grid(surface, conv1, (x0 + 180, top), 4, 44, 4)
    theme.draw_text(surface, "2ª convolución (32 filtros)", (x0 + 385, top - 22), 13, theme.TEXT_DIM)
    _grid(surface, conv2, (x0 + 385, top), 8, 25, 3)

    # --- fila de abajo: capa densa y salidas
    low = y0 + 290
    theme.draw_text(surface, "Capa densa (128 neuronas)", (x0 + 24, low - 22), 13, theme.TEXT_DIM)
    peak = float(dense.max())
    levels = dense / peak if peak > 1e-9 else np.zeros_like(dense)
    for k, level in enumerate(levels):
        row, col = divmod(k, 16)
        pos = (x0 + 30 + col * 14, low + 8 + row * 14)
        active = dense[k] > 0
        color = theme.lerp_color(theme.NEURON_OFF, theme.NEURON_ON, level if active else 0.0)
        pygame.draw.circle(surface, color, pos, 5)
        pygame.draw.circle(surface, (255, 240, 190) if active else (90, 99, 130), pos, 5, 1)

    theme.draw_text(surface, "Salidas", (x0 + 400, low - 22), 13, theme.TEXT_DIM)
    lo, hi = float(q.min()), float(q.max())
    levels = (q - lo) / (hi - lo) if hi - lo > 1e-9 else np.full_like(q, 0.5)
    for k, level in enumerate(levels):
        x, y = x0 + 420, low + 18 + k * 44
        pygame.draw.circle(surface, theme.lerp_color(theme.NEURON_OFF, theme.NEURON_ON, level), (x, y), 11)
        pygame.draw.circle(surface, (255, 240, 190), (x, y), 11, 1)
        is_choice = chosen is not None and k == chosen
        if is_choice:
            pygame.draw.circle(surface, theme.RANDOM if explored else theme.DECISION, (x, y), 17, 3)
        theme.draw_text(surface, output_labels[k], (x + 22, y - 9), 15,
                        theme.TEXT if is_choice else theme.TEXT_DIM, bold=is_choice, anchor="midleft")
        theme.draw_text(surface, f"Q={q[k]:+.2f}", (x + 22, y + 9), 12, theme.TEXT_DIM, anchor="midleft")
