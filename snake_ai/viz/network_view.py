"""Dibuja la red neuronal "pensando" en tiempo real.

Qué significa cada cosa:
  * Círculo brillante  = neurona activa (cuanto más amarillo, más fuerte).
  * Círculo apagado    = neurona desactivada (ReLU devolvió 0).
  * Línea azul         = conexión que EXCITA a la neurona de destino.
  * Línea roja         = conexión que la INHIBE.
  * Grosor/opacidad    = cuánto aporta ahora mismo esa conexión
                         (activación de origen x peso). Si la neurona de
                         origen está apagada, su conexión desaparece.
  * Salida             = valores Q de cada acción; la elegida lleva un aro.
"""

import numpy as np
import pygame

from ..nn import NeuralNetwork
from . import theme

_EDGE_THRESHOLD = 0.06  # las conexiones por debajo de este % del máximo se pintan casi invisibles
_WIDE_LAYER = 32        # a partir de este nº de neuronas una capa se dibuja como cuadrícula de puntos
_TOP_EDGES = 70         # en redes anchas solo se dibujan las conexiones que más aportan (de 65.000 sería ilegible)


def _layer_positions(rect: pygame.Rect, sizes, left: int, right: int, top: int, bottom: int):
    xs = np.linspace(rect.x + left, rect.right - right, len(sizes))
    positions = []
    for x, n in zip(xs, sizes):
        ys = [rect.y + top + offset for offset in _spread(n, rect.height - top - bottom)]
        positions.append([(int(x), int(y)) for y in ys])
    return positions


def _wide_positions(rect: pygame.Rect, sizes, top: int, bottom: int):
    """Como `_layer_positions`, pero las capas muy anchas (p. ej. 256 neuronas) se dibujan como una cuadrícula
    compacta de puntos en vez de una columna imposible de leer."""
    xs = np.linspace(rect.x + 150, rect.right - 118, len(sizes))
    middle = rect.y + top + (rect.height - top - bottom) / 2
    positions = []
    for x, n in zip(xs, sizes):
        if n <= _WIDE_LAYER:
            ys = [rect.y + top + offset for offset in _spread(n, rect.height - top - bottom)]
            positions.append([(int(x), int(y)) for y in ys])
        else:
            cols = 12
            rows = -(-n // cols)
            positions.append([(int(x + (k % cols - (cols - 1) / 2) * 8), int(middle + (k // cols - (rows - 1) / 2) * 8))
                              for k in range(n)])
    return positions


def _spread(n: int, height: int) -> list[float]:
    if n == 1:
        return [height / 2]
    step = height / (n - 1)
    return [i * step for i in range(n)]


def draw_network(
    surface: pygame.Surface,
    rect: pygame.Rect,
    network: NeuralNetwork,
    obs: np.ndarray,
    input_labels,
    output_labels,
    chosen: int | None = None,
    explored: bool = False,
) -> None:
    pygame.draw.rect(surface, theme.PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, theme.PANEL_BORDER, rect, 1, border_radius=10)
    theme.draw_text(surface, "Cerebro: red neuronal", (rect.x + 16, rect.y + 10), 18, bold=True)
    _legend(surface, rect)

    layers = network.activations(obs)
    sizes = [len(layer) for layer in layers]
    wide = max(sizes) > _WIDE_LAYER
    if wide:
        pos = _wide_positions(rect, sizes, top=84, bottom=26)
    else:
        pos = _layer_positions(rect, sizes, left=150, right=118, top=84, bottom=26)

    # --- conexiones (de más débiles a más fuertes, para que las fuertes queden encima)
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    shift = np.array([rect.x, rect.y])
    for li, w in enumerate(network.weights):
        contrib = layers[li][:, None] * w
        scale = float(np.abs(contrib).max())
        if scale < 1e-9:
            scale = 1.0
        norm = np.abs(contrib) / scale
        if wide:  # solo las conexiones más fuertes (ordenadas de menos a más, para que las fuertes queden encima)
            top_k = min(_TOP_EDGES, norm.size)
            strongest = np.argpartition(norm, -top_k, axis=None)[-top_k:]
            order = strongest[np.argsort(norm.ravel()[strongest])]
        else:
            order = np.argsort(norm, axis=None)
        for flat in order:
            i, j = divmod(int(flat), contrib.shape[1])
            s = float(norm[i, j])
            a = np.array(pos[li][i]) - shift
            b = np.array(pos[li + 1][j]) - shift
            if s < _EDGE_THRESHOLD:
                color = (70, 78, 105, 28)
                width = 1
            else:
                base = theme.EXCITE if contrib[i, j] > 0 else theme.INHIBIT
                color = (*base, int(60 + 195 * s))
                width = 1 if s < 0.5 else 2
            pygame.draw.line(overlay, color, tuple(a), tuple(b), width)
    surface.blit(overlay, rect.topleft)

    # --- neuronas
    for li, (layer, nodes) in enumerate(zip(layers, pos)):
        is_output = li == len(layers) - 1
        if is_output:
            lo, hi = float(layer.min()), float(layer.max())
            levels = (layer - lo) / (hi - lo) if hi - lo > 1e-9 else np.full_like(layer, 0.5)
        else:
            peak = float(layer.max())
            levels = layer / peak if peak > 1e-9 else np.zeros_like(layer)
        radius = 11 if sizes[li] <= 12 else (7 if sizes[li] <= _WIDE_LAYER else 3)
        for k, ((x, y), level) in enumerate(zip(nodes, levels)):
            active = layer[k] > 0 if not is_output else True
            color = theme.lerp_color(theme.NEURON_OFF, theme.NEURON_ON, level if active else 0.0)
            pygame.draw.circle(surface, color, (x, y), radius)
            pygame.draw.circle(surface, (90, 99, 130) if not active else (255, 240, 190), (x, y), radius, 1)

    # --- etiquetas de entrada
    for k, (x, y) in enumerate(pos[0]):
        on = obs[k] > 0.5
        theme.draw_text(surface, input_labels[k], (x - 18, y), 14,
                        theme.TEXT if on else theme.TEXT_DIM, anchor="midright")

    # --- etiquetas y valores Q de salida
    q = layers[-1]
    for k, (x, y) in enumerate(pos[-1]):
        is_choice = chosen is not None and k == chosen
        if is_choice:
            ring = theme.RANDOM if explored else theme.DECISION
            pygame.draw.circle(surface, ring, (x, y), 17, 3)
        theme.draw_text(surface, output_labels[k], (x + 22, y - 9), 15,
                        theme.TEXT if is_choice else theme.TEXT_DIM, bold=is_choice, anchor="midleft")
        theme.draw_text(surface, f"Q={q[k]:+.2f}", (x + 22, y + 9), 12, theme.TEXT_DIM, anchor="midleft")

    # --- cabeceras de capa
    titles = ["Entradas"] + [f"Oculta {i + 1}" for i in range(len(layers) - 2)] + ["Salidas"]
    for title, nodes in zip(titles, pos):
        theme.draw_text(surface, title, (nodes[0][0], rect.y + 38), 13, theme.TEXT_DIM, anchor="midtop")


def _legend(surface: pygame.Surface, rect: pygame.Rect) -> None:
    x = rect.right - 16
    y = rect.y + 14
    for text, color in (("inhibe", theme.INHIBIT), ("excita", theme.EXCITE)):
        r = theme.draw_text(surface, text, (x, y), 13, theme.TEXT_DIM, anchor="topright")
        pygame.draw.line(surface, color, (r.left - 30, y + 8), (r.left - 6, y + 8), 3)
        x = r.left - 40
    r = theme.draw_text(surface, "apagada", (x, y), 13, theme.TEXT_DIM, anchor="topright")
    pygame.draw.circle(surface, theme.NEURON_OFF, (r.left - 10, y + 8), 6)
    pygame.draw.circle(surface, (90, 99, 130), (r.left - 10, y + 8), 6, 1)
    r2 = theme.draw_text(surface, "activa", (r.left - 28, y), 13, theme.TEXT_DIM, anchor="topright")
    pygame.draw.circle(surface, theme.NEURON_ON, (r2.left - 10, y + 8), 6)
