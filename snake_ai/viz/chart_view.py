"""Gráfica de puntuación por episodio dentro de la ventana."""

import numpy as np
import pygame

from . import theme


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Media de los últimos `window` valores en cada punto (al inicio, de los que haya)."""
    csum = np.concatenate([[0.0], np.cumsum(values)])
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (csum[end] - csum[start]) / (end - start)


def draw_chart(surface: pygame.Surface, rect: pygame.Rect, scores: list[int],
               title: str = "Puntuación por partida", window: int = 50) -> None:
    pygame.draw.rect(surface, theme.PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, theme.PANEL_BORDER, rect, 1, border_radius=10)
    theme.draw_text(surface, title, (rect.x + 16, rect.y + 10), 18, bold=True)

    plot = pygame.Rect(rect.x + 48, rect.y + 42, rect.width - 68, rect.height - 66)
    if len(scores) < 2:
        theme.draw_text(surface, "Esperando datos...", plot.center, 15, theme.TEXT_DIM, anchor="center")
        return

    data = np.asarray(scores, dtype=float)
    avg = _moving_average(data, window)
    top = max(5.0, float(data.max()) * 1.1)

    for frac in (0, 0.5, 1):  # líneas guía
        y = plot.bottom - frac * plot.height
        pygame.draw.line(surface, theme.PANEL_BORDER, (plot.x, y), (plot.right, y), 1)
        theme.draw_text(surface, f"{top * frac:.0f}", (plot.x - 8, y), 12, theme.TEXT_DIM, anchor="midright")

    def to_points(values: np.ndarray):
        xs = plot.x + np.linspace(0, plot.width, len(values))
        ys = plot.bottom - values / top * plot.height
        return list(zip(xs, ys))

    # Con miles de episodios, se agrupan para no dibujar más puntos que píxeles.
    if len(data) > plot.width:
        bins = np.array_split(data, plot.width)
        data = np.array([b.mean() for b in bins])
        avg = np.array([b.mean() for b in np.array_split(avg, plot.width)])
    pygame.draw.lines(surface, (70, 100, 150), False, to_points(data), 1)
    pygame.draw.lines(surface, theme.NEURON_ON, False, to_points(avg), 3)

    theme.draw_text(surface, f"media móvil ({window})", (plot.right, rect.y + 12), 13,
                    theme.NEURON_ON, anchor="topright")
