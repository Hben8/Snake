"""Dibuja el tablero, la serpiente y la manzana."""

import pygame

from ..game import Direction, SnakeGame
from . import theme


def draw_board(surface: pygame.Surface, rect: pygame.Rect, game: SnakeGame) -> None:
    cell = min(rect.width // game.width, rect.height // game.height)
    board = pygame.Rect(0, 0, cell * game.width, cell * game.height)
    board.center = rect.center

    for x in range(game.width):
        for y in range(game.height):
            color = theme.BOARD_A if (x + y) % 2 == 0 else theme.BOARD_B
            pygame.draw.rect(surface, color, (board.x + x * cell, board.y + y * cell, cell, cell))
    pygame.draw.rect(surface, theme.PANEL_BORDER, board.inflate(4, 4), 2, border_radius=4)

    if game.food is not None:
        _draw_apple(surface, board, cell, game.food)

    body = list(game.snake.body)
    n = len(body)
    pad = max(1, cell // 12)
    for i in range(n - 1, -1, -1):  # de cola a cabeza, para que la cabeza quede encima
        x, y = body[i]
        t = i / max(1, n - 1)
        color = theme.lerp_color(theme.SNAKE_HEAD, theme.SNAKE_TAIL, t) if i else theme.SNAKE_HEAD
        r = pygame.Rect(board.x + x * cell + pad, board.y + y * cell + pad, cell - 2 * pad, cell - 2 * pad)
        pygame.draw.rect(surface, color, r, border_radius=max(3, cell // 4))
    _draw_eyes(surface, board, cell, game.snake.head, game.snake.direction)


def _draw_apple(surface, board, cell, food) -> None:
    cx = board.x + food[0] * cell + cell // 2
    cy = board.y + food[1] * cell + cell // 2 + 1
    pygame.draw.circle(surface, theme.APPLE, (cx, cy), max(3, int(cell * 0.36)))
    pygame.draw.ellipse(surface, theme.LEAF, (cx, cy - int(cell * 0.46), max(3, cell // 5), max(2, cell // 8)))


def _draw_eyes(surface, board, cell, head, direction: Direction) -> None:
    dx, dy = direction.vector
    px, py = -dy, dx  # perpendicular
    cx = board.x + head[0] * cell + cell / 2
    cy = board.y + head[1] * cell + cell / 2
    for side in (-1, 1):
        ex = cx + dx * cell * 0.18 + px * side * cell * 0.2
        ey = cy + dy * cell * 0.18 + py * side * cell * 0.2
        pygame.draw.circle(surface, (255, 255, 255), (ex, ey), max(2, cell // 8))
        pygame.draw.circle(surface, (20, 20, 30), (ex + dx * cell * 0.04, ey + dy * cell * 0.04), max(1, cell // 16))
