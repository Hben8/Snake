"""Modo humano: juega tú con las flechas (o WASD)."""

import pygame

from .game import Action, Direction, SnakeGame
from .viz import theme
from .viz.board_view import draw_board

_KEYS = {
    pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
    pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT,
    pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
}


def run_human(width: int, height: int, fps: int = 8) -> None:
    pygame.init()
    pygame.display.set_caption("Snake - modo humano")
    screen = pygame.display.set_mode((640, 700))
    clock = pygame.time.Clock()
    game = SnakeGame(width, height)
    best = 0
    wanted = game.snake.direction
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if game.done and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    game.reset()
                    wanted = game.snake.direction
                elif event.key in _KEYS:
                    wanted = _KEYS[event.key]

        if not game.done:
            # Un giro de 180 grados no es posible: se ignora y sigue recto.
            action = game.snake.direction.action_towards(wanted) or Action.STRAIGHT
            game.step(action)
            best = max(best, game.score)

        screen.fill(theme.BG)
        theme.draw_text(screen, f"Manzanas: {game.score}    Récord: {best}", (30, 18), 22, bold=True)
        draw_board(screen, pygame.Rect(20, 60, 600, 600), game)
        footer = "Has perdido. Pulsa Enter para reiniciar" if game.done else "Flechas / WASD para moverte"
        theme.draw_text(screen, footer, (30, 672), 16, theme.TEXT_DIM)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
