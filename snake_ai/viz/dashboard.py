"""La ventana principal: tablero + cerebro + gráfica + controles.

Se usa igual para ver el entrenamiento y para ver jugar a una red ya entrenada.
"""

import time

import numpy as np
import pygame

from ..game import ACTION_NAMES, SnakeGame
from ..nn import NeuralNetwork
from ..rl.observation import FEATURE_NAMES
from ..rl.trainer import Observer
from . import theme
from .board_view import draw_board
from .chart_view import draw_chart
from .network_view import draw_network
from .recorder import GifRecorder

WINDOW = (1280, 720)
SPEEDS = (("Lento", 6), ("Normal", 15), ("Rápido", 45), ("Turbo", None))  # (nombre, fps)
_TURBO_DRAW_INTERVAL = 1 / 30


class Dashboard:
    def __init__(self, title: str, speed: str = "Normal", recorder: GifRecorder | None = None):
        pygame.init()
        pygame.display.set_caption(title)
        self.title = title
        self.screen = pygame.display.set_mode(WINDOW)
        self.clock = pygame.time.Clock()
        self.recorder = recorder
        self.speed_index = next((i for i, (n, _) in enumerate(SPEEDS) if n.lower() == speed.lower()), 1)
        self.paused = False
        self.closed = False
        self._last_draw = 0.0
        self._last_poll = 0.0

    @property
    def fps(self) -> int | None:
        return SPEEDS[self.speed_index][1]

    # -------------------------------------------------------------- eventos
    def poll(self) -> bool:
        """Procesa teclado/ratón. Devuelve False si el usuario cierra la ventana."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.closed = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_UP, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_RIGHT):
                    self.speed_index = min(len(SPEEDS) - 1, self.speed_index + 1)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_LEFT):
                    self.speed_index = max(0, self.speed_index - 1)
        self._last_poll = time.perf_counter()
        return not self.closed

    # ---------------------------------------------------------------- dibujo
    def frame(self, game: SnakeGame, network: NeuralNetwork, obs: np.ndarray,
              action: int | None, explored: bool, stats: list[tuple[str, str]],
              scores: list[int], chart_title: str = "Puntuación por partida") -> bool:
        """Dibuja un fotograma respetando la velocidad elegida.

        Devuelve False cuando hay que parar (ventana cerrada). En modo Turbo se
        salta casi todos los fotogramas para no frenar el entrenamiento.
        """
        now = time.perf_counter()
        if self.fps is None and now - self._last_draw < _TURBO_DRAW_INTERVAL and not self.paused:
            if now - self._last_poll > 0.1:
                self.poll()
            return not self.closed

        while True:
            if not self.poll():
                return False
            self._render(game, network, obs, action, explored, stats, scores, chart_title)
            if not self.paused:
                break
            self.clock.tick(30)

        self._last_draw = time.perf_counter()
        if self.recorder is not None:
            self.recorder.capture(self.screen)
            if self.recorder.full:
                self.closed = True
        if self.fps is not None:
            self.clock.tick(self.fps)
        return not self.closed

    def _render(self, game, network, obs, action, explored, stats, scores, chart_title) -> None:
        screen = self.screen
        screen.fill(theme.BG)
        theme.draw_text(screen, self.title, (30, 14), 24, bold=True)

        x = 30
        for label, value in stats:  # barra de estadísticas
            r = theme.draw_text(screen, label, (x, 52), 14, theme.TEXT_DIM)
            r2 = theme.draw_text(screen, value, (r.right + 6, 50), 17, theme.TEXT, bold=True)
            x = r2.right + 22

        draw_board(screen, pygame.Rect(30, 90, 590, 590), game)
        draw_network(screen, pygame.Rect(650, 12, 620, 440), network, obs,
                     FEATURE_NAMES, ACTION_NAMES, action, explored)
        draw_chart(screen, pygame.Rect(650, 466, 620, 214), scores, chart_title)

        if action is not None:
            who = "AZAR (exploración)" if explored else "LA RED decide"
            color = theme.RANDOM if explored else theme.DECISION
            # En la fila del título (y=18): la barra de estadísticas de debajo puede ocupar todo el ancho.
            theme.draw_text(screen, f"{who}: {ACTION_NAMES[action]}", (620, 18), 18, color, bold=True,
                            anchor="topright")

        name, _ = SPEEDS[self.speed_index]
        hint = "PAUSA  |  " if self.paused else ""
        theme.draw_text(
            screen,
            f"{hint}Velocidad: {name}   [↑/↓ cambiar]   [Espacio pausa]   [Esc salir]",
            (30, 696), 14, theme.TEXT_DIM,
        )
        pygame.display.flip()

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.save()
        pygame.quit()


class DashboardObserver(Observer):
    """Conecta el entrenador con el dashboard (se importa en la CLI para no
    obligar al núcleo de RL a depender de pygame)."""

    def __init__(self, dashboard: Dashboard):
        self.dashboard = dashboard

    def on_decision(self, trainer, obs, action, explored) -> bool:
        game = trainer.env.game
        stats = [
            ("Episodio", f"{trainer.episode}/{trainer.config.episodes}"),
            ("Manzanas", str(game.score)),
            ("Récord", str(trainer.record)),
            ("Media(50)", f"{trainer.history.moving_average(50):.1f}"),
            ("Azar ε", f"{trainer.agent.epsilon:.2f}"),
        ]
        return self.dashboard.frame(game, trainer.agent.online, obs, action, explored,
                                    stats, trainer.history.scores)

    def on_episode_end(self, trainer, stats) -> bool:
        return not self.dashboard.closed
