"""La ventana principal: tablero + cerebro + gráfica + controles.

Se usa igual para ver el entrenamiento y para ver jugar a una red ya entrenada.
"""

import time

import numpy as np
import pygame

from ..game import ACTION_NAMES, SnakeGame
from ..nn import NeuralNetwork
from ..rl.observation import FEATURE_NAMES as DEFAULT_FEATURE_NAMES
from ..rl.evolution import EvolutionObserver, play_visible
from ..rl.trainer import Observer
from . import theme
from .board_view import draw_board
from .chart_view import draw_chart
from .cnn_view import draw_cnn
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
        self.feature_names = DEFAULT_FEATURE_NAMES  # nombres de las entradas de la red
        self.chart_slots: int | None = None  # nº final de puntos de la gráfica (ver draw_chart)
        self.chart_window = 50  # ventana de la media móvil de la gráfica (1 = sin suavizar)
        self.footer: str | None = None  # None = controles por defecto; "" = nada
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
            self.render(game, network, obs, action, explored, stats, scores, chart_title)
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

    def render(self, game, network, obs, action, explored, stats, scores, chart_title) -> None:
        screen = self.screen
        screen.fill(theme.BG)
        decision = None
        if action is not None:
            who = "AZAR (exploración)" if explored else "LA RED decide"
            decision = (f"{who}: {ACTION_NAMES[action]}", theme.RANDOM if explored else theme.DECISION)
        self._draw_title(screen, decision[0] if decision else "")

        x = 30
        for label, value in stats:  # barra de estadísticas
            r = theme.draw_text(screen, label, (x, 52), 14, theme.TEXT_DIM)
            r2 = theme.draw_text(screen, value, (r.right + 6, 50), 17, theme.TEXT, bold=True)
            x = r2.right + 22

        draw_board(screen, pygame.Rect(30, 90, 590, 590), game)
        if getattr(network, "is_conv", False):  # red convolucional: se dibujan mapas, no neuronas sueltas
            draw_cnn(screen, pygame.Rect(650, 12, 620, 440), network, obs, ACTION_NAMES, action, explored)
        else:
            draw_network(screen, pygame.Rect(650, 12, 620, 440), network, obs,
                         self.feature_names, ACTION_NAMES, action, explored)
        draw_chart(screen, pygame.Rect(650, 466, 620, 214), scores, chart_title, window=self.chart_window,
                   slots=self.chart_slots)

        if decision is not None:
            # En la fila del título (y=18): la barra de estadísticas de debajo puede ocupar todo el ancho.
            theme.draw_text(screen, decision[0], (620, 18), 18, decision[1], bold=True, anchor="topright")

        if self.footer is None:
            name, _ = SPEEDS[self.speed_index]
            hint = "PAUSA  |  " if self.paused else ""
            footer = f"{hint}Velocidad: {name}   [↑/↓ cambiar]   [Espacio pausa]   [Esc salir]"
        else:
            footer = self.footer
        theme.draw_text(screen, footer, (30, 696), 14, theme.TEXT_DIM)
        pygame.display.flip()

    def _draw_title(self, screen: pygame.Surface, decision_text: str) -> None:
        """Dibuja el título, encogiéndolo si hiciera falta para no pisar el texto de decisión."""
        limit = 620 - (theme.font(18, True).size(decision_text)[0] + 24 if decision_text else 0)
        size = 24
        while size > 13 and theme.font(size, True).size(self.title)[0] > limit - 30:
            size -= 1
        theme.draw_text(screen, self.title, (30, 14 + (24 - size) // 2), size, bold=True)

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
        self.dashboard.feature_names = trainer.env.feature_names
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


class EvolutionDashboardObserver(EvolutionObserver):
    """Tras cada generación, el CAMPEÓN juega una partida a la vista: se ve cómo, generación a generación,
    la red del mejor individuo va jugando mejor."""

    def __init__(self, dashboard: Dashboard):
        self.dashboard = dashboard

    def on_generation(self, trainer, stats) -> bool:
        dash = self.dashboard
        dash.feature_names = trainer.env_names
        dash.chart_window = 1
        history = trainer.history.generations
        scores = [g.best_score for g in history]

        def stats_bar(game):
            return [("Generación", f"{stats.generation}/{trainer.config.generations}"),
                    ("Manzanas", str(game.score)),
                    ("Mejor (media)", f"{stats.best_score:.1f}"),
                    ("Población", f"{stats.mean_score:.1f}")]

        return play_visible(trainer, stats.generation, lambda game, obs, action: dash.frame(
            game, trainer.champion, obs, action, False, stats_bar(game), scores,
            "Manzanas del mejor, por generación"))
