"""Entorno de aprendizaje: envuelve el juego con la interfaz clásica de RL
(`reset()` / `step(accion)`) y decide las RECOMPENSAS."""

from dataclasses import dataclass

import numpy as np

from ..game import Action, SnakeGame
from .observation import N_FEATURES, observe


@dataclass
class RewardConfig:
    eat: float = 10.0       # comer una manzana
    die: float = -10.0      # chocar o morir de hambre
    closer: float = 0.1     # acercarse a la manzana (pequeña "pista")
    farther: float = -0.1   # alejarse de ella


class SnakeEnv:
    n_inputs = N_FEATURES
    n_actions = len(Action)

    def __init__(self, width: int = 10, height: int = 10, seed: int | None = None,
                 rewards: RewardConfig | None = None):
        self.game = SnakeGame(width, height, seed=seed)
        self.rewards = rewards or RewardConfig()

    def reset(self) -> np.ndarray:
        self.game.reset()
        return observe(self.game)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        before = self._distance_to_food()
        outcome = self.game.step(Action(action))

        if outcome.died:
            reward = self.rewards.die
        elif outcome.ate:
            reward = self.rewards.eat
        else:
            reward = self.rewards.closer if self._distance_to_food() < before else self.rewards.farther

        info = {"score": self.game.score, "cause": outcome.cause, "won": outcome.won}
        return observe(self.game), reward, self.game.done, info

    def _distance_to_food(self) -> int:
        if self.game.food is None:
            return 0
        (hx, hy), (fx, fy) = self.game.snake.head, self.game.food
        return abs(hx - fx) + abs(hy - fy)
