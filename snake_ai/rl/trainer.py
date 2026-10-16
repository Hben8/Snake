"""Bucle de entrenamiento.

El entrenador no sabe nada de gráficos: avisa a unos "observadores" en cada
decisión y al final de cada episodio. La visualización, el log por consola,
etc. son simples observadores enchufados desde fuera.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dqn_agent import DQNAgent
from .environment import SnakeEnv


@dataclass
class EpisodeStats:
    episode: int
    score: int
    reward: float
    steps: int
    epsilon: float
    loss: float
    cause: str | None


@dataclass
class TrainingHistory:
    episodes: list[EpisodeStats] = field(default_factory=list)

    @property
    def scores(self) -> list[int]:
        return [e.score for e in self.episodes]

    def moving_average(self, window: int = 50) -> float:
        recent = self.scores[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "score", "reward", "steps", "epsilon", "loss", "cause"])
            for e in self.episodes:
                writer.writerow([e.episode, e.score, f"{e.reward:.3f}", e.steps,
                                 f"{e.epsilon:.4f}", f"{e.loss:.5f}", e.cause or ""])

    @classmethod
    def load_csv(cls, path: str | Path) -> "TrainingHistory":
        with Path(path).open(newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            return cls([
                EpisodeStats(int(r["episode"]), int(r["score"]), float(r["reward"]), int(r["steps"]),
                             float(r["epsilon"]), float(r["loss"]), r["cause"] or None)
                for r in rows
            ])


class Observer:
    """Plantilla de observador. Los métodos devuelven False para parar el entrenamiento."""

    def on_decision(self, trainer: "Trainer", obs: np.ndarray, action: int, explored: bool) -> bool:
        return True

    def on_episode_end(self, trainer: "Trainer", stats: EpisodeStats) -> bool:
        return True


@dataclass
class TrainConfig:
    episodes: int = 1000
    out_dir: str = "models"
    snapshot_every: int = 100      # guarda una "foto" de la red cada N episodios
    best_window: int = 50          # el mejor modelo se elige por la media de los últimos N episodios


class Trainer:
    def __init__(self, env: SnakeEnv, agent: DQNAgent, config: TrainConfig | None = None,
                 observers: list[Observer] | None = None):
        self.env = env
        self.agent = agent
        self.config = config or TrainConfig()
        self.observers = observers or []
        self.history = TrainingHistory()
        self.episode = 0
        self.total_steps = 0
        self.best_average = -np.inf
        self.record = 0

    def run(self) -> TrainingHistory:
        cfg = self.config
        out = Path(cfg.out_dir)
        meta = {"width": self.env.game.width, "height": self.env.game.height}

        for self.episode in range(1, cfg.episodes + 1):
            stats = self._run_episode()
            self.history.episodes.append(stats)
            self.record = max(self.record, stats.score)
            self.agent.end_episode()

            if self.episode % cfg.snapshot_every == 0:
                self.agent.save(out / "snapshots" / f"ep_{self.episode:05d}.npz", **meta)
            if self.episode >= cfg.best_window:
                average = self.history.moving_average(cfg.best_window)
                if average > self.best_average:
                    self.best_average = average
                    self.agent.save(out / "best.npz", **meta)

            if not all([obs.on_episode_end(self, stats) for obs in self.observers]):
                break

        self.agent.save(out / "last.npz", **meta)
        self.history.save_csv(out / "history.csv")
        return self.history

    def _run_episode(self) -> EpisodeStats:
        obs = self.env.reset()
        done, total_reward, steps, losses = False, 0.0, 0, []
        info = {"score": 0, "cause": None}

        while not done:
            action, explored = self.agent.act(obs)
            if not all([o.on_decision(self, obs, action, explored) for o in self.observers]):
                done = True
                break
            next_obs, reward, done, info = self.env.step(action)
            # "terminó" para el aprendizaje = muerte real. Morir de hambre también cuenta.
            self.agent.remember(obs, action, reward, next_obs, done)
            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            total_reward += reward
            steps += 1
            self.total_steps += 1

        return EpisodeStats(
            episode=self.episode, score=self.env.game.score, reward=total_reward, steps=steps,
            epsilon=self.agent.epsilon, loss=float(np.mean(losses)) if losses else 0.0,
            cause=info["cause"],
        )


class ConsoleLogger(Observer):
    """Imprime una línea de resumen cada `every` episodios."""

    def __init__(self, every: int = 25):
        self.every = every

    def on_episode_end(self, trainer: Trainer, stats: EpisodeStats) -> bool:
        if stats.episode % self.every == 0 or stats.episode == trainer.config.episodes:
            print(
                f"Episodio {stats.episode:>5}/{trainer.config.episodes} | "
                f"media(50): {trainer.history.moving_average(50):5.2f} | "
                f"récord: {trainer.record:>3} | epsilon: {stats.epsilon:.3f} | "
                f"pérdida: {stats.loss:.4f}",
                flush=True,
            )
        return True
