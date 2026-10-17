"""Neuroevolución: aprender a jugar SIN retropropagación, imitando a la evolución natural.

En vez de ajustar los pesos de UNA red con gradientes (DQN), se mantiene una POBLACIÓN de redes:

    1. Cada red juega unas partidas y recibe una nota (su "fitness").
    2. Las mejores sobreviven tal cual (élite) y las demás se reemplazan por HIJOS:
       se eligen dos padres buenos, se mezclan sus pesos (cruce) y se alteran un poco al
       azar (mutación).
    3. Se repite durante muchas generaciones.

Nadie le dice a la red qué pesos tocar: solo sobreviven las que juegan mejor. La red es exactamente la
misma que en DQN (mismas entradas, misma estructura, mismo formato `.npz`), así que se ve y se compara igual.
"""

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..nn import NeuralNetwork
from .environment import SnakeEnv


@dataclass
class EvolutionConfig:
    generations: int = 150
    population: int = 150
    elite: int = 10               # los N mejores pasan sin cambios a la siguiente generación
    games_per_individual: int = 3  # partidas que juega cada red para recibir su nota
    tournament: int = 4           # para elegir un padre: se sortean N redes y gana la mejor
    mutation_rate: float = 0.05   # probabilidad de que cada peso sufra una mutación
    mutation_sigma: float = 0.2   # tamaño típico de la mutación
    hidden: tuple[int, ...] = (24, 24)
    features: str = "cola"
    width: int = 10
    height: int = 10
    workers: int = 0              # procesos en paralelo (0 = automático)
    out_dir: str = "models"


# ------------------------------------------------------------------ genoma <-> red
def network_to_genome(net: NeuralNetwork) -> np.ndarray:
    """Todos los pesos y sesgos de la red en un único vector (el "ADN")."""
    return np.concatenate([p.ravel() for p in net.parameters()]).astype(np.float32)


def genome_to_network(genome: np.ndarray, sizes: tuple[int, ...]) -> NeuralNetwork:
    net = NeuralNetwork(sizes)
    start = 0
    for p in net.parameters():
        p[...] = genome[start:start + p.size].reshape(p.shape)
        start += p.size
    return net


# ------------------------------------------------------------------ evaluación
def play(net: NeuralNetwork, env: SnakeEnv) -> tuple[float, int]:
    """Una partida completa sin azar. Devuelve (recompensa total, manzanas)."""
    obs, done, total = env.reset(), False, 0.0
    while not done:
        obs, reward, done, _ = env.step(int(np.argmax(net.activations(obs)[-1])))
        total += reward
    return total, env.game.score


def evaluate_genome(job: tuple) -> tuple[float, float]:
    """Nota de un individuo: media de recompensa y de manzanas en varias partidas (con semillas dadas).

    La nota es la MISMA recompensa que usa DQN (+10 comer, -10 morir, ±0,1 acercarse/alejarse).
    Es una función de nivel superior para poder repartirla entre procesos.
    """
    genome, sizes, features, width, height, seeds = job
    net = genome_to_network(genome, sizes)
    rewards, scores = [], []
    for seed in seeds:
        reward, score = play(net, SnakeEnv(width, height, seed=seed, features=features))
        rewards.append(reward)
        scores.append(score)
    return float(np.mean(rewards)), float(np.mean(scores))


# ------------------------------------------------------------------ historial
@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_score: float   # manzanas de media del mejor individuo
    mean_score: float   # manzanas de media de toda la población


@dataclass
class EvolutionHistory:
    generations: list[GenerationStats] = field(default_factory=list)

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "mean_fitness", "best_score", "mean_score"])
            for g in self.generations:
                writer.writerow([g.generation, f"{g.best_fitness:.3f}", f"{g.mean_fitness:.3f}",
                                 f"{g.best_score:.3f}", f"{g.mean_score:.3f}"])

    @classmethod
    def load_csv(cls, path: str | Path) -> "EvolutionHistory":
        with Path(path).open(newline="", encoding="utf-8") as f:
            return cls([GenerationStats(int(r["generation"]), float(r["best_fitness"]), float(r["mean_fitness"]),
                                        float(r["best_score"]), float(r["mean_score"]))
                        for r in csv.DictReader(f)])


class EvolutionObserver:
    """Plantilla de observador. `on_generation` devuelve False para parar."""

    def on_generation(self, trainer: "EvolutionTrainer", stats: GenerationStats) -> bool:
        return True


class EvolutionConsoleLogger(EvolutionObserver):
    def __init__(self, every: int = 5):
        self.every = every

    def on_generation(self, trainer, stats) -> bool:
        if stats.generation == 1 or stats.generation % self.every == 0:
            print(f"Generación {stats.generation:4d}/{trainer.config.generations} | "
                  f"mejor: {stats.best_score:5.1f} manzanas | media población: {stats.mean_score:5.2f} | "
                  f"nota del mejor: {stats.best_fitness:7.2f}", flush=True)
        return True


# ------------------------------------------------------------------ el algoritmo genético
class EvolutionTrainer:
    def __init__(self, config: EvolutionConfig | None = None, observers: list[EvolutionObserver] | None = None,
                 seed: int | None = None):
        self.config = config or EvolutionConfig()
        self.observers = observers or []
        self.rng = np.random.default_rng(seed)
        env = SnakeEnv(self.config.width, self.config.height, features=self.config.features)
        self.sizes = (env.n_inputs, *self.config.hidden, env.n_actions)
        self.env_names = env.feature_names
        self.history = EvolutionHistory()
        self.generation = 0
        self.champion: NeuralNetwork | None = None  # el mejor de la última generación
        # Población inicial: redes aleatorias (inicialización de He, igual que en DQN).
        self.population = [network_to_genome(NeuralNetwork(self.sizes, seed=int(self.rng.integers(2**31))))
                           for _ in range(self.config.population)]
        self.fitness = np.zeros(self.config.population)

    # -------------------------------------------------------------- una generación
    def _evaluate(self, pool: ProcessPoolExecutor | None) -> np.ndarray:
        cfg = self.config
        # Todos juegan las MISMAS partidas (mismas semillas): la comparación es justa. Cambian cada generación
        # para que nadie se especialice en unas manzanas concretas.
        seeds = [int(s) for s in self.rng.integers(2**31, size=cfg.games_per_individual)]
        jobs = [(g, self.sizes, cfg.features, cfg.width, cfg.height, seeds) for g in self.population]
        results = list(pool.map(evaluate_genome, jobs, chunksize=4)) if pool else [evaluate_genome(j) for j in jobs]
        self.fitness = np.array([r[0] for r in results])
        return np.array([r[1] for r in results])

    def _tournament(self) -> np.ndarray:
        picks = self.rng.integers(len(self.population), size=self.config.tournament)
        return self.population[int(picks[np.argmax(self.fitness[picks])])]

    def _child(self) -> np.ndarray:
        a, b = self._tournament(), self._tournament()
        child = np.where(self.rng.random(a.shape) < 0.5, a, b)  # cruce: cada peso viene de un padre u otro
        mutate = self.rng.random(child.shape) < self.config.mutation_rate
        child = child + mutate * self.rng.normal(0.0, self.config.mutation_sigma, child.shape)
        return child.astype(np.float32)

    def _next_generation(self) -> None:
        order = np.argsort(-self.fitness)
        elite = [self.population[i] for i in order[:self.config.elite]]
        children = [self._child() for _ in range(self.config.population - len(elite))]
        self.population = elite + children

    def run(self) -> EvolutionHistory:
        cfg = self.config
        workers = cfg.workers or max(1, min(12, (__import__("os").cpu_count() or 2) - 2))
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        out = Path(cfg.out_dir)
        meta = {"width": cfg.width, "height": cfg.height, "features": cfg.features}
        try:
            for self.generation in range(1, cfg.generations + 1):
                scores = self._evaluate(pool)
                best = int(np.argmax(self.fitness))
                self.champion = genome_to_network(self.population[best], self.sizes)
                stats = GenerationStats(self.generation, float(self.fitness[best]), float(self.fitness.mean()),
                                        float(scores[best]), float(scores.mean()))
                self.history.generations.append(stats)
                if self.generation % 25 == 0:
                    self.champion.save(out / "snapshots" / f"gen_{self.generation:04d}.npz", **meta)
                keep_going = all([obs.on_generation(self, stats) for obs in self.observers])
                if not keep_going:
                    break
                if self.generation < cfg.generations:
                    self._next_generation()
        finally:
            if pool:
                pool.shutdown()
        self.champion.save(out / "last.npz", **meta)
        self.history.save_csv(out / "history.csv")
        return self.history

    def best_individuals(self, n: int = 5) -> list[NeuralNetwork]:
        """Los n mejores de la última generación (para elegir el definitivo con más partidas)."""
        order = np.argsort(-self.fitness)[:n]
        return [genome_to_network(self.population[i], self.sizes) for i in order]


def play_visible(trainer: EvolutionTrainer, seed: int, on_frame) -> bool:
    """Juega UNA partida del campeón, avisando a `on_frame(juego, obs, acción)` en cada paso.
    Devuelve False si `on_frame` pide parar (p. ej. se cerró la ventana)."""
    cfg = trainer.config
    env = SnakeEnv(cfg.width, cfg.height, seed=seed, features=cfg.features)
    obs, done = env.reset(), False
    while not done:
        action = int(np.argmax(trainer.champion.activations(obs)[-1]))
        if not on_frame(env.game, obs, action):
            return False
        obs, _, done, _ = env.step(action)
    return True
