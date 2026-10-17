"""Gráfica de resultados del entrenamiento con matplotlib (para el README)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..rl.evolution import EvolutionHistory  # noqa: E402
from ..rl.trainer import TrainingHistory  # noqa: E402


def plot_history(history: TrainingHistory, out_path: str | Path, window: int = 50) -> None:
    eps = history.episodes
    x = np.array([e.episode for e in eps])
    scores = np.array([e.score for e in eps], dtype=float)
    epsilon = np.array([e.epsilon for e in eps])
    kernel = np.ones(window) / window
    average = np.convolve(scores, kernel, mode="valid") if len(scores) >= window else scores

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0e111a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161a27")

    ax1.plot(x, scores, color="#4664a0", alpha=0.45, linewidth=0.8, label="Puntuación de cada partida")
    ax1.plot(x[window - 1:] if len(scores) >= window else x, average, color="#ffd65a",
             linewidth=2.4, label=f"Media móvil ({window})")
    ax1.set_title("La serpiente aprende: manzanas por partida")
    ax1.set_xlabel("Episodio (partida)")
    ax1.set_ylabel("Manzanas comidas")
    ax1.legend(loc="upper left")

    ax2.plot(x, epsilon, color="#ffa046", linewidth=2)
    ax2.set_title("De probar al azar a decidir con la red")
    ax2.set_xlabel("Episodio (partida)")
    ax2.set_ylabel("ε (probabilidad de jugar al azar)")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Gráfica guardada en {out_path}")


def plot_evolution(history: EvolutionHistory, out_path: str | Path) -> None:
    """Manzanas por generación: la media de toda la población y la del mejor individuo."""
    gens = history.generations
    x = np.array([g.generation for g in gens])
    best = np.array([g.best_score for g in gens])
    mean = np.array([g.mean_score for g in gens])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#0e111a")
    ax.set_facecolor("#161a27")
    ax.plot(x, best, color="#ffd65a", linewidth=2.2, label="El mejor de la generación")
    ax.plot(x, mean, color="#4aa3ff", linewidth=2.0, label="Media de toda la población")
    ax.set_title("La evolución: manzanas por generación (sin retropropagación)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Manzanas comidas (media de sus partidas)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Gráfica guardada en {out_path}")


def plot_eval_curve(csv_path: str | Path, out_path: str | Path) -> None:
    """Manzanas SIN azar medidas cada cierto número de partidas (entrenamiento en paralelo)."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#0e111a")
    ax.set_facecolor("#161a27")
    ax.plot(data["episodes"], data["eval_score"], color="#ffd65a", linewidth=2.2, marker="o", markersize=4)
    ax.set_title("Entrenamiento en paralelo: manzanas sin azar a lo largo del entrenamiento")
    ax.set_xlabel("Partidas de entrenamiento (todas las paralelas)")
    ax.set_ylabel("Manzanas de media (100 partidas sin azar)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Gráfica guardada en {out_path}")
