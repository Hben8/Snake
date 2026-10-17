"""Ver jugar a una red ya entrenada (sin azar, sin aprender)."""

from pathlib import Path

import numpy as np

from .rl import DEFAULT_FEATURES, FEATURE_SETS, SnakeEnv
from .rl.model_io import load_model
from .viz.dashboard import Dashboard
from .viz.recorder import GifRecorder


def run_watch(model_path: str | Path, width: int | None, height: int | None, games: int,
              speed: str, record: str | None, seed: int | None) -> None:
    network, meta = load_model(model_path)
    # Cada modelo guarda qué pistas usa (los antiguos no lo dicen: son las "basico").
    features = str(meta.get("features", DEFAULT_FEATURES))
    n_inputs = int(np.prod(network.obs_shape)) if getattr(network, "is_conv", False) else network.sizes[0]
    if features not in FEATURE_SETS or (features != "imagen" and len(FEATURE_SETS[features]) != n_inputs):
        raise SystemExit(
            f"Este modelo espera {n_inputs} entradas y no coincide con ningún juego de pistas "
            f"conocido ({', '.join(f'{k}={len(v)}' for k, v in FEATURE_SETS.items())}). Es de una versión "
            "anterior: vuelve a entrenarlo con  python main.py train  o usa models/pretrained.npz."
        )
    # Si no se indica, se usa el tamaño de tablero con el que se entrenó.
    width = width or int(meta.get("width", 10))
    height = height or int(meta.get("height", 10))
    env = SnakeEnv(width, height, seed=seed, features=features)

    recorder = GifRecorder(record, fps=15, scale=0.45, max_frames=150) if record else None
    dash = Dashboard(f"Snake IA - {Path(model_path).name}", speed=speed, recorder=recorder)
    dash.feature_names = env.feature_names
    scores: list[int] = []
    try:
        for game_number in range(1, games + 1):
            obs, done = env.reset(), False
            while not done:
                action = int(np.argmax(network.activations(obs)[-1]))
                stats = [
                    ("Partida", f"{game_number}/{games}"),
                    ("Manzanas", str(env.game.score)),
                    ("Récord", str(max(scores + [env.game.score]))),
                    ("Media", f"{np.mean(scores):.1f}" if scores else "-"),
                ]
                if not dash.frame(env.game, network, obs, action, False, stats, scores,
                                  "Puntuación de cada partida"):
                    return
                obs, _, done, _ = env.step(action)
            scores.append(env.game.score)
            print(f"Partida {game_number}: {env.game.score} manzanas ({env.game.death_cause})")
        print(f"Media: {np.mean(scores):.2f} | Récord: {max(scores)}")
    finally:
        dash.close()
