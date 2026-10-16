"""Compara la velocidad de entrenamiento: NumPy (CPU) vs PyTorch (CPU) vs PyTorch (GPU).

Mide "pasos de aprendizaje por segundo": cada paso es jugar un turno + guardar la
experiencia + un paso de aprendizaje. Uso:  python scripts/benchmark_backends.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from snake_ai.cli import make_agent  # noqa: E402
from snake_ai.rl import DQNConfig, SnakeEnv  # noqa: E402

STEPS = 3000
CASES = [  # (nombre, hidden, batch)
    ("Pequeña (la del proyecto)", (24, 24), 64),
    ("Mediana", (256, 256), 256),
    ("Grande", (1024, 1024, 1024), 1024),
]


def run(backend: str, device: str, hidden, batch) -> float:
    env = SnakeEnv(10, 10, seed=0)
    config = DQNConfig(hidden=hidden, batch_size=batch, min_buffer=batch)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        agent = make_agent(backend, env, config, seed=0, device=device)
    obs = env.reset()
    for i in range(STEPS + 300):
        if i == 300:  # las primeras iteraciones son de calentamiento
            if device != "cpu" and backend == "torch":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        action, _ = agent.act(obs)
        nxt, reward, done, _ = env.step(action)
        agent.remember(obs, action, reward, nxt, done)
        agent.learn()
        obs = env.reset() if done else nxt
    if backend == "torch" and device != "cpu":
        torch.cuda.synchronize()
    return STEPS / (time.perf_counter() - t0)


def main() -> None:
    options = [("NumPy CPU", "numpy", "cpu"), ("PyTorch CPU", "torch", "cpu")]
    if torch.cuda.is_available():
        options.append((f"PyTorch GPU", "torch", "cuda"))
    print(f"{'Red':<28}" + "".join(f"{name:>16}" for name, *_ in options) + "   (pasos/segundo)")
    for label, hidden, batch in CASES:
        row = [run(b, d, hidden, batch) for _, b, d in options]
        print(f"{label:<28}" + "".join(f"{v:>16.0f}" for v in row))


if __name__ == "__main__":
    main()
