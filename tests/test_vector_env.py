from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_ai.game import SnakeGame  # noqa: E402
from snake_ai.nn import NeuralNetwork  # noqa: E402
from snake_ai.rl import SnakeEnv  # noqa: E402
from snake_ai.rl.observation import observe  # noqa: E402
from snake_ai.rl.vector_env import VectorSnakeEnv  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MODEL = Path(__file__).resolve().parents[1] / "models" / "pretrained_cola.npz"


def _cpu_envs(n, seed):
    return [SnakeEnv(10, 10, seed=seed + i, features="cola") for i in range(n)]


@pytest.mark.parametrize("device", DEVICES)
def test_vector_env_matches_the_normal_game_step_by_step(device):
    """Mismas reglas, recompensas y pistas que el juego normal, durante cientos de pasos con serpientes largas."""
    n = 24
    net, _ = NeuralNetwork.load(MODEL)
    rng = np.random.default_rng(0)
    cpu = _cpu_envs(n, 100)
    venv = VectorSnakeEnv(n, 10, 10, device=device, seed=0)
    for e in cpu:
        e.reset()
    venv.load_from_games([e.game for e in cpu])
    checked_long = 0
    for step in range(250):
        obs_cpu = np.stack([observe(e.game, "cola") for e in cpu])
        assert np.allclose(venv.observe().cpu().numpy(), obs_cpu, atol=1e-6), f"pistas distintas en el paso {step}"
        actions = [int(np.argmax(net.activations(o)[-1])) if rng.random() > 0.02 else int(rng.integers(3))
                   for o in obs_cpu]
        results = [e.step(a) for e, a in zip(cpu, actions)]
        _, reward, done, _ = venv.step(torch.tensor(actions, device=device))
        for i, (e, (_, r, d, _)) in enumerate(zip(cpu, results)):
            assert bool(done[i]) == d and abs(float(reward[i]) - r) < 1e-5, f"partida {i}, paso {step}"
        checked_long += sum(len(e.game.snake) > 15 for e in cpu)
        # Las que terminaron ya se reiniciaron en venv: se reinician también en la CPU y se vuelve a copiar todo.
        for e in cpu:
            if e.game.done:
                e.reset()
        venv.load_from_games([e.game for e in cpu])
    assert checked_long > 150  # sí se probaron serpientes largas, no solo los primeros pasos


@pytest.mark.parametrize("device", DEVICES)
def test_vector_env_runs_many_games_and_reports_finished_ones(device):
    venv = VectorSnakeEnv(64, 10, 10, device=device, seed=1)
    finished = 0
    for _ in range(300):
        obs, reward, done, info = venv.step(torch.randint(0, 3, (64,), device=device))
        assert obs.shape == (64, 17) and reward.shape == done.shape == (64,)
        assert info["score"].shape[0] == int(done.sum())
        finished += int(done.sum())
    assert finished > 64  # con acciones al azar mueren enseguida y se reinician solas
    assert bool((venv.ttl.flatten(1).max(1).values == venv.length).all())  # la cabeza vale la longitud


@pytest.mark.parametrize("device", DEVICES)
def test_parallel_trainer_runs_and_saves_a_model_watch_can_load(device, tmp_path):
    from snake_ai.rl.model_io import load_model
    from snake_ai.rl.parallel_trainer import ParallelConfig, ParallelTrainer
    cfg = ParallelConfig(episodes=300, n_envs=32, min_buffer=200, batch_size=64, eval_every=150, eval_games=16,
                         buffer_size=5000, device=device, out_dir=str(tmp_path))
    trainer = ParallelTrainer(cfg, seed=0)
    history = trainer.run(log=lambda msg: None)
    assert len(history.episodes) >= 300
    assert (tmp_path / "history.csv").exists() and (tmp_path / "last.npz").exists()
    net, meta = load_model(tmp_path / "best.npz")  # mismo formato que el resto de modelos
    assert net.sizes == (17, 24, 24, 3) and meta["features"] == "cola"
