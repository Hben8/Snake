import numpy as np

from snake_ai.nn import Adam, NeuralNetwork
from snake_ai.rl import DQNAgent, DQNConfig, SnakeEnv, TrainConfig, Trainer
from snake_ai.rl.replay_buffer import ReplayBuffer


def test_backward_matches_numerical_gradient():
    """La retropropagación debe coincidir con el gradiente calculado "a fuerza bruta"."""
    rng = np.random.default_rng(0)
    net = NeuralNetwork([4, 6, 5, 3], seed=0)
    for b in net.biases:  # con sesgos a 0 una neurona puede quedar justo en z=0, el pico de ReLU
        b[...] = rng.normal(size=b.shape)
    x = rng.normal(size=(7, 4)).astype(np.float32)
    target = rng.normal(size=(7, 3)).astype(np.float32)

    def loss() -> float:
        return float(0.5 * np.sum((net.forward(x) - target) ** 2))

    net.forward(x)
    grads_w, grads_b = net.backward(net.forward(x) - target)
    analytic = [g for pair in zip(grads_w, grads_b) for g in pair]

    eps = 1e-3
    for param, grad in zip(net.parameters(), analytic):
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            up = loss()
            param[idx] = original - eps
            down = loss()
            param[idx] = original
            numeric = (up - down) / (2 * eps)
            assert abs(numeric - grad[idx]) < 1e-2 * max(1.0, abs(numeric))


def test_activations_do_not_disturb_backward_cache():
    net = NeuralNetwork([3, 4, 2], seed=0)
    x = np.ones((5, 3), dtype=np.float32)
    net.forward(x)
    net.activations(np.zeros(3))  # visualización en mitad del entrenamiento
    grads_w, _ = net.backward(np.ones((5, 2), dtype=np.float32))
    assert grads_w[0].shape == (3, 4)


def test_adam_fits_a_simple_function():
    net = NeuralNetwork([1, 16, 1], seed=1)
    opt = Adam(net.parameters(), lr=1e-2)
    x = np.linspace(-1, 1, 64, dtype=np.float32)[:, None]
    y = x**2
    for _ in range(600):
        err = net.forward(x) - y
        gw, gb = net.backward(2 * err / len(x))
        opt.step([g for pair in zip(gw, gb) for g in pair])
    assert float(np.mean((net.forward(x) - y) ** 2)) < 0.01


def test_save_and_load_roundtrip(tmp_path):
    net = NeuralNetwork([11, 8, 3], seed=3)
    net.save(tmp_path / "net.npz", width=12, height=9)
    loaded, meta = NeuralNetwork.load(tmp_path / "net.npz")
    x = np.random.default_rng(0).normal(size=11).astype(np.float32)
    assert np.allclose(net.forward(x), loaded.forward(x))
    assert meta == {"width": 12, "height": 9}


def test_replay_buffer_overwrites_oldest():
    buf = ReplayBuffer(3, 2, seed=0)
    for i in range(5):
        buf.add([i, i], 0, float(i), [i, i], False)
    assert len(buf) == 3
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0]


def test_short_training_runs_and_saves(tmp_path):
    env = SnakeEnv(8, 8, seed=0)
    agent = DQNAgent(env.n_inputs, env.n_actions,
                     DQNConfig(hidden=(8, 8), min_buffer=50, batch_size=16), seed=0)
    history = Trainer(env, agent, TrainConfig(episodes=15, out_dir=str(tmp_path), snapshot_every=5,
                                              best_window=5)).run()
    assert len(history.episodes) == 15
    assert (tmp_path / "last.npz").exists() and (tmp_path / "history.csv").exists()
    assert (tmp_path / "snapshots" / "ep_00005.npz").exists()
    assert agent.epsilon < 1.0
