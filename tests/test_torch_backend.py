import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_ai.nn import NeuralNetwork  # noqa: E402
from snake_ai.rl import DQNConfig, SnakeEnv, TrainConfig, Trainer  # noqa: E402
from snake_ai.rl.torch_agent import QNetwork, TorchDQNAgent, resolve_device  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_numpy_and_torch_networks_agree():
    """Con los mismos pesos, las dos implementaciones calculan exactamente lo mismo."""
    np_net = NeuralNetwork([11, 8, 8, 3], seed=0)
    for b in np_net.biases:
        b[...] = np.random.default_rng(1).normal(size=b.shape)
    t_net = QNetwork([11, 8, 8, 3])
    with torch.no_grad():
        for layer, w, b in zip(t_net.layers, np_net.weights, np_net.biases):
            layer.weight.copy_(torch.from_numpy(w.T.copy()))
            layer.bias.copy_(torch.from_numpy(b))
    x = np.random.default_rng(2).normal(size=11).astype(np.float32)
    for a, b in zip(np_net.activations(x), t_net.activations(x)):
        assert np.allclose(a, b, atol=1e-5)
    assert all(np.allclose(a, b) for a, b in zip(np_net.weights, t_net.weights))


def test_torch_model_loads_in_numpy_version(tmp_path):
    net = QNetwork([11, 6, 3])
    net.save(tmp_path / "m.npz", width=10, height=10)
    loaded, meta = NeuralNetwork.load(tmp_path / "m.npz")
    x = np.random.default_rng(0).normal(size=11).astype(np.float32)
    assert np.allclose(net.activations(x)[-1], loaded.forward(x), atol=1e-5)
    assert meta == {"width": 10, "height": 10}


@pytest.mark.parametrize("device", DEVICES)
def test_short_training_runs_on_device(tmp_path, device):
    env = SnakeEnv(8, 8, seed=0)
    agent = TorchDQNAgent(env.n_inputs, env.n_actions,
                          DQNConfig(hidden=(8, 8), min_buffer=50, batch_size=16),
                          seed=0, device=device)
    assert agent.device.type == device
    history = Trainer(env, agent, TrainConfig(episodes=10, out_dir=str(tmp_path), snapshot_every=5,
                                              best_window=5)).run()
    assert len(history.episodes) == 10 and (tmp_path / "last.npz").exists()


def test_learning_moves_weights_and_syncs_target():
    env = SnakeEnv(8, 8, seed=0)
    agent = TorchDQNAgent(env.n_inputs, env.n_actions,
                          DQNConfig(hidden=(8,), min_buffer=20, batch_size=8, target_sync_every=5),
                          seed=0, device="cpu")
    before = [w.copy() for w in agent.online.weights]
    obs = env.reset()
    for _ in range(60):
        action, _ = agent.act(obs)
        nxt, reward, done, _ = env.step(action)
        agent.remember(obs, action, reward, nxt, done)
        agent.learn()
        obs = env.reset() if done else nxt
    assert any(not np.allclose(a, b) for a, b in zip(before, agent.online.weights))
    assert agent.learn_steps > 5


def test_cuda_request_without_gpu_gives_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    assert resolve_device("auto").type == "cpu"
