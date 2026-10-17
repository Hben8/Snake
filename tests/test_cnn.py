import numpy as np
import pytest

from snake_ai.game import Action, Direction, SnakeGame
from snake_ai.rl import SnakeEnv
from snake_ai.rl.observation import observe_image
from snake_ai.rl.replay_buffer import ReplayBuffer


def test_image_is_rotated_so_the_snake_always_looks_up():
    for direction in Direction:
        game = SnakeGame(10, 10, seed=1)
        game.snake.body.clear()
        game.snake.body.extend([(5, 5), (5 - direction.vector[0], 5 - direction.vector[1])])
        game.snake.direction = direction
        # La comida justo delante de la cabeza, sea cual sea hacia dónde mire.
        game.food = (5 + direction.vector[0], 5 + direction.vector[1])
        img = observe_image(game)
        assert img.shape == (4, 12, 12)
        (hy, hx), (fy, fx) = np.argwhere(img[2] == 1)[0], np.argwhere(img[3] == 1)[0]
        assert (fy, fx) == (hy - 1, hx), f"con la serpiente mirando a {direction.name} la comida debe quedar arriba"
        assert img[1][hy + 1, hx] > 0  # el cuerpo queda justo detrás (abajo)


def test_image_has_wall_border_and_body_ages():
    game = SnakeGame(10, 10, seed=1)
    img = observe_image(game)
    assert img[0].sum() == 12 * 4 - 4  # el marco de pared
    body = img[1][img[1] > 0]
    assert sorted(body) == pytest.approx([1 / 3, 2 / 3])  # cola = 1/3 (se aparta antes), cuello = 2/3


def test_image_env_needs_square_board_and_has_obs_shape():
    env = SnakeEnv(8, 8, seed=0, features="imagen")
    assert env.reset().shape == env.obs_shape == (4, 10, 10)
    with pytest.raises(ValueError):
        SnakeEnv(8, 10, features="imagen")


def test_replay_buffer_stores_images():
    buf = ReplayBuffer(10, (4, 6, 6))
    state = np.ones((4, 6, 6), dtype=np.float32)
    buf.add(state, 1, 0.5, state * 2, False)
    s, a, r, ns, d = buf.sample(3)
    assert s.shape == ns.shape == (3, 4, 6, 6) and ns.max() == 2


torch = pytest.importorskip("torch")

from snake_ai.rl import DQNConfig  # noqa: E402
from snake_ai.rl.model_io import load_model  # noqa: E402
from snake_ai.rl.torch_agent import ConvQNetwork, TorchDQNAgent  # noqa: E402


def test_conv_network_activations_and_save_load_roundtrip(tmp_path):
    net = ConvQNetwork((4, 12, 12), 3)
    x = np.random.default_rng(0).random((4, 12, 12)).astype(np.float32)
    layers = net.activations(x)
    assert [a.shape for a in layers] == [(4, 12, 12), (16, 12, 12), (32, 12, 12), (128,), (3,)]
    net.save(tmp_path / "cnn.npz", width=10, height=10, features="imagen")
    loaded, meta = load_model(tmp_path / "cnn.npz")
    assert loaded.is_conv and meta["features"] == "imagen" and meta["width"] == 10
    assert np.allclose(loaded.activations(x)[-1], layers[-1], atol=1e-5)


def test_cnn_agent_learns_a_batch_without_errors():
    env = SnakeEnv(10, 10, seed=0, features="imagen")
    agent = TorchDQNAgent(env.n_inputs, env.n_actions, DQNConfig(batch_size=8, min_buffer=8), seed=0,
                          device="cpu", obs_shape=env.obs_shape)
    obs = env.reset()
    for _ in range(12):
        action, _ = agent.act(obs)
        nxt, reward, done, _ = env.step(action)
        agent.remember(obs, action, reward, nxt, done)
        obs = env.reset() if done else nxt
    assert agent.learn() is not None
