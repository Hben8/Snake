"""Versión PyTorch del agente DQN (con soporte para GPU/CUDA).

Es el mismo algoritmo que `dqn_agent.py`, pero la red, la retropropagación y
Adam los hace PyTorch (`autograd`) y todo el cálculo pesado puede ejecutarse en
la tarjeta gráfica. Tiene la misma interfaz que `DQNAgent`, así que el
`Trainer` y la visualización funcionan igual con cualquiera de los dos.

Los modelos se guardan en el mismo formato .npz que la versión NumPy, de modo
que una red entrenada aquí se puede ver con `python main.py watch` y viceversa.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .dqn_agent import DQNConfig
from .replay_buffer import ReplayBuffer


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' usa la GPU si hay una disponible; 'cuda' falla con un mensaje claro si no."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Has pedido CUDA pero PyTorch no ve ninguna GPU. Probablemente tienes instalada la versión "
            "para CPU. Reinstala con:  pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
    return torch.device(name)


class QNetwork(nn.Module):
    """El mismo MLP que `nn.NeuralNetwork`, pero con capas de PyTorch.

    Expone `activations()` y `weights` con el mismo formato que la versión
    NumPy, para que la ventana de visualización la dibuje sin cambios.
    """

    def __init__(self, sizes: list[int] | tuple[int, ...]):
        super().__init__()
        self.sizes = tuple(int(s) for s in sizes)
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(self.sizes[:-1], self.sizes[1:]))
        for layer in self.layers:  # misma inicialización que la versión NumPy (He, sesgos a 0)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    @property
    def device(self) -> torch.device:
        return self.layers[0].weight.device

    # ---- interfaz común con NeuralNetwork (para visualizar) ----
    @torch.no_grad()
    def activations(self, x: np.ndarray) -> list[np.ndarray]:
        a = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        layers = [a]
        for layer in self.layers[:-1]:
            a = F.relu(layer(a))
            layers.append(a)
        layers.append(self.layers[-1](a))
        return [t.cpu().numpy() for t in layers]

    @property
    def weights(self) -> list[np.ndarray]:
        """Pesos como matrices (entradas, salidas), igual que en `NeuralNetwork`."""
        return [layer.weight.detach().cpu().numpy().T for layer in self.layers]

    def save(self, path: str | Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"sizes": np.array(self.sizes)}
        for i, layer in enumerate(self.layers):
            data[f"W{i}"] = layer.weight.detach().cpu().numpy().T.copy()
            data[f"b{i}"] = layer.bias.detach().cpu().numpy().copy()
        data.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
        np.savez(path, **data)


class TorchDQNAgent:
    def __init__(self, n_inputs: int, n_actions: int, config: DQNConfig | None = None,
                 seed: int | None = None, device: str = "auto", loss_log_every: int = 50):
        self.config = config or DQNConfig()
        self.n_actions = n_actions
        self.device = resolve_device(device)
        if seed is not None:
            torch.manual_seed(seed)

        sizes = [n_inputs, *self.config.hidden, n_actions]
        self.online = QNetwork(sizes).to(self.device)
        self.target = QNetwork(sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)

        # La memoria vive en la CPU (añadir 1 experiencia por paso a la GPU saldría más caro);
        # lo que viaja a la GPU es el lote entero, de una vez.
        self.buffer = ReplayBuffer(self.config.buffer_size, n_inputs, seed=seed)
        self.epsilon = self.config.epsilon_start
        self.learn_steps = 0
        self._rng = np.random.default_rng(seed)

        # `loss.item()` obliga a la CPU a esperar a la GPU; solo se hace cada N pasos.
        self._loss_log_every = loss_log_every
        self._last_loss = 0.0

    # ------------------------------------------------------------------ actuar
    def act(self, obs: np.ndarray, explore: bool = True) -> tuple[int, bool]:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions)), True
        with torch.no_grad():
            q = self.online(torch.as_tensor(obs, device=self.device))
            return int(q.argmax().item()), False

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    # ----------------------------------------------------------------- aprender
    def learn(self) -> float | None:
        cfg = self.config
        if len(self.buffer) < max(cfg.min_buffer, cfg.batch_size):
            return None

        # Un solo viaje CPU -> GPU con todo el lote empaquetado (varias copias pequeñas serían más lentas).
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)
        n = states.shape[1]
        packed = np.concatenate([states, next_states, actions[:, None], rewards[:, None], dones[:, None]],
                                axis=1).astype(np.float32, copy=False)
        packed = torch.as_tensor(packed).to(self.device, non_blocking=True)
        s, ns = packed[:, :n], packed[:, n:2 * n]
        a, r, d = packed[:, 2 * n].long(), packed[:, 2 * n + 1], packed[:, 2 * n + 2]

        with torch.no_grad():
            # Double DQN: la red online ELIGE la mejor acción futura y la objetivo la EVALÚA.
            best_next = self.online(ns).argmax(dim=1, keepdim=True)
            next_q = self.target(ns).gather(1, best_next).squeeze(1)
            targets = r + cfg.gamma * next_q * (1.0 - d)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, targets)  # pérdida de Huber

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # retropropagación automática (autograd)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % cfg.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        if self.learn_steps % self._loss_log_every == 0:
            self._last_loss = float(loss.item())
        return self._last_loss

    def end_episode(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    # --------------------------------------------------------------- persistencia
    def save(self, path: str | Path, **meta) -> None:
        self.online.save(path, **meta)
