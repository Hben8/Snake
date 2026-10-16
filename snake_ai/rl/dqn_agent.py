"""Agente DQN (Deep Q-Network) con red objetivo y Double DQN.

Idea en una frase: la red estima, para cada acción posible, cuánta recompensa
total futura espera conseguir (el "valor Q"). Para jugar elige la acción con
mayor Q. Para aprender, ajusta sus estimaciones hacia

        Q(s, a)  ≈  recompensa + γ · mejor Q(s')

es decir: lo que ganas ahora más lo que esperas ganar después.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..nn import Adam, NeuralNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (24, 24)
    lr: float = 1e-3
    gamma: float = 0.95              # cuánto valora el futuro frente al presente
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 1_000          # experiencias mínimas antes de empezar a aprender
    epsilon_start: float = 1.0       # al principio todo es azar...
    epsilon_end: float = 0.01        # ...y al final casi todo es decisión de la red
    epsilon_decay: float = 0.995     # multiplicador por episodio
    target_sync_every: int = 500     # cada cuántos pasos se actualiza la red objetivo


class DQNAgent:
    def __init__(self, n_inputs: int, n_actions: int, config: DQNConfig | None = None,
                 seed: int | None = None):
        self.config = config or DQNConfig()
        self.n_actions = n_actions
        sizes = [n_inputs, *self.config.hidden, n_actions]
        self.online = NeuralNetwork(sizes, seed=seed)    # la red que juega y aprende
        self.target = NeuralNetwork(sizes, seed=seed)    # copia "congelada" que da objetivos estables
        self.target.copy_from(self.online)
        self.optimizer = Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_size, n_inputs, seed=seed)
        self.epsilon = self.config.epsilon_start
        self.learn_steps = 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ actuar
    def act(self, obs: np.ndarray, explore: bool = True) -> tuple[int, bool]:
        """Devuelve (acción, ¿fue exploración al azar?)."""
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions)), True
        return int(np.argmax(self.online.activations(obs)[-1])), False

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    # ----------------------------------------------------------------- aprender
    def learn(self) -> float | None:
        """Un paso de entrenamiento con un lote aleatorio. Devuelve la pérdida."""
        cfg = self.config
        if len(self.buffer) < max(cfg.min_buffer, cfg.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)
        rows = np.arange(cfg.batch_size)

        # Double DQN: la red online ELIGE la mejor acción futura y la objetivo la EVALÚA.
        best_next = np.argmax(self.online.forward(next_states), axis=1)
        next_q = self.target.forward(next_states)[rows, best_next]
        targets = rewards + cfg.gamma * next_q * (1.0 - dones)

        q = self.online.forward(states)  # (debe ir el último: guarda la caché de backward)
        error = q[rows, actions] - targets

        # Pérdida de Huber: cuadrática para errores pequeños, lineal para grandes.
        loss = float(np.mean(np.where(np.abs(error) <= 1, 0.5 * error**2, np.abs(error) - 0.5)))
        grad = np.zeros_like(q)
        grad[rows, actions] = np.clip(error, -1.0, 1.0) / cfg.batch_size

        grads_w, grads_b = self.online.backward(grad)
        flat = [g for pair in zip(grads_w, grads_b) for g in pair]
        self.optimizer.step(flat)

        self.learn_steps += 1
        if self.learn_steps % cfg.target_sync_every == 0:
            self.target.copy_from(self.online)
        return loss

    def end_episode(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    # --------------------------------------------------------------- persistencia
    def save(self, path: str | Path, **meta) -> None:
        self.online.save(path, **meta)
