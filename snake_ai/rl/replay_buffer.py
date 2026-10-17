"""Memoria de experiencias (experience replay).

Guardamos las últimas N transiciones (estado, acción, recompensa, estado
siguiente, ¿terminó?) y entrenamos con muestras ALEATORIAS de ellas. Así la
red no se obsesiona con lo último que ha vivido y aprende de forma más estable.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, n_inputs: int | tuple[int, ...], seed: int | None = None):
        """`n_inputs`: tamaño del vector de entrada, o su forma completa (p. ej. una imagen)."""
        shape = (n_inputs,) if isinstance(n_inputs, int) else tuple(n_inputs)
        self.capacity = capacity
        self.states = np.zeros((capacity, *shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._next = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, state, action, reward, next_state, done) -> None:
        i = self._next
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self._next = (i + 1) % self.capacity  # al llenarse, sobrescribe lo más antiguo
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])
