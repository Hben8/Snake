"""DQN con cientos de partidas en paralelo, todo dentro de la GPU.

Es el mismo algoritmo (ε-greedy, memoria de experiencias, red objetivo, Double DQN, pérdida de Huber), pero:

  * `VectorSnakeEnv` juega N partidas a la vez en la GPU (en vez de una en la CPU);
  * en cada paso, la red decide las N jugadas de una sola vez (un lote);
  * la memoria de experiencias también vive en la GPU: nada viaja entre CPU y GPU durante el entrenamiento.

Resultado: se generan cientos de veces más experiencia por segundo, que es lo que más le falta a una red que
aprende de partidas. Guarda los modelos en el mismo formato `.npz`: se ven con `python main.py watch`.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .torch_agent import QNetwork, resolve_device
from .trainer import EpisodeStats, TrainingHistory
from .vector_env import VectorSnakeEnv

_CAUSES = ("wall", "body", "starved", "won")


@dataclass
class ParallelConfig:
    episodes: int = 20000            # partidas completadas EN TOTAL (entre todas las paralelas)
    n_envs: int = 256                # partidas simultáneas
    hidden: tuple[int, ...] = (24, 24)
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 256
    updates_per_step: int = 2        # pasos de aprendizaje por cada paso de TODAS las partidas
    buffer_size: int = 200_000
    min_buffer: int = 5_000
    target_sync_every: int = 500     # pasos de aprendizaje entre copias a la red objetivo
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_steps: int = 300_000     # jugadas (entre todas las partidas) que tarda el azar en bajar a epsilon_end
    eval_every: int = 1000           # cada N partidas se mide la red SIN azar (para elegir la mejor)
    eval_games: int = 200
    width: int = 10
    height: int = 10
    device: str = "auto"
    out_dir: str = "models"


class GPUReplayBuffer:
    """Memoria de experiencias que vive en la GPU (guarda lotes enteros de una vez)."""

    def __init__(self, capacity: int, n_inputs: int, device: torch.device):
        self.capacity, self.device = capacity, device
        self.states = torch.zeros(capacity, n_inputs, device=device)
        self.next_states = torch.zeros(capacity, n_inputs, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.next_slot, self.size = 0, 0

    def add(self, s, a, r, ns, d) -> None:
        n = s.shape[0]
        idx = (self.next_slot + torch.arange(n, device=self.device)) % self.capacity  # al llenarse, sobrescribe lo antiguo
        self.states[idx], self.actions[idx], self.rewards[idx] = s, a, r
        self.next_states[idx], self.dones[idx] = ns, d.float()
        self.next_slot = (self.next_slot + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]


def evaluate(net: QNetwork, games: int, width: int, height: int, device: torch.device, seed: int = 12345) -> float:
    """Manzanas de media jugando SIN azar: cada partida paralela cuenta solo su primera partida completa."""
    env = VectorSnakeEnv(games, width, height, device=device, seed=seed)
    finished = torch.zeros(games, dtype=torch.bool, device=device)
    scores = torch.zeros(games, dtype=torch.long, device=device)
    obs = env.observe()
    with torch.no_grad():
        for _ in range(20000):  # tope de seguridad: el límite de hambre hace que toda partida termine
            obs, _, done, info = env.step(net(obs).argmax(1))
            final = torch.zeros_like(scores)
            final[done] = info["score"]  # puntuación final de las que acaban de terminar
            scores = torch.where(done & ~finished, final, scores)
            finished |= done
            if bool(finished.all()):
                break
    return float(scores.float().mean())


class ParallelTrainer:
    def __init__(self, config: ParallelConfig | None = None, seed: int | None = None):
        self.config = cfg = config or ParallelConfig()
        self.device = resolve_device(cfg.device)
        if seed is not None:
            torch.manual_seed(seed)
        self.env = VectorSnakeEnv(cfg.n_envs, cfg.width, cfg.height, device=self.device, seed=seed)
        sizes = [self.env.n_inputs, *cfg.hidden, self.env.n_actions]
        self.online = QNetwork(sizes).to(self.device)
        self.target = QNetwork(sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = GPUReplayBuffer(cfg.buffer_size, self.env.n_inputs, self.device)
        self.history = TrainingHistory()
        self.learn_steps = 0
        self.total_transitions = 0
        self.epsilon = cfg.epsilon_start
        self._loss = torch.zeros((), device=self.device)
        self.best_eval = -np.inf
        self.eval_history: list[tuple[int, float]] = []

    # ---------------------------------------------------------------- aprender
    def _learn(self) -> None:
        cfg = self.config
        s, a, r, ns, d = self.buffer.sample(cfg.batch_size)
        with torch.no_grad():  # Double DQN: la red online ELIGE la mejor acción futura y la objetivo la EVALÚA
            best_next = self.online(ns).argmax(1, keepdim=True)
            next_q = self.target(ns).gather(1, best_next).squeeze(1)
            targets = r + cfg.gamma * next_q * (1.0 - d)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._loss = loss.detach()
        self.learn_steps += 1
        if self.learn_steps % cfg.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _epsilon(self) -> float:
        cfg = self.config
        frac = min(1.0, self.total_transitions / cfg.epsilon_steps)
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac

    # ---------------------------------------------------------------- bucle
    def run(self, log=lambda msg: print(msg, flush=True)) -> TrainingHistory:
        cfg = self.config
        out = Path(cfg.out_dir)
        meta = {"width": cfg.width, "height": cfg.height, "features": self.env.features}
        env, n = self.env, cfg.n_envs
        obs = env.observe()
        started = time.perf_counter()
        next_eval = cfg.eval_every
        episodes = 0
        while episodes < cfg.episodes:
            self.epsilon = self._epsilon()
            with torch.no_grad():
                greedy = self.online(obs).argmax(1)
            explore = torch.rand(n, device=self.device) < self.epsilon
            actions = torch.where(explore, torch.randint(0, env.n_actions, (n,), device=self.device), greedy)
            next_obs, reward, done, info = env.step(actions)
            self.buffer.add(obs, actions, reward, next_obs, done)
            obs = next_obs
            self.total_transitions += n
            if self.buffer.size >= cfg.min_buffer:
                for _ in range(cfg.updates_per_step):
                    self._learn()

            finished = info["score"].shape[0]
            if finished:
                loss = float(self._loss)
                scores, rewards = info["score"].tolist(), info["reward"].tolist()
                steps, causes = info["steps"].tolist(), info["cause"].tolist()
                for sc, rw, st, ca in zip(scores, rewards, steps, causes):
                    episodes += 1
                    self.history.episodes.append(EpisodeStats(episodes, sc, rw, st, self.epsilon, loss, _CAUSES[ca]))
            if episodes >= next_eval or episodes >= cfg.episodes:
                next_eval += cfg.eval_every
                score = evaluate(self.online, cfg.eval_games, cfg.width, cfg.height, self.device)
                self.eval_history.append((episodes, score))
                if score > self.best_eval:
                    self.best_eval = score
                    self.online.save(out / "best.npz", **meta)
                elapsed = time.perf_counter() - started
                log(f"Partidas {episodes:>7}/{cfg.episodes} | sin azar: {score:5.1f} manzanas | media(500): "
                    f"{self.history.moving_average(500):5.1f} | epsilon: {self.epsilon:.3f} | "
                    f"{self.total_transitions / elapsed:,.0f} jugadas/s | {elapsed / 60:.1f} min")
        self.online.save(out / "last.npz", **meta)
        self.history.save_csv(out / "history.csv")
        self.save_eval_csv(out / "eval.csv")
        return self.history

    def save_eval_csv(self, path: str | Path) -> None:
        """Las mediciones SIN azar a lo largo del entrenamiento (la curva que de verdad importa)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("episodes,eval_score\n")
            for episodes, score in self.eval_history:
                f.write(f"{episodes},{score:.2f}\n")
