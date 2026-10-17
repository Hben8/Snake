"""Cientos de partidas de Snake A LA VEZ, calculadas en la GPU.

El `SnakeEnv` normal juega UNA partida y la calcula la CPU casilla a casilla. Aquí todas las partidas viven
dentro de unos pocos tensores (una "pila" de tableros) y un solo paso de la GPU hace avanzar a todas: es el
tipo de cálculo en el que una GPU gana de verdad, porque son miles de operaciones iguales a la vez.

El truco para representar la serpiente: cada casilla guarda cuántos pasos le quedan al cuerpo ahí
(`ttl`, de "time to live"). La cabeza vale la longitud de la serpiente, el cuello longitud-1... y la cola 1.
En cada paso todo baja 1 y la cola (1) desaparece sola: moverse, crecer y chocar son solo comparaciones.

Reglas, recompensas y pistas ("cola", 17 números) son EXACTAMENTE las del juego normal; los tests lo comprueban
comparando ambas implementaciones estado a estado.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .environment import RewardConfig
from .observation import feature_names

_DX = (0, 1, 0, -1)  # ARRIBA, DERECHA, ABAJO, IZQUIERDA (como en game.Direction)
_DY = (-1, 0, 1, 0)
_TURN = (0, 1, -1)   # recto, derecha, izquierda
_CAUSES = ("wall", "body", "starved", "won")


class VectorSnakeEnv:
    features = "cola"

    def __init__(self, n_envs: int, width: int = 10, height: int = 10, device: str | torch.device = "cpu",
                 seed: int | None = None, rewards: RewardConfig | None = None, idle_limit_factor: int = 40):
        self.n, self.width, self.height = n_envs, width, height
        self.device = torch.device(device)
        self.rewards = rewards or RewardConfig()
        self.idle_limit_factor = idle_limit_factor
        self.n_actions = 3
        self.feature_names = feature_names(self.features)
        self.n_inputs = len(self.feature_names)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)
        dev = self.device
        self.dx, self.dy = torch.tensor(_DX, device=dev), torch.tensor(_DY, device=dev)
        self.turn = torch.tensor(_TURN, device=dev)
        self.ar = torch.arange(n_envs, device=dev)

        self.ttl = torch.zeros(n_envs, height, width, dtype=torch.int32, device=dev)
        self.head = torch.zeros(n_envs, 2, dtype=torch.long, device=dev)   # (x, y)
        self.food = torch.zeros(n_envs, 2, dtype=torch.long, device=dev)
        self.dir = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.length = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.score = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.idle = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.ep_reward = torch.zeros(n_envs, device=dev)
        self.ep_steps = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self._reset(torch.ones(n_envs, dtype=torch.bool, device=dev))

    # ------------------------------------------------------------------ reinicio
    def _reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero().squeeze(1)
        if idx.numel() == 0:
            return
        cx, cy = self.width // 2, self.height // 2
        self.ttl[idx] = 0
        for k, x in enumerate((cx, cx - 1, cx - 2)):  # cabeza (3), cuello (2), cola (1); mira a la derecha
            self.ttl[idx, cy, x] = 3 - k
        self.head[idx] = torch.tensor([cx, cy], device=self.device)
        self.dir[idx] = 1
        self.length[idx] = 3
        self.score[idx] = 0
        self.idle[idx] = 0
        self.ep_reward[idx] = 0
        self.ep_steps[idx] = 0
        self._spawn_food(mask)

    def _spawn_food(self, mask: torch.Tensor) -> torch.Tensor:
        """Manzana nueva en una casilla libre al azar. Devuelve qué partidas se han quedado SIN sitio (victoria)."""
        free = (self.ttl == 0).flatten(1).float()
        full = free.sum(1) == 0
        free[full, 0] = 1.0  # evita el error de muestrear sin opciones; esas partidas terminan igual
        cell = torch.multinomial(free, 1, generator=self.gen).squeeze(1)
        new_food = torch.stack([cell % self.width, cell // self.width], dim=1)
        self.food = torch.where(mask.unsqueeze(1), new_food, self.food)
        return full & mask

    # --------------------------------------------------------------------- paso
    def step(self, actions: torch.Tensor):
        """Avanza TODAS las partidas un paso. Reinicia solas las que terminan.

        Devuelve (observación siguiente, recompensa, terminó, info). `info` trae, para las partidas que acaban,
        sus manzanas, recompensa total, pasos y causa (índice de ("wall", "body", "starved", "won")).
        """
        dev = self.device
        new_dir = (self.dir + self.turn[actions]) % 4
        nx, ny = self.head[:, 0] + self.dx[new_dir], self.head[:, 1] + self.dy[new_dir]
        wall = (nx < 0) | (nx >= self.width) | (ny < 0) | (ny >= self.height)
        cx, cy = nx.clamp(0, self.width - 1), ny.clamp(0, self.height - 1)
        body = ~wall & (self.ttl[self.ar, cy, cx] > 1)  # la cola (1) se aparta: se puede pisar
        died = wall | body
        alive = ~died
        ate = alive & (nx == self.food[:, 0]) & (ny == self.food[:, 1])

        dist_before = (self.head[:, 0] - self.food[:, 0]).abs() + (self.head[:, 1] - self.food[:, 1]).abs()
        dist_after = (nx - self.food[:, 0]).abs() + (ny - self.food[:, 1]).abs()

        # El cuerpo avanza: todo baja 1 (la cola desaparece), salvo si come (entonces la serpiente crece).
        dec = (alive & ~ate).view(-1, 1, 1).to(torch.int32)
        self.ttl = (self.ttl - dec * (self.ttl > 0)).to(torch.int32)
        self.length = self.length + ate.long()
        move = alive.nonzero().squeeze(1)
        self.ttl[move, cy[move], cx[move]] = self.length[move].to(torch.int32)
        self.head = torch.where(alive.unsqueeze(1), torch.stack([nx, ny], dim=1), self.head)
        self.dir = torch.where(alive, new_dir, self.dir)
        self.score = self.score + ate.long()
        self.idle = torch.where(alive, torch.where(ate, torch.zeros_like(self.idle), self.idle + 1), self.idle)
        starved = alive & ~ate & (self.idle > self.idle_limit_factor * self.length)
        won = self._spawn_food(ate)  # solo las que comieron necesitan manzana nueva

        reward = torch.where(
            died | starved, torch.full_like(self.ep_reward, self.rewards.die),
            torch.where(ate, torch.full_like(self.ep_reward, self.rewards.eat),
                        torch.where(dist_after < dist_before, self.rewards.closer, self.rewards.farther)))
        done = died | starved | won
        self.ep_reward = self.ep_reward + reward
        self.ep_steps = self.ep_steps + 1

        cause = torch.where(wall, 0, torch.where(body, 1, torch.where(starved, 2, 3)))
        info = {"score": self.score[done], "reward": self.ep_reward[done], "steps": self.ep_steps[done],
                "cause": cause[done]}
        self._reset(done)
        return self.observe(), reward, done, info

    # ------------------------------------------------------------- observación
    def observe(self) -> torch.Tensor:
        """Las 17 pistas "cola" de TODAS las partidas: tensor (n_envs, 17). Igual que `observation.observe`."""
        n, dev = self.n, self.device
        hx, hy = self.head[:, 0], self.head[:, 1]
        # Las 3 casillas a las que podría ir la cabeza: recto, derecha, izquierda -> (n, 3)
        dirs = (self.dir.unsqueeze(1) + self.turn.unsqueeze(0)) % 4
        tx, ty = hx.unsqueeze(1) + self.dx[dirs], hy.unsqueeze(1) + self.dy[dirs]
        wall = (tx < 0) | (tx >= self.width) | (ty < 0) | (ty >= self.height)
        cx, cy = tx.clamp(0, self.width - 1), ty.clamp(0, self.height - 1)
        ttl_at = self.ttl[self.ar.unsqueeze(1), cy, cx]
        danger = wall | (ttl_at > 1)

        heading = F.one_hot(self.dir, 4).float()
        fx, fy = self.food[:, 0], self.food[:, 1]
        food = torch.stack([fy < hy, fx > hx, fy > hy, fx < hx], dim=1).float()

        count, reaches_tail = self._escape_info(cx, cy, danger)
        needed = (self.length + 1).unsqueeze(1).float()
        space = count.minimum(needed) / needed
        return torch.cat([danger.float(), heading, food, space, reaches_tail.float()], dim=1)

    def _escape_info(self, cx: torch.Tensor, cy: torch.Tensor, danger: torch.Tensor):
        """Versión en GPU de `SnakeGame.escape_info`: búsqueda por inundación que tiene en cuenta que la cola se
        aparta. Se hace "ensanchando" todas las zonas alcanzables a la vez, un paso cada vez."""
        n, h, w = self.n, self.height, self.width
        flat_ttl = self.ttl.repeat_interleave(3, dim=0)  # (3n, h, w): una copia por cada acción
        start = torch.zeros(3 * n, h, w, dtype=torch.bool, device=self.device)
        rows = torch.arange(3 * n, device=self.device)
        start[rows, cy.reshape(-1), cx.reshape(-1)] = ~danger.reshape(-1)
        reach, frontier = start.clone(), start
        is_tail = flat_ttl == 1
        for step in range(2, h * w + 2):
            # Solo se expande desde la "frontera" (lo descubierto en el paso anterior), igual que la búsqueda
            # normal: una casilla de cuerpo que se libera más tarde NO se alcanza desde casillas antiguas.
            p = F.pad(frontier, (1, 1, 1, 1))
            grown = p[:, :-2, 1:-1] | p[:, 2:, 1:-1] | p[:, 1:-1, :-2] | p[:, 1:-1, 2:]  # vecinos (4 direcciones)
            # Una casilla de cuerpo con ttl t queda libre cuando la cabeza llega en el paso t o después.
            frontier = grown & ~reach & (flat_ttl <= step)
            reach = reach | frontier
            if step % 4 == 0 and not frontier.any():  # comprobar cada 4 pasos: cada comprobación espera a la GPU
                break
        count = reach.flatten(1).sum(1).view(n, 3)
        reaches_tail = (reach & is_tail).flatten(1).any(1).view(n, 3)
        return count.float(), reaches_tail

    # ------------------------------------------------------------- para los tests
    def load_from_games(self, games) -> None:
        """Copia el estado de varias `SnakeGame` (una por partida) a este entorno. Solo para comparar en tests."""
        assert len(games) == self.n
        self.ttl.zero_()
        for i, g in enumerate(games):
            body = list(g.snake.body)
            for j, (x, y) in enumerate(body):
                self.ttl[i, y, x] = len(body) - j
            self.head[i] = torch.tensor(g.snake.head)
            self.dir[i] = int(g.snake.direction)
            self.food[i] = torch.tensor(g.food)
            self.length[i] = len(body)
            self.score[i] = g.score
            self.idle[i] = g.steps_since_food
