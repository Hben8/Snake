"""El tablero y las reglas del juego (sin dibujar nada, sin IA)."""

import random
from dataclasses import dataclass

from .actions import Action, Direction
from .snake import Position, Snake


@dataclass(frozen=True)
class StepOutcome:
    """Qué ocurrió en un paso del juego."""

    ate: bool = False
    died: bool = False
    won: bool = False
    cause: str | None = None  # "wall", "body" o "starved" si murió


class SnakeGame:
    """Reglas: la serpiente avanza una celda por turno, crece al comer una
    manzana y muere si choca con una pared, consigo misma o si pasa demasiado
    tiempo sin comer (evita bucles infinitos)."""

    MIN_SIZE = 5

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int | None = None,
        idle_limit_factor: int = 40,
    ):
        if width < self.MIN_SIZE or height < self.MIN_SIZE:
            raise ValueError(f"El tablero debe medir al menos {self.MIN_SIZE}x{self.MIN_SIZE}")
        self.width = width
        self.height = height
        self.idle_limit_factor = idle_limit_factor
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.snake = Snake((self.width // 2, self.height // 2), Direction.RIGHT)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.won = False
        self.death_cause: str | None = None
        self.food: Position | None = self._spawn_food()

    # ------------------------------------------------------------------ reglas
    def is_wall(self, pos: Position) -> bool:
        x, y = pos
        return not (0 <= x < self.width and 0 <= y < self.height)

    def is_dangerous(self, pos: Position) -> bool:
        """¿Morir si la cabeza se mueve a `pos` en el próximo turno?"""
        return self.is_wall(pos) or self.snake.occupies(pos, ignore_tail=True)

    def reachable_cells(self, start: Position, limit: int) -> int:
        """Cuántas celdas libres se pueden alcanzar caminando desde `start`.

        Es una búsqueda por inundación ("flood fill") que se detiene al contar
        `limit` celdas, porque a partir de ahí ya da igual: hay sitio de sobra.
        La cola cuenta como libre, ya que se aparta al avanzar.
        """
        if self.is_dangerous(start):
            return 0
        blocked = set(list(self.snake.body)[:-1])
        seen = {start}
        frontier = [start]
        while frontier and len(seen) < limit:
            x, y = frontier.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt not in seen and not self.is_wall(nxt) and nxt not in blocked:
                    seen.add(nxt)
                    frontier.append(nxt)
        return min(len(seen), limit)

    def escape_info(self, start: Position, limit: int) -> tuple[int, bool]:
        """Como `reachable_cells`, pero teniendo en cuenta que la cola SE VA APARTANDO.

        Al avanzar, la serpiente libera su cola trozo a trozo: la celda que ocupa el trozo k-ésimo
        (contando desde la cola, que es el 0) estará libre cuando la cabeza llegue allí en el
        paso k o posterior. Así un "pasillo" cerrado por la cola ya no cuenta como callejón: se
        abrirá a tiempo. Devuelve (celdas alcanzables, hasta `limit`; ¿se puede llegar hasta la cola?).
        Poder llegar a la cola es la señal clásica de que la serpiente siempre tendrá por dónde salir.
        """
        if self.is_dangerous(start):
            return 0, False
        body = list(self.snake.body)
        frees_at = {cell: len(body) - 1 - i for i, cell in enumerate(body)}  # cola = 0
        tail = body[-1]
        seen = {start}
        frontier = [start]
        step = 1  # la cabeza llega a `start` en el paso 1
        while frontier:
            step += 1
            nxt_frontier = []
            for x, y in frontier:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (x + dx, y + dy)
                    if nxt in seen or self.is_wall(nxt) or frees_at.get(nxt, -1) >= step:
                        continue
                    seen.add(nxt)
                    nxt_frontier.append(nxt)
            frontier = nxt_frontier
        return min(len(seen), limit), tail in seen

    @property
    def idle_limit(self) -> int:
        return self.idle_limit_factor * len(self.snake)

    def step(self, action: Action) -> StepOutcome:
        if self.done:
            raise RuntimeError("La partida ha terminado: llama a reset().")

        direction = self.snake.direction.turn(action)
        dx, dy = direction.vector
        hx, hy = self.snake.head
        new_head = (hx + dx, hy + dy)
        ate = new_head == self.food

        if self.is_wall(new_head):
            return self._die("wall")
        if self.snake.occupies(new_head, ignore_tail=not ate):
            return self._die("body")

        self.snake.move(direction, grow=ate)
        self.steps += 1

        if ate:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            if self.food is None:  # el tablero está lleno: victoria
                self.done = self.won = True
                return StepOutcome(ate=True, won=True)
            return StepOutcome(ate=True)

        self.steps_since_food += 1
        if self.steps_since_food > self.idle_limit:
            return self._die("starved")
        return StepOutcome()

    # --------------------------------------------------------------- internos
    def _die(self, cause: str) -> StepOutcome:
        self.done = True
        self.death_cause = cause
        return StepOutcome(died=True, cause=cause)

    def _spawn_food(self) -> Position | None:
        free = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if not self.snake.occupies((x, y))
        ]
        return self.rng.choice(free) if free else None
