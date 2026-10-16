"""La serpiente: un cuerpo ordenado de celdas, de la cabeza a la cola."""

from collections import deque

from .actions import Direction

Position = tuple[int, int]


class Snake:
    def __init__(self, head: Position, direction: Direction, length: int = 3):
        dx, dy = direction.vector
        # La cola se extiende en sentido contrario al que mira la cabeza.
        self.body: deque[Position] = deque(
            (head[0] - dx * i, head[1] - dy * i) for i in range(length)
        )
        self.direction = direction

    @property
    def head(self) -> Position:
        return self.body[0]

    @property
    def tail(self) -> Position:
        return self.body[-1]

    def __len__(self) -> int:
        return len(self.body)

    def occupies(self, pos: Position, ignore_tail: bool = False) -> bool:
        """¿Hay un trozo de serpiente en `pos`?

        `ignore_tail=True` sirve para saber si moverse ahí chocaría: si la
        serpiente no come, la cola se aparta y esa celda queda libre.
        """
        if ignore_tail:
            return any(cell == pos for i, cell in enumerate(self.body) if i < len(self.body) - 1)
        return pos in self.body

    def move(self, direction: Direction, grow: bool) -> None:
        dx, dy = direction.vector
        hx, hy = self.head
        self.direction = direction
        self.body.appendleft((hx + dx, hy + dy))
        if not grow:
            self.body.pop()
