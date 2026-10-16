"""Acciones y direcciones del juego.

La IA no decide "arriba/abajo/izquierda/derecha" (acciones absolutas) sino
"sigue recto / gira a la derecha / gira a la izquierda" (acciones relativas a
donde mira la serpiente). Así la red tiene solo 3 salidas y es imposible que
elija darse la vuelta de 180 grados.
"""

from enum import IntEnum


class Action(IntEnum):
    STRAIGHT = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2


ACTION_NAMES = ("Recto", "Derecha", "Izquierda")


class Direction(IntEnum):
    # El orden es el de las agujas del reloj: sumar 1 = girar a la derecha.
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def vector(self) -> tuple[int, int]:
        """(dx, dy). El eje Y crece hacia abajo, como en pantalla."""
        return _VECTORS[self]

    def turn(self, action: Action) -> "Direction":
        offset = {Action.STRAIGHT: 0, Action.TURN_RIGHT: 1, Action.TURN_LEFT: -1}[action]
        return Direction((self + offset) % 4)

    def action_towards(self, target: "Direction") -> Action | None:
        """Acción necesaria para pasar de esta dirección a `target`.

        Devuelve None si `target` es la dirección opuesta (giro de 180 grados).
        """
        diff = (target - self) % 4
        return {0: Action.STRAIGHT, 1: Action.TURN_RIGHT, 3: Action.TURN_LEFT}.get(diff)


_VECTORS = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}
