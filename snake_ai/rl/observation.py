"""Lo que "ve" la red: 14 números que resumen la situación.

La serpiente NO ve la pantalla: solo recibe estas pistas. Son relativas a
hacia dónde mira, así que la red funciona igual en tableros de cualquier
tamaño.

  * 11 pistas de 0/1: peligro inmediato, hacia dónde mira y dónde está la comida.
  * 3 pistas entre 0 y 1: el ESPACIO LIBRE que quedaría tras cada movimiento.
    Sin ellas, la serpiente solo veía peligros a una casilla y acababa
    encerrándose a sí misma (9 de cada 10 muertes eran por chocar con su cuerpo).
"""

import numpy as np

from ..game import Action, Direction, SnakeGame

FEATURE_NAMES = (
    "Peligro recto",
    "Peligro derecha",
    "Peligro izquierda",
    "Mira arriba",
    "Mira derecha",
    "Mira abajo",
    "Mira izquierda",
    "Comida arriba",
    "Comida derecha",
    "Comida abajo",
    "Comida izquierda",
    "Espacio recto",
    "Espacio derecha",
    "Espacio izquierda",
)
N_FEATURES = len(FEATURE_NAMES)


def observe(game: SnakeGame) -> np.ndarray:
    head_x, head_y = game.snake.head
    direction = game.snake.direction

    def danger(action: Action) -> float:
        dx, dy = direction.turn(action).vector
        return float(game.is_dangerous((head_x + dx, head_y + dy)))

    heading = [float(direction == d) for d in (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)]

    if game.food is None:
        food = [0.0, 0.0, 0.0, 0.0]
    else:
        fx, fy = game.food
        food = [float(fy < head_y), float(fx > head_x), float(fy > head_y), float(fx < head_x)]

    # Espacio alcanzable tras cada movimiento, como fracción de lo que necesita la serpiente
    # (su longitud + 1). 1.0 = hay sitio de sobra; menos de 1.0 = callejón sin salida.
    needed = len(game.snake) + 1

    def space(action: Action) -> float:
        dx, dy = direction.turn(action).vector
        return game.reachable_cells((head_x + dx, head_y + dy), needed) / needed

    return np.array(
        [danger(Action.STRAIGHT), danger(Action.TURN_RIGHT), danger(Action.TURN_LEFT), *heading, *food,
         space(Action.STRAIGHT), space(Action.TURN_RIGHT), space(Action.TURN_LEFT)],
        dtype=np.float32,
    )
