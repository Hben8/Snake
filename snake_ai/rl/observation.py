"""Lo que "ve" la red: unos pocos números que resumen la situación.

La serpiente NO ve la pantalla: solo recibe estas pistas. Son relativas a
hacia dónde mira, así que la red funciona igual en tableros de cualquier
tamaño. Hay varios "juegos de pistas" (cada modelo nuevo del proyecto añade
mejoras):

  * "basico" (14 números): peligro inmediato, hacia dónde mira, dónde está la
    comida y el ESPACIO LIBRE que quedaría tras cada movimiento. Sin el espacio,
    la serpiente solo veía peligros a una casilla y acababa encerrándose a sí
    misma (9 de cada 10 muertes eran por chocar con su cuerpo).
  * "cola" (17 números): lo mismo, pero el espacio libre tiene en cuenta que la
    cola se va apartando, y añade 3 pistas: ¿podría llegar hasta mi propia cola
    tras ese movimiento? Si puede, siempre tendrá una salida.
"""

import numpy as np

from ..game import Action, Direction, SnakeGame

BASIC_NAMES = (
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
TAIL_NAMES = BASIC_NAMES + ("Cola recto", "Cola derecha", "Cola izquierda")

FEATURE_SETS = {"basico": BASIC_NAMES, "cola": TAIL_NAMES}
DEFAULT_FEATURES = "basico"

# Compatibilidad: el juego de pistas por defecto.
FEATURE_NAMES = FEATURE_SETS[DEFAULT_FEATURES]
N_FEATURES = len(FEATURE_NAMES)


def feature_names(features: str) -> tuple[str, ...]:
    if features not in FEATURE_SETS:
        raise ValueError(f"Juego de pistas desconocido '{features}'. Opciones: {', '.join(FEATURE_SETS)}")
    return FEATURE_SETS[features]


def observe(game: SnakeGame, features: str = DEFAULT_FEATURES) -> np.ndarray:
    feature_names(features)  # valida el nombre
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
    actions = (Action.STRAIGHT, Action.TURN_RIGHT, Action.TURN_LEFT)

    def target(action: Action):
        dx, dy = direction.turn(action).vector
        return head_x + dx, head_y + dy

    if features == "basico":
        space = [game.reachable_cells(target(a), needed) / needed for a in actions]
        tail: list[float] = []
    else:
        info = [game.escape_info(target(a), needed) for a in actions]
        space = [cells / needed for cells, _ in info]
        tail = [float(reaches) for _, reaches in info]

    return np.array(
        [*(danger(a) for a in actions), *heading, *food, *space, *tail],
        dtype=np.float32,
    )
