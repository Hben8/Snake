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
  * "imagen" (4 × (alto+2) × (ancho+2) números): NO hay pistas calculadas a mano. La red recibe el
    tablero entero como una imagen de 4 canales (pared, cuerpo, cabeza, comida), girada para que la
    serpiente siempre mire hacia arriba. Es la entrada de una red CONVOLUCIONAL: tiene que descubrir
    ella sola qué es un peligro, una trampa o un camino.
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

IMAGE_NAMES = ("Pared", "Cuerpo", "Cabeza", "Comida")  # los 4 canales de la imagen

FEATURE_SETS = {"basico": BASIC_NAMES, "cola": TAIL_NAMES, "imagen": IMAGE_NAMES}
DEFAULT_FEATURES = "basico"

# Compatibilidad: el juego de pistas por defecto.
FEATURE_NAMES = FEATURE_SETS[DEFAULT_FEATURES]
N_FEATURES = len(FEATURE_NAMES)


def feature_names(features: str) -> tuple[str, ...]:
    if features not in FEATURE_SETS:
        raise ValueError(f"Juego de pistas desconocido '{features}'. Opciones: {', '.join(FEATURE_SETS)}")
    return FEATURE_SETS[features]


def observation_shape(features: str, width: int, height: int) -> tuple[int, ...]:
    """Forma de lo que recibe la red: un vector de pistas, o una imagen (canales, alto, ancho)."""
    if features == "imagen":
        return (len(IMAGE_NAMES), height + 2, width + 2)  # +2: el borde de pared
    return (len(feature_names(features)),)


def observe_image(game: SnakeGame) -> np.ndarray:
    """El tablero como imagen de 4 canales, girado para que la serpiente mire siempre hacia arriba.

    Canales: pared (el borde exterior), cuerpo (más cercano a 1 = tardará más en apartarse; la cola vale
    poco), cabeza y comida. Girarlo hace que "recto" sea siempre "hacia arriba" y que la red no tenga que
    aprender cuatro veces lo mismo.
    """
    h, w = game.height, game.width
    img = np.zeros((4, h + 2, w + 2), dtype=np.float32)
    img[0, 0, :] = img[0, -1, :] = img[0, :, 0] = img[0, :, -1] = 1.0
    body = list(game.snake.body)
    n = len(body)
    for i, (x, y) in enumerate(body):  # i = 0 es la cabeza; la cola es el último
        img[1, y + 1, x + 1] = (n - i) / n if i else 0.0
    hx, hy = game.snake.head
    img[2, hy + 1, hx + 1] = 1.0
    if game.food is not None:
        img[3, game.food[1] + 1, game.food[0] + 1] = 1.0
    return np.ascontiguousarray(np.rot90(img, k=int(game.snake.direction), axes=(1, 2)))


def observe(game: SnakeGame, features: str = DEFAULT_FEATURES) -> np.ndarray:
    feature_names(features)  # valida el nombre
    if features == "imagen":
        return observe_image(game)
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
