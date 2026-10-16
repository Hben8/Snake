import pytest

from snake_ai.game import Action, Direction, SnakeGame
from snake_ai.rl import observe


def make_game(**kw) -> SnakeGame:
    return SnakeGame(10, 10, seed=1, **kw)


def test_initial_state():
    game = make_game()
    assert len(game.snake) == 3
    assert game.snake.head == (5, 5)
    assert game.snake.direction == Direction.RIGHT
    assert game.food is not None and not game.snake.occupies(game.food)


def test_turning_is_relative_to_heading():
    assert Direction.UP.turn(Action.TURN_RIGHT) == Direction.RIGHT
    assert Direction.UP.turn(Action.TURN_LEFT) == Direction.LEFT
    assert Direction.LEFT.turn(Action.TURN_LEFT) == Direction.DOWN
    assert Direction.UP.action_towards(Direction.DOWN) is None


def test_moves_forward_without_growing():
    game = make_game()
    game.food = (0, 0)
    game.step(Action.STRAIGHT)
    assert game.snake.head == (6, 5) and len(game.snake) == 3


def test_eating_grows_and_scores():
    game = make_game()
    game.food = (6, 5)
    outcome = game.step(Action.STRAIGHT)
    assert outcome.ate and game.score == 1 and len(game.snake) == 4
    assert game.food is not None and game.food != game.snake.head


def test_wall_collision():
    game = make_game()
    game.food = (0, 0)
    outcome = None
    for _ in range(10):
        outcome = game.step(Action.STRAIGHT)
        if game.done:
            break
    assert outcome.died and outcome.cause == "wall"


def test_self_collision():
    game = make_game()
    game.food = (0, 0)
    game.snake.body.clear()
    game.snake.body.extend([(5, 5), (5, 4), (4, 4), (4, 5), (4, 6), (5, 6)])
    game.snake.direction = Direction.DOWN
    # Girar a la derecha desde DOWN = LEFT -> (4,5), que es cuerpo.
    outcome = game.step(Action.TURN_RIGHT)
    assert outcome.died and outcome.cause == "body"


def test_moving_into_tail_cell_is_safe():
    game = make_game()
    game.food = (0, 0)
    game.snake.body.clear()
    game.snake.body.extend([(5, 5), (5, 6), (4, 6), (4, 5)])  # cola en (4,5)
    game.snake.direction = Direction.UP
    outcome = game.step(Action.TURN_LEFT)  # UP -> LEFT: (4,5) = la cola, que se aparta
    assert not outcome.died


def test_starvation():
    game = make_game(idle_limit_factor=1)  # 3 pasos de margen
    game.food = (0, 9)
    game.snake.direction = Direction.RIGHT
    died = False
    for action in [Action.STRAIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT]:
        outcome = game.step(action)
        if outcome.died:
            died = outcome.cause == "starved"
            break
    assert died


def test_step_after_done_raises():
    game = make_game()
    game.done = True
    with pytest.raises(RuntimeError):
        game.step(Action.STRAIGHT)


def test_board_too_small():
    with pytest.raises(ValueError):
        SnakeGame(3, 3)


def test_observation_features():
    game = make_game()
    game.food = (8, 5)  # a la derecha, la serpiente mira a la derecha
    obs = observe(game)
    assert obs.shape == (14,)
    assert list(obs[:3]) == [0, 0, 0]  # sin peligro inmediato
    assert list(obs[11:]) == [1, 1, 1]  # y espacio de sobra en las tres direcciones
    assert obs[4] == 1 and obs[3] == obs[5] == obs[6] == 0  # mira a la derecha
    assert obs[8] == 1 and obs[7] == obs[9] == obs[10] == 0  # comida a la derecha


def test_observation_danger_next_to_wall():
    game = make_game()
    game.snake.body.clear()
    game.snake.body.extend([(9, 5), (8, 5), (7, 5)])
    game.snake.direction = Direction.RIGHT
    obs = observe(game)
    assert obs[0] == 1  # pared justo delante


def test_reachable_cells_counts_open_space_and_stops_at_limit():
    game = make_game()
    assert game.reachable_cells((6, 5), limit=10) == 10  # tablero abierto: llega al límite
    assert game.reachable_cells((6, 5), limit=1000) == 100 - 2  # todo salvo cuerpo (cola cuenta como libre)
    assert game.reachable_cells((-1, 5), limit=10) == 0  # pared


def test_reachable_cells_detects_dead_end():
    game = make_game()
    game.snake.body.clear()
    # (0,0) queda rodeada por el cuerpo en (1,0) y (0,1): es un callejón de una sola casilla.
    game.snake.body.extend([(5, 5), (0, 1), (1, 0), (1, 1), (4, 5)])
    assert game.reachable_cells((0, 0), limit=10) == 1


def test_observation_space_feature_detects_pocket():
    game = make_game()
    game.snake.body.clear()
    #  . . . .       La cabeza H en (2,0) mira a la izquierda. Seguir recto es entrar en
    #  . X H .       (1,0), pero ahí solo caben 2 casillas, (1,0) y (0,0): una trampa.
    #  . X X .       Girar a derecha/izquierda es chocar (pared / cuerpo).
    game.snake.body.extend([(2, 0), (2, 1), (1, 1), (0, 1), (0, 2)])
    game.snake.direction = Direction.LEFT
    game.food = (9, 9)
    obs = observe(game)
    needed = len(game.snake) + 1  # 6
    assert obs[11] == pytest.approx(2 / needed)  # recto: 2 casillas de las 6 necesarias
    assert obs[12] == 0 and obs[13] == 0  # los otros dos giros son choque directo
