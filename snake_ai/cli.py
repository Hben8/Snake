"""Línea de comandos:  python main.py {play,train,watch,plot} ..."""

import argparse
import sys

from .rl import (DEFAULT_FEATURES, FEATURE_SETS, ConsoleLogger, DQNAgent, DQNConfig, SnakeEnv, TrainConfig,
                 Trainer, TrainingHistory)


def _add_board_args(p: argparse.ArgumentParser, default: int | None = 10) -> None:
    p.add_argument("--width", type=int, default=default, help="ancho del tablero (celdas)")
    p.add_argument("--height", type=int, default=default, help="alto del tablero (celdas)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Snake + Deep Learning")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="juega tú")
    _add_board_args(play)

    train = sub.add_parser("train", help="entrena a la serpiente")
    _add_board_args(train)
    train.add_argument("--episodes", type=int, default=1500, help="partidas de entrenamiento")
    train.add_argument("--hidden", type=int, nargs="+", default=[24, 24], help="neuronas por capa oculta")
    train.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje")
    train.add_argument("--batch-size", type=int, default=64, help="experiencias por paso de aprendizaje")
    train.add_argument("--backend", choices=["numpy", "torch"], default="numpy",
                       help="numpy = red hecha a mano; torch = PyTorch (con GPU si hay)")
    train.add_argument("--device", default="auto",
                       help="solo con --backend torch: auto, cpu o cuda")
    train.add_argument("--features", choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURES,
                       help="qué pistas recibe la red: basico (14) o cola (17, la serpiente sabe si puede salir)")
    train.add_argument("--gamma", type=float, default=0.95, help="importancia del futuro (0-1)")
    train.add_argument("--seed", type=int, default=None, help="semilla para repetir resultados")
    train.add_argument("--out", default="models", help="carpeta donde guardar modelos e historial")
    train.add_argument("--render", action="store_true", help="ver el entrenamiento en una ventana")
    train.add_argument("--speed", default="Rápido", choices=["Lento", "Normal", "Rápido", "Turbo"],
                       help="velocidad inicial de la ventana (con --render)")
    train.add_argument("--record", default=None, help="guardar un GIF de la ventana (con --render)")

    watch = sub.add_parser("watch", help="ver jugar a una red entrenada")
    _add_board_args(watch, default=None)
    watch.add_argument("--model", default="models/pretrained.npz", help="archivo .npz de la red")
    watch.add_argument("--games", type=int, default=20, help="partidas a jugar")
    watch.add_argument("--speed", default="Normal", choices=["Lento", "Normal", "Rápido", "Turbo"])
    watch.add_argument("--seed", type=int, default=None)
    watch.add_argument("--record", default=None, help="guardar un GIF de la ventana")

    plot = sub.add_parser("plot", help="gráfica de un entrenamiento guardado")
    plot.add_argument("--history", default="models/history.csv")
    plot.add_argument("--out", default="docs/img/training.png")
    return parser


def make_agent(backend: str, env: SnakeEnv, config: DQNConfig, seed: int | None, device: str = "auto"):
    if backend == "torch":
        from .rl.torch_agent import TorchDQNAgent  # import perezoso: torch es opcional
        agent = TorchDQNAgent(env.n_inputs, env.n_actions, config, seed=seed, device=device)
        print(f"Backend: PyTorch en {agent.device}"
              + (f" ({__import__('torch').cuda.get_device_name(agent.device)})" if agent.device.type == "cuda" else ""))
        return agent
    print("Backend: NumPy (CPU)")
    return DQNAgent(env.n_inputs, env.n_actions, config, seed=seed)


def _train(args: argparse.Namespace) -> None:
    env = SnakeEnv(args.width, args.height, seed=args.seed, features=args.features)
    config = DQNConfig(hidden=tuple(args.hidden), lr=args.lr, gamma=args.gamma,
                       batch_size=args.batch_size)
    agent = make_agent(args.backend, env, config, args.seed, args.device)
    observers = [ConsoleLogger(every=25)]
    dashboard = None

    if args.render:
        from .viz.dashboard import Dashboard, DashboardObserver
        from .viz.recorder import GifRecorder
        recorder = GifRecorder(args.record, fps=15, every=3) if args.record else None
        dashboard = Dashboard("Snake IA - entrenando", speed=args.speed, recorder=recorder)
        observers.append(DashboardObserver(dashboard))

    trainer = Trainer(env, agent, TrainConfig(episodes=args.episodes, out_dir=args.out), observers)
    try:
        history = trainer.run()
    except KeyboardInterrupt:
        history = trainer.history
        print("\nEntrenamiento interrumpido.")
    finally:
        if dashboard is not None:
            dashboard.close()
    print(f"Listo. Mejor modelo en {args.out}/best.npz | media final(50): {history.moving_average(50):.2f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "play":
        from .human import run_human
        run_human(args.width, args.height)
    elif args.command == "train":
        _train(args)
    elif args.command == "watch":
        from pathlib import Path

        from .watch import run_watch
        if not Path(args.model).exists():
            print(f"No encuentro el modelo '{args.model}'. Entrena uno con: python main.py train", file=sys.stderr)
            return 1
        run_watch(args.model, args.width, args.height, args.games, args.speed, args.record, args.seed)
    elif args.command == "plot":
        from .viz.plots import plot_history
        plot_history(TrainingHistory.load_csv(args.history), args.out)
    return 0
