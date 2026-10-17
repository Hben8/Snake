"""Línea de comandos:  python main.py {play,train,evolve,watch,plot} ..."""

import argparse
import sys

from .rl import (DEFAULT_FEATURES, FEATURE_SETS, ConsoleLogger, DQNAgent, DQNConfig, EvolutionConfig,
                 EvolutionConsoleLogger, EvolutionHistory, EvolutionTrainer, SnakeEnv, TrainConfig, Trainer,
                 TrainingHistory)


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
    train.add_argument("--features", choices=sorted(FEATURE_SETS), default=None,
                       help="qué recibe la red: basico (14 pistas, por defecto), cola (17 pistas; por defecto con --envs) o imagen (el tablero entero; "
                            "red convolucional, necesita --backend torch)")
    train.add_argument("--envs", type=int, default=1,
                       help="partidas simultáneas en la GPU (>1 = entrenamiento en paralelo; --episodes pasa a contar "
                            "las partidas de TODAS; necesita --backend torch y pistas cola; sin ventana)")
    train.add_argument("--updates-per-step", type=int, default=2,
                       help="solo con --envs: pasos de aprendizaje por cada paso de todas las partidas")
    train.add_argument("--epsilon-steps", type=int, default=300_000,
                       help="solo con --envs: jugadas (entre todas las partidas) que tarda el azar en bajar a 0,01")
    train.add_argument("--epsilon-end", type=float, default=0.01,
                       help="solo con --envs: probabilidad mínima de jugar al azar (con red grande va mejor 0,1)")
    train.add_argument("--buffer-size", type=int, default=200_000,
                       help="solo con --envs: experiencias que guarda la memoria (en la GPU)")
    train.add_argument("--eval-every", type=int, default=1000,
                       help="solo con --envs: cada cuántas partidas se mide la red sin azar (y se guarda la mejor)")
    train.add_argument("--gamma", type=float, default=0.95, help="importancia del futuro (0-1)")
    train.add_argument("--seed", type=int, default=None, help="semilla para repetir resultados")
    train.add_argument("--out", default="models", help="carpeta donde guardar modelos e historial")
    train.add_argument("--render", action="store_true", help="ver el entrenamiento en una ventana")
    train.add_argument("--speed", default="Rápido", choices=["Lento", "Normal", "Rápido", "Turbo"],
                       help="velocidad inicial de la ventana (con --render)")
    train.add_argument("--record", default=None, help="guardar un GIF de la ventana (con --render)")

    evolve = sub.add_parser("evolve", help="entrena con algoritmo genético (neuroevolución, sin DQN)")
    _add_board_args(evolve)
    evolve.add_argument("--generations", type=int, default=150, help="generaciones")
    evolve.add_argument("--population", type=int, default=150, help="redes en cada generación")
    evolve.add_argument("--elite", type=int, default=10, help="mejores que pasan intactos a la siguiente")
    evolve.add_argument("--games", type=int, default=3, help="partidas que juega cada red para ser evaluada")
    evolve.add_argument("--mutation-rate", type=float, default=0.05, help="probabilidad de mutar cada peso")
    evolve.add_argument("--mutation-sigma", type=float, default=0.2, help="tamaño de las mutaciones")
    evolve.add_argument("--hidden", type=int, nargs="+", default=[24, 24], help="neuronas por capa oculta")
    evolve.add_argument("--features", choices=sorted(FEATURE_SETS), default="cola", help="pistas de la red")
    evolve.add_argument("--workers", type=int, default=0, help="procesos en paralelo (0 = automático)")
    evolve.add_argument("--seed", type=int, default=None, help="semilla para repetir resultados")
    evolve.add_argument("--out", default="models/evolution_run", help="carpeta donde guardar el resultado")
    evolve.add_argument("--render", action="store_true", help="ver al campeón jugar tras cada generación")
    evolve.add_argument("--speed", default="Rápido", choices=["Lento", "Normal", "Rápido", "Turbo"])

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
    if env.features == "imagen" and backend != "torch":
        raise SystemExit("Las pistas 'imagen' usan una red convolucional: añade  --backend torch  "
                         "(y mejor  --device cuda  si tienes GPU NVIDIA).")
    if backend == "torch":
        from .rl.torch_agent import TorchDQNAgent  # import perezoso: torch es opcional
        agent = TorchDQNAgent(env.n_inputs, env.n_actions, config, seed=seed, device=device,
                              obs_shape=env.obs_shape)
        print(f"Backend: PyTorch en {agent.device}"
              + (f" ({__import__('torch').cuda.get_device_name(agent.device)})" if agent.device.type == "cuda" else ""))
        return agent
    print("Backend: NumPy (CPU)")
    return DQNAgent(env.n_inputs, env.n_actions, config, seed=seed)


def _train_parallel(args: argparse.Namespace) -> None:
    if args.backend != "torch":
        raise SystemExit("El entrenamiento en paralelo (--envs) vive en la GPU: añade  --backend torch  "
                         "(y  --device cuda  si tienes GPU NVIDIA).")
    if args.features not in (None, "cola"):
        raise SystemExit("El entrenamiento en paralelo solo está implementado para las pistas 'cola' (--features cola).")
    if args.render or args.record:
        raise SystemExit("Con --envs no hay ventana (las partidas están dentro de la GPU). "
                         "Mira el resultado después con  python main.py watch --model <carpeta>/best.npz")
    from .rl.parallel_trainer import ParallelConfig, ParallelTrainer
    config = ParallelConfig(
        episodes=args.episodes, n_envs=args.envs, hidden=tuple(args.hidden), lr=args.lr, gamma=args.gamma,
        batch_size=args.batch_size, updates_per_step=args.updates_per_step, epsilon_steps=args.epsilon_steps,
        epsilon_end=args.epsilon_end, buffer_size=args.buffer_size, eval_every=args.eval_every, eval_games=100,
        width=args.width, height=args.height, device=args.device, out_dir=args.out)
    trainer = ParallelTrainer(config, seed=args.seed)
    print(f"Backend: PyTorch en {trainer.device} | {args.envs} partidas en paralelo")
    try:
        history = trainer.run()
    except KeyboardInterrupt:
        history = trainer.history
        print("\nEntrenamiento interrumpido.")
    print(f"Listo. Mejor modelo en {args.out}/best.npz | mejor medida sin azar: {trainer.best_eval:.1f} manzanas")


def _train(args: argparse.Namespace) -> None:
    if args.envs > 1:
        return _train_parallel(args)
    env = SnakeEnv(args.width, args.height, seed=args.seed, features=args.features or DEFAULT_FEATURES)
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


def _evolve(args: argparse.Namespace) -> None:
    import numpy as np

    from .rl.evolution import play
    config = EvolutionConfig(
        generations=args.generations, population=args.population, elite=args.elite,
        games_per_individual=args.games, mutation_rate=args.mutation_rate, mutation_sigma=args.mutation_sigma,
        hidden=tuple(args.hidden), features=args.features, width=args.width, height=args.height,
        workers=args.workers, out_dir=args.out)
    observers = [EvolutionConsoleLogger(every=5)]
    dashboard = None
    if args.render:
        from .viz.dashboard import Dashboard, EvolutionDashboardObserver
        dashboard = Dashboard("Snake IA - evolucionando", speed=args.speed)
        observers.append(EvolutionDashboardObserver(dashboard))
    trainer = EvolutionTrainer(config, observers, seed=args.seed)
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\nEvolución interrumpida.")
    finally:
        if dashboard is not None:
            dashboard.close()
    if trainer.champion is None:
        return
    # El ganador "de verdad": entre los mejores de la última generación, el que más manzanas hace en 100 partidas nuevas.
    print("Eligiendo al mejor de los finalistas (100 partidas nuevas cada uno)...")
    finalists = trainer.best_individuals(5)
    averages = []
    for net in finalists:
        env = SnakeEnv(args.width, args.height, seed=2024, features=args.features)
        averages.append(float(np.mean([play(net, env)[1] for _ in range(100)])))
    best = int(np.argmax(averages))
    finalists[best].save(f"{args.out}/best.npz", width=args.width, height=args.height, features=args.features)
    print(f"Listo. Mejor modelo en {args.out}/best.npz | media en 100 partidas: {averages[best]:.1f} manzanas")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "play":
        from .human import run_human
        run_human(args.width, args.height)
    elif args.command == "train":
        _train(args)
    elif args.command == "evolve":
        _evolve(args)
    elif args.command == "watch":
        from pathlib import Path

        from .watch import run_watch
        if not Path(args.model).exists():
            print(f"No encuentro el modelo '{args.model}'. Entrena uno con: python main.py train", file=sys.stderr)
            return 1
        run_watch(args.model, args.width, args.height, args.games, args.speed, args.record, args.seed)
    elif args.command == "plot":
        from .viz.plots import plot_eval_curve, plot_evolution, plot_history
        with open(args.history, encoding="utf-8") as f:
            header = f.readline()
        if header.startswith("episodes,eval_score"):
            plot_eval_curve(args.history, args.out)
        elif header.startswith("generation"):
            plot_evolution(EvolutionHistory.load_csv(args.history), args.out)
        else:
            plot_history(TrainingHistory.load_csv(args.history), args.out)
    return 0
