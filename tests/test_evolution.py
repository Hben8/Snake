import numpy as np

from snake_ai.nn import NeuralNetwork
from snake_ai.rl import EvolutionConfig, EvolutionHistory, EvolutionTrainer
from snake_ai.rl.evolution import evaluate_genome, genome_to_network, network_to_genome


def test_genome_roundtrip_keeps_every_weight():
    net = NeuralNetwork([17, 24, 24, 3], seed=3)
    genome = network_to_genome(net)
    assert genome.shape == (17 * 24 + 24 + 24 * 24 + 24 + 24 * 3 + 3,)  # 1107 parámetros
    clone = genome_to_network(genome, net.sizes)
    x = np.random.default_rng(0).random(17).astype(np.float32)
    assert np.allclose(net.forward(x), clone.forward(x))


def test_evaluation_is_deterministic_for_same_seeds():
    net = NeuralNetwork([17, 8, 3], seed=1)
    job = (network_to_genome(net), net.sizes, "cola", 10, 10, [1, 2])
    assert evaluate_genome(job) == evaluate_genome(job)


def test_children_mix_parents_and_elite_survives_unchanged():
    cfg = EvolutionConfig(generations=2, population=12, elite=3, games_per_individual=1, hidden=(6,),
                          workers=1, out_dir="unused")
    trainer = EvolutionTrainer(cfg, seed=5)
    trainer._evaluate(None)
    best = trainer.population[int(np.argmax(trainer.fitness))].copy()
    trainer._next_generation()
    assert len(trainer.population) == 12
    assert np.array_equal(trainer.population[0], best)  # la élite pasa intacta, y el mejor va primero
    assert not np.array_equal(trainer.population[5], best)  # los hijos son distintos


def test_short_run_saves_files_and_history(tmp_path):
    cfg = EvolutionConfig(generations=2, population=8, elite=2, games_per_individual=1, hidden=(6,),
                          workers=1, out_dir=str(tmp_path))
    history = EvolutionTrainer(cfg, seed=2).run()
    assert [g.generation for g in history.generations] == [1, 2]
    assert (tmp_path / "last.npz").exists()
    loaded = EvolutionHistory.load_csv(tmp_path / "history.csv")
    assert len(loaded.generations) == 2
    net, meta = NeuralNetwork.load(tmp_path / "last.npz")
    assert net.sizes == (17, 6, 3) and meta["features"] == "cola"
