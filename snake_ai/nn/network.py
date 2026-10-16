"""Una red neuronal totalmente conectada (MLP) escrita desde cero con NumPy.

    entrada ──► [capa oculta + ReLU] ──► ... ──► salida (lineal)

Cada capa calcula  z = a_anterior @ W + b  y aplica una función de activación:
ReLU en las capas ocultas (`max(0, z)`: la neurona está "apagada" si z <= 0) y
ninguna en la salida (porque la salida son valores Q, que pueden ser negativos).
"""

from pathlib import Path

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int] | tuple[int, ...], seed: int | None = None):
        if len(layer_sizes) < 2:
            raise ValueError("Una red necesita al menos capa de entrada y de salida")
        self.sizes = tuple(int(s) for s in layer_sizes)
        rng = np.random.default_rng(seed)
        # Inicialización de He: pesos ~ N(0, 2/n_entradas), la adecuada para ReLU.
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)).astype(np.float32)
            for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.biases = [np.zeros(n_out, dtype=np.float32) for n_out in self.sizes[1:]]
        self._cache: tuple[list[np.ndarray], list[np.ndarray]] | None = None

    # ------------------------------------------------------------------ forward
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Calcula la salida y guarda lo necesario para `backward`.

        `x` puede ser un vector (n_entradas,) o un lote (n_lote, n_entradas).
        """
        a = np.asarray(x, dtype=np.float32)
        activations = [a]
        pre_activations = []
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre_activations.append(z)
            a = z if i == last else np.maximum(z, 0.0)
            activations.append(a)
        self._cache = (activations, pre_activations)
        return a

    def activations(self, x: np.ndarray) -> list[np.ndarray]:
        """Valores de TODAS las capas para una entrada (para visualizar).

        No toca la caché de `backward`, así que se puede llamar en mitad de un
        entrenamiento sin estropearlo.
        """
        a = np.asarray(x, dtype=np.float32)
        layers = [a]
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            a = z if i == last else np.maximum(z, 0.0)
            layers.append(a)
        return layers

    # ----------------------------------------------------------------- backward
    def backward(self, grad_output: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Retropropagación del error.

        `grad_output` es dLoss/dSalida (misma forma que la salida del último
        `forward`). Devuelve los gradientes de pesos y sesgos de cada capa.
        """
        if self._cache is None:
            raise RuntimeError("Llama a forward() antes que a backward()")
        activations, pre_activations = self._cache
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        delta = np.asarray(grad_output, dtype=np.float32)
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            grads_w[i] = np.atleast_2d(a_prev).T @ np.atleast_2d(delta)
            grads_b[i] = np.atleast_2d(delta).sum(axis=0)
            if i > 0:
                # Se propaga hacia atrás; la derivada de ReLU es 1 si z>0 y 0 si no.
                delta = (delta @ self.weights[i].T) * (pre_activations[i - 1] > 0)
        return grads_w, grads_b

    # ------------------------------------------------------------------ utilidades
    def parameters(self) -> list[np.ndarray]:
        """Lista plana [W0, b0, W1, b1, ...] (referencias, se modifican in situ)."""
        params: list[np.ndarray] = []
        for w, b in zip(self.weights, self.biases):
            params += [w, b]
        return params

    def copy_from(self, other: "NeuralNetwork") -> None:
        for mine, theirs in zip(self.parameters(), other.parameters()):
            mine[...] = theirs

    def save(self, path: str | Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"sizes": np.array(self.sizes)}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f"W{i}"] = w
            data[f"b{i}"] = b
        data.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> tuple["NeuralNetwork", dict]:
        """Carga una red guardada. Devuelve (red, metadatos)."""
        with np.load(Path(path)) as data:
            net = cls(tuple(int(s) for s in data["sizes"]))
            for i in range(len(net.weights)):
                net.weights[i][...] = data[f"W{i}"]
                net.biases[i][...] = data[f"b{i}"]
            meta = {k[5:]: data[k].item() for k in data.files if k.startswith("meta_")}
        return net, meta
