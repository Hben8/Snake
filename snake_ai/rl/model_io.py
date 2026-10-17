"""Cargar cualquier modelo guardado: un MLP de NumPy (.npz clásico) o una red convolucional de PyTorch."""

from pathlib import Path

import numpy as np

from ..nn import NeuralNetwork


def load_model(path: str | Path):
    """Devuelve (red, metadatos). La red convolucional necesita PyTorch, que solo se importa si hace falta."""
    with np.load(Path(path)) as data:
        is_cnn = "arch" in data.files and str(data["arch"]) == "cnn"
    if is_cnn:
        from .torch_agent import ConvQNetwork
        return ConvQNetwork.load(path)
    return NeuralNetwork.load(path)
