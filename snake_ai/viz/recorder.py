"""Graba la ventana en un GIF animado (para el README, para compartir, etc.)."""

from pathlib import Path

import pygame
from PIL import Image


class GifRecorder:
    def __init__(self, path: str | Path, fps: int = 15, scale: float = 0.6,
                 every: int = 1, max_frames: int = 600):
        self.path = Path(path)
        self.fps = fps
        self.scale = scale
        self.every = every
        self.max_frames = max_frames
        self._frames: list[Image.Image] = []
        self._calls = 0

    @property
    def full(self) -> bool:
        return len(self._frames) >= self.max_frames

    def capture(self, surface: pygame.Surface) -> None:
        self._calls += 1
        if self.full or self._calls % self.every:
            return
        size = surface.get_size()
        image = Image.frombytes("RGB", size, pygame.image.tobytes(surface, "RGB"))
        target = (int(size[0] * self.scale), int(size[1] * self.scale))
        self._frames.append(image.resize(target, Image.LANCZOS))

    def save(self) -> None:
        if not self._frames:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frames = [f.quantize(colors=48, method=Image.Quantize.FASTOCTREE) for f in self._frames]
        frames[0].save(self.path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / self.fps), loop=0, optimize=True)
        print(f"GIF guardado en {self.path} ({len(frames)} fotogramas)")
