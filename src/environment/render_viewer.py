"""Tiny Tk/Pillow viewer for numpy RGB frames returned by crafter.Env.render()."""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageTk
import tkinter as tk


class RenderViewer:
    """Persistent window that displays one RGB frame at a time."""

    def __init__(self, title: str = "Voyager Crafter") -> None:
        self._root = tk.Tk()
        self._root.title(title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._image_ref: Any = None

    def show(self, frame) -> None:
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        self._image_ref = photo
        self._label.configure(image=photo)
        self._root.update_idletasks()
        self._root.update()
