"""Compatibility shim for running from the project directory itself.

When Python starts with this directory as the working directory, importing
`sudoku_rl` would normally fail because the parent directory of the package is
not on `sys.path`. This module makes `import sudoku_rl` work in that case while
still behaving like a package so `sudoku_rl.models` and similar imports resolve
correctly.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


__path__ = [str(Path(__file__).resolve().parent)]

__all__ = [
    "SudokuRlAction",
    "SudokuRlObservation",
    "SudokuRlState",
    "SudokuRlEnv",
]


def __getattr__(name: str):
    if name in {"SudokuRlAction", "SudokuRlObservation", "SudokuRlState"}:
        return getattr(import_module(f"{__name__}.models"), name)
    if name == "SudokuRlEnv":
        return import_module(f"{__name__}.client").SudokuRlEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
