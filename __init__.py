# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sudoku Rl Environment."""

from .models import SudokuRlAction, SudokuRlObservation, SudokuRlState

__all__ = [
    "SudokuRlAction",
    "SudokuRlObservation",
    "SudokuRlState",
    "SudokuRlEnv",
]


def __getattr__(name: str):
    if name == "SudokuRlEnv":
        from .client import SudokuRlEnv

        return SudokuRlEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
