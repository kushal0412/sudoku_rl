from __future__ import annotations

import random
from typing import Any


Grid = list[list[int]]
Cell = tuple[int, int]

BASE_SOLUTION_GRID: Grid = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def shuffled_groups(rng: random.Random) -> list[int]:
    groups = list(range(3))
    rng.shuffle(groups)
    positions: list[int] = []
    for group in groups:
        offsets = [0, 1, 2]
        rng.shuffle(offsets)
        positions.extend(group * 3 + offset for offset in offsets)
    return positions


def remap_digits(grid: Grid, rng: random.Random) -> Grid:
    digits = list(range(1, 10))
    shuffled_digits = digits[:]
    rng.shuffle(shuffled_digits)
    digit_map = {source: target for source, target in zip(digits, shuffled_digits)}
    return [[digit_map[value] for value in row] for row in grid]


def generate_valid_sudoku(rng: random.Random) -> Grid:
    row_order = shuffled_groups(rng)
    column_order = shuffled_groups(rng)
    shuffled_grid = [
        [BASE_SOLUTION_GRID[row_index][column_index] for column_index in column_order]
        for row_index in row_order
    ]
    return remap_digits(shuffled_grid, rng)


def build_initial_puzzle(valid_sudoku: Grid, holes: int, rng: random.Random) -> Grid:
    puzzle = copy_grid(valid_sudoku)
    positions = [(row_index, column_index) for row_index in range(9) for column_index in range(9)]
    rng.shuffle(positions)
    for row_index, column_index in positions[:holes]:
        puzzle[row_index][column_index] = 0
    return puzzle


def normalize_hole_count(value: Any, default: int = 35) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        if int(value) != value:
            return default
        parsed = int(value)
    else:
        text = str(value).strip()
        if not text or not text.lstrip("-").isdigit():
            return default
        parsed = int(text)
    return max(0, min(81, parsed))


def calculate_score_step(initial_puzzle: Grid) -> int:
    empty_cells = sum(1 for row in initial_puzzle for value in row if value == 0)
    if empty_cells <= 0:
        return 0
    return round(100 / empty_cells)


def is_sudoku_solved(grid: Grid, valid_sudoku: Grid) -> bool:
    return grid == valid_sudoku


def encode_invalid_cells(cells: set[Cell]) -> list[list[int]]:
    return [[row_index, column_index] for row_index, column_index in sorted(cells)]


def invalid_indices(cells: set[Cell]) -> list[int]:
    return [row_index * 9 + column_index for row_index, column_index in sorted(cells)]


def count_incorrect_cells(grid: Grid, valid_sudoku: Grid) -> int:
    count = 0
    for row_index in range(9):
        for column_index in range(9):
            value = grid[row_index][column_index]
            if value != 0 and value != valid_sudoku[row_index][column_index]:
                count += 1
    return count


def format_board_text(grid: Grid) -> str:
    lines: list[str] = []
    for row_index, row in enumerate(grid):
        chunks = []
        for chunk_start in range(0, 9, 3):
            chunk = row[chunk_start : chunk_start + 3]
            chunks.append(" ".join("." if value == 0 else str(value) for value in chunk))
        lines.append(" | ".join(chunks))
        if row_index in {2, 5}:
            lines.append("-" * 21)
    return "\n".join(lines)


def validate_move(
    initial_puzzle: Grid,
    user_input: Grid,
    valid_sudoku: Grid,
    row: int,
    column: int,
    number: int,
) -> str:
    if row < 1 or row > 9 or column < 1 or column > 9:
        return "Other error: row and column must be between 1 and 9."

    if number < 1 or number > 9:
        return "Other error: the move number must be between 1 and 9."

    row_index = row - 1
    column_index = column - 1

    if initial_puzzle[row_index][column_index] != 0:
        return "Other error: the selected cell is part of the original puzzle and cannot be changed."

    for current_column, current_number in enumerate(user_input[row_index]):
        if current_column != column_index and current_number == number:
            return "Error: number in row."

    for current_row_index, current_row in enumerate(user_input):
        if current_row_index != row_index and current_row[column_index] == number:
            return "Error: number in column."

    box_row_start = (row_index // 3) * 3
    box_column_start = (column_index // 3) * 3
    for box_row in range(box_row_start, box_row_start + 3):
        for box_column in range(box_column_start, box_column_start + 3):
            if (box_row, box_column) != (row_index, column_index) and user_input[box_row][box_column] == number:
                return "Error: number in box."

    if valid_sudoku[row_index][column_index] != number:
        return "Other error: value does not match the valid Sudoku solution."

    return "Valid move: this number can be placed here."


def resolve_target(index: int | None, row: int | None, column: int | None) -> tuple[int | None, int | None, str | None]:
    if index is not None:
        if index < 0 or index > 80:
            return None, None, "Other error: index must be between 0 and 80."
        row_index, column_index = divmod(index, 9)
        if row is not None and column is not None and (row != row_index + 1 or column != column_index + 1):
            return None, None, "Other error: provided index does not match the provided row and column."
        return row_index, column_index, None

    if row is None or column is None:
        return None, None, "Other error: provide either index or both row and column."

    if row < 1 or row > 9 or column < 1 or column > 9:
        return None, None, "Other error: row and column must be between 1 and 9."

    return row - 1, column - 1, None
