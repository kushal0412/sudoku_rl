from __future__ import annotations

from sudoku_rl.models import SudokuRlAction
from sudoku_rl.server.sudoku_rl_environment import SudokuRlEnvironment


def test_reset_respects_empty_boxes() -> None:
    env = SudokuRlEnvironment()
    observation = env.reset(seed=7, empty_boxes=12)

    assert observation.status == "ready"
    assert observation.empty_boxes == 12
    assert observation.moves == 0
    assert observation.mistakes == 0
    assert observation.score == 0
    assert sum(1 for row in observation.board for value in row if value == 0) == 12


def test_invalid_move_keeps_attempted_value_and_penalizes() -> None:
    env = SudokuRlEnvironment()
    env.reset(seed=3, empty_boxes=6)
    state = env.state

    row_index = column_index = solution_value = None
    duplicate_value = None
    for current_row in range(9):
        filled_values = [value for value in state.board[current_row] if value != 0]
        empty_columns = [idx for idx, value in enumerate(state.board[current_row]) if value == 0]
        if filled_values and empty_columns:
            row_index = current_row
            column_index = empty_columns[0]
            solution_value = state.solution_board[current_row][column_index]
            duplicate_value = next(value for value in filled_values if value != solution_value)
            break

    assert row_index is not None
    assert column_index is not None
    assert duplicate_value is not None

    observation = env.step(
        SudokuRlAction(row=row_index + 1, column=column_index + 1, value=duplicate_value)
    )

    assert observation.status == "invalid_move"
    assert observation.move_valid is False
    assert observation.reward == -observation.score_step
    assert observation.score == -observation.score_step
    assert observation.mistakes == 1
    assert observation.board[row_index][column_index] == duplicate_value
    assert observation.last_index == row_index * 9 + column_index
    assert observation.last_value == duplicate_value
    assert observation.mistake_reason
    assert observation.invalid_indices == [row_index * 9 + column_index]


def test_one_hole_puzzle_solves_in_single_step() -> None:
    env = SudokuRlEnvironment()
    observation = env.reset(seed=11, empty_boxes=1)

    missing = [
        (row_index, column_index)
        for row_index in range(9)
        for column_index in range(9)
        if observation.board[row_index][column_index] == 0
    ]
    assert len(missing) == 1
    row_index, column_index = missing[0]
    correct_value = env.state.solution_board[row_index][column_index]

    solved = env.step(
        SudokuRlAction(index=row_index * 9 + column_index, value=correct_value)
    )

    assert solved.status == "solved"
    assert solved.done is True
    assert solved.move_valid is True
    assert solved.reward == 100
    assert solved.score == 100
    assert solved.mistakes == 0
    assert solved.board == env.state.solution_board


def test_previously_filled_editable_cell_cannot_be_overwritten() -> None:
    env = SudokuRlEnvironment()
    observation = env.reset(seed=13, empty_boxes=2)

    missing = [
        (row_index, column_index)
        for row_index in range(9)
        for column_index in range(9)
        if observation.board[row_index][column_index] == 0
    ]
    assert len(missing) == 2

    row_index, column_index = missing[0]
    correct_value = env.state.solution_board[row_index][column_index]
    first_move = env.step(
        SudokuRlAction(index=row_index * 9 + column_index, value=correct_value)
    )

    assert first_move.move_valid is True
    assert first_move.board[row_index][column_index] == correct_value

    overwrite_value = 1 if correct_value != 1 else 2
    overwrite_attempt = env.step(
        SudokuRlAction(index=row_index * 9 + column_index, value=overwrite_value)
    )

    assert overwrite_attempt.status == "invalid_move"
    assert overwrite_attempt.move_valid is False
    assert overwrite_attempt.board_updated is False
    assert overwrite_attempt.board[row_index][column_index] == correct_value
    assert overwrite_attempt.last_index == row_index * 9 + column_index
    assert overwrite_attempt.last_value == overwrite_value
    assert overwrite_attempt.mistake_reason == (
        "Other error: the selected cell was already updated and cannot be changed."
    )
    assert overwrite_attempt.invalid_indices == []
