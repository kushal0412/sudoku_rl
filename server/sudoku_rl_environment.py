"""Sudoku RL environment implementation."""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import SudokuRlAction, SudokuRlObservation, SudokuRlState
    from ..sudoku_logic import (
        build_initial_puzzle,
        calculate_score_step,
        copy_grid,
        count_incorrect_cells,
        encode_invalid_cells,
        format_board_text,
        generate_valid_sudoku,
        invalid_indices,
        is_sudoku_solved,
        normalize_hole_count,
        resolve_target,
        validate_move,
    )
except ImportError:
    from models import SudokuRlAction, SudokuRlObservation, SudokuRlState
    from sudoku_logic import (
        build_initial_puzzle,
        calculate_score_step,
        copy_grid,
        count_incorrect_cells,
        encode_invalid_cells,
        format_board_text,
        generate_valid_sudoku,
        invalid_indices,
        is_sudoku_solved,
        normalize_hole_count,
        resolve_target,
        validate_move,
    )


class SudokuRlEnvironment(Environment):
    """Single-episode Sudoku environment with reward logic copied from sudoku_app.py."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    DEFAULT_EMPTY_BOXES: int = 35

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random()
        self._state = SudokuRlState(episode_id=str(uuid4()), step_count=0)
        self._valid_sudoku: list[list[int]] = []
        self._initial_puzzle: list[list[int]] = []
        self._board: list[list[int]] = []
        self._invalid_positions: set[tuple[int, int]] = set()
        self._empty_boxes = self.DEFAULT_EMPTY_BOXES
        self._moves = 0
        self._mistakes = 0
        self._score = 0
        self._episode_done = False

    def _clear_stale_invalid_cells(self, board: list[list[int]], keep_cell: tuple[int, int] | None) -> list[list[int]]:
        next_board = copy_grid(board)
        stale_cells = [cell for cell in self._invalid_positions if cell != keep_cell]
        for row_index, column_index in stale_cells:
            next_board[row_index][column_index] = self._initial_puzzle[row_index][column_index]
            self._invalid_positions.discard((row_index, column_index))
        return next_board

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        empty_boxes: int | None = None,
        holes: int | None = None,
        empty_cells: int | None = None,
        **kwargs,
    ) -> SudokuRlObservation:
        if seed is not None:
            self._rng.seed(seed)

        requested_empty_boxes = empty_boxes
        if requested_empty_boxes is None:
            requested_empty_boxes = holes
        if requested_empty_boxes is None:
            requested_empty_boxes = empty_cells
        if requested_empty_boxes is None:
            requested_empty_boxes = kwargs.get("empty_boxes")
        if requested_empty_boxes is None:
            requested_empty_boxes = kwargs.get("holes")
        if requested_empty_boxes is None:
            requested_empty_boxes = kwargs.get("empty_cells")

        self._empty_boxes = normalize_hole_count(requested_empty_boxes, default=self.DEFAULT_EMPTY_BOXES)
        self._valid_sudoku = generate_valid_sudoku(self._rng)
        self._initial_puzzle = build_initial_puzzle(self._valid_sudoku, holes=self._empty_boxes, rng=self._rng)
        self._board = copy_grid(self._initial_puzzle)
        self._invalid_positions = set()
        self._moves = 0
        self._mistakes = 0
        self._score = 0
        self._episode_done = False
        self._state = SudokuRlState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        observation = self._build_observation(
            status="ready",
            message="Puzzle ready. Click any box to choose the row and column you want to fill.",
            reward=0,
            move_valid=None,
            board_updated=False,
            mistake_reason="",
            last_index=None,
            last_row=None,
            last_column=None,
            last_value=None,
            done=False,
        )
        self._sync_state(observation)
        return observation

    def step(self, action: SudokuRlAction, timeout_s: float | None = None, **kwargs) -> SudokuRlObservation:  # type: ignore[override]
        del timeout_s, kwargs
        if not self._board:
            observation = self._build_observation(
                status="invalid_move",
                message="No puzzle loaded. Call reset() to start a new puzzle.",
                reward=0,
                move_valid=False,
                board_updated=False,
                mistake_reason="No puzzle loaded. Call reset() to start a new puzzle.",
                last_index=action.index,
                last_row=action.row,
                last_column=action.column,
                last_value=action.value if action.value is not None else action.number,
                done=False,
            )
            self._sync_state(observation)
            return observation

        if self._episode_done:
            observation = self._build_observation(
                status="solved",
                message="Episode is finished. Call reset() to start a new puzzle.",
                reward=0,
                move_valid=None,
                board_updated=False,
                mistake_reason="",
                last_index=None,
                last_row=None,
                last_column=None,
                last_value=None,
                done=True,
            )
            self._sync_state(observation)
            return observation

        self._state.step_count += 1
        self._moves += 1

        value = int(action.value if action.value is not None else action.number)
        row_index, column_index, location_error = resolve_target(action.index, action.row, action.column)
        next_board = copy_grid(self._board)
        target_cell: tuple[int, int] | None = None
        addressed_cell: tuple[int, int] | None = None
        board_updated = False
        last_index = action.index
        last_row = action.row
        last_column = action.column

        if row_index is not None and column_index is not None:
            addressed_cell = (row_index, column_index)
            next_board = self._clear_stale_invalid_cells(next_board, addressed_cell)
            last_index = row_index * 9 + column_index
            last_row = row_index + 1
            last_column = column_index + 1
            if self._initial_puzzle[row_index][column_index] == 0:
                target_cell = (row_index, column_index)
                next_board[row_index][column_index] = value
                board_updated = True

        if location_error is not None:
            result = location_error
        else:
            result = validate_move(
                self._initial_puzzle,
                next_board,
                self._valid_sudoku,
                last_row if last_row is not None else 0,
                last_column if last_column is not None else 0,
                value,
            )

        score_delta = calculate_score_step(self._initial_puzzle)
        if result.startswith("Valid move"):
            if target_cell is not None:
                self._invalid_positions.discard(target_cell)
            self._board = next_board
            self._score += score_delta
            solved = is_sudoku_solved(self._board, self._valid_sudoku)
            self._episode_done = solved
            if solved:
                message = f"Sudoku solved completely. Final score: {self._score}."
                status = "solved"
            else:
                message = (
                    f"{result} Updated row {last_row}, column {last_column} with {value}. "
                    f"Score change: +{score_delta}. Total score: {self._score}."
                )
                status = "in_progress"
            observation = self._build_observation(
                status=status,
                message=message,
                reward=score_delta,
                move_valid=True,
                board_updated=board_updated,
                mistake_reason="",
                last_index=last_index,
                last_row=last_row,
                last_column=last_column,
                last_value=value,
                done=solved,
            )
            self._sync_state(observation)
            return observation

        self._mistakes += 1
        if target_cell is not None:
            self._invalid_positions.add(target_cell)
            self._board = next_board
        negative_delta = -score_delta
        self._score += negative_delta
        message = f"{result} Score change: {negative_delta}. Total score: {self._score}."
        observation = self._build_observation(
            status="invalid_move",
            message=message,
            reward=negative_delta,
            move_valid=False,
            board_updated=board_updated,
            mistake_reason=result,
            last_index=last_index,
            last_row=last_row,
            last_column=last_column,
            last_value=value,
            done=False,
        )
        self._sync_state(observation)
        return observation

    @property
    def state(self) -> SudokuRlState:
        return self._state

    @property
    def solution_board(self) -> list[list[int]]:
        return copy_grid(self._valid_sudoku)

    def _build_observation(
        self,
        *,
        status: str,
        message: str,
        reward: int,
        move_valid: bool | None,
        board_updated: bool,
        mistake_reason: str,
        last_index: int | None,
        last_row: int | None,
        last_column: int | None,
        last_value: int | None,
        done: bool,
    ) -> SudokuRlObservation:
        board = copy_grid(self._board)
        invalid_cells = encode_invalid_cells(self._invalid_positions)
        flat_invalid_indices = invalid_indices(self._invalid_positions)
        filled_cells = sum(1 for row in board for value in row if value != 0)
        empty_cells_remaining = sum(1 for row in board for value in row if value == 0)
        incorrect_cells = count_incorrect_cells(board, self._valid_sudoku)
        summary = (
            f"status={status}; moves={self._moves}; mistakes={self._mistakes}; score={self._score}; "
            f"empty_cells_remaining={empty_cells_remaining}; incorrect_cells={incorrect_cells}"
        )
        return SudokuRlObservation(
            status=status,
            message=message,
            status_summary=summary,
            board=board,
            initial_puzzle=copy_grid(self._initial_puzzle),
            invalid_cells=invalid_cells,
            invalid_indices=flat_invalid_indices,
            board_text=format_board_text(board),
            move_valid=move_valid,
            board_updated=board_updated,
            mistake_reason=mistake_reason,
            empty_boxes=self._empty_boxes,
            score=self._score,
            score_delta=reward,
            score_step=calculate_score_step(self._initial_puzzle),
            moves=self._moves,
            mistakes=self._mistakes,
            filled_cells=filled_cells,
            empty_cells_remaining=empty_cells_remaining,
            incorrect_cells=incorrect_cells,
            last_index=last_index,
            last_row=last_row,
            last_column=last_column,
            last_value=last_value,
            reward=reward,
            done=done,
            metadata={
                "step": self._state.step_count,
                "score_step": calculate_score_step(self._initial_puzzle),
                "index_is_zero_based": True,
                "row_and_column_are_one_based": True,
            },
        )

    def _sync_state(self, observation: SudokuRlObservation) -> None:
        self._state.status = observation.status
        self._state.board = copy_grid(self._board)
        self._state.initial_puzzle = copy_grid(self._initial_puzzle)
        self._state.solution_board = copy_grid(self._valid_sudoku)
        self._state.invalid_indices = list(observation.invalid_indices)
        self._state.moves = self._moves
        self._state.mistakes = self._mistakes
        self._state.score = self._score
        self._state.empty_boxes = self._empty_boxes
        self._state.done = observation.done
