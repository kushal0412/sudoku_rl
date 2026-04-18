"""Data models for the Sudoku RL environment."""

from __future__ import annotations

from pydantic import Field, model_validator

from openenv.core.env_server.types import Action, Observation, State


class SudokuRlAction(Action):
    """Single-step action for updating one Sudoku cell."""

    index: int | None = Field(
        default=None,
        description="0-based flat cell index in the range 0..80. Optional when row and column are provided.",
    )
    row: int | None = Field(
        default=None,
        description="1-based row number in the range 1..9. Optional when index is provided.",
    )
    column: int | None = Field(
        default=None,
        description="1-based column number in the range 1..9. Optional when index is provided.",
    )
    value: int | None = Field(
        default=None,
        description="Integer value to place into the addressed cell. The environment treats values outside 1..9 as mistakes while still returning the updated board status.",
    )
    number: int | None = Field(
        default=None,
        description="Alias for value. When both are provided they must match.",
    )

    @model_validator(mode="after")
    def validate_addressing(self) -> "SudokuRlAction":
        if self.index is None and (self.row is None or self.column is None):
            raise ValueError("Provide either index or both row and column.")
        if self.value is None and self.number is None:
            raise ValueError("Provide a value or number for the move.")
        if self.value is not None and self.number is not None and self.value != self.number:
            raise ValueError("value and number must match when both are provided.")
        if self.value is None:
            self.value = self.number
        if self.number is None:
            self.number = self.value
        return self


class SudokuRlObservation(Observation):
    """Observation returned after reset and after each move."""

    status: str = Field(
        default="ready",
        description="Episode status: ready, in_progress, invalid_move, or solved.",
    )
    message: str = Field(default="", description="Human-readable explanation of the current state.")
    status_summary: str = Field(default="", description="Compact status summary intended for agents and debugging.")
    board: list[list[int]] = Field(default_factory=list, description="Current Sudoku board after the latest update.")
    initial_puzzle: list[list[int]] = Field(default_factory=list, description="Initial puzzle shown at the start of the episode.")
    invalid_cells: list[list[int]] = Field(
        default_factory=list,
        description="Invalid cells encoded as [row_index, column_index] pairs using 0-based coordinates.",
    )
    invalid_indices: list[int] = Field(
        default_factory=list,
        description="Invalid cells encoded as 0-based flat indices.",
    )
    board_text: str = Field(default="", description="Current board rendered as plain text with '.' for empty cells.")
    move_valid: bool | None = Field(default=None, description="Whether the latest move was valid.")
    board_updated: bool = Field(default=False, description="Whether the addressed board cell was updated before validation.")
    mistake_reason: str = Field(default="", description="Reason for an invalid move, if any.")
    empty_boxes: int = Field(default=35, description="Number of empty cells requested at reset.")
    score: int = Field(default=0, description="Current cumulative score.")
    score_delta: int = Field(default=0, description="Score change caused by the latest reset or move.")
    score_step: int = Field(default=0, description="Absolute reward magnitude for one move in this episode.")
    moves: int = Field(default=0, description="Total attempted moves in this episode.")
    mistakes: int = Field(default=0, description="Total mistakes in this episode.")
    filled_cells: int = Field(default=0, description="Number of non-zero cells currently on the board.")
    empty_cells_remaining: int = Field(default=0, description="Number of zero-valued cells currently on the board.")
    incorrect_cells: int = Field(default=0, description="Number of filled cells that do not match the hidden solution.")
    last_index: int | None = Field(default=None, description="0-based flat index targeted by the latest move.")
    last_row: int | None = Field(default=None, description="1-based row targeted by the latest move.")
    last_column: int | None = Field(default=None, description="1-based column targeted by the latest move.")
    last_value: int | None = Field(default=None, description="Value submitted in the latest move.")


class SudokuRlState(State):
    """Server-side state snapshot for the current Sudoku episode."""

    status: str = Field(default="ready", description="Current episode status.")
    board: list[list[int]] = Field(default_factory=list, description="Current board.")
    initial_puzzle: list[list[int]] = Field(default_factory=list, description="Initial puzzle.")
    solution_board: list[list[int]] = Field(default_factory=list, description="Solved board stored on the server.")
    invalid_indices: list[int] = Field(default_factory=list, description="Current invalid flat indices.")
    moves: int = Field(default=0, description="Attempted moves.")
    mistakes: int = Field(default=0, description="Mistake count.")
    score: int = Field(default=0, description="Current score.")
    empty_boxes: int = Field(default=35, description="Requested reset difficulty.")
    done: bool = Field(default=False, description="Whether the puzzle is solved.")
