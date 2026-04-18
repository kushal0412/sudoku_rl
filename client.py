"""Sudoku RL environment client."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SudokuRlAction, SudokuRlObservation, SudokuRlState


class SudokuRlEnv(EnvClient[SudokuRlAction, SudokuRlObservation, SudokuRlState]):
    """Client for the Sudoku RL environment."""

    def _step_payload(self, action: SudokuRlAction) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_name in ("index", "row", "column", "value", "number"):
            field_value = getattr(action, field_name)
            if field_value is not None:
                payload[field_name] = field_value
        return payload

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SudokuRlObservation]:
        observation = SudokuRlObservation.model_validate(
            {
                **payload.get("observation", {}),
                "reward": payload.get("reward"),
                "done": payload.get("done", False),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> SudokuRlState:
        return SudokuRlState.model_validate(payload)

    async def reset_puzzle(self, empty_boxes: int = 35, **kwargs: Any) -> StepResult[SudokuRlObservation]:
        """Reset the environment while choosing the number of empty cells."""
        return await self.reset(empty_boxes=empty_boxes, **kwargs)

    async def update_cell(
        self,
        *,
        index: int | None = None,
        row: int | None = None,
        column: int | None = None,
        value: int,
    ) -> StepResult[SudokuRlObservation]:
        """Convenience helper for one Sudoku move."""
        return await self.step(
            SudokuRlAction(index=index, row=row, column=column, value=value)
        )
