---
title: Sudoku RL
emoji: 🧩
colorFrom: amber
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - sudoku
  - reinforcement-learning
---

# Sudoku RL

An OpenEnv Sudoku environment backed by a real episode state machine instead of a frontend-only app.

## What It Does

- `reset(empty_boxes=...)` starts a new Sudoku puzzle with the requested number of blank cells.
- `step(...)` attempts exactly one cell update using either a flat `index` (`0..80`) or `row` and `column` (`1..9`).
- Every observation returns the full board, score, invalid cells, mistake reason, and a compact `status_summary`.
- Wrong moves stay on the board and are highlighted, matching the behavior in `sudoku_app.py`.
- Previously accepted edits are locked by the environment; only empty cells or env-marked invalid cells can be changed.
- Reward logic matches the original app: each move changes score by `round(100 / empty_boxes)`, positive for valid moves and negative for mistakes.

## Action Schema

`SudokuRlAction` accepts:

- `index`: 0-based flat cell index.
- `row`: 1-based row.
- `column`: 1-based column.
- `value`: number to place.
- `number`: alias for `value`.

Use either `index` or `row` + `column`.

## Observation Schema

`SudokuRlObservation` includes:

- `status`: `ready`, `in_progress`, `invalid_move`, or `solved`
- `message`: human-readable result of the last reset or move
- `status_summary`: compact state summary for agents
- `board`: current board after the update
- `invalid_cells` and `invalid_indices`
- `mistake_reason`
- `score`, `score_delta`, `score_step`
- `moves`, `mistakes`
- `board_text`

## Quick Start

```python
from sudoku_rl import SudokuRlAction, SudokuRlEnv

with SudokuRlEnv(base_url="http://localhost:8000") as env:
    reset_result = env.reset(empty_boxes=35)
    print(reset_result.observation.board_text)

    step_result = env.step(SudokuRlAction(index=0, value=5))
    print(step_result.observation.status)
    print(step_result.observation.message)
```

## Local Development

```bash
cd sudoku_rl
uv sync
pytest
uv run --project . server
```

The custom UI is available at `/web` and mirrors the original Sudoku Gradio layout while talking to the OpenEnv backend.

## Lightning AI / Notebook Usage

If your notebook kernel starts inside the `sudoku_rl` folder, Python does not
normally see the package's parent directory, so `from sudoku_rl import ...`
can fail with `ModuleNotFoundError`.

This repo includes a local compatibility shim so that import works when you
open the project directory directly, but you still need the project
dependencies installed in the notebook environment:

```bash
cd sudoku_rl
pip install -e .
```

If you prefer `uv`:

```bash
cd sudoku_rl
uv sync
```
