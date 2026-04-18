"""Custom Gradio UI for the Sudoku RL environment."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from openenv.core.env_server.serialization import serialize_observation

try:
    from ..sudoku_logic import copy_grid, normalize_hole_count
except ImportError:
    from sudoku_logic import copy_grid, normalize_hole_count


CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(180deg, #f4efe4 0%, #e6dcc7 100%);
}

#sudoku-shell {
    max-width: 900px;
    margin: 0 auto;
}

#sudoku-shell h1,
#sudoku-shell p,
#sudoku-shell label,
#sudoku-shell .prose {
    color: #1f2933;
}

#sudoku-board {
    border: 3px solid #2f4858;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(47, 72, 88, 0.14);
    width: 520px;
}

#sudoku-board table {
    background: #fffdf8;
    width: 520px !important;
    table-layout: fixed;
}

#sudoku-board th {
    background: #2f4858;
    color: #fffaf0;
    font-weight: 700;
    height: 42px;
}

#sudoku-board td {
    background: #fffdf8 !important;
    color: #13212b !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    height: 48px;
    width: 48px;
}

#sudoku-board td,
#sudoku-board th {
    border: 1px solid #8fa3b0 !important;
    text-align: center;
}

#sudoku-board td input,
#sudoku-board td textarea,
#sudoku-board td [contenteditable="true"],
#sudoku-board td span {
    color: #13212b !important;
    background: transparent !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    text-align: center !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #13212b !important;
}

#sudoku-board tr:nth-child(3n) td {
    border-bottom: 3px solid #2f4858 !important;
}

#sudoku-board td:nth-child(3n),
#sudoku-board th:nth-child(3n) {
    border-right: 3px solid #2f4858 !important;
}

#sudoku-board td:focus,
#sudoku-board td:hover {
    background: #ffe8a3 !important;
}

#sudoku-board td:focus-within {
    background: #fff4cc !important;
    box-shadow: inset 0 0 0 2px #c97820;
}

#sudoku-board td:focus-within input,
#sudoku-board td input:focus,
#sudoku-board td textarea:focus,
#sudoku-board td [contenteditable="true"]:focus {
    background: #fffdf8 !important;
    color: #13212b !important;
    outline: none !important;
    box-shadow: none !important;
    caret-color: #13212b !important;
}

.status-box {
    border-radius: 10px;
    padding: 12px 14px;
    font-weight: 600;
    line-height: 1.4;
    margin-bottom: 12px;
}

.status-ok {
    background: #e6f4ea;
    border: 1px solid #7fb48a;
    color: #1e4d2b;
}

.status-error {
    background: #fbe4e6;
    border: 1px solid #d56b74;
    color: #7d1f28;
}

.valid-board-button {
    background: #2f4858;
    color: #fffaf0;
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 700;
    cursor: pointer;
    margin-top: 12px;
}

.valid-board-button:hover {
    background: #243846;
}

.solved-board-shell {
    width: 520px;
    margin-top: 12px;
    padding: 10px 0 0 0;
}

.solved-board-title {
    color: #1f2933;
    font-weight: 700;
    margin-bottom: 8px;
}

.solved-board-table {
    border-collapse: collapse;
    width: 520px;
    table-layout: fixed;
    background: #fffdf8;
    border: 3px solid #2f4858;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(47, 72, 88, 0.14);
}

.solved-board-table td {
    border: 1px solid #8fa3b0;
    color: #13212b;
    font-size: 18px;
    font-weight: 700;
    text-align: center;
    height: 48px;
    width: 48px;
    background: #fffdf8;
}

.solved-board-table td.block-bottom {
    border-bottom: 3px solid #2f4858;
}

.solved-board-table td.block-right {
    border-right: 3px solid #2f4858;
}
"""


def build_status_html(message: str, is_error: bool) -> str:
    status_class = "status-error" if is_error else "status-ok"
    return f'<div class="status-box {status_class}">{message}</div>'


def build_board_html(grid: list[list[int]], title: str) -> str:
    rows = []
    for row_index, row in enumerate(grid):
        cells = []
        for column_index, value in enumerate(row):
            extra_class = []
            if (row_index + 1) % 3 == 0 and row_index != 8:
                extra_class.append("block-bottom")
            if (column_index + 1) % 3 == 0 and column_index != 8:
                extra_class.append("block-right")
            class_name = f' class="{" ".join(extra_class)}"' if extra_class else ""
            cells.append(f"<td{class_name}>{value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<div class="solved-board-shell">'
        f'<div class="solved-board-title">{title}</div>'
        f'<table class="solved-board-table"><tbody>{"".join(rows)}</tbody></table>'
        f"</div>"
    )


def build_valid_board_panel(grid: list[list[int]], visible: bool = False) -> str:
    display_style = "block" if visible else "none"
    return (
        '<div class="valid-board-panel">'
        '<button class="valid-board-button" '
        'onclick="const panel=document.getElementById(\'valid-board-content\'); '
        "if(panel){panel.style.display = panel.style.display === 'none' ? 'block' : 'none';}"
        ' return false;">Show Valid Board</button>'
        f'<div id="valid-board-content" style="display: {display_style};">'
        f"{build_board_html(grid, 'Valid Sudoku Board')}"
        "</div>"
        "</div>"
    )


def empty_grid() -> list[list[int]]:
    return [[0 for _ in range(9)] for _ in range(9)]


def normalize_grid_for_view(grid: list[list[int]] | None) -> list[list[int]]:
    if (
        isinstance(grid, list)
        and len(grid) == 9
        and all(isinstance(row, list) and len(row) == 9 for row in grid)
    ):
        return copy_grid(grid)
    return empty_grid()


def format_json_payload(payload: Any) -> str:
    if payload in (None, ""):
        return ""
    return json.dumps(payload, indent=2, ensure_ascii=False)


def decode_invalid_cells(cells: list[list[int]] | None) -> set[tuple[int, int]]:
    if not cells:
        return set()
    return {(int(row), int(column)) for row, column in cells}


def build_board_view(
    grid: list[list[int]],
    initial_puzzle: list[list[int]],
    invalid_cells: list[list[int]] | None = None,
) -> Any:
    frame = pd.DataFrame(
        [["" if value == 0 else str(value) for value in row] for row in grid],
        columns=[str(index) for index in range(1, 10)],
    )
    invalid_positions = decode_invalid_cells(invalid_cells)
    style_frame = pd.DataFrame("", index=frame.index, columns=frame.columns)

    for row_index in range(9):
        for column_index in range(9):
            styles = ["text-align: center", "font-weight: 700", "color: #13212b"]
            if initial_puzzle[row_index][column_index] != 0:
                styles.append("background-color: #f2ecdd")
            else:
                styles.append("background-color: #fffdf8")
            if (row_index, column_index) in invalid_positions:
                styles.append("color: #b42318")
                styles.append("background-color: #fdebec")
            style_frame.iat[row_index, column_index] = "; ".join(styles)
    return frame.style.apply(lambda _: style_frame, axis=None)


def build_selection_state(row: int, column: int, number: int) -> list[int]:
    return [row, column, number]


def coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def select_cell(
    user_input: list[list[int]],
    puzzle_loaded: bool,
    evt: gr.SelectData,
) -> tuple[int, int, int, str, list[int]]:
    row_index, column_index = evt.index
    value = user_input[row_index][column_index]
    if not puzzle_loaded:
        message = "No puzzle loaded. Click Reset Board to start a new puzzle."
    else:
        message = (
            f"Selected row {row_index + 1}, column {column_index + 1}. "
            f"Current value: {value}."
        )
    return row_index + 1, column_index + 1, value, message, build_selection_state(
        row_index + 1, column_index + 1, value
    )


def _is_error_observation(observation: dict[str, Any]) -> bool:
    return observation.get("move_valid") is False or observation.get("status") == "invalid_move"


def _status_html_for_observation(observation: dict[str, Any]) -> str:
    return build_status_html(observation.get("message", ""), _is_error_observation(observation))


def _sync_web_manager_after_reset(web_manager, observation):
    serialized = serialize_observation(observation)
    state = web_manager.env.state
    web_manager.episode_state.episode_id = state.episode_id
    web_manager.episode_state.step_count = state.step_count
    web_manager.episode_state.current_observation = serialized["observation"]
    web_manager.episode_state.action_logs = []
    web_manager.episode_state.is_reset = True
    return serialized


async def _reset_with_empty_boxes(web_manager, empty_boxes: int) -> dict[str, Any]:
    observation = await web_manager._run_sync_in_thread_pool(
        web_manager.env.reset,
        empty_boxes=empty_boxes,
    )
    serialized = _sync_web_manager_after_reset(web_manager, observation)
    await web_manager._send_state_update()
    return serialized


def _ui_values_from_observation(
    observation: dict[str, Any],
    solution_board: list[list[int]],
    request_payload: dict[str, Any] | None,
    response_payload: dict[str, Any] | None,
) -> tuple[Any, ...]:
    board = normalize_grid_for_view(observation.get("board"))
    initial_puzzle = normalize_grid_for_view(observation.get("initial_puzzle"))
    last_row = observation.get("last_row") or 1
    last_column = observation.get("last_column") or 1
    last_value = observation.get("last_value") or 0
    puzzle_loaded = bool(observation.get("board"))
    return (
        copy_grid(board),
        build_board_view(board, initial_puzzle, observation.get("invalid_cells")),
        last_row,
        last_column,
        last_value,
        observation.get("message", ""),
        observation.get("moves", 0),
        observation.get("mistakes", 0),
        observation.get("score", 0),
        _status_html_for_observation(observation),
        build_valid_board_panel(solution_board, visible=False),
        build_selection_state(last_row, last_column, last_value),
        format_json_payload(request_payload),
        format_json_payload(response_payload),
        puzzle_loaded,
    )


def build_sudoku_gradio_app(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title,
    quick_start_md,
):
    del action_fields, metadata, is_chat_env, title, quick_start_md
    if gr is None:  # pragma: no cover
        raise ImportError("gradio is required to build the Sudoku UI.")

    blank_grid = empty_grid()
    initial_message = "No puzzle loaded. Click Reset Board to start a new puzzle."

    with gr.Blocks(title="Sudoku Move Validator") as demo:
        with gr.Column(elem_id="sudoku-shell"):
            gr.HTML(f"<style>{CUSTOM_CSS}</style>")
            gr.Markdown(
                """
                # Sudoku Move Validator

                Set how many empty cells you want, then click Reset Board to load a puzzle from OpenEnv.
                Click a box in the Sudoku board to select it. The app will show the row and column
                you are filling, then enter a number on the right and update that exact position.
                Wrong values stay on the board in red so you can correct them without losing state.
                """
            )

            board_state = gr.State(copy_grid(blank_grid))
            selection_state = gr.State(build_selection_state(1, 1, 0))
            puzzle_loaded_state = gr.State(False)

            with gr.Row():
                with gr.Column(scale=0, min_width=520):
                    board = gr.Dataframe(
                        value=build_board_view(blank_grid, blank_grid, []),
                        headers=[str(index) for index in range(1, 10)],
                        row_count=(9, "fixed"),
                        column_count=(9, "fixed"),
                        datatype="str",
                        interactive=False,
                        wrap=False,
                        label="Sudoku Board",
                        elem_id="sudoku-board",
                    )

                    with gr.Row():
                        moves_output = gr.Number(label="Moves", value=0, precision=0, interactive=False)
                        mistakes_output = gr.Number(label="Mistakes", value=0, precision=0, interactive=False)
                    score_output = gr.Number(label="Score", value=0, precision=0, interactive=False)
                    valid_board_panel = gr.HTML(
                        value=build_valid_board_panel(blank_grid, visible=False),
                    )

                with gr.Column(scale=0, min_width=300):
                    status_output = gr.HTML(
                        value=build_status_html(initial_message, is_error=False)
                    )
                    result_output = gr.Textbox(
                        label="Result",
                        value=initial_message,
                        interactive=False,
                        lines=8,
                    )

                    holes_input = gr.Number(label="Empty Cells", value=35, precision=0)
                    row_input = gr.Number(label="Selected Row", value=1, precision=0)
                    column_input = gr.Number(label="Selected Column", value=1, precision=0)
                    number_input = gr.Number(label="Number To Place", value=0, precision=0)
                    request_payload_output = gr.Textbox(
                        label="OpenEnv Request Payload",
                        value="",
                        interactive=False,
                        lines=8,
                    )
                    response_payload_output = gr.Textbox(
                        label="OpenEnv Response Payload",
                        value="",
                        interactive=False,
                        lines=16,
                    )

                    update_button = gr.Button("Update Selected Cell", variant="primary")
                    reset_button = gr.Button("Reset Board")

        async def reset_episode(holes: float):
            hole_count = normalize_hole_count(holes, default=35)
            request_payload = {"empty_boxes": hole_count}
            serialized = await _reset_with_empty_boxes(
                web_manager,
                hole_count,
            )
            observation = serialized["observation"]
            return _ui_values_from_observation(
                observation,
                web_manager.env.solution_board,
                request_payload,
                serialized,
            )

        async def update_selected_cell(row: float, column: float, number: float):
            request_payload = {
                "row": coerce_int(row, 1),
                "column": coerce_int(column, 1),
                "value": coerce_int(number, 0),
            }
            serialized = await web_manager.step_environment(
                request_payload
            )
            observation = serialized["observation"]
            return _ui_values_from_observation(
                observation,
                web_manager.env.solution_board,
                request_payload,
                serialized,
            )

        board.select(
            fn=select_cell,
            inputs=[board_state, puzzle_loaded_state],
            outputs=[row_input, column_input, number_input, result_output, selection_state],
            show_progress="hidden",
        )

        update_button.click(
            fn=update_selected_cell,
            inputs=[row_input, column_input, number_input],
            outputs=[
                board_state,
                board,
                row_input,
                column_input,
                number_input,
                result_output,
                moves_output,
                mistakes_output,
                score_output,
                status_output,
                valid_board_panel,
                selection_state,
                request_payload_output,
                response_payload_output,
                puzzle_loaded_state,
            ],
            show_progress="hidden",
        )

        reset_button.click(
            fn=reset_episode,
            inputs=[holes_input],
            outputs=[
                board_state,
                board,
                row_input,
                column_input,
                number_input,
                result_output,
                moves_output,
                mistakes_output,
                score_output,
                status_output,
                valid_board_panel,
                selection_state,
                request_payload_output,
                response_payload_output,
                puzzle_loaded_state,
            ],
            show_progress="hidden",
        )

        number_input.submit(
            fn=update_selected_cell,
            inputs=[row_input, column_input, number_input],
            outputs=[
                board_state,
                board,
                row_input,
                column_input,
                number_input,
                result_output,
                moves_output,
                mistakes_output,
                score_output,
                status_output,
                valid_board_panel,
                selection_state,
                request_payload_output,
                response_payload_output,
                puzzle_loaded_state,
            ],
            show_progress="hidden",
        )

    return demo
