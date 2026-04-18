"""FastAPI application for the Sudoku RL environment."""

from __future__ import annotations

import os

try:
    from openenv.core.env_server.web_interface import create_web_interface_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SudokuRlAction, SudokuRlObservation
    from .gradio_ui import build_sudoku_gradio_app
    from .sudoku_rl_environment import SudokuRlEnvironment
except ImportError:
    from models import SudokuRlAction, SudokuRlObservation
    from server.gradio_ui import build_sudoku_gradio_app
    from server.sudoku_rl_environment import SudokuRlEnvironment


app = create_web_interface_app(
    SudokuRlEnvironment,
    SudokuRlAction,
    SudokuRlObservation,
    env_name="sudoku_rl",
    gradio_builder=build_sudoku_gradio_app,
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    ws_ping_interval = float(os.getenv("UVICORN_WS_PING_INTERVAL", "600"))
    ws_ping_timeout = float(os.getenv("UVICORN_WS_PING_TIMEOUT", "600"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        ws_ping_interval=ws_ping_interval,
        ws_ping_timeout=ws_ping_timeout,
    )


if __name__ == "__main__":
    main()
