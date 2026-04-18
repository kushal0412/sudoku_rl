"""Unsloth-driven inference loop for the Sudoku RL environment.

This script keeps the environment side unchanged and runs a local language
model as the policy. It supports either:

- connecting to an already-running server via ``BASE_URL`` / ``SUDOKU_BASE_URL``
- starting from a Docker image via ``SudokuRlEnv.from_docker_image(...)``

Required extra packages are not part of the base ``sudoku_rl`` environment:

- ``unsloth``
- ``transformers``
- ``torch``
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any

try:
    from sudoku_rl import SudokuRlAction, SudokuRlEnv
except ImportError:
    from __init__ import SudokuRlAction, SudokuRlEnv


IMAGE_NAME = os.getenv("IMAGE_NAME") or "sudoku_rl:latest"
BASE_URL = os.getenv("SUDOKU_BASE_URL") or os.getenv("BASE_URL") or ""
MODEL_NAME = os.getenv("MODEL_NAME") or ""
TASK_NAME = os.getenv("SudokuRlAction_TASK", "solve_sudoku_with_unsloth")
BENCHMARK = os.getenv("SudokuRlEnv_BENCHMARK", "sudoku_rl")
EMPTY_BOXES = int(os.getenv("EMPTY_BOXES", "35"))
MAX_STEPS_ENV = os.getenv("MAX_STEPS")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").strip().lower() not in {"0", "false", "no"}
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").strip().lower() in {"1", "true", "yes"}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving a Sudoku puzzle one move at a time.

    Return exactly one JSON object with this schema:
    {"row": <1-9>, "column": <1-9>, "value": <1-9>}

    Rules:
    - Choose one editable cell and one value from 1 to 9.
    - Prefer correcting currently invalid cells before filling untouched cells.
    - Do not include markdown, explanations, code fences, or extra text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: SudokuRlAction,
    reward: float,
    done: bool,
    status: str,
    score: int,
    invalid_cells: int,
    source: str,
    message: str,
) -> None:
    message_clean = re.sub(r"\s+", " ", message).strip()
    print(
        "[STEP] "
        f"step={step} row={action.row} column={action.column} value={action.value} "
        f"reward={reward:.2f} score={score} invalid_cells={invalid_cells} "
        f"status={status} done={str(done).lower()} source={source} "
        f"message={message_clean}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, final_status: str, raw_score: int) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"final_status={final_status} raw_score={raw_score}",
        flush=True,
    )


def build_user_prompt(step: int, observation: Any, history: list[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    invalid_block = observation.invalid_cells if observation.invalid_cells else "[]"
    return textwrap.dedent(
        f"""
        Step: {step}
        Status: {observation.status}
        Score: {observation.score}
        Score delta from last move: {observation.score_delta}
        Moves attempted: {observation.moves}
        Mistakes: {observation.mistakes}
        Empty cells remaining: {observation.empty_cells_remaining}
        Incorrect cells: {observation.incorrect_cells}
        Invalid cells (0-based [row, col]): {invalid_block}
        Last environment message: {observation.message}

        Current board:
        {observation.board_text}

        Previous recent actions:
        {history_block}

        Return the next move as JSON only.
        """
    ).strip()


def load_unsloth_model() -> tuple[Any, Any]:
    if not MODEL_NAME:
        raise RuntimeError(
            "Set MODEL_NAME to an Unsloth-compatible checkpoint before running "
            "`sudoku_rl/inference_unsloth.py`."
        )

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is not installed. Install `unsloth`, `transformers`, and `torch` "
            "in the environment where you run this script."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_model_text(model: Any, tokenizer: Any, user_prompt: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        rendered = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        prompt_inputs = tokenizer(rendered, return_tensors="pt").input_ids

    prompt_inputs = prompt_inputs.to(model.device)
    generation_kwargs = {
        "input_ids": prompt_inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "do_sample": TEMPERATURE > 0,
        "pad_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if TEMPERATURE > 0:
        generation_kwargs["temperature"] = TEMPERATURE
        generation_kwargs["top_p"] = TOP_P

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    generated = outputs[0][prompt_inputs.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_action(text: str) -> SudokuRlAction | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    candidates: list[dict[str, Any]] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{[^{}]+\}", cleaned, flags=re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)

    row_match = re.search(r'"?row"?\s*[:=]\s*([1-9])', cleaned, flags=re.IGNORECASE)
    column_match = re.search(r'"?(?:column|col)"?\s*[:=]\s*([1-9])', cleaned, flags=re.IGNORECASE)
    value_match = re.search(r'"?(?:value|number)"?\s*[:=]\s*([1-9])', cleaned, flags=re.IGNORECASE)
    if row_match and column_match and value_match:
        candidates.append(
            {
                "row": int(row_match.group(1)),
                "column": int(column_match.group(1)),
                "value": int(value_match.group(1)),
            }
        )

    for candidate in candidates:
        try:
            row = int(candidate["row"])
            column = int(candidate["column"])
            value = int(candidate.get("value", candidate.get("number")))
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= row <= 9 and 1 <= column <= 9 and 1 <= value <= 9:
            return SudokuRlAction(row=row, column=column, value=value)

    return None


def candidates_for_cell(board: list[list[int]], row_index: int, column_index: int) -> list[int]:
    if board[row_index][column_index] != 0:
        return []

    used_values = set(board[row_index])
    used_values.update(board[r][column_index] for r in range(9))

    box_row = (row_index // 3) * 3
    box_column = (column_index // 3) * 3
    for sub_row in range(box_row, box_row + 3):
        for sub_column in range(box_column, box_column + 3):
            used_values.add(board[sub_row][sub_column])

    return [value for value in range(1, 10) if value not in used_values]


def heuristic_action(observation: Any) -> SudokuRlAction:
    board = [row[:] for row in observation.board]
    invalid_targets = {
        (int(row_index), int(column_index))
        for row_index, column_index in observation.invalid_cells
        if 0 <= row_index < 9 and 0 <= column_index < 9
    }

    for row_index, column_index in invalid_targets:
        board[row_index][column_index] = 0

    target_cells: list[tuple[int, int]] = list(invalid_targets)
    for row_index in range(9):
        for column_index in range(9):
            if observation.initial_puzzle[row_index][column_index] != 0:
                continue
            if board[row_index][column_index] == 0 and (row_index, column_index) not in invalid_targets:
                target_cells.append((row_index, column_index))

    if not target_cells:
        for row_index in range(9):
            for column_index in range(9):
                if observation.initial_puzzle[row_index][column_index] == 0:
                    target_cells.append((row_index, column_index))

    for row_index, column_index in target_cells:
        values = candidates_for_cell(board, row_index, column_index)
        if values:
            return SudokuRlAction(row=row_index + 1, column=column_index + 1, value=values[0])

    row_index, column_index = target_cells[0] if target_cells else (0, 0)
    return SudokuRlAction(row=row_index + 1, column=column_index + 1, value=1)


async def create_env() -> Any:
    if BASE_URL:
        return SudokuRlEnv(base_url=BASE_URL)
    return await SudokuRlEnv.from_docker_image(IMAGE_NAME)


async def main() -> None:
    model, tokenizer = load_unsloth_model()
    env = await create_env()

    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    success = False
    final_status = "unknown"
    final_score_raw = 0
    normalized_score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(empty_boxes=EMPTY_BOXES)
        observation = result.observation
        default_max_steps = max(observation.empty_boxes * 3, 40)
        max_steps = int(MAX_STEPS_ENV) if MAX_STEPS_ENV is not None else default_max_steps

        for step in range(1, max_steps + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step=step, observation=observation, history=history)
            raw_text = generate_model_text(model=model, tokenizer=tokenizer, user_prompt=user_prompt)
            action = parse_action(raw_text)
            source = "model"
            if action is None:
                action = heuristic_action(observation)
                source = "heuristic_fallback"

            result = await env.step(action)
            observation = result.observation

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            final_status = observation.status
            final_score_raw = observation.score

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=result.done,
                status=observation.status,
                score=observation.score,
                invalid_cells=len(observation.invalid_cells),
                source=source,
                message=observation.message,
            )

            history.append(
                f"step={step} row={action.row} column={action.column} value={action.value} "
                f"reward={reward:+.2f} status={observation.status}"
            )

            if result.done:
                break

        success = final_status == "solved"
        max_possible_score = max(observation.empty_boxes * max(observation.score_step, 1), 1)
        normalized_score = max(min(final_score_raw / max_possible_score, 1.0), 0.0)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)
        log_end(
            success=success,
            steps=steps_taken,
            score=normalized_score,
            final_status=final_status,
            raw_score=final_score_raw,
        )


if __name__ == "__main__":
    asyncio.run(main())
