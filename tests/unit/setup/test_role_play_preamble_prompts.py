# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Regression tests for role-play prompt output-format instructions."""

from pathlib import Path

import pytest

from pyrit.common.path import EXECUTOR_SEED_PROMPT_PATH, EXECUTOR_SIMULATED_TARGET_PATH

ROLE_PLAY_PROMPT_PATHS = {
    name: Path(EXECUTOR_SEED_PROMPT_PATH) / "red_teaming" / "role_play" / f"{name}.yaml"
    for name in (
        "role_play_movie_script",
        "role_play_video_game",
        "role_play_trivia_game",
        "role_play_persuasion",
        "role_play_persuasion_written",
    )
}
ROLE_PLAY_PROMPT_PATHS["role_play_next_message"] = (
    Path(EXECUTOR_SIMULATED_TARGET_PATH) / "role_play_next_message.yaml"
)


@pytest.mark.parametrize("prompt_name", ROLE_PLAY_PROMPT_PATHS)
def test_role_play_prompt_explicitly_avoids_preamble(prompt_name):
    """All role-play prompts should require the forwarded message to start in character."""
    text = ROLE_PLAY_PROMPT_PATHS[prompt_name].read_text(encoding="utf-8")
    assert "next_message" in text
    assert "preamble" in text.lower()
    assert "first words" in text.lower()


@pytest.mark.parametrize("prompt_name", ROLE_PLAY_PROMPT_PATHS)
def test_role_play_prompt_block_headings_and_top_level_items_keep_base_indent(prompt_name):
    """Guard YAML block-scalar indentation so edits do not accidentally nest prompt lists."""
    lines = ROLE_PLAY_PROMPT_PATHS[prompt_name].read_text(encoding="utf-8").splitlines()
    value_index = next(i for i, line in enumerate(lines) if line.startswith("value:"))
    content_lines = lines[value_index + 1 :]
    first = next(line for line in content_lines if line.strip())
    base_indent = len(first) - len(first.lstrip())
    assert base_indent > 0

    for line in content_lines:
        stripped = line.lstrip()
        is_top_level_item = (
            stripped.startswith(("- ", "## "))
            or (len(stripped) > 2 and stripped[0].isdigit() and ". " in stripped[:5])
        )
        if is_top_level_item:
            indent = len(line) - len(stripped)
            assert indent == base_indent, (
                f"{prompt_name}: {stripped!r} has indent {indent}; expected {base_indent}"
            )
