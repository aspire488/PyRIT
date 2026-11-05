# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from unit.mocks import MockPromptTarget
from pyrit.score import NumericRange, SelfAskGeneralFloatScaleScorer, SelfAskGeneralTrueFalseScorer


_OUTPUT_KEYS = [
    "score_value_output_key",
    "rationale_output_key",
    "description_output_key",
    "metadata_output_key",
    "category_output_key",
]


@pytest.mark.parametrize("output_key", _OUTPUT_KEYS)
def test_general_true_false_identifier_includes_output_keys(output_key: str) -> None:
    target = MockPromptTarget()
    default = SelfAskGeneralTrueFalseScorer(
        chat_target=target, system_prompt_format_string="Judge the response."
    )
    configured = SelfAskGeneralTrueFalseScorer(
        chat_target=target,
        system_prompt_format_string="Judge the response.",
        **{output_key: "custom_" + output_key},
    )

    assert default.get_identifier().unique_name != configured.get_identifier().unique_name


@pytest.mark.parametrize("output_key", _OUTPUT_KEYS)
def test_general_float_scale_identifier_includes_output_keys(output_key: str) -> None:
    target = MockPromptTarget()
    default = SelfAskGeneralFloatScaleScorer(
        chat_target=target,
        system_prompt_format_string="Judge the response.",
        scale=NumericRange(minimum_value=0, maximum_value=1),
    )
    configured = SelfAskGeneralFloatScaleScorer(
        chat_target=target,
        system_prompt_format_string="Judge the response.",
        scale=NumericRange(minimum_value=0, maximum_value=1),
        **{output_key: "custom_" + output_key},
    )

    assert default.get_identifier().unique_name != configured.get_identifier().unique_name
