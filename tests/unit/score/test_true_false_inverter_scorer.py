# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

from pyrit.models import ComponentIdentifier, Score
from pyrit.score.score_utils import ORIGINAL_FLOAT_VALUE_KEY, normalize_score_to_float
from pyrit.score.true_false.true_false_inverter_scorer import TrueFalseInverterScorer
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


def _score(value: bool, original_float: float) -> Score:
    return Score(
        score_value=str(value),
        score_value_description="threshold",
        score_type="true_false",
        score_metadata={ORIGINAL_FLOAT_VALUE_KEY: original_float},
        message_piece_id="test-piece",
        scorer_class_identifier=ComponentIdentifier(
            class_name="FloatScaleThresholdScorer",
            class_module="tests.unit.score",
        ),
    )


def test_inverter_complements_original_float_metadata() -> None:
    wrapped = MagicMock(spec=TrueFalseScorer)
    wrapped.get_identifier.return_value = ComponentIdentifier(
        class_name="FloatScaleThresholdScorer",
        class_module="tests.unit.score",
    )
    inverter = TrueFalseInverterScorer(scorer=wrapped)

    inverted = inverter._invert([_score(True, 0.73)])[0]

    assert inverted.get_value() is False
    assert inverted.score_metadata == {ORIGINAL_FLOAT_VALUE_KEY: 0.27}
    assert normalize_score_to_float(inverted) == 0.27
