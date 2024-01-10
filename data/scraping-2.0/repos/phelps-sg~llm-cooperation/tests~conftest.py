#  MIT License
#
#  Copyright (c) 2023 Steve Phelps
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

from typing import List

import pytest
from openai_pygenerator import Completion, assistant_message, user_message

from llm_cooperation import ConfigValue, Group, Participant
from llm_cooperation.experiments import (
    CONDITION_CASE,
    CONDITION_CHAIN_OF_THOUGHT,
    CONDITION_DEFECT_FIRST,
    CONDITION_GROUP,
    CONDITION_PROMPT_INDEX,
    CONDITION_PRONOUN,
    Case,
    Pronoun,
)
from llm_cooperation.experiments.dilemma import (
    CONDITION_LABEL,
    CONDITION_LABELS_REVERSED,
    Label,
)

COLOR_COOPERATE = "Green"
COLOR_DEFECT = "Blue"


@pytest.fixture
def conversation() -> List[Completion]:
    return [
        user_message("Role prompt.  What is your choice in the first round?"),
        assistant_message(f"project {COLOR_COOPERATE}"),
        user_message(f"Your partner chose project {COLOR_DEFECT}"),
        assistant_message(f"project {COLOR_DEFECT}"),
        user_message(f"Your partner chose project {COLOR_COOPERATE}"),
        assistant_message(f"project {COLOR_DEFECT}"),
        user_message(f"Your partner chose project {COLOR_DEFECT}"),
        assistant_message(f"project {COLOR_DEFECT}"),
        user_message(f"Your partner chose project {COLOR_DEFECT}"),
        assistant_message(f"project {COLOR_COOPERATE}"),
        user_message(f"project {COLOR_COOPERATE}"),
    ]


@pytest.fixture
def base_condition() -> Participant:
    return Participant(
        {
            CONDITION_GROUP: Group.Control.value,
            CONDITION_PROMPT_INDEX: 0,
            CONDITION_LABEL: Label.COLORS.value,
            CONDITION_LABELS_REVERSED: False,
            CONDITION_CHAIN_OF_THOUGHT: False,
            CONDITION_DEFECT_FIRST: False,
            CONDITION_CASE: Case.STANDARD.value,
            CONDITION_PRONOUN: Pronoun.SHE.value,
        }
    )


def modify_condition(
    base_condition: Participant, key: str, value: ConfigValue
) -> Participant:
    result = base_condition.copy()
    result[key] = value
    return Participant(result)


@pytest.fixture
def with_chain_of_thought(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_CHAIN_OF_THOUGHT, True)


@pytest.fixture
def with_numerals(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_LABEL, Label.NUMERALS.value)


@pytest.fixture
def with_numbers(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_LABEL, Label.NUMBERS.value)


@pytest.fixture
def with_upper_case(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_CASE, Case.UPPER.value)


@pytest.fixture
def with_lower_case(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_CASE, Case.LOWER.value)


@pytest.fixture
def with_defect_first(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_DEFECT_FIRST, True)


@pytest.fixture
def with_gender_neutral_pronoun(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_PRONOUN, Pronoun.THEY.value)


@pytest.fixture
def with_labels_reversed(base_condition: Participant) -> Participant:
    return modify_condition(base_condition, CONDITION_LABELS_REVERSED, True)
