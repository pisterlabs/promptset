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
import re

import pytest
from openai_pygenerator import user_message
from pytest_lazyfixture import lazy_fixture

from llm_cooperation import ConfigValue, Group, Participant
from llm_cooperation.experiments import (
    AI_PARTICIPANTS,
    CONDITION_CASE,
    CONDITION_CHAIN_OF_THOUGHT,
    CONDITION_DEFECT_FIRST,
    CONDITION_PRONOUN,
    Case,
)
from llm_cooperation.experiments.dictator import (
    BLACK,
    BLUE,
    BROWN,
    GREEN,
    TOTAL_SHARE,
    WHITE,
    DictatorChoice,
    DictatorEnum,
    all_dictator_choices,
    choice_menu,
    compute_freq_dictator,
    describe_payoffs,
    extract_choice_dictator,
    get_prompt_dictator,
    payoffs_dictator,
    payout_allo,
    payout_ego,
)


@pytest.mark.parametrize(
    "condition, enum, expected_description, expected_payoff_ego, expected_payoff_allo",
    [
        (lazy_fixture("base_condition"), DictatorEnum.BLACK, "black", 4.0, 0.0),
        (lazy_fixture("base_condition"), DictatorEnum.BROWN, "brown", 3.0, 1.0),
        (lazy_fixture("base_condition"), DictatorEnum.GREEN, "green", 2.0, 2.0),
        (lazy_fixture("base_condition"), DictatorEnum.BLUE, "blue", 1.0, 3.0),
        (lazy_fixture("base_condition"), DictatorEnum.WHITE, "white", 0.0, 4.0),
    ],
)
def test_dictator_choice(
    condition: Participant,
    enum: DictatorEnum,
    expected_description: str,
    expected_payoff_ego: float,
    expected_payoff_allo,
):
    choice = DictatorChoice(enum)
    assert expected_description in choice.description(condition).lower()
    assert choice.payoff_ego(condition) == expected_payoff_ego
    assert choice.payoff_allo(condition) == expected_payoff_allo


@pytest.mark.parametrize(
    "condition, text, expected_result",
    [
        (lazy_fixture("base_condition"), "Choice: Black", BLACK),
        (lazy_fixture("base_condition"), "Choice: 'project black'", BLACK),
        (lazy_fixture("base_condition"), "choice: 'Project BLACK'", BLACK),
        (lazy_fixture("base_condition"), "choice:Brown", BROWN),
        (lazy_fixture("base_condition"), "choice: Green", GREEN),
        (lazy_fixture("base_condition"), "Choice:   Blue", BLUE),
        (lazy_fixture("base_condition"), "Choice: White", WHITE),
        (lazy_fixture("with_numerals"), "Choice: 5", WHITE),
        (lazy_fixture("with_numbers"), "Choice: four", BLUE),
        (lazy_fixture("with_numbers"), "Choice: Three", GREEN),
    ],
)
def test_extract_choice_dictator(
    condition: Participant, text: str, expected_result: DictatorChoice
):
    assert extract_choice_dictator(condition, user_message(text)) == expected_result
    assert (
        extract_choice_dictator(condition, user_message(text.upper()))
        == expected_result
    )
    assert (
        extract_choice_dictator(condition, user_message(text.lower()))
        == expected_result
    )


@pytest.mark.parametrize(
    "condition, test_choice, expected_payoff",
    [
        (lazy_fixture("base_condition"), BLACK, 4),
        (lazy_fixture("base_condition"), BROWN, 3),
        (lazy_fixture("base_condition"), GREEN, 2),
        (lazy_fixture("base_condition"), BLUE, 1),
        (lazy_fixture("base_condition"), WHITE, 0),
        (lazy_fixture("with_labels_reversed"), BLACK, 0),
        (lazy_fixture("with_labels_reversed"), BROWN, 1),
        (lazy_fixture("with_labels_reversed"), GREEN, 2),
        (lazy_fixture("with_labels_reversed"), BLUE, 3),
        (lazy_fixture("with_labels_reversed"), WHITE, 4),
    ],
)
def test_payoffs_dictator(
    condition: Participant, test_choice: DictatorChoice, expected_payoff
):
    result = payoffs_dictator(condition, test_choice)
    assert result == expected_payoff


@pytest.mark.parametrize(
    "condition, test_choice, expected_payoff",
    [
        (lazy_fixture("base_condition"), BLACK, 0),
        (lazy_fixture("base_condition"), BROWN, 1),
        (lazy_fixture("base_condition"), GREEN, 2),
        (lazy_fixture("base_condition"), BLUE, 3),
        (lazy_fixture("base_condition"), WHITE, 4),
        (lazy_fixture("with_labels_reversed"), BLACK, 4),
        (lazy_fixture("with_labels_reversed"), BROWN, 3),
        (lazy_fixture("with_labels_reversed"), GREEN, 2),
        (lazy_fixture("with_labels_reversed"), BLUE, 1),
        (lazy_fixture("with_labels_reversed"), WHITE, 0),
    ],
)
def test_payoff_allo(
    condition: Participant, test_choice: DictatorChoice, expected_payoff
):
    result = test_choice.payoff_allo(condition)
    assert result == expected_payoff


@pytest.mark.parametrize("test_choice", all_dictator_choices)
def test_compute_freq_dictator(
    base_condition: Participant, test_choice: DictatorChoice
):
    result = compute_freq_dictator(base_condition, test_choice)
    assert result == test_choice.payoff_allo(base_condition) / TOTAL_SHARE


@pytest.mark.parametrize(
    "condition",
    [
        lazy_fixture("base_condition"),
        lazy_fixture("with_gender_neutral_pronoun"),
        lazy_fixture("with_upper_case"),
        lazy_fixture("with_chain_of_thought"),
        lazy_fixture("with_numerals"),
        lazy_fixture("with_numbers"),
    ],
)
def test_get_prompt_dictator(condition: Participant):
    prompt = get_prompt_dictator(condition)

    def contains(text: ConfigValue) -> bool:
        return str(text).lower() in prompt.lower()

    assert contains("explanation:") == condition[CONDITION_CHAIN_OF_THOUGHT]
    assert contains(condition[CONDITION_PRONOUN])
    assert contains(AI_PARTICIPANTS[Group.Control][0])
    for choice in all_dictator_choices:
        assert contains(describe_payoffs(condition, choice))
    if condition[CONDITION_CASE] == Case.UPPER.value:
        assert "THIS IS A STUDY" in prompt


@pytest.mark.parametrize(
    "condition", [lazy_fixture("base_condition"), lazy_fixture("with_defect_first")]
)
def test_choice_menu(condition: Participant):
    result = choice_menu(condition)
    black = BLACK.description(condition)
    white = WHITE.description(condition)
    for choice in all_dictator_choices:
        assert choice.description(condition) in result
    if condition[CONDITION_DEFECT_FIRST]:
        assert re.search(rf"{black}.*{white}", result)
    else:
        assert re.search(rf"{white}.*{black}", result)


@pytest.mark.parametrize(
    "condition, expected",
    [
        (lazy_fixture("base_condition"), "$4.00"),
        (lazy_fixture("with_labels_reversed"), "$0.00"),
    ],
)
def test_payout_ego(condition: Participant, expected: str):
    assert payout_ego(condition, BLACK) == expected


@pytest.mark.parametrize(
    "condition, expected",
    [
        (lazy_fixture("base_condition"), "$0.00"),
        (lazy_fixture("with_labels_reversed"), "$4.00"),
    ],
)
def test_payout_allo(condition: Participant, expected: str):
    assert payout_allo(condition, BLACK) == expected
