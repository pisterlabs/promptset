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
from openai_pygenerator import Completion, user_message

from llm_cooperation import Participant, assistant_message
from llm_cooperation.experiments.ultimatum import (
    Accept,
    ProposerChoice,
    Reject,
    extract_choice_ultimatum,
    payoffs_ultimatum,
)
from llm_cooperation.gametypes import alternating
from llm_cooperation.gametypes.repeated import Choices


@pytest.mark.parametrize(
    "i, expected_choices",
    [
        (0, Choices(ai=ProposerChoice(10.0), user=Accept)),  # type: ignore
        (1, Choices(user=ProposerChoice(10.0), ai=Accept)),  # type: ignore
        (2, Choices(ai=ProposerChoice(7.0), user=Reject)),  # type: ignore
        (3, Choices(user=ProposerChoice(10.0), ai=Accept)),  # type: ignore
        (4, Choices(ai=ProposerChoice(5.00), user=Accept)),  # type: ignore
    ],
)
def test_analyse_round(
    i: int, expected_choices: Choices, alternating_history, base_condition: Participant
):
    __scores__, choices = alternating.analyse_round(
        i,
        alternating_history,
        payoffs_ultimatum,
        extract_choice_ultimatum,
        base_condition,
    )
    assert choices == expected_choices


@pytest.fixture
def base_condition() -> Participant:
    return Participant(dict())


@pytest.fixture
def alternating_history() -> List[Completion]:
    return [
        assistant_message("Propose $10"),
        user_message("Accept,\n Propose $10"),
        assistant_message(
            """I accept your offer.
For the next round I propose $7"""
        ),
        user_message("Reject / Propose $10"),
        assistant_message("I Accept, and then I propose $5"),
        user_message("Accept"),
    ]
