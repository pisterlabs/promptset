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
from unittest.mock import Mock

import numpy as np
import pytest
from openai_pygenerator import content, user_message

from llm_cooperation import Group, Participant, Payoffs, amount_as_str
from llm_cooperation.experiments import AI_PARTICIPANTS
from llm_cooperation.experiments.ultimatum import (
    MAX_AMOUNT,
    NUM_ROUNDS,
    Accept,
    ProposerChoice,
    Reject,
    ResponderChoice,
    ResponderEnum,
    UltimatumChoice,
    amount_from_str,
    compute_freq_ultimatum,
    extract_choice_ultimatum,
    get_prompt_ultimatum,
    next_round_ultimatum,
    payoffs_ultimatum,
    strategy_cooperate,
)
from llm_cooperation.gametypes.repeated import Choices, GameState


@pytest.mark.parametrize(
    "amount, expected", [(10, "$10.00"), (0, "$0.00"), (0.5, "$0.50")]
)
def test_amount_as_str(amount: float, expected: str):
    assert amount_as_str(amount) == expected


def test_get_instruction_prompt(base_condition):
    prompt = get_prompt_ultimatum(base_condition)  # type: ignore
    assert f"{NUM_ROUNDS} rounds" in prompt
    assert AI_PARTICIPANTS[Group.Control][0] in prompt


@pytest.mark.parametrize(
    "amount_str, expected",
    [
        ("$10.00", 10.0),
        ("$3", 3.0),
        ("$3.50", 3.50),
        ("$0.00", 0.0),
        ("$0", 0.0),
    ],
)
def test_amount_from_str(amount_str: str, expected: float):
    assert amount_from_str(amount_str) == expected


@pytest.mark.parametrize(
    "text, proposer, expected",
    [
        ("accept", False, Accept),
        ("reject", False, Reject),
        ("ACCEPT", False, Accept),
        ("REJECT", False, Reject),
        ("Accept", False, Accept),
        ("Reject", False, Reject),
        ("Rejected", False, Reject),
        (" Rejected", False, Reject),
        ("Accepted", False, Accept),
        (" Rejected", False, Reject),
        ("'Rejected'", False, Reject),
        (" 'Rejected'", False, Reject),
        ("$5.00", True, ProposerChoice(5.0)),
        ("$1.00", True, ProposerChoice(1.0)),
        (" $1.00", True, ProposerChoice(1.0)),
        ("$1.00 ", True, ProposerChoice(1.0)),
        ("$0", True, ProposerChoice(0)),
    ],
)
def test_extract_choice(
    text: str, proposer: bool, expected: UltimatumChoice, base_condition: Participant
):
    assert (
        extract_choice_ultimatum(base_condition, user_message(text), proposer=proposer)
        == expected
    )


@pytest.mark.parametrize(
    "player1, player2, expected",
    [
        (ProposerChoice(10.0), Reject, (0.0, 0.0)),
        (ProposerChoice(10.0), Accept, (0.0, 10.0)),
        (ProposerChoice(3.0), Accept, (7.0, 3.0)),
        (ProposerChoice(3.0), Reject, (0.0, 0.0)),
        (ProposerChoice(0.0), Accept, (10.0, 0.0)),
        (ProposerChoice(0.0), Reject, (0.0, 0.0)),
        (Reject, ProposerChoice(10.0), (0.0, 0.0)),
        (Accept, ProposerChoice(10.0), (10.0, 0.0)),
        (Accept, ProposerChoice(3.0), (3.0, 7.0)),
        (Reject, ProposerChoice(3.0), (0.0, 0.0)),
        (Accept, ProposerChoice(0.0), (0.0, 10.0)),
        (Reject, ProposerChoice(0.0), (0.0, 0.0)),
    ],
)
def test_payoffs_ultimatum(
    player1: UltimatumChoice, player2: UltimatumChoice, expected: Payoffs
):
    assert payoffs_ultimatum(player1, player2) == expected


@pytest.mark.parametrize(
    "choices, expected",
    [
        ([ProposerChoice(10.0), Accept, ProposerChoice(5.0), Reject], 0.75),
        ([ProposerChoice(10.0), Accept, ProposerChoice(10.0), Reject], 1.0),
        ([ProposerChoice(0.0), Accept, ProposerChoice(0.0), Reject], 0.0),
        ([ProposerChoice(np.nan), ProposerChoice(5.0)], 0.5),
    ],
)
def test_compute_freq_ultimatum(choices: List[Choices], expected: float):
    assert np.isclose(compute_freq_ultimatum(choices), expected)


@pytest.mark.parametrize(
    "propose, expected",
    [(True, ProposerChoice(MAX_AMOUNT)), (False, Accept)],
)
def test_cooperate(propose: bool, expected: UltimatumChoice):
    state = Mock()
    assert strategy_cooperate(state, propose=propose) == expected


def test_ultimatum_choice():
    assert ProposerChoice(10.0) == ProposerChoice(10.0)
    assert ProposerChoice(0.0) != ProposerChoice(5.0)
    assert ProposerChoice(10.0) != Accept
    assert Accept != Reject
    assert Accept == ResponderChoice(ResponderEnum.Accept)
    assert Reject == ResponderChoice(ResponderEnum.Reject)


@pytest.mark.parametrize(
    "user_response, user_proposal",
    [(Accept, ProposerChoice(5.0)), (Reject, ProposerChoice(10.0))],
)
def test_next_round_ultimatum(
    user_response: UltimatumChoice,
    user_proposal: UltimatumChoice,
    base_condition: Participant,
):
    # pylint: disable=unused-argument
    def test_strategy(state: GameState, **kwargs: bool) -> UltimatumChoice:
        if kwargs["propose"]:
            return user_proposal
        else:
            return user_response

    state = Mock()
    result = content(next_round_ultimatum(test_strategy, state)[0]).lower()
    assert (
        f"your partner proposes {user_proposal.description(base_condition).lower()}"
        in result
    )
    assert (
        f"your partner responded with {user_response.description(base_condition).lower()}"
        in result
    )
