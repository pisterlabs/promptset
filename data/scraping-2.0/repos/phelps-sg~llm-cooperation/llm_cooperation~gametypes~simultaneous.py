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

from __future__ import annotations

import logging
from typing import List, Tuple

from openai_pygenerator import Completion, is_assistant_role, user_message

from llm_cooperation import CT, Participant
from llm_cooperation.experiments import apply_case_condition
from llm_cooperation.gametypes.repeated import (
    ChoiceExtractor,
    Choices,
    GameState,
    PayoffFunction,
    ResultForRound,
    Scores,
    Strategy,
)

logger = logging.getLogger(__name__)


def analyse_round(
    i: int,
    conversation: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    participant_condition: Participant,
) -> Tuple[Scores, Choices[CT]]:
    assert is_assistant_role(conversation[i * 2])
    ai_choice = extract_choice(participant_condition, conversation[i * 2])
    user_choice = extract_choice(participant_condition, conversation[i * 2 + 1])
    logger.debug("round = %d", i)
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice)
    return Scores(user, ai), Choices(user_choice, ai_choice)


def analyse_rounds(
    history: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    participant_condition: Participant,
) -> List[ResultForRound[CT]]:
    num_messages = len(history)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    return [
        analyse_round(i, history, payoffs, extract_choice, participant_condition)
        for i in range(num_messages // 2)
    ]


def next_round(
    partner_strategy: Strategy[CT], state: GameState[CT]
) -> List[Completion]:
    previous_message = state.messages[-1]
    logger.debug("previous_message = %s", previous_message)
    ai_choice = state.game_setup.extract_choice(
        state.participant_condition, previous_message
    )
    user_choice = partner_strategy(state)
    logger.debug("ai_choice = %s", ai_choice)
    logger.debug("user_choice = %s", user_choice)
    ai_payoff, user_payoff = state.game_setup.payoffs(ai_choice, user_choice)
    return [
        user_message(
            apply_case_condition(
                state.participant_condition,
                f"Your partner chose {user_choice.description(state.participant_condition)}"
                f" in that round, and therefore you earned {ai_payoff} and your partner"
                f" earned {user_payoff}. "
                "Now we will move on the next round. "
                "What is your choice for the next round?",
            )
        )
    ]
