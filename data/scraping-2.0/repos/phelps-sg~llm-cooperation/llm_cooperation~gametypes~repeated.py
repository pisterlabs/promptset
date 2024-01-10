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
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, transcript

from llm_cooperation import (
    CT,
    CT_co,
    CT_contra,
    ModelSetup,
    Participant,
    Payoffs,
    Results,
)
from llm_cooperation.gametypes import PromptGenerator, start_game

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Choices(Generic[CT]):
    user: CT
    ai: CT


@dataclass(frozen=True)
class Scores:
    user: float
    ai: float


@dataclass(frozen=True)
class GameSetup(Generic[CT]):
    num_rounds: int
    generate_instruction_prompt: PromptGenerator
    next_round: RoundGenerator[CT]
    analyse_rounds: RoundsAnalyser[CT]
    payoffs: PayoffFunction[CT]
    extract_choice: ChoiceExtractor[CT]
    model_setup: ModelSetup


@dataclass(frozen=True)
class ExperimentSetup(Generic[CT]):
    num_replications: int
    compute_freq: CooperationFrequencyFunction[CT]


@dataclass(frozen=True)
class GameState(Generic[CT]):
    messages: List[Completion]
    round: int
    game_setup: GameSetup[CT]
    participant_condition: Participant


class ChoiceExtractor(Protocol[CT_co]):
    def __call__(
        self, participant: Participant, completion: Completion, **kwargs: bool
    ) -> CT_co:
        ...


class Strategy(Protocol[CT_co]):
    def __call__(self, state: GameState, **kwargs: bool) -> CT_co:
        ...


RoundGenerator = Callable[[Strategy[CT], GameState[CT]], List[Completion]]


class CooperationFrequencyFunction(Protocol[CT]):
    def __call__(self, choices: List[Choices[CT]]) -> float:
        ...


class PayoffFunction(Protocol[CT_contra]):
    def __call__(self, player1: CT_contra, player2: CT_contra) -> Payoffs:
        ...


ResultForRound = Tuple[Scores, Choices[CT]]
RoundsAnalyser = Callable[
    [List[Completion], PayoffFunction[CT], ChoiceExtractor[CT], Participant],
    List[ResultForRound[CT]],
]

ResultRepeatedGame = Tuple[
    Participant,
    str,
    float,
    float,
    Optional[List[Choices]],
    List[str],
    str,
    float,
]


class RepeatedGameResults(Results):
    def __init__(self, rows: Iterable[ResultRepeatedGame]):
        self._rows: Iterable[ResultRepeatedGame] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            # pylint: disable=unnecessary-comprehension
            [
                (
                    condition,
                    strategy,
                    score,
                    freq,
                    choices,
                    history,
                    model,
                    temp,
                )
                # pylint: disable=line-too-long
                for condition, strategy, score, freq, choices, history, model, temp in self._rows
            ],
            columns=[
                "Participant Condition",
                "Partner Condition",
                "Score",
                "Cooperation frequency",
                "Choices",
                "Transcript",
                "Model",
                "Temperature",
            ],
        )


def play_game(
    participant: Participant,
    partner_strategy: Strategy[CT],
    game_setup: GameSetup[CT],
) -> List[Completion]:
    gpt_completions, messages = start_game(
        game_setup.generate_instruction_prompt,
        game_setup.model_setup,
        participant,
    )
    for i in range(game_setup.num_rounds):
        completion = gpt_completions(messages, 1)
        messages += completion
        partner_response = game_setup.next_round(
            partner_strategy,
            GameState(messages, i, game_setup, participant),
        )
        messages += partner_response
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    analyse_rounds: RoundsAnalyser[CT],
    participant_condition: Participant,
) -> Tuple[Scores, List[Choices[CT]]]:
    conversation = conversation[1:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    results = analyse_rounds(
        conversation, payoffs, extract_choice, participant_condition
    )
    user_score = sum((scores.user for scores, _ in results))
    ai_score = sum((scores.ai for scores, _ in results))
    return Scores(user_score, ai_score), [choices for _, choices in results]


def analyse(
    conversation: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    compute_freq: CooperationFrequencyFunction[CT],
    analyse_rounds: RoundsAnalyser,
    participant_condition: Participant,
) -> Tuple[float, float, Optional[List[Choices[CT]]], List[str]]:
    try:
        history = transcript(conversation)
        result: Tuple[Scores, List[Choices[CT]]] = compute_scores(
            list(conversation),
            payoffs,
            extract_choice,
            analyse_rounds,
            participant_condition,
        )
        scores, choices = result
        freq = compute_freq(choices)
        return scores.ai, freq, choices, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def generate_replications(
    participant: Participant,
    partner_strategy: Strategy[CT],
    measurement_setup: ExperimentSetup[CT],
    game_setup: GameSetup[CT],
) -> Iterable[Tuple[float, float, Optional[List[Choices[CT]]], List[str]]]:
    # pylint: disable=R0801
    for __i__ in range(measurement_setup.num_replications):
        try:
            conversation = play_game(
                partner_strategy=partner_strategy,
                game_setup=game_setup,
                participant=participant,
            )
            yield analyse(
                conversation,
                game_setup.payoffs,
                game_setup.extract_choice,
                measurement_setup.compute_freq,
                game_setup.analyse_rounds,
                participant,
            )
        except ValueError as ex:
            logger.exception(ex)
            yield np.nan, np.nan, None, [str(ex)]


def run_experiment(
    participants: Iterable[Participant],
    partner_conditions: Dict[str, Strategy[CT]],
    experiment_setup: ExperimentSetup[CT],
    game_setup: GameSetup[CT],
) -> RepeatedGameResults:
    return RepeatedGameResults(
        (
            participant,
            strategy_name,
            score,
            freq,
            choices,
            history,
            game_setup.model_setup.model,
            game_setup.model_setup.temperature,
        )
        for participant in participants
        for strategy_name, strategy_fn in partner_conditions.items()
        for score, freq, choices, history in generate_replications(
            participant=participant,
            partner_strategy=strategy_fn,
            measurement_setup=experiment_setup,
            game_setup=game_setup,
        )
    )
