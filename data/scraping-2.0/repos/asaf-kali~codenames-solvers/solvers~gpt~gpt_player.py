import json
import logging
import re
from abc import ABC
from typing import List, Optional

from codenames.game.color import TeamColor
from codenames.game.move import (
    GivenGuess,
    GivenHint,
    GuessMove,
    HintMove,
    Move,
    PassMove,
)
from codenames.game.player import Player
from codenames.game.score import Score
from codenames.game.state import BaseGameState
from openai import ChatCompletion

from solvers.gpt.instructions import load_instructions

log = logging.getLogger(__name__)
INSTRUCTIONS = load_instructions()
FULL_INSTRUCTIONS = INSTRUCTIONS["full_instructions"]
SHORT_INSTRUCTIONS = INSTRUCTIONS["short_instructions"]
HINTER_TURN_COMMAND = INSTRUCTIONS["hinter_turn_command"]
GUESSER_TURN_COMMAND = INSTRUCTIONS["guesser_turn_command"]


class GPTPlayer(Player, ABC):
    def __init__(
        self,
        name: str,
        api_key: str,
        team_color: Optional[TeamColor] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
    ):
        super().__init__(name=name, team_color=team_color)
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def build_team_repr(self):
        return f"You are the {self.team_color} team {self.role}."

    @classmethod
    def build_score_repr(cls, score: Score) -> str:
        return (
            f"The current score is: "
            f"Red: {score.red.revealed}/{score.red.total}, "
            f"Blue: {score.blue.revealed}/{score.blue.total}."
        )

    @classmethod
    def build_moves_repr(cls, state: BaseGameState) -> Optional[str]:
        moves: List[Move] = state.moves
        if not moves:
            return None
        moves_repr = []
        for move in moves:
            if isinstance(move, HintMove):
                moves_repr.append(hint_repr(hint=move.given_hint))
            elif isinstance(move, GuessMove):
                moves_repr.append(guess_repr(guess=move.given_guess))
            elif isinstance(move, PassMove):
                moves_repr.append(pass_repr(move=move))
        return "\n".join(moves_repr)

    def generate_completion(self, messages: List[dict]) -> dict:
        log.debug("Sending completion request", extra={"payload_size": len(str(messages)), "messages": messages})
        response = ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        usage = response.get("usage")
        log.debug(f"Got completion response, usage: {usage}", extra={"response": response})
        return response


def extract_data_from_response(completion_result: dict) -> dict:
    response_content: str = completion_result["choices"][0]["message"]["content"]
    data_raw = find_json_in_string(response_content)
    log.debug(f"Parsing content: {data_raw}")
    data = json.loads(data_raw)
    return data


def find_json_in_string(data: str) -> str:
    match = re.search(r"\{.*}", data)
    if match:
        return match.group(0)
    raise ValueError("No JSON found in string")


def hint_repr(hint: GivenHint) -> str:
    return f"{hint.team_color} hinter said: '{hint.word}', {hint.card_amount} cards."


def guess_repr(guess: GivenGuess) -> str:
    return f"{guess.team} guesser said: {guess}."


def pass_repr(move: PassMove) -> str:
    return f"{move.team_color} team guesser passed the turn."
