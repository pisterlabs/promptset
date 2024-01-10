from __future__ import annotations

import json
from collections import UserDict, UserList
from dataclasses import dataclass
from typing import Any, Callable

from cohere import responses
from cohere.error import CohereError

from clippy.controllers.utils import truncate_left


@dataclass
class Embeddings:
    embeddings: list[list[float]]
    meta: dict

    id: str = None


@dataclass
class Tokens:
    id: str
    tokens: list[int]
    token_strings: list[str]
    meta: dict


@dataclass
class TokenLikelihood:
    token: str
    likelihood: float


@dataclass
class Generation:
    id: str  # uuid
    prompt: str
    text: str
    likelihood: float
    finish_reason: str
    token_likelihoods: list[TokenLikelihood]

    @classmethod
    def from_response(cls, resp: responses.Generations) -> Generation:
        return cls(
            id=resp.id,
            prompt=resp.prompt,
            text=resp.text,
            likelihood=resp.likelihood,
            finish_reason=resp.finish_reason,
            token_likelihoods=[TokenLikelihood(token=v.token, likelihood=v.likelihood) for v in resp.token_likelihoods],
        )


@dataclass
class Generations:
    # generations: list
    generations: list[Generation]
    meta: dict
    return_likelihoods: str

    @classmethod
    def from_response(cls, resp: responses.Generations) -> Generations:
        return cls(
            generations=[Generation.from_response(vals) for vals in resp.generations],
            meta=resp.meta,
            return_likelihoods=resp.return_likelihoods,
        )

    def __getitem__(self, idx: int) -> Generation:
        return self.generations[idx]


class CohereJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        try:
            return json.dumps(o)
        except json.JSONDecodeError:
            if hasattr(o, "__dict__"):
                return self.default(o.__dict__)
            elif isinstance(o, (UserList, list)):
                return [self.default(item) for item in o]
            elif isinstance(o, (UserDict, dict)):
                return {key: self.default(value) for key, value in o.items()}

            return super().default(o)


def make_fn(generate_func, tokenize_func, model):
    """helper to make func for threadpool"""

    def _fn(x):
        """func that is actually called by threadpool

        this takes a prompt and returns the likelihood of that prompt (hence max_tokens=0)
        """
        if len(x) == 2:
            option, prompt = x
            return_likelihoods = "ALL"
        elif len(x) == 3:
            option, prompt, return_likelihoods = x

        while True:
            try:
                if len(tokenize_func(prompt)) > 2048:
                    prompt = truncate_left(tokenize_func, prompt)
                response = generate_func(
                    prompt=prompt, max_tokens=0, model=model, return_likelihoods=return_likelihoods
                )
                return (response.generations[0].likelihood, option)
            except CohereError as e:
                print(f"Cohere fucked up: {e}")
                continue
            except ConnectionError as e:
                print(f"Connection error: {e}")
                continue

    return _fn


def _generate_func(co_client: "cohere.Client") -> Callable:
    return co_client.generate


def _tokenize_func(co_client: "cohere.Client") -> Callable:
    return co_client.tokenize
