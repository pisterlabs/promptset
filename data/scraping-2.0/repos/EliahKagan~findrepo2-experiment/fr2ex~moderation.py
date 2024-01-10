# Copyright (c) 2023 Eliah Kagan
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""
Checking OpenAI moderation endpoint results for repository names.

This project uses an OpenAI service (embeddings). But it does not involve
generating any content with OpenAI services. So checking the moderation
endpoint would only need to be done if the repository names themselves (which
on a small server likely already have scrutiny) would be considered a violation
of OpenAI's content policy.

Even beyond that, it may still be interesting. One use is to point out to the
user that the repository name they're interested in may inadvertently have
unintended interpretations. For example, a UI element that points out that a
repo name is a poor choice because it has whitespace, control characters, or
other confusing characters, could also point out if it seems to be not safe for
work, or not safe for life.

Note that, per https://beta.openai.com/docs/guides/moderation/overview:

    The moderation endpoint is free to use when monitoring the inputs and
    outputs of OpenAI APIs. We currently do not support monitoring of
    third-party traffic.

We are sending repository names as input to the text-embedding-ada-002, so this
should be fine. But it is important to keep in mind:

 1. Except in limited manual-testing scenarios, if we cache a repository name's
    embedding, and its moderation scores are also computed (including before),
    then we should also cache the moderation endpoint's output, so we don't end
    up sending significantly more requests to the moderation API than to the
    embeddings API.

 2. If we do an experiment that does not involve using any OpenAI models, for
    example if we compute embeddings with a non-OpenAI model without also
    computing them with an OpenAI model, then that experiment MUST NOT send the
    repository names (nor other data) to the OpenAI moderation endpoint.
"""

__all__ = [
    'Categories',
    'CategoryScores',
    'Result',
    'any_flagged',
    'get_moderation',
]

from typing import Any, TypedDict, cast

import more_itertools
import openai

from fr2ex import _task

_CHUNK_SIZE = 32
"""Number of moderations to retrieve per API request."""


Categories = TypedDict('Categories', {
    'hate': bool,
    'hate/threatening': bool,
    'harassment': bool,
    'harassment/threatening': bool,
    'self-harm': bool,
    'self-harm/intent': bool,
    'self-harm/instructions': bool,
    'sexual': bool,
    'sexual/minors': bool,
    'violence': bool,
    'violence/graphic': bool,
})
"""Each moderation category and whether or not it was flagged."""


CategoryScores = TypedDict('CategoryScores', {
    'hate': float,
    'hate/threatening': float,
    'harassment': float,
    'harassment/threatening': float,
    'self-harm': float,
    'self-harm/intent': float,
    'self-harm/instructions': float,
    'sexual': float,
    'sexual/minors': float,
    'violence': float,
    'violence/graphic': float,
})
"""Each moderation category and the score for that category."""


class Result(TypedDict):
    """Result data, for a single text, from the OpenAI moderation endpoint."""

    categories: Categories
    """Each category and whether this moderation result shows it as flagged."""

    category_scores: CategoryScores
    """Each category's score as returned in this moderation result."""

    flagged: bool
    """
    Whether the text is considered flagged.

    Usually this has been equivalent to checking if any of the categories are
    flagged, but I am not sure if that is officially guaranteed.
    """


def any_flagged(result: Result) -> bool:
    """
    Check if the moderation result shows any category flagged.

    Usually this has been equivalent to the values associated with the
    ``flagged`` key in the result, but I am not sure if that is officially
    guaranteed.
    """
    return any(result['categories'].values())


@_task.api_task('moderation')
def get_moderation(texts: list[str]) -> list[Result]:
    """Load or query the API for a list of moderation results for all texts."""
    chunks = more_itertools.chunked(texts, _CHUNK_SIZE)
    moderations = (openai.Moderation.create(input=chunk) for chunk in chunks)
    return [result for mod in moderations for result in cast(Any, mod).results]
