import asyncio
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

from anthropic import HUMAN_PROMPT, AI_PROMPT
from anthropic.types import Completion

from langbridge.generation import AnthropicGeneration
from langbridge.schema import GenerationResponse
from langbridge.trackers import ProgressTracker
from langbridge.callbacks import BaseCallbackManager


@pytest.fixture
def fake_generation() -> AnthropicGeneration:
    generation = AnthropicGeneration(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0, "max_tokens_to_sample": 50},
        prompt=HUMAN_PROMPT + " " + "Tell me a joke." + AI_PROMPT,
        metadata={"index": 0},
        callback_manager=BaseCallbackManager(handlers=[], run_id=uuid4())
    )

    return generation


def test_prompt_tokens_computation(fake_generation: AnthropicGeneration) -> None:
    assert fake_generation.usage.prompt_tokens == 13


def test_completion_tokens(fake_generation: AnthropicGeneration) -> None:
    assert fake_generation.usage.completion_tokens == 50


@pytest.mark.asyncio
async def test_execution(
    fake_generation: AnthropicGeneration,
    monkeypatch
) -> None:
    async def mock_call_api() -> Completion:
        return Completion(
            completion="\n\nThis is a test.",
            model="claude-2",
            stop_reason="max_tokens"
        )

    # Patch the execute method of the specific instance
    monkeypatch.setattr("langbridge.generation.AnthropicGeneration._call_api", AsyncMock(side_effect=mock_call_api))

    response: GenerationResponse = await fake_generation.invoke(
        retry_queue=asyncio.Queue(),
        progress_tracker=ProgressTracker()
    )

    assert response.completion is not None
    assert fake_generation.usage.completion_tokens == 7
