import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import openai
import openai.error
import pytest
from langchain.schema import BaseMessage

from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from tests.conftest import TEST_SERVER_URL
from tests.utils.langchain import (
    ai,
    create_model,
    run_model,
    sanitize_test_name,
    sys,
    user,
)


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: BedrockDeployment
    streaming: bool
    max_tokens: Optional[int]
    stop: Optional[List[str]]

    messages: List[BaseMessage]
    test: Callable[[str], bool] | Exception

    def get_id(self):
        max_tokens_str = str(self.max_tokens) if self.max_tokens else "inf"
        stop_sequence_str = str(self.stop) if self.stop else "nonstop"
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {max_tokens_str} {stop_sequence_str} {self.name}"
        )


chat_deployments = [
    BedrockDeployment.AMAZON_TITAN_TG1_LARGE,
    BedrockDeployment.AI21_J2_GRANDE_INSTRUCT,
    BedrockDeployment.AI21_J2_JUMBO_INSTRUCT,
    BedrockDeployment.ANTHROPIC_CLAUDE_INSTANT_V1,
    BedrockDeployment.ANTHROPIC_CLAUDE_V1,
    BedrockDeployment.ANTHROPIC_CLAUDE_V2,
    BedrockDeployment.META_LLAMA2_70B_CHAT_V1,
    BedrockDeployment.COHERE_COMMAND_TEXT_V14,
]


def get_test_cases(
    deployment: BedrockDeployment, streaming: bool
) -> List[TestCase]:
    ret: List[TestCase] = []

    ret.append(
        TestCase(
            name="dialog recall",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=None,
            messages=[
                user("my name is Anton"),
                ai("nice to meet you"),
                user("what's my name?"),
            ],
            test=lambda s: "anton" in s.lower(),
        )
    )

    ret.append(
        TestCase(
            name="2+3=5",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=None,
            messages=[user("compute 2+3")],
            test=lambda s: "5" in s,
        )
    )

    ret.append(
        TestCase(
            name="empty system message",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=None,
            messages=[sys(""), user("compute 2+4")],
            test=lambda s: "6" in s,
        )
    )

    query = 'Reply with "Hello"'
    if deployment == BedrockDeployment.ANTHROPIC_CLAUDE_INSTANT_V1:
        query = 'Print "Hello"'

    ret.append(
        TestCase(
            name="hello",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=None,
            messages=[user(query)],
            test=lambda s: "hello" in s.lower() or "hi" in s.lower(),
        )
    )

    ret.append(
        TestCase(
            name="empty dialog",
            deployment=deployment,
            streaming=streaming,
            max_tokens=1,
            stop=None,
            messages=[],
            test=Exception("List of messages must not be empty"),
        )
    )

    ret.append(
        TestCase(
            name="empty user message",
            deployment=deployment,
            streaming=streaming,
            max_tokens=1,
            stop=None,
            messages=[user("")],
            test=lambda s: True,
        )
    )

    ret.append(
        TestCase(
            name="single space user message",
            deployment=deployment,
            streaming=streaming,
            max_tokens=1,
            stop=None,
            messages=[user(" ")],
            test=lambda s: True,
        )
    )

    ret.append(
        TestCase(
            name="max tokens 1",
            deployment=deployment,
            streaming=streaming,
            max_tokens=1,
            stop=None,
            messages=[user("tell me the full story of Pinocchio")],
            test=lambda s: len(s.split()) <= 1,
        )
    )

    # ai21 models do not support more than one stop word
    stop = ["world", "World"]
    if "ai21" in deployment.model_id:
        stop = ["world"]

    ret.append(
        TestCase(
            name="stop sequence",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=stop,
            messages=[user('Reply with "hello world"')],
            test=lambda s: "world" not in s.lower(),
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment in chat_deployments
        for streaming in [False, True]
        for test in get_test_cases(deployment, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_langchain(server, test: TestCase):
    model = create_model(
        TEST_SERVER_URL, test.deployment.value, test.streaming, test.max_tokens
    )

    if isinstance(test.test, Exception):
        with pytest.raises(Exception) as exc_info:
            await run_model(model, test.messages, test.streaming, test.stop)

        assert isinstance(exc_info.value, openai.error.OpenAIError)
        assert exc_info.value.http_status == 422
        assert re.search(str(test.test), str(exc_info.value))
    else:
        actual_output = await run_model(
            model, test.messages, test.streaming, test.stop
        )
        assert test.test(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
