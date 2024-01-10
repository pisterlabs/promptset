import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import openai.error
import pytest
from langchain.schema import BaseMessage

from aidial_adapter_vertexai.llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
)
from tests.conftest import TEST_SERVER_URL
from tests.utils.llm import (
    assert_dialog,
    create_chat_model,
    sanitize_test_name,
    sys,
    user,
)

deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CODECHAT_BISON_1,
]


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: ChatCompletionDeployment
    streaming: bool
    max_tokens: Optional[int]
    stop: Optional[List[str]]

    messages: List[BaseMessage]
    expected: Callable[[str], bool] | Exception

    def get_id(self):
        max_tokens_str = str(self.max_tokens) if self.max_tokens else "inf"
        stop_sequence_str = str(self.stop) if self.stop else "nonstop"
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {max_tokens_str} {stop_sequence_str} {self.name}"
        )


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
) -> List[TestCase]:
    ret: List[TestCase] = []

    ret.append(
        TestCase(
            name="2+3=5",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=None,
            messages=[user("2+3=?")],
            expected=lambda s: "5" in s,
        )
    )

    ret.append(
        TestCase(
            name="hello",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=None,
            messages=[user('Reply with "Hello"')],
            expected=lambda s: "hello" in s.lower(),
        )
    )

    ret.append(
        TestCase(
            name="empty sys message",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=None,
            messages=[sys(""), user("2+4=?")],
            expected=lambda s: "6" in s,
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
            expected=lambda s: len(s.split()) == 1,
        )
    )

    ret.append(
        TestCase(
            name="stop sequence",
            deployment=deployment,
            streaming=streaming,
            max_tokens=None,
            stop=["world"],
            messages=[user('Reply with "hello world"')],
            expected=Exception(
                "stop sequences are not supported for code chat model"
            )
            if deployment == ChatCompletionDeployment.CODECHAT_BISON_1
            else lambda s: "world" not in s.lower(),
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test_case
        for model in deployments
        for streaming in [False, True]
        for test_case in get_test_cases(model, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_langchain(server, test: TestCase):
    model = create_chat_model(
        TEST_SERVER_URL,
        test.deployment,
        test.streaming,
        test.max_tokens,
    )

    if isinstance(test.expected, Exception):
        with pytest.raises(Exception) as exc_info:
            await assert_dialog(
                model=model,
                messages=test.messages,
                output_predicate=lambda s: True,
                streaming=test.streaming,
                stop=test.stop,
            )

        assert isinstance(exc_info.value, openai.error.OpenAIError)
        assert exc_info.value.http_status == 422
        assert re.search(str(test.expected), str(exc_info.value))
    else:
        await assert_dialog(
            model=model,
            messages=test.messages,
            output_predicate=test.expected,
            streaming=test.streaming,
            stop=test.stop,
        )
