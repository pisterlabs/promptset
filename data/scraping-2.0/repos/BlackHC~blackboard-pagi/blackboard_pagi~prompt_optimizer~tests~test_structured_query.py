#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import re

import pytest
from langchain.schema import AIMessage, HumanMessage, OutputParserException

from blackboard_pagi.prompt_optimizer.structured_query import structured_query
from blackboard_pagi.prompt_optimizer.track_hyperparameters import HyperparametersBuilder, track_hyperparameters
from blackboard_pagi.prompts.chat_chain import ChatChain
from blackboard_pagi.testing.fake_chat_model import FakeChatModel


def test_structured_query():
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(
                    content='Return 1 as string\n\nThe output should be formatted as a JSON instance that conforms to '
                    'the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": '
                    '"Foo", "description": "a list of strings", "type": "array", "items": {"type": '
                    '"string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a '
                    'well-formatted instance of the schema. The object {"properties": {"foo": ["bar", '
                    '"baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {'
                    '"result": {"title": "Result", "type": "string"}}, "required": ["result"]}\n```',
                    additional_kwargs={},
                ),
                AIMessage(content='{"result": "1"}', additional_kwargs={}),
            ]
        ]
    )

    def f():
        chain = ChatChain(chat_model, [])
        result, new_chain = structured_query(chain, "Return 1 as string", str)
        assert result == "1"
        assert len(new_chain.messages) == 2

    f()


def test_structured_query_retry():
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(
                    content='Return 1 as string\n\nThe output should be formatted as a JSON instance that conforms to '
                    'the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": '
                    '"Foo", "description": "a list of strings", "type": "array", "items": {"type": '
                    '"string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a '
                    'well-formatted instance of the schema. The object {"properties": {"foo": ["bar", '
                    '"baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {'
                    '"result": {"title": "Result", "type": "string"}}, "required": ["result"]}\n```',
                    additional_kwargs={},
                ),
                AIMessage(content='The result is: "1".', additional_kwargs={}),
                HumanMessage(
                    content='Tried to parse your last output but failed:\n\nFailed to parse StructuredOutput from '
                    'completion The result is: "1".. Got: Expecting value: line 1 column 1 (char 0)\n\nPlease '
                    'try again and avoid this issue.',
                    additional_kwargs={},
                ),
                AIMessage(content='My apologies. The result should be: {"result": "1"}', additional_kwargs={}),
            ]
        ]
    )

    def f():
        chain = ChatChain(chat_model, [])
        result, new_chain = structured_query(chain, "Return 1 as string", str)
        assert result == "1"
        assert len(new_chain.messages) == 4

    f()


def test_structured_query_retry_fail():
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(
                    content='Return 1 as string\n\nThe output should be formatted as a JSON instance that conforms to '
                    'the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": '
                    '"Foo", "description": "a list of strings", "type": "array", "items": {"type": '
                    '"string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a '
                    'well-formatted instance of the schema. The object {"properties": {"foo": ["bar", '
                    '"baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {'
                    '"result": {"title": "Result", "type": "string"}}, "required": ["result"]}\n```',
                    additional_kwargs={},
                ),
                AIMessage(content='The result is: "1".', additional_kwargs={}),
                HumanMessage(
                    content='Tried to parse your last output but failed:\n\nFailed to parse StructuredOutput from '
                    'completion The result is: "1".. Got: Expecting value: line 1 column 1 (char 0)\n\nPlease '
                    'try again and avoid this issue.',
                    additional_kwargs={},
                ),
                AIMessage(content='My apologies. The result should be: {"result": "1"}', additional_kwargs={}),
            ]
        ]
    )

    @track_hyperparameters
    def f():
        chain = ChatChain(chat_model, [])
        result, new_chain = structured_query(chain, "Return 1 as string", str)
        assert result == "1"
        assert len(new_chain.messages) == 4

    builder = HyperparametersBuilder()
    builder[structured_query]['num_retries_on_parser_failure'] = 0

    with pytest.raises(OutputParserException, match=re.escape("Failed to parse output")):
        with builder.scope():
            f()
