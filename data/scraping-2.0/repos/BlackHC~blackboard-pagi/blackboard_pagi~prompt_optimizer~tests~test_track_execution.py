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

import pytest
from langchain.schema import AIMessage, HumanMessage

from blackboard_pagi.prompt_optimizer.track_execution import TrackedChatModel
from blackboard_pagi.prompts.chat_chain import ChatChain
from blackboard_pagi.testing.fake_chat_model import FakeChatModel


def test_chat_chain():
    # Only test ChatChain
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(content="Hello"),
                AIMessage(content="World"),
                HumanMessage(content="How are you?"),
                AIMessage(content="Good. How are you?"),
            ],
            [
                HumanMessage(content="Hello"),
                AIMessage(content="World"),
                HumanMessage(content="What's up?"),
                AIMessage(content="Nothing. You?"),
            ],
        ]
    )

    tracked_chat_model = TrackedChatModel(chat_model=chat_model)

    chat_chain = ChatChain(tracked_chat_model, [HumanMessage(content="Hello")])

    assert chat_chain.response == "Hello"

    assert chat_chain.messages == [
        HumanMessage(content="Hello"),
    ]

    chat_chain_2 = ChatChain(tracked_chat_model, [])

    with pytest.raises(AssertionError):
        chat_chain_2.response

    assert chat_chain_2.messages == []

    response, chat_chain_3 = chat_chain_2.query("Hello")

    assert response == "World"

    assert chat_chain_3.messages == [
        HumanMessage(content="Hello"),
        AIMessage(content="World"),
    ]

    assert chat_chain_3.response == "World"
    assert tracked_chat_model.tracked_chats.build_compact_dict() == {
        'children': [],
        'messages': [{'content': 'Hello', 'role': 'user'}, {'content': 'World', 'role': 'assistant'}],
    }

    chat_chain_4 = chat_chain_3.branch()

    assert chat_chain_4.messages == [
        HumanMessage(content="Hello"),
        AIMessage(content="World"),
    ]

    response, chat_chain_5 = chat_chain_4.query("How are you?")
    assert response == "Good. How are you?"
    assert chat_chain_5.messages == [
        HumanMessage(content="Hello"),
        AIMessage(content="World"),
        HumanMessage(content="How are you?"),
        AIMessage(content="Good. How are you?"),
    ]

    assert tracked_chat_model.tracked_chats.build_compact_dict() == {
        'children': [],
        'messages': [
            {'content': 'Hello', 'role': 'user'},
            {'content': 'World', 'role': 'assistant'},
            {'content': 'How are you?', 'role': 'user'},
            {'content': 'Good. How are you?', 'role': 'assistant'},
        ],
    }

    response, chat_chain_6 = chat_chain_4.query("What's up?")
    assert response == "Nothing. You?"
    assert chat_chain_6.messages == [
        HumanMessage(content="Hello"),
        AIMessage(content="World"),
        HumanMessage(content="What's up?"),
        AIMessage(content="Nothing. You?"),
    ]

    assert tracked_chat_model.tracked_chats.build_compact_dict() == {
        'messages': [{'content': 'Hello', 'role': 'user'}, {'content': 'World', 'role': 'assistant'}],
        'children': [
            {
                'children': [],
                'messages': [
                    {'content': 'How are you?', 'role': 'user'},
                    {'content': 'Good. How are you?', 'role': 'assistant'},
                ],
            },
            {
                'children': [],
                'messages': [
                    {'content': "What's up?", 'role': 'user'},
                    {'content': 'Nothing. You?', 'role': 'assistant'},
                ],
            },
        ],
    }
