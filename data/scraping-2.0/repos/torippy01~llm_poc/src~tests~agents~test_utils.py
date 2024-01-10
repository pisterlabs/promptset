import pytest

from langchain.agents.agent_types import AgentType

from xecretary_core.agents.utils import agent_types_from_string


def test_agent_types_from_string():
    assert (
        agent_types_from_string("zero-shot-react-description")
        == AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    assert (
        agent_types_from_string("structured-chat-zero-shot-react-description")
        == AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    assert (
        agent_types_from_string("self-ask-with-search")
        == AgentType.SELF_ASK_WITH_SEARCH
    )

    assert agent_types_from_string("react-docstore") == AgentType.REACT_DOCSTORE

    assert (
        agent_types_from_string("openai-multi-functions")
        == AgentType.OPENAI_MULTI_FUNCTIONS
    )

    assert agent_types_from_string("openai-functions") == AgentType.OPENAI_FUNCTIONS

    assert (
        agent_types_from_string("conversational-react-description")
        == AgentType.CONVERSATIONAL_REACT_DESCRIPTION
    )

    assert (
        agent_types_from_string("chat-zero-shot-react-description")
        == AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    assert (
        agent_types_from_string("chat-conversational-react-description")
        == AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION
    )


def test_agent_type_from_string_error():
    with pytest.raises(ValueError) as e:
        agent_types_from_string("test")

    assert str(e.value) == "Invalid value : 'agent_type_str'"
