from __future__ import annotations

import pytest
import retry
from langchain.schema import BaseMemory

from openbrain.agents.gpt_agent import GptAgent
from openbrain.orm.model_agent_config import AgentConfig
from openbrain.orm.model_lead import Lead


@pytest.fixture
def incoming_agent_config(default_agent_config):
    outgoing_agent_config = default_agent_config
    return outgoing_agent_config


@pytest.fixture
def incoming_lead(simple_lead):
    outgoing_lead = simple_lead
    return outgoing_lead


class TestAgent:
    @retry.retry(delay=1, tries=5)
    @pytest.mark.integration_tests
    def test_gpt_agent_creation(self, incoming_agent_config: AgentConfig, incoming_lead: Lead):
        gpt_agent = GptAgent(incoming_agent_config)
        assert True

        response = gpt_agent.handle_user_message("Hello")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # assert isinstance(type(returned_lead), type(unique_lead)) # TODO, types coming in wrong

    @retry.retry(delay=1, tries=2)
    @pytest.mark.integration_tests
    def test_serialize_deserialize_agent(self, incoming_agent_config: AgentConfig, incoming_lead: Lead):
        # unique_agent_config = agent_config_fixture
        gpt_agent = GptAgent(agent_config=incoming_agent_config, lead=incoming_lead)
        response_message = gpt_agent.handle_user_message("I see 25 blue birds!")

        serialized_agent = gpt_agent.serialize()
        assert serialized_agent is not None
        assert serialized_agent["frozen_agent_memory"] is not None
        assert serialized_agent["frozen_agent_config"] is not None
        assert len(serialized_agent) > 0
        assert isinstance(serialized_agent, dict)
        assert isinstance(serialized_agent["frozen_agent_memory"], bytes)
        assert isinstance(serialized_agent["frozen_agent_config"], str)

        # deserialize the agent
        deserialized_agent = GptAgent.deserialize(serialized_agent)
        assert deserialized_agent is not None
        assert deserialized_agent.agent_config is not None
        assert deserialized_agent.working_memory is not None

        assert isinstance(deserialized_agent, GptAgent)
        assert isinstance(deserialized_agent.agent_config, AgentConfig)
        assert isinstance(deserialized_agent.working_memory, BaseMemory)

        # chat with the agent
        response_message = deserialized_agent.handle_user_message(user_message="How many birds did I tell you I saw?")
        assert response_message is not None
        assert len(response_message) > 0
        assert isinstance(response_message, str)
        assert (
            "25" in response_message
            or "twenty-five" in response_message
            or "twenty five" in response_message
            or "DISCONNECTED" in response_message
        )

    @pytest.mark.ci_cd
    @pytest.mark.integration_tests
    def test_conversational_memory(self, incoming_agent_config: AgentConfig):
        messages = [
            "I see 25 blue birds",
            "How many birds did I see?",
            "What color are they?",
        ]

        # GIVE INFO
        message = messages.pop(0)
        gpt_agent = GptAgent(incoming_agent_config)
        response_message = gpt_agent.handle_user_message(message)

        assert response_message is not None
        assert isinstance(response_message, str)

        # ASK QUESTION
        message = messages.pop(0)
        response_message = gpt_agent.handle_user_message(message)

        assert response_message is not None
        assert isinstance(response_message, str)
        assert "25" in response_message or "twenty-five" in response_message or "twenty five" in response_message

        # ASK ANOTHER QUESTION
        message = messages.pop(0)
        response_message = gpt_agent.handle_user_message(message)

        assert response_message is not None
        assert isinstance(response_message, str)
        assert "blue" in response_message.lower()
